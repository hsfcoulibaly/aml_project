"""Builds Charan/fusion/colab_fusion.ipynb: the refiner stage of fusion_unet2048_refiner.ipynb on Colab.

Data comes as one tar (fusion_pack.tar, output of the Kaggle CPU kernel charanpeeriga/filament-fusion-pack).
Refiner checkpoints, logs, pickles and submissions live on Google Drive (MyDrive/filament_fusion), so a
disconnect costs at most one epoch: rerun all cells and every step resumes or skips what is done.

    python Charan/fusion/make_colab_notebook.py
"""
import json
from pathlib import Path

BRANCH, EPOCHS, TAG = "charan/fusion-self", 60, "u2r"

cells = [
    ("md", f"""# U-Net-2048 proposals + instance refiner on Colab (5 folds + full, pooled OOF)

**Runtime:** GPU (A100 if available, else L4), **High-RAM** on. Then *Runtime > Run all*.

**Kaggle token**, one of:
* Colab secret `KAGGLE_API_TOKEN` (new-style token), or secrets `KAGGLE_USERNAME` + `KAGGLE_KEY`; notebook access on; or
* `kaggle.json` in the root of your Google Drive.

Everything is written to `MyDrive/filament_fusion/`. If the runtime disconnects, reconnect and *Run all*
again: training resumes from the last epoch, finished steps are skipped."""),
    ("code", """# 1. Drive, Kaggle token, packages
import os, sys, json, subprocess, time, shutil
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive/filament_fusion')
for d in ['refiner', 'logs', 'results']:
    (DRIVE / d).mkdir(parents=True, exist_ok=True)
from google.colab import userdata

def secret(k):
    try:
        return userdata.get(k)
    except Exception:
        return None

if secret('KAGGLE_API_TOKEN'):  # new-style token (Kaggle settings > API > Generate New Token)
    os.environ['KAGGLE_API_TOKEN'] = secret('KAGGLE_API_TOKEN')
elif secret('KAGGLE_USERNAME') and secret('KAGGLE_KEY'):  # legacy kaggle.json values
    os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'] = secret('KAGGLE_USERNAME'), secret('KAGGLE_KEY')
else:
    tok = Path('/content/drive/MyDrive/kaggle.json')
    assert tok.exists(), 'add a KAGGLE_API_TOKEN secret (or KAGGLE_USERNAME + KAGGLE_KEY), or put kaggle.json in MyDrive'
    Path('~/.kaggle').expanduser().mkdir(exist_ok=True)
    shutil.copy(tok, Path('~/.kaggle/kaggle.json').expanduser())
    os.chmod(Path('~/.kaggle/kaggle.json').expanduser(), 0o600)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle>=1.8', 'segmentation-models-pytorch', 'timm',
                'pycocotools'], check=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv
import psutil
RAM_GB = psutil.virtual_memory().total / 2**30
print(f'RAM {RAM_GB:.0f} GB, CPUs {os.cpu_count()}')"""),
    ("code", f"""# 2. Code (Azumah/ from main, unchanged) and data pack from Kaggle
REPO = Path('/content/aml_project')
if not REPO.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', '{BRANCH}',
                    'https://github.com/hsfcoulibaly/aml_project.git', str(REPO)], check=True)
print(subprocess.run(['git', '-C', str(REPO), 'log', '-1', '--oneline'], capture_output=True, text=True).stdout)
CODE = REPO / 'Azumah'
WORK = Path('/content/fseg')
if not (WORK / 'manifest.csv').exists():
    !kaggle kernels output charanpeeriga/filament-fusion-pack -p /content/pack --file-pattern 'fusion_pack.tar'
    !tar -xf /content/pack/fusion_pack.tar -C /content && rm /content/pack/fusion_pack.tar
if not (WORK / 'refiner').is_symlink():
    shutil.rmtree(WORK / 'refiner', ignore_errors=True)
    (WORK / 'refiner').symlink_to(DRIVE / 'refiner')
os.environ.update(FSEG_WORK_DIR=str(WORK), FSEG_DATA_ROOT='/content/no_raw_data_needed',
                  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
os.chdir(CODE)
print('maps', len(list((WORK / 'unet/maps/unet2048/oof').glob('*.png'))), len(list((WORK / 'unet/maps/unet2048/test').glob('*.png'))),
      '| proposals', sorted(p.name for p in (WORK / 'runs').glob('*/*.pkl')))"""),
    ("code", f"""# 3. Train 6 refiners ({EPOCHS} epochs; folds 0-4 held out + full data). Two at a time with High-RAM.
def name(k):
    return '{TAG}_full' if k == 'none' else f'{TAG}_v{{k}}'

def done(k):
    h = DRIVE / 'refiner' / name(k) / 'history.csv'
    return h.exists() and len(h.read_text().strip().splitlines()) - 1 >= {EPOCHS}

def cmd(k):
    return (f'python scripts/train_refiner.py --name {{name(k)}} --val-fold {{k}} --unet-name unet2048 --arch unet '
            f'--epochs {EPOCHS} --workers 4 --max-matched 12 --max-neg 8 --resume >> {{DRIVE}}/logs/{{name(k)}}.log 2>&1')

todo = [k for k in ['0', '1', '2', '3', '4', 'none'] if not done(k)]
parallel = 2 if RAM_GB > 40 else 1
print('to train:', todo, '| parallel', parallel)
T0 = time.time()
running = []
while todo or running:
    while todo and len(running) < parallel:
        k = todo.pop(0)
        running.append((k, subprocess.Popen(cmd(k), shell=True)))
    time.sleep(120)
    for k, p in list(running):
        if p.poll() is not None:
            running.remove((k, p))
            if p.returncode:
                print(f'{{name(k)}} FAILED, see logs/{{name(k)}}.log'); print(open(f'{{DRIVE}}/logs/{{name(k)}}.log').read()[-3000:])
    for k, _ in running:
        lines = open(f'{{DRIVE}}/logs/{{name(k)}}.log').read().strip().splitlines()
        if lines:
            print(f'[{{(time.time() - T0) / 60:5.0f}} min] {{name(k)}}: {{lines[-1][:160]}}', flush=True)
print('all refiners trained:', all(done(k) for k in ['0', '1', '2', '3', '4', 'none']))"""),
    ("code", f"""# 4. Out-of-fold refinement of all 707 images (each fold by the refiner that never saw it), pooled PQ
for k in range(5):
    if not (DRIVE / 'refiner' / f'{TAG}_f{{k}}' / f'refined_fold{{k}}.pkl').exists():
        !python scripts/predict_refiner.py oof --fold {{k}} --refiner {TAG}_v{{k}} --tag {TAG}_f{{k}} --unet-name unet2048 --tta d4 > {{DRIVE}}/logs/oof_f{{k}}.log 2>&1
    print(k, open(f'{{DRIVE}}/logs/oof_f{{k}}.log').read().strip().splitlines()[-1][:250])
!python scripts/predict_refiner.py pool --tag {TAG} --folds 0 1 2 3 4 --tta d4 2>&1 | tee {{DRIVE}}/logs/pool.log"""),
    ("code", f"""# 5. Test: 6-refiner vote with the pooled config, and the full-data refiner alone -> Drive
P = WORK / 'refiner' / '{TAG}_pooled'
cfg = P / 'pq_config.json'
raw = WORK / 'runs' / 'unet2048_test' / 'test_raw.pkl'
six = ' '.join([f'{TAG}_v{{k}}' for k in range(5)] + ['{TAG}_full'])
R = DRIVE / 'results'
!python scripts/predict_refiner.py test --refiner {{six}} --config {{cfg}} --raw {{raw}} --unet-name unet2048 --tag test_{TAG}x6 --out {{R}}/submission_{TAG}x6_pooled.csv 2>&1 | tail -2
!python scripts/predict_refiner.py test --refiner {TAG}_full --config {{cfg}} --raw {{raw}} --unet-name unet2048 --tag test_{TAG}full --out {{R}}/submission_{TAG}full_pooled.csv 2>&1 | tail -2
shutil.copy(cfg, R / 'pq_config_pooled.json')
shutil.copy(P / 'per_fold.csv', R / 'per_fold.csv')
shutil.copy(P / 'pq_sweep_pooled.csv', R / 'pq_sweep_pooled.csv')
print(open(cfg).read()); print(open(P / 'per_fold.csv').read())
!ls -la {{R}}"""),
]


def to_nb(cells):
    out = []
    for kind, src in cells:
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": src.splitlines(True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        out.append(cell)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU",
                                       "colab": {"machine_shape": "hm", "gpuType": "A100"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "colab_fusion.ipynb"
    path.write_text(json.dumps(to_nb(cells), indent=1))
    print("wrote", path)
