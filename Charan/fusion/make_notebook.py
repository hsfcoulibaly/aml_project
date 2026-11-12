"""Builds Charan/fusion/fusion_unet2048_refiner.ipynb (Kaggle, GPU T4 x2, Internet on).

Self-contained second system: proposals from Charan's native-2048 U-Net (5-fold OOF + test maps from the
kernel charanpeeriga/filament-seg-unet2048), refined by the stage-3 instance refiner from Azumah/ (code from
main, used unchanged), trained once per held-out fold so all 707 images get an out-of-fold prediction.
Settings are tuned on the pooled 707-image OOF PQ; the test set is refined by the 5 fold refiners + the
full-data refiner voting together.

    python Charan/fusion/make_notebook.py
"""
import json
from pathlib import Path

BRANCH, EPOCHS, TAG = "charan/fusion-self", 60, "u2r"

cells = [
    ("md", f"""# Native-2048 U-Net proposals + instance refiner (5 folds, pooled OOF tuning)

Inputs: competition data and the output of `charanpeeriga/filament-seg-unet2048` (prepared data, folds,
labels, 5-fold OOF and test U-Net maps). Code: `Azumah/` from branch `{BRANCH}` (identical to main), plus
`Charan/fusion/make_unet_proposals.py`. No YOLO and no outputs from any other kernel.

1. U-Net maps -> instance proposals (components at thresholds 0.3/0.5/0.7, deduplicated) in the stage-1 format.
2. ResNet-34 refiner, {EPOCHS} epochs, one per held-out fold + one on all data, U-Net channels = unet2048 maps.
3. Out-of-fold refinement of every training image, pooled 707-image PQ sweep (ranking, threshold, min area).
4. Test: 6-refiner vote, D4 TTA, pooled config."""),
    ("code", f"""# 1. Setup
import os, subprocess, sys, json, time
from pathlib import Path

T0 = time.time()
REPO = Path('/tmp/aml_project')
if not REPO.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', '{BRANCH}',
                    'https://github.com/hsfcoulibaly/aml_project.git', str(REPO)], check=True)
print(subprocess.run(['git', '-C', str(REPO), 'log', '-1', '--oneline'], capture_output=True, text=True).stdout)
CODE = Path('/kaggle/working/filament-seg')
if not CODE.exists():
    subprocess.run(['cp', '-r', str(REPO / 'Azumah'), str(CODE)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch', 'timm',
                'pycocotools'], check=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.chdir(CODE)
sys.path.insert(0, str(CODE))
os.makedirs('logs', exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """# 2. Link the U-Net-2048 run into one work dir
WORK = Path('/kaggle/working/fseg')
(WORK / 'unet' / 'maps').mkdir(parents=True, exist_ok=True)
inp = Path('/kaggle/input')
base = next(p.parent for p in inp.rglob('manifest.csv') if (p.parent / 'references.pkl').exists())
for name in ['manifest.csv', 'targets.pkl', 'references.pkl', 'preprocessed']:
    if not (WORK / name).exists():
        (WORK / name).symlink_to(base / name)
maps = next(p for p in inp.rglob('maps/unet2048') if (p / 'oof').exists() and (p / 'test').exists())
if not (WORK / 'unet' / 'maps' / 'unet2048').exists():
    (WORK / 'unet' / 'maps' / 'unet2048').symlink_to(maps)
os.environ['FSEG_WORK_DIR'] = str(WORK)
print('base', base, '| maps', maps, len(list((maps / 'oof').glob('*.png'))), len(list((maps / 'test').glob('*.png'))))"""),
    ("code", """# 3. Proposals from the U-Net maps (CPU, ~10 min)
!python {REPO}/Charan/fusion/make_unet_proposals.py --code {CODE} --maps {WORK}/unet/maps/unet2048 --workers 4
print(f'elapsed {(time.time() - T0) / 3600:.2f} h')"""),
    ("code", f"""# 4. Train 6 refiners (folds 0-4 held out + full data), three per GPU in sequence
JOBS = {{0: ['0', '1', '2'], 1: ['3', '4', 'none']}}
name = lambda k: '{TAG}_full' if k == 'none' else f'{TAG}_v{{k}}'
procs = []
for gpu, folds in JOBS.items():
    chain = '; '.join(f'python scripts/train_refiner.py --name {{name(k)}} --val-fold {{k}} --unet-name unet2048 '
                      f'--arch unet --epochs {EPOCHS} --workers 2 --max-matched 12 --max-neg 8 --resume '
                      f'> logs/{{name(k)}}.log 2>&1' for k in folds)
    procs.append(subprocess.Popen(f'CUDA_VISIBLE_DEVICES={{gpu}} bash -c "{{chain}}"', shell=True))
for p in procs:
    p.wait()
for k in ['0', '1', '2', '3', '4', 'none']:
    print(k, open(f'logs/{{name(k)}}.log').read().strip().splitlines()[-1])
print(f'elapsed {{(time.time() - T0) / 3600:.2f}} h')"""),
    ("code", f"""# 5. Out-of-fold refinement of all 707 images (each fold by the refiner that never saw it)
JOBS = {{0: [0, 1, 2], 1: [3, 4]}}
procs = []
for gpu, folds in JOBS.items():
    chain = '; '.join(f'python scripts/predict_refiner.py oof --fold {{k}} --refiner {TAG}_v{{k}} --tag {TAG}_f{{k}} '
                      f'--unet-name unet2048 --tta d4 > logs/oof_f{{k}}.log 2>&1' for k in folds)
    procs.append(subprocess.Popen(f'CUDA_VISIBLE_DEVICES={{gpu}} bash -c "{{chain}}"', shell=True))
for p in procs:
    p.wait()
for k in range(5):
    print(k, open(f'logs/oof_f{{k}}.log').read().strip().splitlines()[-1][:250])
print(f'elapsed {{(time.time() - T0) / 3600:.2f}} h')"""),
    ("code", f"""# 6. Pooled 707-image OOF PQ ('yolo' column = U-Net-2048 alone, thr .5 / area 400 / mean .7)
!python scripts/predict_refiner.py pool --tag {TAG} --folds 0 1 2 3 4 --tta d4"""),
    ("code", f"""# 7. Test: 6-refiner vote with the pooled config, and the full-data refiner alone
P = WORK / 'refiner' / '{TAG}_pooled'
cfg = P / 'pq_config.json'
raw = WORK / 'runs' / 'unet2048_test' / 'test_raw.pkl'
six = ' '.join([f'{TAG}_v{{k}}' for k in range(5)] + ['{TAG}_full'])
!python scripts/predict_refiner.py test --refiner {{six}} --config {{cfg}} --raw {{raw}} --unet-name unet2048 --tag test_{TAG}x6 --out /kaggle/working/submission_{TAG}x6_pooled.csv
!python scripts/predict_refiner.py test --refiner {TAG}_full --config {{cfg}} --raw {{raw}} --unet-name unet2048 --tag test_{TAG}full --out /kaggle/working/submission_{TAG}full_pooled.csv
print(open(cfg).read())
print(open(P / 'per_fold.csv').read())
print(f'total {{(time.time() - T0) / 3600:.2f}} h')"""),
    ("code", """# 8. Keep the output small: drop symlinks and crop caches, keep weights, pickles, logs, CSVs
import shutil
for name in ['manifest.csv', 'targets.pkl', 'references.pkl', 'preprocessed']:
    (WORK / name).unlink(missing_ok=True)
(WORK / 'unet' / 'maps' / 'unet2048').unlink(missing_ok=True)
for f in WORK.rglob('last_state.pt'):
    f.unlink()
!du -sh /kaggle/working/* ; ls -la /kaggle/working/*.csv"""),
]


def to_nb(cells):
    out = []
    for kind, src in cells:
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": src.splitlines(True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        out.append(cell)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "fusion_unet2048_refiner.ipynb"
    path.write_text(json.dumps(to_nb(cells), indent=1))
    print("wrote", path)
