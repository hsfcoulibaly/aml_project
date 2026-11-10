"""Builds notebooks/02b_unet_native.ipynb: stage 2 at native resolution (2048 px image, 1024 px crops).

Same model, loss and 5 folds as 02_unet_stage.ipynb; only the input resolution changes. The whole-disk
model sees the image downsampled to 1024 (median filament width ~5 px); this one sees it at 2048
(~11 px), which raised standalone validation PQ by 0.02 to 0.03 for a ResNet-34 U-Net on a 7-day split.
Maps go to work/unet/maps/unet2048/, so stages 3 and 4 select them with --unet-name unet2048.
"""
import json
from pathlib import Path

NAME, BRANCH = "unet2048", "charan/fusion-native-unet"
EPOCHS, BATCH, CROP, CROPS_PER_IMAGE = 30, 4, 1024, 2

cells = [
    ("md", f"""# Solar filament segmentation: stage 2 at native resolution (5 folds)

Run on Kaggle with **GPU T4 x2** and **Internet on**, competition data attached. Prepares the data from
scratch (folds are deterministic, seed 2026, so they match every other stage).

Trains the stage-2 U-Net (ConvNeXt-tiny + ASPP, filament / centerline / boundary) on random {CROP} px
crops of the 2048 px image ({CROPS_PER_IMAGE} crops per image per epoch, 70% centred on a filament),
two folds at a time, then writes out-of-fold maps for all 707 images, the fold-0 standalone PQ, and
5-fold-average test maps under `maps/{NAME}/`."""),
    ("code", f"""# 1. Setup: code from the team repo, extra packages
import os, subprocess, sys, time
from pathlib import Path

T0 = time.time()
CODE = Path('/kaggle/working/filament-seg')
if not CODE.exists():
    subprocess.run(['git', 'clone', '--depth', '1', '--branch', '{BRANCH}',
                    'https://github.com/hsfcoulibaly/aml_project.git', '/tmp/aml_project'], check=True)
    subprocess.run(['cp', '-r', '/tmp/aml_project/Azumah', str(CODE)], check=True)
    print(subprocess.run(['git', '-C', '/tmp/aml_project', 'log', '-1', '--oneline'],
                         capture_output=True, text=True).stdout)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch', 'timm',
                'pycocotools'], check=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.chdir(CODE)
sys.path.insert(0, str(CODE))
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """# 2. Stage 0 data: manifest, folds, labels, preprocessed images (~20 min)
WORK = Path('/kaggle/working/fseg')
WORK.mkdir(parents=True, exist_ok=True)
os.environ['FSEG_WORK_DIR'] = str(WORK)
if not (WORK / 'references.pkl').exists():
    !python scripts/prepare.py --modes consensus
import pandas as pd
print(pd.read_csv(WORK / 'manifest.csv').fold.value_counts().sort_index().to_dict())
print(f'elapsed {(time.time() - T0) / 3600:.2f} h')"""),
    ("code", f"""# 3. Train the 5 fold models, two at a time (one per GPU). Rerunning resumes from last_state.pt.
os.makedirs('logs', exist_ok=True)

def launch(fold, gpu):
    cmd = (f'CUDA_VISIBLE_DEVICES={{gpu}} python scripts/train_unet.py --fold {{fold}} --name {NAME}_f{{fold}} '
           f'--size 2048 --crop {CROP} --crops-per-image {CROPS_PER_IMAGE} --epochs {EPOCHS} --batch {BATCH} '
           f'--workers 2 --pq-every 2 --patience 8 --resume > logs/{NAME}_f{{fold}}.log 2>&1')
    return subprocess.Popen(cmd, shell=True)

for pair in [(0, 1), (2, 3), (4,)]:
    procs = [launch(f, g) for g, f in enumerate(pair)]
    for p in procs:
        p.wait()
    for f in pair:
        print(f'fold {{f}}:', open(f'logs/{NAME}_f{{f}}.log').read().strip().splitlines()[-1])
    print(f'elapsed {{(time.time() - T0) / 3600:.2f}} h', flush=True)"""),
    ("code", f"""# 4. Out-of-fold maps for all 707 images + standalone U-Net PQ on fold 0
!python scripts/predict_unet.py oof --name {NAME} --folds 0 1 2 3 4 --sweep-fold 0 --tta flip
print(f'elapsed {{(time.time() - T0) / 3600:.2f}} h')"""),
    ("code", f"""# 5. Test maps (5-fold average) + standalone U-Net submission
!python scripts/predict_unet.py test --name {NAME} --folds 0 1 2 3 4 --tta flip \\
    --config /kaggle/working/fseg/unet/runs/{NAME}_f0/pq_config.json --out /kaggle/working/submission_{NAME}.csv
print(f'elapsed {{(time.time() - T0) / 3600:.2f}} h')"""),
    ("code", f"""# 6. Summary; drop the YOLO export (not needed downstream) to keep the output small
import shutil
shutil.rmtree(WORK / 'yolo', ignore_errors=True)
for f in range(5):
    h = pd.read_csv(WORK / 'unet' / 'runs' / f'{NAME}_f{{f}}' / 'history.csv')
    b = h.loc[h.val_pq.idxmax()]
    print(f'fold {{f}}: {{len(h)}} epochs, best val PQ {{b.val_pq:.4f}} at epoch {{int(b.epoch)}}, '
          f'{{h.epoch_s.mean() / 60:.1f}} min/epoch')
print('whole-disk 1024 reference (BEST_PIPELINE.md): U-Net alone fold-0 OOF PQ 0.406')
print(pd.read_csv(WORK / 'unet' / 'runs' / '{NAME}_f0' / 'pq_sweep_fold0.csv').head(3).to_string())
print(f'total {{(time.time() - T0) / 3600:.2f}} h')"""),
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
    path = Path(__file__).resolve().parents[1] / "notebooks" / "02b_unet_native.ipynb"
    path.write_text(json.dumps(to_nb(cells), indent=1))
    print("wrote", path)
