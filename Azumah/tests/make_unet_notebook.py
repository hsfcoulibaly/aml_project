"""Builds notebooks/02_unet_stage.ipynb (kept as code so the notebook stays in sync with the scripts)."""
import json
from pathlib import Path

EPOCHS, BATCH = 30, 4

cells = [
    ("md", """# Solar filament segmentation: stage 2 (dense U-Net, 5 folds)

Run on Kaggle with **GPU T4 x2** and **Internet on**. Attach the competition data and the output of the
stage-1 notebook (`filament-seg-yolo-stage`): its manifest, labels and preprocessed images are reused, so
nothing is prepared twice.

Trains one U-Net per fold, two folds at a time (one per GPU), then writes out-of-fold maps for all 707
training images (the refiner trains on them), scores the standalone U-Net with PQ on fold 0, and predicts
the test set with the 5-fold average."""),
    ("code", """# 1. Setup: code from the team repo, extra packages
import os, subprocess, sys
from pathlib import Path

CODE = Path('/kaggle/working/filament-seg')
if not CODE.exists():
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/hsfcoulibaly/aml_project.git',
                    '/tmp/aml_project'], check=True)
    subprocess.run(['cp', '-r', '/tmp/aml_project/Azumah', str(CODE)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch', 'timm',
                'pycocotools'], check=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.chdir(CODE)
sys.path.insert(0, str(CODE))
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """# 2. Reuse stage-1 outputs (read-only input -> symlinks in the writable work dir)
WORK = Path('/kaggle/working/fseg')
WORK.mkdir(parents=True, exist_ok=True)
src = next((p.parent for p in Path('/kaggle/input').rglob('manifest.csv')
            if (p.parent / 'references.pkl').exists()), None)
if src is not None:
    for name in ['manifest.csv', 'targets.pkl', 'references.pkl', 'preprocessed']:
        if not (WORK / name).exists():
            (WORK / name).symlink_to(src / name)
    print('reusing stage-1 outputs from', src)
else:
    print('stage-1 outputs not attached: preparing from scratch (~20 min)')
    !python scripts/prepare.py --modes consensus
os.environ['FSEG_WORK_DIR'] = str(WORK)"""),
    ("code", f"""# 3. Train the 5 fold models, two at a time (one per GPU). Rerunning resumes from last_state.pt.
import subprocess, time
EPOCHS, BATCH = {EPOCHS}, {BATCH}
os.makedirs('logs', exist_ok=True)

def launch(fold, gpu):
    cmd = (f'CUDA_VISIBLE_DEVICES={{gpu}} python scripts/train_unet.py --fold {{fold}} --epochs {{EPOCHS}} '
           f'--batch {{BATCH}} --workers 2 --pq-every 2 --patience 8 --resume > logs/unet_f{{fold}}.log 2>&1')
    return subprocess.Popen(cmd, shell=True)

t0 = time.time()
for pair in [(0, 1), (2, 3), (4,)]:
    procs = [launch(f, g) for g, f in enumerate(pair)]
    for p in procs:
        p.wait()
    for f in pair:
        print(f'fold {{f}}:', open(f'logs/unet_f{{f}}.log').read().strip().splitlines()[-1])
    print(f'elapsed {{(time.time() - t0) / 3600:.2f}} h')"""),
    ("code", """# 4. Out-of-fold maps for all 707 images + standalone U-Net PQ on fold 0
!python scripts/predict_unet.py oof --folds 0 1 2 3 4 --sweep-fold 0 --tta flip"""),
    ("code", """# 5. Test maps (5-fold average) + standalone U-Net submission
!python scripts/predict_unet.py test --folds 0 1 2 3 4 --config /kaggle/working/fseg/unet/runs/unet_f0/pq_config.json \\
    --out /kaggle/working/submission_unet.csv"""),
    ("code", """import pandas as pd
for f in range(5):
    h = pd.read_csv(f'/kaggle/working/fseg/unet/runs/unet_f{f}/history.csv')
    best = h.loc[h.val_pq.idxmax()] if 'val_pq' in h else None
    print(f'fold {f}: {len(h)} epochs, best val PQ {best.val_pq:.4f} at epoch {int(best.epoch)}' if best is not None else f)"""),
]

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": src} if kind == "md" else
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
        for kind, src in cells
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_unet_stage.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(out)
