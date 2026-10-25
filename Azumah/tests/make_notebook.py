"""Builds notebooks/01_yolo_stage.ipynb (kept as code so the notebook stays in sync with the scripts)."""
import json
from pathlib import Path

cells = [
    ("md", """# Solar filament segmentation: stage 0 + stage 1 (YOLO proposer)

Run on Kaggle with **GPU T4 x2** and **Internet on**. Each heavy step is its own cell so you can run
them in separate notebook versions if you hit the 12-hour limit.

Order: setup -> prepare -> train fold 0 -> out-of-fold PQ -> (dilation ablation) -> final model -> submission."""),
    ("code", """# 1. Setup: get the code (either an attached Kaggle dataset named filament-seg or your GitHub repo)
import os, subprocess, sys
from pathlib import Path

CODE = Path('/kaggle/working/filament-seg')
attached = next(Path('/kaggle/input').rglob('fseg/__init__.py'), None)
if attached:
    subprocess.run(['cp', '-r', str(attached.parents[1]), str(CODE)], check=True)
elif not CODE.exists():
    REPO = 'https://github.com/Ewenam/filament-seg.git'   # <- change to your team's repo
    subprocess.run(['git', 'clone', REPO, str(CODE)], check=True)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics', 'pycocotools'], check=True)
os.chdir(CODE)
sys.path.insert(0, str(CODE))
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """# 2. Stage 0: folds, consensus targets, human ceiling, preprocessing, YOLO export (~15-25 min)
!python scripts/prepare.py --size 1280 --modes consensus largest"""),
    ("md", """Check the printed **human agreement ceiling**: it is the PQ of one annotator scored against the others,
the natural reference line for the report. Then eyeball a preprocessed image:"""),
    ("code", """import cv2, matplotlib.pyplot as plt, pandas as pd
m = pd.read_csv('/kaggle/working/fseg/manifest.csv')
img = cv2.imread(f"/kaggle/working/fseg/preprocessed/train/{m.observation_id[0]}.jpg")
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
for i, t in enumerate(['raw', 'limb-flattened', 'CLAHE']):
    ax[i].imshow(img[:, :, i], cmap='gray'); ax[i].set_title(t); ax[i].axis('off')
plt.show()"""),
    ("code", """# 3. Train the baseline proposer on fold 0 (~1.5-2.5 h on 2xT4). Lower --batch to 4 if you hit OOM.
!python scripts/train_yolo.py --mode consensus --fold 0 --model yolo11l-seg.pt --imgsz 1280 --epochs 60 --batch 8"""),
    ("code", """# 4. Out-of-fold PQ on fold 0: sweeps confidence and min-area thresholds for PQ, saves pq_config.json
!python scripts/predict_yolo.py oof --weights /kaggle/working/fseg/runs/consensus_f0/weights/best.pt --fold 0 --imgsz 1280"""),
    ("md", """### Ablations (run as separate versions, compare the OOF PQ line)
* labels: `--mode largest` vs `--mode consensus`
* dilation (your teammate's idea, zero-init branch in the neck): `--dilated p4,p5` or `--dilated p3,p4,p5`
* TTA: `--tta id` vs the default four flips"""),
    ("code", """# 5. (optional) dilation ablation on the same fold
# !python scripts/train_yolo.py --mode consensus --fold 0 --dilated p4,p5 --epochs 60 --batch 8
# !python scripts/predict_yolo.py oof --weights /kaggle/working/fseg/runs/consensus_f0_dilp4p5/weights/best.pt --fold 0"""),
    ("code", """# 6. Final model on all training images, then predict the test set with the fold-0 calibrated thresholds
!python scripts/train_yolo.py --mode consensus --fold full --model yolo11l-seg.pt --imgsz 1280 --epochs 60 --batch 8
!python scripts/predict_yolo.py test --weights /kaggle/working/fseg/runs/consensus_ffull/weights/best.pt \\
    --config /kaggle/working/fseg/runs/consensus_f0/pq_config.json --imgsz 1280"""),
    ("code", """import pandas as pd
sub = pd.read_csv('/kaggle/working/submission.csv')
print(sub.shape, sub.filament_id.str.rsplit('_', n=1).str[0].nunique(), 'images')
sub.head()"""),
]

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": src}
        if kind == "md"
        else {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
        for kind, src in cells
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "01_yolo_stage.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(out)
