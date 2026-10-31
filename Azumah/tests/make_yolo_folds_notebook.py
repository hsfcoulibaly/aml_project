"""Builds notebooks/01b_yolo_folds.ipynb: YOLO on extra folds, out-of-fold proposals for the refiner.

    python tests/make_yolo_folds_notebook.py 1 2      # folds to train (default 1 2)
"""
import json
import sys
from pathlib import Path

folds = [int(f) for f in sys.argv[1:]] or [1, 2]

cells = [
    ("md", f"""# Solar filament segmentation: YOLO folds {", ".join(map(str, folds))}

Run on Kaggle with **GPU T4 x2** and **Internet on**. Attach the competition data and the output of the
stage-1 notebook (`filament-seg-yolo-stage`), whose manifest, labels, preprocessed images and YOLO
datasets are reused. Same settings as fold 0 (YOLO11l-seg, 1280 px, 60 epochs, batch 4 on 2x T4).

Each fold writes `runs/consensus_f<k>/oof_raw_fold<k>.pkl`: proposals for images that fold's model never
saw. With all five folds, the refiner trains on real YOLO mistakes for every training image."""),
    ("code", """# 1. Setup
import os, subprocess, sys
from pathlib import Path

CODE = Path('/kaggle/working/filament-seg')
if not CODE.exists():
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/hsfcoulibaly/aml_project.git',
                    '/tmp/aml_project'], check=True)
    subprocess.run(['cp', '-r', '/tmp/aml_project/Azumah', str(CODE)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics', 'pycocotools'], check=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.chdir(CODE)
sys.path.insert(0, str(CODE))"""),
    ("code", """# 2. Reuse stage-1 outputs. The YOLO yamls hold absolute /kaggle/working/fseg paths, so link the
# read-only input into that location; labels.cache cannot be written there, which Ultralytics tolerates.
WORK = Path('/kaggle/working/fseg')
WORK.mkdir(parents=True, exist_ok=True)
src = next((p.parent for p in Path('/kaggle/input').rglob('manifest.csv')
            if (p.parent / 'yolo' / 'consensus').exists()), None)
if src is not None:
    for name in ['manifest.csv', 'targets.pkl', 'references.pkl', 'preprocessed', 'yolo']:
        if not (WORK / name).exists():
            (WORK / name).symlink_to(src / name)
    print('reusing stage-1 outputs from', src)
else:
    print('stage-1 outputs not attached: preparing from scratch (~20 min)')
    !python scripts/prepare.py --size 1280 --modes consensus"""),
    ("code", f"""# 3. Train each fold, then predict its validation images (out-of-fold proposals + PQ sweep)
FOLDS = {folds}
for k in FOLDS:
    !python scripts/train_yolo.py --mode consensus --fold {{k}} --model yolo11l-seg.pt --imgsz 1280 --epochs 60 --batch 4 --resume
    !python scripts/predict_yolo.py oof --weights /kaggle/working/fseg/runs/consensus_f{{k}}/weights/best.pt --fold {{k}} --imgsz 1280"""),
    ("code", """import json
for k in FOLDS:
    d = Path(f'/kaggle/working/fseg/runs/consensus_f{k}')
    print(k, json.loads((d / 'pq_config.json').read_text()) if (d / 'pq_config.json').exists() else 'missing')"""),
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
out = Path(__file__).resolve().parents[1] / "notebooks" / "01b_yolo_folds.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out)
