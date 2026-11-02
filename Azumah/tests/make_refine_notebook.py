"""Builds notebooks/03_refine_fuse.ipynb (kept as code so the notebook stays in sync with the scripts)."""
import json
from pathlib import Path

cells = [
    ("md", """# Solar filament segmentation: stages 3 and 4 (refiner + fusion)

Run on Kaggle with **GPU T4 x2** and **Internet on**. Attach the competition data and the outputs of
`filament-seg-yolo-stage`, `filament-seg-yolo-folds-1-2`, `filament-seg-yolo-folds-3-4` and
`filament-seg-unet-stage`: out-of-fold YOLO proposals and U-Net maps for all 707 training images, plus
test-set proposals and maps. Nothing upstream is retrained.

1. Train the refiner twice in parallel: `refiner_v0` (folds 1-4, validated on fold 0) for evaluation and
   `refiner_full` (all folds) for the test set.
2. Fold-0 PQ: stage 1 vs stage 3 on the same images.
3. Fusion sweeps on fold 0 (refined instances, and YOLO proposals as an ablation).
4. Test submissions: `submission_refiner.csv`, `submission_fused.csv`, `submission_yolo_fused.csv`."""),
    ("code", """# 1. Setup
import os, subprocess, sys, json, shutil
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
sys.path.insert(0, str(CODE))"""),
    ("code", """# 2. Link every upstream output into one work dir (inputs are read-only; new results go next to them)
WORK = Path('/kaggle/working/fseg')
(WORK / 'runs').mkdir(parents=True, exist_ok=True)
(WORK / 'unet' / 'maps').mkdir(parents=True, exist_ok=True)
inp = Path('/kaggle/input')
base = next(p.parent for p in inp.rglob('manifest.csv') if (p.parent / 'references.pkl').exists())
for name in ['manifest.csv', 'targets.pkl', 'references.pkl', 'preprocessed']:
    if not (WORK / name).exists():
        (WORK / name).symlink_to(base / name)
for run in sorted({p.parent for p in inp.rglob('oof_raw_fold*.pkl')} | {p.parent for p in inp.rglob('test_raw.pkl')}):
    dst = WORK / 'runs' / run.name
    if not dst.exists():
        dst.symlink_to(run)
maps = next(p for p in inp.rglob('maps/unet') if (p / 'oof').exists())
if not (WORK / 'unet' / 'maps' / 'unet').exists():
    (WORK / 'unet' / 'maps' / 'unet').symlink_to(maps)
os.environ['FSEG_WORK_DIR'] = str(WORK)
print('runs:', sorted(p.name for p in (WORK / 'runs').iterdir()))
print('U-Net oof maps:', len(list((maps / 'oof').glob('*.png'))), '| test maps:', len(list((maps / 'test').glob('*.png'))))"""),
    ("code", """# 3. Train both refiners at once, one per GPU (rerunning resumes)
procs = []
for gpu, vf in [(0, '0'), (1, 'none')]:
    log = f'logs/refiner_{vf}.log'
    os.makedirs('logs', exist_ok=True)
    procs.append(subprocess.Popen(f'CUDA_VISIBLE_DEVICES={gpu} python scripts/train_refiner.py --val-fold {vf} '
                                  f'--epochs 30 --workers 2 --resume > {log} 2>&1', shell=True))
for p in procs:
    p.wait()
for vf in ['0', 'none']:
    print(vf, open(f'logs/refiner_{vf}.log').read().strip().splitlines()[-1])"""),
    ("code", """# 4. Stage 3 on fold 0: refine all out-of-fold proposals, calibrate, compare with stage 1
!python scripts/predict_refiner.py oof --fold 0"""),
    ("code", """# 5. Stage 4 sweeps on fold 0 (refined instances; YOLO proposals as the no-refiner ablation)
!python scripts/fuse.py oof --source refiner --fold 0 --workers 4
!python scripts/fuse.py oof --source yolo --fold 0 --workers 4"""),
    ("code", """# 6. Test submissions with the calibrated configs
shutil.copy(WORK / 'refiner' / 'refiner_v0' / 'pq_config.json', WORK / 'refiner' / 'refiner_full' / 'pq_config.json')
!python scripts/predict_refiner.py test --config /kaggle/working/fseg/refiner/refiner_v0/pq_config.json --out /kaggle/working/submission_refiner.csv
!python scripts/fuse.py test --source refiner --config /kaggle/working/fseg/fusion/refiner_f0/fusion_config.json --out /kaggle/working/submission_fused.csv --workers 4
!python scripts/fuse.py test --source yolo --config /kaggle/working/fseg/fusion/yolo_f0/fusion_config.json --out /kaggle/working/submission_yolo_fused.csv --workers 4"""),
    ("code", """# 7. Summary: fold-0 PQ of every variant (same 142 images)
import pandas as pd
rows = []
s3 = pd.read_csv(WORK / 'refiner' / 'refiner_v0' / 'pq_sweep_fold0.csv').iloc[0]
rows.append(('stage 3: YOLO + refiner', s3.pq, s3.sq, s3.rq))
for src, label in [('refiner', 'stage 4: YOLO + refiner + fusion'), ('yolo', 'YOLO + fusion (no refiner)')]:
    t = pd.read_csv(WORK / 'fusion' / f'{src}_f0' / 'fusion_sweep_fold0.csv')
    rows.append((label, t.pq.max(), t.loc[t.pq.idxmax(), 'sq'], t.loc[t.pq.idxmax(), 'rq']))
    base = t[t.step == 'baseline'].iloc[0]
    rows.append((f'  {src} before fusion', base.pq, base.sq, base.rq))
print(pd.DataFrame(rows, columns=['variant', 'pq', 'sq', 'rq']).round(4).to_string(index=False))"""),
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
out = Path(__file__).resolve().parents[1] / "notebooks" / "03_refine_fuse.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out)
