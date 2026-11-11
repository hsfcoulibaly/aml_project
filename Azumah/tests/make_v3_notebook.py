"""Builds notebooks/05_refiner_5fold.ipynb (kept as code so the notebook stays in sync with the scripts)."""
import json
from pathlib import Path

cells = [
    ("md", """# Solar filament segmentation: 5-fold refiner, pooled OOF tuning, fold-refiner ensemble

Run on Kaggle with **GPU T4 x2** and **Internet on**. Attach the competition data and the outputs of
`filament-seg-yolo-stage`, `filament-seg-yolo-folds-1-2`, `filament-seg-yolo-folds-3-4`,
`filament-seg-unet-stage` and `filament-seg-meshnet-ensemble`.

* Until now the refiner was scored and tuned on fold 0 only (142 images). Here the ResNet-34 refiner
  (60 epochs) is trained once per held-out fold, so every one of the 707 images gets a refiner prediction
  from a model that never saw it.
* Score threshold, ranking mode and min area are tuned on the pooled 707-image OOF PQ (one yardstick).
* Test: the 5 fold refiners + the full-data refiner vote together (D4 TTA) on the 6-model YOLO proposals."""),
    ("code", """# 1. Setup
import os, subprocess, sys, json, shutil
from pathlib import Path

CODE = Path('/kaggle/working/filament-seg')
if not CODE.exists():
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/hsfcoulibaly/aml_project.git',
                    '/tmp/aml_project'], check=True)
    subprocess.run(['cp', '-r', '/tmp/aml_project/Azumah', str(CODE)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch', 'timm',
                'ultralytics', 'pycocotools'], check=True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.chdir(CODE)
sys.path.insert(0, str(CODE))
os.makedirs('logs', exist_ok=True)"""),
    ("code", """# 2. Link every upstream output into one work dir
WORK = Path('/kaggle/working/fseg')
for d in ['runs/ensemble', 'unet/maps', 'refiner']:
    (WORK / d).mkdir(parents=True, exist_ok=True)
inp = Path('/kaggle/input')
base = next(p.parent for p in inp.rglob('manifest.csv') if (p.parent / 'references.pkl').exists())
for name in ['manifest.csv', 'targets.pkl', 'references.pkl', 'preprocessed']:
    if not (WORK / name).exists():
        (WORK / name).symlink_to(base / name)
for run in sorted({p.parents[1] for p in inp.rglob('weights/best.pt') if p.parents[1].name.startswith('consensus_f')}):
    if not (WORK / 'runs' / run.name).exists():
        (WORK / 'runs' / run.name).symlink_to(run)
maps = next(p for p in inp.rglob('maps/unet') if (p / 'oof').exists())
if not (WORK / 'unet' / 'maps' / 'unet').exists():
    (WORK / 'unet' / 'maps' / 'unet').symlink_to(maps)
for name in ['resnet60_v0', 'resnet60_full', 'mesh_full']:
    src = next(p.parent for p in inp.rglob(f'refiner/{name}/best.pt'))
    (WORK / 'refiner' / name).mkdir(exist_ok=True)
    shutil.copy(src / 'best.pt', WORK / 'refiner' / name / 'best.pt')
shutil.copy(next(inp.rglob('runs/ensemble/test_raw.pkl')), WORK / 'runs' / 'ensemble' / 'test_raw.pkl')
os.environ['FSEG_WORK_DIR'] = str(WORK)
print('runs:', sorted(p.name for p in (WORK / 'runs').iterdir()))
print('oof proposals:', sorted(p.name for p in (WORK / 'runs').glob('consensus_f*/oof_raw_fold*.pkl')))"""),
    ("code", """# 3. Train the fold-1..4 refiners (fold 0 comes from the previous run), two per GPU in sequence
JOBS = {0: [1, 2], 1: [3, 4]}
procs = []
for gpu, folds in JOBS.items():
    chain = '; '.join(f'python scripts/train_refiner.py --name resnet60_v{k} --val-fold {k} --arch unet --epochs 60 '
                      f'--workers 2 --max-matched 12 --max-neg 8 --resume > logs/resnet60_v{k}.log 2>&1'
                      for k in folds)
    procs.append(subprocess.Popen(f'CUDA_VISIBLE_DEVICES={gpu} bash -c "{chain}"', shell=True))
for p in procs:
    p.wait()
for k in range(1, 5):
    print(k, open(f'logs/resnet60_v{k}.log').read().strip().splitlines()[-1])"""),
    ("code", """# 4. Out-of-fold refiner predictions for all 5 folds (each fold by the refiner that never saw it)
JOBS = {0: [0, 1, 2], 1: [3, 4]}
procs = []
for gpu, folds in JOBS.items():
    chain = '; '.join(f'python scripts/predict_refiner.py oof --fold {k} --refiner resnet60_v{k} --tag r60_f{k} '
                      f'--tta d4 > logs/oof_f{k}.log 2>&1' for k in folds)
    procs.append(subprocess.Popen(f'CUDA_VISIBLE_DEVICES={gpu} bash -c "{chain}"', shell=True))
for p in procs:
    p.wait()
for k in range(5):
    print(k, open(f'logs/oof_f{k}.log').read().strip().splitlines()[-1][:250])"""),
    ("code", """# 5. Pooled 707-image OOF PQ: tune once on all folds, report per fold
!python scripts/predict_refiner.py pool --tag r60 --folds 0 1 2 3 4 --tta d4"""),
    ("code", """# 6. Submissions: 6-refiner vote (5 folds + full) and the full refiner alone, pooled config,
#    plus the next stricter threshold of the pooled sweep
import pandas as pd
P = WORK / 'refiner' / 'r60_pooled'
cfg = P / 'pq_config.json'
t = pd.read_csv(P / 'pq_sweep_pooled.csv')
b = t.iloc[0]
same = t[(t['mode'] == b['mode']) & (t.min_area == b.min_area) & (t.thr > b.thr)].sort_values('thr')
hi = same.iloc[0] if len(same) else b
(P / 'strict.json').write_text(json.dumps({**json.loads(cfg.read_text()), 'thr': float(hi.thr)}))
print(f'pooled best thr {b.thr} (PQ {b.pq:.4f}) | strict thr {hi.thr} (PQ {hi.pq:.4f})')
raw = WORK / 'runs' / 'ensemble' / 'test_raw.pkl'
six = ' '.join([f'resnet60_v{k}' for k in range(5)] + ['resnet60_full'])
!python scripts/predict_refiner.py test --refiner {six} --config {cfg} --raw {raw} --tag test_r60x6 --out /kaggle/working/submission_r60x6_pooled.csv
!python scripts/predict_refiner.py test --refiner resnet60_full --config {cfg} --raw {raw} --tag test_r60full --out /kaggle/working/submission_r60full_pooled.csv
!python scripts/predict_refiner.py test --refiner {six} --config {P / 'strict.json'} --raw {raw} --tag test_r60x6_strict --out /kaggle/working/submission_r60x6_strict.csv
!ls -la /kaggle/working/*.csv"""),
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
out = Path(__file__).resolve().parents[1] / "notebooks" / "05_refiner_5fold.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out)
