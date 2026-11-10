"""Builds notebooks/04_meshnet_ensemble.ipynb (kept as code so the notebook stays in sync with the scripts)."""
import json
from pathlib import Path

cells = [
    ("md", """# Solar filament segmentation: MeshNet refiner, refiner ensembles, YOLO fold ensemble

Run on Kaggle with **GPU T4 x2** and **Internet on**. Attach the competition data and the outputs of
`filament-seg-yolo-stage`, `filament-seg-yolo-folds-1-2`, `filament-seg-yolo-folds-3-4`,
`filament-seg-unet-stage` and `filament-seg-refine-fuse`.

* Refiners (crop cache, GPU-bound): 2D MeshNet with the hdc_deep dilation schedule trained with the catalyst
  recipe (150 epochs), and the ResNet-34 U-Net refiner retrained for 60 epochs. Each as `*_v0` (folds 1-4,
  scored on fold 0) and `*_full` (all folds, for the test set), one GPU each.
* Fold-0 PQ with D4 TTA for every refiner and refiner ensemble; the best one by OOF PQ makes the submissions.
* Test proposals from the 5 fold models + the full model voting together (6 models x 4 flips)."""),
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
for d in ['runs', 'unet/maps', 'refiner']:
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
for old, new in [('refiner_v0', 'resnet_v0'), ('refiner_full', 'resnet_full')]:
    src = next(p.parent for p in inp.rglob(f'refiner/{old}/best.pt'))
    (WORK / 'refiner' / new).mkdir(exist_ok=True)
    shutil.copy(src / 'best.pt', WORK / 'refiner' / new / 'best.pt')
os.environ['FSEG_WORK_DIR'] = str(WORK)
print('runs:', sorted(p.name for p in (WORK / 'runs').iterdir()))
print('U-Net maps oof/test:', len(list((maps / 'oof').glob('*.png'))), len(list((maps / 'test').glob('*.png'))))"""),
    ("code", """# 3. Train four refiners, two per GPU in sequence (rerunning resumes)
JOBS = {
    0: [('mesh_v0', '--val-fold 0 --arch meshnet --channels 8 --recipe catalyst --epochs 150'),
        ('resnet60_v0', '--val-fold 0 --arch unet --epochs 60')],
    1: [('mesh_full', '--val-fold none --arch meshnet --channels 8 --recipe catalyst --epochs 150'),
        ('resnet60_full', '--val-fold none --arch unet --epochs 60')],
}
procs = []
for gpu, jobs in JOBS.items():
    chain = '; '.join(f'python scripts/train_refiner.py --name {n} {a} --workers 2 --max-matched 12 --max-neg 8 '
                      f'--resume > logs/{n}.log 2>&1'
                        for n, a in jobs)
    procs.append(subprocess.Popen(f'CUDA_VISIBLE_DEVICES={gpu} bash -c "{chain}"', shell=True))
for p in procs:
    p.wait()
for jobs in JOBS.values():
    for n, _ in jobs:
        lines = open(f'logs/{n}.log').read().strip().splitlines()
        print(n, '|', next((l for l in lines if 'parameters' in l), ''), '|', lines[-1])"""),
    ("code", """# 4. Test proposals from the YOLO fold ensemble (5 fold models + full model, 4 flips each)
w = ' '.join(str(WORK / 'runs' / r / 'weights' / 'best.pt') for r in
             ['consensus_f0', 'consensus_f1', 'consensus_f2', 'consensus_f3', 'consensus_f4', 'consensus_ffull'])
!python scripts/predict_yolo.py test --weights {w} --config /kaggle/working/fseg/runs/consensus_f0/pq_config.json \\
    --raw-out /kaggle/working/fseg/runs/ensemble/test_raw.pkl --out /kaggle/working/submission_yolo_ensemble.csv"""),
    ("code", """# 5. Fold-0 PQ of every refiner and refiner ensemble (D4 TTA), same 142 images
VARIANTS = {
    'resnet30': ['resnet_v0'], 'resnet60': ['resnet60_v0'], 'mesh': ['mesh_v0'],
    'resnet60+mesh': ['resnet60_v0', 'mesh_v0'], 'all3': ['resnet_v0', 'resnet60_v0', 'mesh_v0'],
}
import pandas as pd
rows = []
for tag, names in VARIANTS.items():
    subprocess.run(f'python scripts/predict_refiner.py oof --fold 0 --tta d4 --tag eval_{tag} --refiner {" ".join(names)} '
                   f'> logs/eval_{tag}.log 2>&1', shell=True)
    f = WORK / 'refiner' / f'eval_{tag}' / 'pq_sweep_fold0.csv'
    if not f.exists():
        print(tag, 'failed:', open(f'logs/eval_{tag}.log').read()[-500:])
        continue
    t = pd.read_csv(f)
    rows.append({'variant': tag, **t.iloc[0][['mode', 'thr', 'min_area', 'pq', 'sq', 'rq', 'precision', 'recall']].to_dict()})
res = pd.DataFrame(rows).sort_values('pq', ascending=False)
print('stage 1 (YOLO) fold 0: 0.4172 | previous best (resnet30, hflip, + fusion): 0.4472')
print(res.round(4).to_string(index=False))
BEST = res.iloc[0].variant
print('best:', BEST)"""),
    ("code", """# 6. Submissions with the best variant's full-data refiners
full = [n.replace('_v0', '_full') for n in VARIANTS[BEST]]
cfg = WORK / 'refiner' / f'eval_{BEST}' / 'pq_config.json'
t = pd.read_csv(WORK / 'refiner' / f'eval_{BEST}' / 'pq_sweep_fold0.csv')
b = t.iloc[0]
same = t[(t['mode'] == b['mode']) & (t.min_area == b.min_area) & (t.thr > b.thr)].sort_values('thr')
hi = same.iloc[0] if len(same) else b  # next stricter threshold: fewer, surer filaments for sparser test labels
hicfg = {**json.loads(cfg.read_text()), 'thr': float(hi.thr)}
(WORK / 'refiner' / 'hiprec.json').write_text(json.dumps(hicfg))
print(f'best thr {b.thr} (PQ {b.pq:.4f}, precision {b.precision:.3f}) | strict thr {hi.thr} (PQ {hi.pq:.4f}, precision {hi.precision:.3f})')
for raw, suffix in [(WORK / 'runs' / 'consensus_ffull' / 'test_raw.pkl', 'single'), (WORK / 'runs' / 'ensemble' / 'test_raw.pkl', 'ens')]:
    for c, name in [(cfg, BEST), (WORK / 'refiner' / 'hiprec.json', BEST + '_strict')]:
        !python scripts/predict_refiner.py test --refiner {' '.join(full)} --config {c} --raw {raw} --tag test_{name}_{suffix} \\
            --out /kaggle/working/submission_{name}_{suffix}.csv
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
out = Path(__file__).resolve().parents[1] / "notebooks" / "04_meshnet_ensemble.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out)
