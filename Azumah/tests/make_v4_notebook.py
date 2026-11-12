"""Builds notebooks/06_unet_longer.ipynb (kept as code so the notebook stays in sync with the scripts)."""
import json
from pathlib import Path

cells = [
    ("md", """# Solar filament segmentation: longer U-Net, then the 5-fold refiner on its maps

Run on Kaggle with **GPU T4 x2** and **Internet on**. Attach the competition data and the outputs of
`filament-seg-yolo-stage`, `filament-seg-yolo-folds-1-2`, `filament-seg-yolo-folds-3-4`,
`filament-seg-unet-stage` and `filament-seg-meshnet-ensemble`.

* The 30-epoch U-Net was still improving at its last epoch. Each fold model is warm-started from its
  30-epoch weights and trained 20 more epochs (fresh warmup + cosine, peak lr 1e-4): 50 epochs in total.
* New out-of-fold and test maps (`unet50`), then the same 5-fold ResNet-34 refiner recipe as notebook 05
  (60 epochs, D4, pooled 707-image tuning) on top of them.
* Compare: pooled OOF PQ 0.4579 with the 30-epoch maps (notebook 05). Keep this only if it beats that."""),
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
os.makedirs('logs', exist_ok=True)

def run_on_gpus(jobs):
    \"\"\"jobs: {gpu: [shell command, ...]}; commands on one GPU run in sequence, GPUs in parallel.\"\"\"
    procs = [subprocess.Popen(f'CUDA_VISIBLE_DEVICES={g} bash -c "{"; ".join(cmds)}"', shell=True)
             for g, cmds in jobs.items()]
    for p in procs:
        p.wait()

def last(log, n=1):
    lines = open(log).read().strip().splitlines()
    return ' / '.join(l[:220] for l in lines[-n:])"""),
    ("code", """# 2. Link every upstream output into one work dir
WORK = Path('/kaggle/working/fseg')
for d in ['runs/ensemble', 'unet/runs', 'unet/init', 'refiner']:
    (WORK / d).mkdir(parents=True, exist_ok=True)
inp = Path('/kaggle/input')
base = next(p.parent for p in inp.rglob('manifest.csv') if (p.parent / 'references.pkl').exists())
for name in ['manifest.csv', 'targets.pkl', 'references.pkl', 'preprocessed']:
    if not (WORK / name).exists():
        (WORK / name).symlink_to(base / name)
for run in sorted({p.parents[1] for p in inp.rglob('weights/best.pt') if p.parents[1].name.startswith('consensus_f')}):
    if not (WORK / 'runs' / run.name).exists():
        (WORK / 'runs' / run.name).symlink_to(run)
for k in range(5):
    shutil.copy(next(inp.rglob(f'unet/runs/unet_f{k}/best.pt')), WORK / 'unet' / 'init' / f'unet_f{k}.pt')
shutil.copy(next(inp.rglob('runs/ensemble/test_raw.pkl')), WORK / 'runs' / 'ensemble' / 'test_raw.pkl')
os.environ['FSEG_WORK_DIR'] = str(WORK)
from fseg.config import get_paths
from fseg.unet_data import build_targets
import pandas as pd
paths = get_paths(None, str(WORK))
build_targets(paths, pd.read_csv(paths.manifest).observation_id.tolist(), workers=4)  # once, before the GPU jobs
print('runs:', sorted(p.name for p in (WORK / 'runs').iterdir()))
print('U-Net init:', sorted(p.name for p in (WORK / 'unet' / 'init').iterdir()))"""),
    ("code", """# 3. U-Net: 20 more epochs per fold from the 30-epoch weights (3 folds on GPU 0, 2 on GPU 1)
def unet_cmd(k):
    return (f'python scripts/train_unet.py --fold {k} --name unet50_f{k} --init {WORK}/unet/init/unet_f{k}.pt '
            f'--epochs 20 --lr 1e-4 --warmup-epochs 0.5 --batch 4 --workers 2 --patience 100 --pq-every 2 '
            f'--resume > logs/unet50_f{k}.log 2>&1')
run_on_gpus({0: [unet_cmd(k) for k in [0, 1, 2]], 1: [unet_cmd(k) for k in [3, 4]]})
old = {0: 0.3778, 1: 0.3863, 2: 0.3797, 3: 0.4035, 4: 0.3883}  # 30-epoch best val PQ (standalone, fixed thr)
for k in range(5):
    print(f'fold {k}: 30 ep {old[k]:.4f} |', last(f'logs/unet50_f{k}.log'))"""),
    ("code", """# 4. New U-Net maps: out-of-fold for all 707 training images, 5-fold average for test
run_on_gpus({
    0: ['python scripts/predict_unet.py oof --name unet50 --folds 0 1 2 --tta flip > logs/maps_oof_a.log 2>&1',
        'python scripts/predict_unet.py test --name unet50 --folds 0 1 2 3 4 --tta flip > logs/maps_test.log 2>&1'],
    1: ['python scripts/predict_unet.py oof --name unet50 --folds 3 4 --tta flip > logs/maps_oof_b.log 2>&1'],
})
m = WORK / 'unet' / 'maps' / 'unet50'
print('maps oof/test:', len(list((m / 'oof').glob('*.png'))), len(list((m / 'test').glob('*.png'))))
!python scripts/predict_unet.py oof --name unet50 --folds 0 --sweep-fold 0 --tta flip 2>&1 | tail -1"""),
    ("code", """# 5. Refiners on the new maps: 5 fold models + full model, 3 per GPU in sequence
def ref_cmd(name, val):
    return (f'python scripts/train_refiner.py --name {name} --val-fold {val} --unet-name unet50 --arch unet '
            f'--epochs 60 --workers 2 --max-matched 12 --max-neg 8 --resume > logs/{name}.log 2>&1')
run_on_gpus({0: [ref_cmd(f'r60u_v{k}', k) for k in [0, 1, 2]],
             1: [ref_cmd(f'r60u_v{k}', k) for k in [3, 4]] + [ref_cmd('r60u_full', 'none')]})
for n in [f'r60u_v{k}' for k in range(5)] + ['r60u_full']:
    print(n, '|', last(f'logs/{n}.log'))"""),
    ("code", """# 6. Out-of-fold refiner predictions for every fold, then pooled 707-image tuning
def oof_cmd(k):
    return (f'python scripts/predict_refiner.py oof --fold {k} --refiner r60u_v{k} --tag r60u_f{k} '
            f'--unet-name unet50 --tta d4 > logs/oof_u_f{k}.log 2>&1')
run_on_gpus({0: [oof_cmd(k) for k in [0, 1, 2]], 1: [oof_cmd(k) for k in [3, 4]]})
!python scripts/predict_refiner.py pool --tag r60u --folds 0 1 2 3 4 --tta d4
print('previous (30-epoch U-Net maps, notebook 05): pooled 0.4579 | per fold 0.4517 0.4579 0.4444 0.4738 0.4617')"""),
    ("code", """# 7. Submissions: 6-refiner vote on the 6-model YOLO proposals with the new maps, plus the next stricter threshold
P = WORK / 'refiner' / 'r60u_pooled'
cfg = P / 'pq_config.json'
t = pd.read_csv(P / 'pq_sweep_pooled.csv')
b = t.iloc[0]
same = t[(t['mode'] == b['mode']) & (t.min_area == b.min_area) & (t.thr > b.thr)].sort_values('thr')
hi = same.iloc[0] if len(same) else b
(P / 'strict.json').write_text(json.dumps({**json.loads(cfg.read_text()), 'thr': float(hi.thr)}))
print(f'pooled best {dict(b[["mode", "min_area", "thr", "pq"]])} | strict thr {hi.thr} (PQ {hi.pq:.4f})')
raw = WORK / 'runs' / 'ensemble' / 'test_raw.pkl'
six = ' '.join([f'r60u_v{k}' for k in range(5)] + ['r60u_full'])
!python scripts/predict_refiner.py test --refiner {six} --unet-name unet50 --config {cfg} --raw {raw} --tag test_r60u_x6 --out /kaggle/working/submission_r60u_x6_pooled.csv
!python scripts/predict_refiner.py test --refiner {six} --unet-name unet50 --config {P / 'strict.json'} --raw {raw} --tag test_r60u_x6_strict --out /kaggle/working/submission_r60u_x6_strict.csv
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
out = Path(__file__).resolve().parents[1] / "notebooks" / "06_unet_longer.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out)
