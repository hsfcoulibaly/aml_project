"""Builds notebooks/07_end_to_end.ipynb (kept as code so the notebook stays in sync with the scripts)."""
import json
from pathlib import Path

cells = [
    ("md", """# Solar filament segmentation: end-to-end inference of the final pipeline

Pooled 5-fold out-of-fold PQ **0.460** on all 707 training images, public leaderboard **0.40**
(file `union_c2r.csv`). Run on Kaggle with **GPU T4** and **Internet on**; attach the competition data and
the public dataset **`richmondazumah/filament-seg-weights`** (28 fp16 checkpoints, 1.6 GB, plus the tuned
configs). Locally: `bash scripts/run_inference.sh` runs the same stages.

| Stage | Model | Output |
|---|---|---|
| 0 | disk fit, limb flattening, CLAHE | 3-channel 2048 px images |
| 1 | YOLO11l-seg, 5 fold models + full model, 4 flips | instance proposals (conf >= 0.05) |
| 2 | ConvNeXt-T + ASPP U-Net, 1024 px, 5-fold average | filament, centerline, boundary maps |
| 3 | ResNet-34 refiner on 256 px crops, 6 models, D4 | refined mask + predicted IoU per proposal |
| 4a | U-Net-2048 (native resolution, 1024 tiles), 5-fold average | second set of maps |
| 4b | components at p > 0.3/0.5/0.7, own 6 refiners, D4 | refined second proposal set |
| 5 | score = quality x conf (x 0.25 for 4b), greedy overlap resolution | `submission.csv` |

Training notebooks: `01` YOLO, `01b` YOLO folds, `02` U-Net, `03` refiner + fusion, `04` refiner ensemble,
`05` 5-fold refiner, `06` longer U-Net (ablation); U-Net-2048 by Charan (branch `charan/fusion-native-unet`)."""),
    ("code", """# Setup
import os, subprocess, sys, time, json
from pathlib import Path

CODE = Path('/kaggle/working/filament-seg')
if not CODE.exists():
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/hsfcoulibaly/aml_project.git',
                    '/tmp/aml_project'], check=True)
    subprocess.run(['cp', '-r', '/tmp/aml_project/Azumah', str(CODE)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch==0.6.0',
                'timm==1.0.30', 'ultralytics==8.4.189', 'pycocotools==2.0.11'], check=True)
os.chdir(CODE)
sys.path.insert(0, str(CODE))
WEIGHTS = next(p.parent for p in Path('/kaggle/input').rglob('manifest.csv') if (p.parent / 'configs').exists())
WORK = Path('/kaggle/working/fseg')
os.environ.update(WEIGHTS=str(WEIGHTS), FSEG_WORK_DIR=str(WORK))
print('weights:', WEIGHTS)
T = {}

def run(name, cmd):
    t = time.time()
    subprocess.run(cmd, shell=True, check=True)
    T[name] = time.time() - t
    print(f'{name}: {T[name] / 60:.1f} min')"""),
    ("code", """# Stage 0: link the weights into the work dir and preprocess the 180 test images
for d in ['runs/ensemble', 'unet/runs', 'refiner']:
    (WORK / d).mkdir(parents=True, exist_ok=True)
if not (WORK / 'manifest.csv').exists():
    (WORK / 'manifest.csv').symlink_to(WEIGHTS / 'manifest.csv')
for sub in ['runs', 'unet/runs', 'refiner']:
    for d in (WEIGHTS / 'fseg' / sub).iterdir():
        if not (WORK / sub / d.name).exists():
            (WORK / sub / d.name).symlink_to(d)
run('preprocess', '''python -c "
from fseg.config import get_paths; from fseg import yolo_data
p = get_paths(); f = sorted(x.name for x in p.test_images.glob('*') if x.suffix.lower() in {'.jpg', '.jpeg', '.png'})
yolo_data.preprocess_all(p, 'test', f, workers=4); print(len(f), 'test images')"''')"""),
    ("code", """# Stage 1: YOLO11l-seg proposals (5 fold models + full model, 4 flips each, mask voting)
yolo = ' '.join(str(WORK / 'runs' / f'consensus_{r}' / 'weights' / 'best.pt') for r in ['f0', 'f1', 'f2', 'f3', 'f4', 'ffull'])
run('yolo', f'python scripts/predict_yolo.py test --weights {yolo} --config {WORK}/runs/consensus_f0/pq_config.json '
            f'--raw-out {WORK}/runs/ensemble/test_raw.pkl --out {WORK}/submission_yolo.csv')"""),
    ("code", """# Stage 2: U-Net maps (filament, centerline, boundary), 5-fold average, flip TTA
run('unet', 'python scripts/predict_unet.py test --name unet --folds 0 1 2 3 4 --tta flip')"""),
    ("code", """# Stage 3: refine every YOLO proposal (5 fold refiners + full refiner, D4 TTA)
six = ' '.join([f'resnet60_v{k}' for k in range(5)] + ['resnet60_full'])
run('refine_yolo', f'python scripts/predict_refiner.py test --refiner {six} --config {WEIGHTS}/configs/r60_pooled.json '
                   f'--raw {WORK}/runs/ensemble/test_raw.pkl --tag test_r60x6 --out {WORK}/submission_yolo_refined.csv')"""),
    ("code", """# Stage 4: second proposal source: native-resolution U-Net-2048 maps -> components -> its own refiners
run('unet2048', 'python scripts/predict_unet.py test --name unet2048 --folds 0 1 2 3 4 --tta flip')
run('unet2048_proposals', f'python scripts/make_unet_proposals.py --code . --maps {WORK}/unet/maps/unet2048 --test-only')
six = ' '.join([f'u2r_v{k}' for k in range(5)] + ['u2r_full'])
run('refine_unet2048', f'python scripts/predict_refiner.py test --refiner {six} --unet-name unet2048 '
                       f'--config {WEIGHTS}/configs/u2r_pooled.json --raw {WORK}/runs/unet2048_test/test_raw.pkl '
                       f'--tag test_u2rx6 --out {WORK}/submission_unet2048_refined.csv')"""),
    ("code", """# Stage 5: merge both refined sets (U-Net-2048 weight 0.25, threshold 0.1, min area 150) -> submission.csv
print(json.loads((WEIGHTS / 'configs' / 'union.json').read_text()))
run('merge', f'python scripts/union_proposals.py submit --yolo-test {WORK}/refiner/test_r60x6/refined_test.pkl '
             f'--unet-test {WORK}/refiner/test_u2rx6/refined_test.pkl --config {WEIGHTS}/configs/union.json '
             f'--out /kaggle/working/submission.csv')
print({k: f'{v / 60:.1f} min' for k, v in T.items()}, f'total {sum(T.values()) / 60:.1f} min')"""),
    ("code", """# Sanity check and a look at three test images (red: predicted filaments)
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from fseg import rle as R
from fseg.config import get_paths
sub = pd.read_csv('/kaggle/working/submission.csv')
sub['obs'] = sub.filament_id.str.rsplit('_', n=1).str[0]
print(len(sub), 'filaments on', sub.obs.nunique(), 'images;', round(len(sub) / sub.obs.nunique(), 1), 'per image')
paths = get_paths()
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, obs in zip(axs, sub.obs.value_counts().index[[0, len(sub.obs.unique()) // 2, -1]]):
    img = cv2.imread(str(paths.preprocessed / 'test' / f'{obs}.jpg'))[..., 1]
    rgb = np.repeat(img[..., None], 3, 2)
    for c in sub.loc[sub.obs == obs, 'segmentation_rle']:
        m = R.decode(R.from_counts(c)).astype(bool)
        rgb[m] = (0.4 * rgb[m] + 0.6 * np.array([230, 80, 40])).astype(np.uint8)
    ax.imshow(rgb); ax.set_title(f'{obs}: {(sub.obs == obs).sum()} filaments'); ax.axis('off')
plt.tight_layout()"""),
    ("md", """## Validation results (pooled 5-fold OOF, 707 images, every image predicted by models that never saw it)

| Stage | PQ | SQ | RQ | Precision | Recall | one-to-many | many-to-one |
|---|---|---|---|---|---|---|---|
| Human agreement | 0.344 | 0.635 | 0.542 | 0.542 | 0.542 | 2.9% | 2.9% |
| YOLO11l-seg | 0.421 | 0.662 | 0.635 | 0.669 | 0.605 | 1.5% | 3.9% |
| YOLO + refiner | 0.458 | 0.684 | 0.670 | 0.726 | 0.622 | 1.2% | 3.6% |
| U-Net-2048 + refiner | 0.452 | 0.682 | 0.663 | 0.686 | 0.641 | 3.3% | 2.4% |
| **Union (final)** | **0.460** | 0.682 | 0.674 | 0.698 | 0.653 | 2.4% | 3.2% |

Reproduce: `scripts/report_figures.py` (tables, IoU/Dice distributions, qualitative figures);
details and ablations in `docs/BEST_PIPELINE.md` and `report/report.tex`."""),
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
out = Path(__file__).resolve().parents[1] / "notebooks" / "07_end_to_end.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(out)
