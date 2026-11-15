#!/usr/bin/env bash
# End-to-end test inference of the final pipeline (pooled OOF PQ 0.460, public LB 0.40).
#
#   WEIGHTS=<weights dataset dir> FSEG_DATA_ROOT=<competition data> FSEG_WORK_DIR=<scratch> \
#       bash scripts/run_inference.sh [submission.csv]
#
# WEIGHTS holds fseg/{runs,unet/runs,refiner}, configs/ and manifest.csv (Kaggle dataset
# richmondazumah/filament-seg-weights). Run from the Azumah/ directory. Stages and their timings are logged.
set -euo pipefail
OUT=${1:-submission.csv}
: "${WEIGHTS:?set WEIGHTS}" "${FSEG_WORK_DIR:?set FSEG_WORK_DIR}"
PY=${PYTHON:-python}
W=$FSEG_WORK_DIR
mkdir -p "$W/runs/ensemble" "$W/unet/runs" "$W/refiner"

stage() { echo "=== [$(date +%T)] $*"; }

stage "0. link weights"
[ -e "$W/manifest.csv" ] || cp "$WEIGHTS/manifest.csv" "$W/manifest.csv"
for d in "$WEIGHTS"/fseg/runs/*/ "$WEIGHTS"/fseg/unet/runs/*/ "$WEIGHTS"/fseg/refiner/*/; do
  rel=${d#"$WEIGHTS"/fseg/}
  [ -e "$W/${rel%/}" ] || ln -s "${d%/}" "$W/${rel%/}"
done

stage "1. preprocess test images (disk fit, limb flattening, CLAHE)"
$PY - <<'EOF'
from fseg.config import get_paths
from fseg import yolo_data
p = get_paths()
files = sorted(f.name for f in p.test_images.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
yolo_data.preprocess_all(p, "test", files, workers=4)
print(len(files), "test images")
EOF

stage "2. YOLO11l-seg proposals: 5 fold models + full model, 4 flips each"
YOLO=$(for r in f0 f1 f2 f3 f4 ffull; do printf '%s ' "$W/runs/consensus_$r/weights/best.pt"; done)
$PY scripts/predict_yolo.py test --weights $YOLO --config "$W/runs/consensus_f0/pq_config.json" \
  --raw-out "$W/runs/ensemble/test_raw.pkl" --out "$W/submission_yolo.csv"

stage "3. U-Net maps (ConvNeXt-T + ASPP, 1024 px, 5-fold average, flip TTA)"
$PY scripts/predict_unet.py test --name unet --folds 0 1 2 3 4 --tta flip

stage "4. refine YOLO proposals: 5 fold refiners + full refiner, D4 TTA"
$PY scripts/predict_refiner.py test --refiner resnet60_v0 resnet60_v1 resnet60_v2 resnet60_v3 resnet60_v4 resnet60_full \
  --config "$WEIGHTS/configs/r60_pooled.json" --raw "$W/runs/ensemble/test_raw.pkl" --tag test_r60x6 \
  --out "$W/submission_yolo_refined.csv"

stage "5. U-Net-2048 maps (native resolution, 1024 px tiles, 5-fold average, flip TTA)"
$PY scripts/predict_unet.py test --name unet2048 --folds 0 1 2 3 4 --tta flip

stage "6. U-Net-2048 proposals (components at p > 0.3, 0.5, 0.7)"
$PY scripts/make_unet_proposals.py --code . --maps "$W/unet/maps/unet2048" --test-only

stage "7. refine U-Net-2048 proposals with their own 6 refiners, D4 TTA"
$PY scripts/predict_refiner.py test --refiner u2r_v0 u2r_v1 u2r_v2 u2r_v3 u2r_v4 u2r_full --unet-name unet2048 \
  --config "$WEIGHTS/configs/u2r_pooled.json" --raw "$W/runs/unet2048_test/test_raw.pkl" --tag test_u2rx6 \
  --out "$W/submission_unet2048_refined.csv"

stage "8. merge both refined sets (U-Net-2048 weight 0.25) -> $OUT"
$PY scripts/union_proposals.py submit --yolo-test "$W/refiner/test_r60x6/refined_test.pkl" \
  --unet-test "$W/refiner/test_u2rx6/refined_test.pkl" --config "$WEIGHTS/configs/union.json" --out "$OUT"
stage "done"
