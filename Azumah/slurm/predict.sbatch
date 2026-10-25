#!/bin/bash
# Out-of-fold PQ calibration (default) or test-set submission.
#
#   sbatch slurm/predict.sbatch                                   # OOF on fold 0 of consensus_f0
#   NAME=consensus_f0_dilp4p5 sbatch slurm/predict.sbatch         # OOF for an ablation run
#   SPLIT=test NAME=consensus_ffull CONFIG_FROM=consensus_f0 sbatch slurm/predict.sbatch
#
#SBATCH --job-name=fseg-predict
#SBATCH --partition=CHANGE_ME_GPU_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source slurm/env.sh

SPLIT="${SPLIT:-oof}"
FOLD="${FOLD:-0}"
NAME="${NAME:-consensus_f${FOLD}}"
WEIGHTS="${FSEG_WORK_DIR}/runs/${NAME}/weights/best.pt"

if [ "${SPLIT}" = "oof" ]; then
    python scripts/predict_yolo.py oof --weights "${WEIGHTS}" --fold "${FOLD}" --imgsz "${IMGSZ}"
else
    CONFIG_FROM="${CONFIG_FROM:-consensus_f0}"
    python scripts/predict_yolo.py test --weights "${WEIGHTS}" --imgsz "${IMGSZ}" \
        --config "${FSEG_WORK_DIR}/runs/${CONFIG_FROM}/pq_config.json" --out submission.csv
fi
