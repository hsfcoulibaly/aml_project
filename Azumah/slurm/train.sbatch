#!/bin/bash
# Stage 1: train the YOLO proposer. Resubmit the same command if it hits the time limit:
# --resume continues from last.pt.
#
#   sbatch slurm/train.sbatch                          # fold 0, consensus labels
#   FOLD=full sbatch slurm/train.sbatch                # final model on all data
#   DILATED=p4,p5 sbatch slurm/train.sbatch            # dilation ablation
#   MODE=largest sbatch slurm/train.sbatch             # label ablation
#
#SBATCH --job-name=fseg-train
#SBATCH --partition=qTRDGPUH
#SBATCH --account=trends53c17
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source slurm/env.sh

FOLD="${FOLD:-0}"
MODE="${MODE:-consensus}"
DILATED="${DILATED:-}"
MODEL="${MODEL:-yolo11l-seg.pt}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-8}"   # total across GPUs; halve on out-of-memory
NAME="${MODE}_f${FOLD}${DILATED:+_dil${DILATED//,/}}"

nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/train_yolo.py --mode "${MODE}" --fold "${FOLD}" --model "${MODEL}" --imgsz "${IMGSZ}" \
    --epochs "${EPOCHS}" --batch "${BATCH}" --dilated "${DILATED}" --name "${NAME}" \
    --workers "${SLURM_CPUS_PER_TASK}" --resume
