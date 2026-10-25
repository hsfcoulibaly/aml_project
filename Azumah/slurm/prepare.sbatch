#!/bin/bash
# Stage 0 (CPU only): folds, consensus targets, preprocessing, YOLO export. ~20-40 min.
#SBATCH --job-name=fseg-prepare
#SBATCH --partition=CHANGE_ME_CPU_PARTITION
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source slurm/env.sh

python scripts/prepare.py --size "${IMGSZ}" --modes consensus largest --workers "${SLURM_CPUS_PER_TASK}"
