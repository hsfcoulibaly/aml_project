# Shared environment for every job. Edit the CHANGE_ME lines once for your cluster.

# 1. Load Python / CUDA modules if your cluster uses them (see `module avail python cuda`):
module load python/3.12.7   # the .venv was built from this module
# (no CUDA module needed: the PyTorch wheel bundles CUDA 12.1)

# 2. Activate the environment created during setup (venv or conda, whichever you used)
source .venv/bin/activate
# conda activate fseg

# 3. Data and outputs. Point FSEG_DATA_ROOT at the folder that contains train/ and test/.
#    Put the work dir on scratch storage if your home quota is small (~5 GB needed).
export FSEG_DATA_ROOT="${FSEG_DATA_ROOT:-$PWD/data/MAGFiLO_1.0_Kaggle_2026}"
export FSEG_WORK_DIR="${FSEG_WORK_DIR:-$PWD/work}"

# 4. YOLO training resolution; must match the --size used by prepare
export IMGSZ="${IMGSZ:-1280}"

# Compute nodes are usually offline: never try to download at run time
export YOLO_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p logs
echo "[env] host=$(hostname) data=${FSEG_DATA_ROOT} work=${FSEG_WORK_DIR} imgsz=${IMGSZ}"
