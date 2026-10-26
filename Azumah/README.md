# filament-seg

Topology-aware YOLO to U-Net cascade for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(MAGFiLO, GONG H-alpha, scored with Panoptic Quality).

This release covers **stage 0** (data) and **stage 1** (YOLO proposer). Stages 2 to 4 (dense U-Net branch,
instance-conditioned refiner, topology-aware fusion) build on the files these stages write.

## Pipeline so far

| Stage | What it does | Code |
|---|---|---|
| 0a | One row per physical observation, 5 folds stratified by site and filament count (no record-ID leakage) | `fseg/labels.py` |
| 0b | Clusters the 1 to 3 annotation sets per image; each filament gets an agreement score; `consensus` targets keep filaments marked by at least half the annotators | `fseg/labels.py` |
| 0c | Fits the solar disk, divides out limb darkening, builds a 3-channel image (raw, limb-flattened, CLAHE), zeroes the sky | `fseg/preprocess.py` |
| 0d | Local PQ with the organizers' formula, scored against every annotation set; human-agreement ceiling; fragmentation and over-merge rates | `fseg/metrics.py` |
| 1a | Exports single-class YOLO-seg polygons (multi-part filaments joined) | `fseg/yolo_data.py` |
| 1b | Optional zero-initialized dilated context branches (rates 1, 2, 5) on the P3/P4/P5 neck outputs | `fseg/dilated.py` |
| 1c | Full-resolution inference, flip TTA with mask voting, overlap resolution, off-disk filtering, submission | `fseg/yolo_infer.py`, `fseg/postprocess.py` |
| 1d | Dark tightening: YOLO prototype masks are ~4x coarser than the image and bleed into bright surroundings, so each mask is optionally cut to pixels darker than a ratio of the quiet Sun (0.85 to 0.97, or off) | `fseg/yolo_infer.py` |
| 1e | PQ sweep over confidence, minimum area and dark ratio on out-of-fold predictions (exact and fast: greedy overlap resolution makes every confidence threshold a prefix of one ranked list) | `fseg/yolo_infer.py` |

## Run

On Kaggle, attach the competition data, enable GPU T4 x2 and Internet, and run
`notebooks/01_yolo_stage.ipynb`. The same steps from a shell:

```bash
pip install -r requirements.txt
python scripts/prepare.py --size 1280 --modes consensus largest
python scripts/train_yolo.py --mode consensus --fold 0 --model yolo11l-seg.pt --imgsz 1280 --epochs 60 --batch 8
python scripts/predict_yolo.py oof --weights work/runs/consensus_f0/weights/best.pt --fold 0
python scripts/train_yolo.py --mode consensus --fold full --epochs 60
python scripts/predict_yolo.py test --weights work/runs/consensus_ffull/weights/best.pt \
    --config work/runs/consensus_f0/pq_config.json
```

Outside Kaggle set `FSEG_DATA_ROOT` (folder containing `train/` and `test/`) and optionally `FSEG_WORK_DIR`.

## Run on a SLURM cluster

Compute nodes are usually offline, so everything that needs the internet happens once on the login node.

```bash
# --- login node, once ---
git clone <your repo> filament-seg && cd filament-seg     # or scp/rsync the folder
module load <python module>                               # if your cluster uses modules
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match the cluster's CUDA
pip install -r requirements.txt
pip install kaggle                                        # needs ~/.kaggle/kaggle.json
kaggle competitions download -c filament-segmentation-2026
unzip -q filament-segmentation-2026.zip -d data
python scripts/fetch_assets.py yolo11l-seg.pt             # pre-download YOLO weights and fonts
# edit slurm/env.sh (modules, env activation, FSEG_DATA_ROOT) and the
# CHANGE_ME partitions in slurm/*.sbatch  (`sinfo` lists partitions)

# --- jobs ---
jid=$(sbatch --parsable slurm/prepare.sbatch)
jid=$(sbatch --parsable --dependency=afterok:$jid slurm/train.sbatch)
sbatch --dependency=afterok:$jid slurm/predict.sbatch
squeue -u $USER ; tail -f logs/fseg-train-*.out
```

* If a training job hits its time limit, submit the same `sbatch slurm/train.sbatch` again: it resumes
  from `last.pt`.
* Ablations and the final model are environment variables on the same scripts, e.g.
  `DILATED=p4,p5 sbatch slurm/train.sbatch` then `NAME=consensus_f0_dilp4p5 sbatch slurm/predict.sbatch`;
  `FOLD=full sbatch slurm/train.sbatch` then
  `SPLIT=test NAME=consensus_ffull sbatch slurm/predict.sbatch` (writes `submission.csv`).
* For 2 GPUs set `#SBATCH --gres=gpu:2` and `BATCH=16`; the script uses every visible GPU.

## Ablations to report

| Flag | Options |
|---|---|
| `--mode` | `consensus` (default), `largest` (single most complete annotation set, as in the public Mask R-CNN notebook) |
| `--dilated` | `""` (off), `p4,p5`, `p3,p4,p5` |
| `--tta` | `id` or any of `id,hflip,vflip,rot180` |
| `--model` | `yolo11m-seg.pt`, `yolo11l-seg.pt`, `yolo11x-seg.pt` |
| `--imgsz` | 1024, 1280, 1536 (re-run `prepare.py --size` to match) |

Every OOF run writes, next to its weights: `pq_config.json` (best thresholds), `pq_sweep_fold*.csv`,
`oof_per_image_fold*.csv` and `oof_raw_fold*.pkl` (raw instances reused by later stages).

## Tests

```bash
python tests/test_metrics.py                 # PQ unit tests
python tests/make_synthetic.py /tmp/synth    # tiny fake dataset for an end-to-end smoke run
FSEG_DATA_ROOT=/tmp/synth FSEG_WORK_DIR=/tmp/work python scripts/prepare.py --size 640
```

## Notes

* The training JSON also contains spine annotations. The rules forbid training on "other ground-truth
  metadata", so later stages derive centerlines by skeletonizing the masks (`fseg.labels.skeleton`).
* Submissions contain RLE counts only; size is fixed at 2048 x 2048.
