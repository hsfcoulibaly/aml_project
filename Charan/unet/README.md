# PQ-evaluated U-Net (Charan)

U-Net track for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(MAGFiLO, GONG H-alpha, scored with Panoptic Quality). Self-contained; uses the team's grouped split
(`Sheikh/outputs/threshold_eval/grouped_split.json`, 566 train / 141 val physical images), so Dice numbers
line up with Sheikh's U-Net (0.631) and Hassan's YOLO11s-seg (0.650).

What it adds over the first U-Net baseline: instance output and PQ scoring, 1024 px input instead of 512
(median filament width is ~11 px at 2048, so 512 leaves ~3 px), limb-darkening correction, an
ImageNet-pretrained ResNet34 encoder, and a tuned post-processing step.

## Pipeline

| Stage | What | Code |
|---|---|---|
| Prep | Fit the solar disk, divide out limb darkening, 3 channels (raw, limb-flattened, CLAHE), sky zeroed, resize to 1024. One binary mask per annotation record; PQ references keep every record's filaments at 2048 | `fseg_unet/preprocess.py`, `fseg_unet/labels.py` |
| Model | `segmentation_models_pytorch` U-Net, ResNet34 ImageNet encoder, 1 output channel | `fseg_unet/model.py` |
| Train | 0.5 BCE (pos_weight 10) + 0.5 soft Dice; dihedral flips/rotations + gain/bias jitter; AdamW 3e-4, 1-epoch warmup + cosine; AMP; all visible GPUs; best checkpoint by val PQ; early stopping | `scripts/train.py`, `fseg_unet/data.py` |
| Instances | Upsample probabilities to 2048, threshold, keep the disk, 8-connected components, drop components below a min area | `fseg_unet/postprocess.py` |
| Eval | PQ (IoU > 0.5 match) scored against every annotator's record and averaged; pooled and per-image PQ, SQ, RQ, precision/recall, split and merge rates, human-agreement ceiling; per-record Dice/IoU at 512 like the baselines | `fseg_unet/pq.py`, `fseg_unet/evaluate.py` |
| Tune + submit | Sweep threshold x min area on val PQ (optional flip TTA), then write `submission.csv` (`filament_id,segmentation_rle`, RLE counts at 2048 x 2048) | `scripts/infer.py` |

Only segmentation polygons are used for training: the rules forbid other ground-truth metadata
(spines, boxes, category labels).

## Time-blocked split (branch `charan/newdatasplit`)

`splits/week_split.json` (from `scripts/make_split.py`) assigns whole 7-day blocks to train or val, stratified
by year: 546 train / 161 val images. Filaments persist for days and the Sun rotates ~13 deg/day, so the
by-image split above leaks near-duplicates (20 val images share a day with a training image, 45 more are one
day away). The blocked split has no same-day pairs and a median gap of 5 days to the nearest training day.
It is not chronological on purpose: the test set is a random sample of the same 2011-2022 span, and the year
stratification keeps every phase of the solar cycle on both sides. Use it with `prepare.py --split splits/week_split.json`.
Val PQ on this split is not comparable with numbers on Sheikh's split; compare on the leaderboard.

## Run on Kaggle (GPU T4 x2, Internet on)

Open `notebooks/kaggle_train_unet.ipynb` on Kaggle, add the competition data, and **Save Version -> Save & Run All**
so it runs without the browser open. Or from a shell:

```bash
pip install segmentation-models-pytorch pycocotools
python scripts/prepare.py --size 1024                      # ~5 min, also prints the annotator ceiling
python scripts/train.py --epochs 40 --batch 8               # ~1.5-2.5 h on 2x T4; --resume continues
python scripts/infer.py sweep --weights work/runs/unet_r34/best.pt --tta
python scripts/infer.py test  --weights work/runs/unet_r34/best.pt \
    --config work/runs/unet_r34/pq_config.json --out submission.csv
```

Outside Kaggle set `UNET_DATA_ROOT` (the `MAGFiLO_1.0_Kaggle_2026` folder) and optionally `UNET_WORK_DIR`.
`best.pt` is saved as fp16 weights (~49 MB for ResNet34), under GitHub's 100 MB limit, because the organizers
require weights in the repo.

## Tests

```bash
python tests/test_pq.py                                    # PQ + post-processing unit tests
python tests/make_synthetic.py /tmp/synth                  # fake disk images for a CPU smoke run
UNET_DATA_ROOT=/tmp/synth UNET_WORK_DIR=/tmp/work python scripts/prepare.py --size 256 --split /tmp/synth/split.json
UNET_DATA_ROOT=/tmp/synth UNET_WORK_DIR=/tmp/work python scripts/train.py --encoder resnet18 --no-pretrained \
    --epochs 40 --batch 2 --lr 1e-3 --pq-min-area 100 --pq-every 5
```

On the synthetic set the smoke run reaches val PQ 0.78 (annotator ceiling 0.875), which confirms the pipeline learns end to end.

## Next

Distance/boundary head + watershed to split touching filaments; sliding-window inference at full resolution;
skeleton-aware loss for thin barbs; encoder and resolution ablations.
