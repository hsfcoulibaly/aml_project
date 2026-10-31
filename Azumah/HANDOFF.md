# Handoff: solar filament segmentation (CS 8850 group project)

Read this whole file before writing code. It records what exists, the conventions the code relies on,
and the specification for every remaining stage.

## 1. The task

* Kaggle: Solar Filament Segmentation Challenge 2026 (`filament-segmentation-2026`), part of the IEEE
  BigData 2026 Cup. Data: MAGFiLO, GONG H-alpha full-disk images, 2048 x 2048 grayscale JPEG.
* Train: 707 physical images, 1,154 COCO "image" records (1 to 3 independent annotation sets per
  image), 8,199 filament polygons. Test: 180 images.
* **Leaderboard metric: Panoptic Quality (PQ)**, not Dice. A prediction matches a ground-truth
  filament when IoU > 0.5. PQ = sum of matched IoU / (TP + 0.5 FP + 0.5 FN). Every spurious fragment
  and every missed filament costs score; splitting one filament into pieces is punished heavily.
* Final judging rubric: 70% quantitative (PQ, Dice and IoU distributions, one-to-many and
  many-to-one counts, i.e. fragmentation and over-merging), 30% qualitative (pipeline description,
  visual morphology, code quality and documentation). End-to-end computational efficiency is also
  considered.
* **Deadline: Nov 15, 2026, 11:59 pm CDT.** Deliverables: a 4-page technical report (organizers'
  Overleaf template, single PDF), a public Git repo with `requirements.txt` (pinned versions) and a
  Jupyter notebook showing the whole pipeline, submitted through the organizers' Google form.
* Rules: only the provided test images at inference. **Do not train on other ground-truth metadata**:
  the JSON has `spine` and chirality fields; we use masks only and derive centerlines by skeletonizing
  masks (`fseg.labels.skeleton`).
* Reference points: best honest public notebook (Mask R-CNN + crop U-Net refiner) scores PQ 0.33.
  The organizers said any PQ above 0.35 is of great value; the cluster of leaderboard scores near 0.55
  is believed to exploit the metric. Two human annotators agree at roughly PQ 0.35 to 0.5
  (`prepare.py` prints the exact ceiling).

## 2. The approach (topology-aware YOLO to U-Net cascade)

YOLO knows *which* filament but draws coarse masks; U-Net draws precise pixels but cannot separate
instances. So:

1. Preprocess: remove limb darkening, mask the sky. **(done)**
2. YOLO11-seg proposes every filament instance with a confidence. **(done)**
3. U-Net predicts full-resolution filament probability, centerline and boundary maps. **(to build)**
4. An instance-conditioned refiner turns each YOLO proposal into a precise mask for that one filament,
   plus a quality score. **(to build)**
5. Topology-aware fusion joins collinear fragments, recovers U-Net-only detections, removes blobs, and
   resolves overlaps. **(to build)**
6. Calibrate every threshold for PQ on out-of-fold predictions; ensemble; write the submission. **(to build)**

The group's dilated-convolution idea is implemented as zero-initialized parallel dilated branches in
the YOLO neck (`--dilated p4,p5`), reported as an ablation.

## 3. Status

### Done and tested (on a synthetic dataset; unit tests pass)

| Area | File | What it does |
|---|---|---|
| Paths | `fseg/config.py` | Auto-finds data on Kaggle; elsewhere `FSEG_DATA_ROOT`, `FSEG_WORK_DIR` |
| RLE | `fseg/rle.py` | encode/decode, pairwise IoU and intersection on COCO RLE |
| Labels | `fseg/labels.py` | Per-observation manifest, 5 folds (stratified by site and count), multi-annotator clustering into `consensus` targets with an agreement score, `largest` targets, references |
| Preprocess | `fseg/preprocess.py` | Disk fit, radial limb-darkening correction, 3-channel image (raw, flattened, CLAHE), sky zeroed |
| Metric | `fseg/metrics.py` | Organizer PQ formula against every annotation set (expectation over annotators); SQ, RQ, precision, recall, one-to-many and many-to-one rates; human ceiling |
| Post-process | `fseg/postprocess.py` | Greedy overlap resolution by score, min area, off-disk filter; `resolve_ranked` makes any confidence threshold an exact prefix (fast sweeps) |
| YOLO data | `fseg/yolo_data.py` | Preprocess all images, export single-class YOLO polygons (multi-part filaments joined), fold yamls |
| Dilation | `fseg/dilated.py` | Zero-init dilated context (rates 1, 2, 5) on P3/P4/P5 neck outputs; custom trainer; restores wrapped weights on resume |
| YOLO inference | `fseg/yolo_infer.py` | Full-res inference, flip TTA with mask voting, "dark tightening" of coarse masks, PQ sweep over conf, min area, dark ratio; submission writer |
| Scripts | `scripts/prepare.py`, `train_yolo.py` (`--resume`, MPS/CUDA/CPU), `predict_yolo.py` (`oof`, `test`), `fetch_assets.py` (offline clusters) |
| Cluster | `slurm/*.sbatch`, `slurm/env.sh` | prepare, train (resumable), predict jobs |
| Kaggle | `notebooks/01_yolo_stage.ipynb` | Stage 0 + 1 end to end |
| Tests | `tests/test_metrics.py`, `tests/make_synthetic.py` | PQ unit tests; fake MAGFiLO-shaped dataset for smoke runs |

### Real-data results so far

* YOLO11l-seg, 1280 px, consensus labels, fold 0, 60 epochs: best validation **mask mAP50 0.61,
  mAP50-95 0.19** (epoch 58), no train/val gap, still creeping up at the end
  (`figures/yolo_val_curves.png`). For context, the public YOLO11x notebook reached 0.29 mAP50, but on a
  leaky split with 4 classes, so the numbers are not directly comparable.
* Sample per-image PQ 0.28 to 0.58 (`figures/yolo_predictions.png`).
* Fold-0 out-of-fold PQ (from `predict_yolo.py oof`): **PQ 0.417, SQ 0.668, RQ 0.625** (precision
  0.645, recall 0.607, 6.8 predicted vs 7.2 true filaments per image); best config conf 0.25,
  min area 300, dark tightening off (no dark ratio from 0.85 to 0.97 helped).
* Human-agreement ceiling (printed by `prepare.py`): **PQ 0.344** (SQ 0.635, RQ 0.542).
* Kaggle public leaderboard: **PQ 0.35** for the full-data YOLO11l model (YOLO11s baseline: 0.28).
  Public notebooks report the same gap of about 0.06 between OOF and public PQ.

### Observed failure modes (drive the next stages)

1. **False positives** (dark features no annotator marked, often near active regions or the limb).
   Target: refiner quality score plus PQ-calibrated threshold, shape filter.
2. **Missed small, faint filaments**. Target: U-Net probability map plus recovery in fusion.
3. **Partial overlap / wrong extent** (IoU just under 0.5 counts as an FP plus an FN). Target: refiner.
4. Some images have a bright instrumental ring near or outside the limb; check the disk fit on those
   and make sure nothing is predicted on it.

## 4. Conventions the existing code relies on

* All masks are COCO RLE dicts at **2048 x 2048** (`fseg.rle`). Submission rows are
  `filament_id = <observation_id>_<k>` and `segmentation_rle` = RLE counts string only.
* `observation_id` = image file stem (e.g. `20140609195854Bh`; last 2 chars = GONG site).
  Split and cross-validate by observation, never by COCO record id.
* Work dir layout (`FSEG_WORK_DIR`, default `work/` or `/kaggle/working/fseg`):
  * `manifest.csv`: `observation_id, file_name, site, year, n_sets, n_largest, n_consensus, n_clusters, fold`
  * `targets.pkl`: `{"consensus"|"largest": {obs: [{"rle", "agreement", "n_sets"}]}}`
  * `references.pkl`: `{obs: [[rle, ...] for each annotation set]}`; always evaluate against this
  * `preprocessed/{train,test}/<obs>.jpg`: 2048 px, 3 channels read by cv2 as
    `[:,:,0]` raw, `[:,:,1]` limb-flattened (quiet Sun = 127), `[:,:,2]` CLAHE; sky = 0
  * `yolo/<mode>/...`: YOLO datasets; `runs/<name>/weights/{best,last}.pt`
  * `runs/<name>/oof_raw_fold{k}.pkl` and `test_raw.pkl`:
    `{obs: {"instances": [(rle, score)], "dark": {ratio: [rle]}, "disk": (cx, cy, r)}}`;
    **reuse these as the proposals for stages 3 and 4 instead of re-running YOLO**
  * `runs/<name>/pq_config.json`, `pq_sweep_fold{k}.csv`, `oof_per_image_fold{k}.csv`
* Every new stage must report `fseg.metrics.evaluate(preds, references, observations)` on the same
  fold-0 validation observations, so the ablation table is comparable. Calibrate thresholds on OOF
  data only; never tune on test.
* Every new script must work on the synthetic dataset
  (`python tests/make_synthetic.py <dir>` then `FSEG_DATA_ROOT=<dir>`), on CPU, MPS and CUDA, and be
  runnable as a SLURM job (add a `slurm/*.sbatch` next to the existing ones).
* Writing style for docs and the report: crisp and precise, no em dashes.

## 5. What to build next, with specifications

Suggested order and time budget (today is Oct 31): stage 2 by Nov 4, stage 3 by Nov 8, stage 4 by
Nov 10, stage 5 and full-data runs by Nov 12, report and repo polish Nov 12 to 15.

### Stage 2: dense U-Net branch

Files: `fseg/unet.py`, `fseg/unet_data.py`, `scripts/train_unet.py`, `scripts/predict_unet.py`.

* Input: the 3-channel preprocessed image at native resolution.
* Training: random 1024 px crops (or 768 if memory is tight) from the 2048 image, sampled so at least
  ~70% of crops contain a filament; D4 augmentation (flips, 90 degree rotations), mild gamma and
  contrast. Mixed precision.
* Model: `segmentation_models_pytorch` U-Net (encoder `tu-convnext_tiny` or `efficientnet-b4`,
  ImageNet weights) with an **ASPP block** (parallel dilated 3x3 convs, rates 1, 6, 12, 18, plus image
  pooling) on the deepest encoder feature. Add `segmentation-models-pytorch` and `timm` to
  `requirements.txt`; `fetch_assets.py` must pre-download the encoder weights.
* Three output channels:
  1. filament: **soft target** = fraction of annotation sets marking the pixel (from `references.pkl`);
  2. centerline: union of per-instance skeletons of consensus masks, dilated by 2 px;
  3. boundary: morphological gradient of consensus instance masks (3 px).
* Loss: filament BCE + soft Dice + **clDice** (soft-skeleton centerline Dice, Shit et al., CVPR 2021);
  centerline BCE with positive weight; boundary BCE. Weights as config.
* Inference: sliding window 1024 px, stride 768, Gaussian blending, optional D4 TTA, sky set to 0.
  Save per observation an 8-bit 3-channel PNG at 2048 px in `work/unet/<name>/{oof,test}/<obs>.png`
  (prob x 255) to keep disk use small.
* **Out-of-fold maps for all 707 training images are required** (the refiner trains on them), so
  train 5 fold models. If compute is short: smaller encoder, 768 crops, fewer epochs, before skipping folds.
* Standalone baseline for the ablation table: instances from the U-Net alone (threshold, then split
  with centerline-seeded watershed or connected components) scored with PQ.

### Stage 3: instance-conditioned refiner (main novelty)

Files: `fseg/refiner.py`, `scripts/train_refiner.py`, `scripts/predict_refiner.py`.

* For each proposal: square crop around its bbox with 1.5x context (min 128 px at 2048 scale), resized
  to 256 (try 384). **Input channels (5)**: limb-flattened, CLAHE, proposal prior mask, U-Net filament
  prob, U-Net centerline.
* Model: small U-Net (ResNet-34 encoder, `in_channels=5`) with two outputs: refined mask for **this
  filament only**, and a scalar **quality head** predicting the IoU of the refined mask with its matched
  ground truth (0 when the proposal matches nothing).
* Training samples, built on the fly from OOF data:
  * every consensus ground-truth instance, with the prior = a **corrupted copy** of the true mask:
    random erosion or dilation, x4 down-then-up sampling (mimics YOLO prototype coarseness), random
    shift of a few px, dropped segments, partial bleed into a neighbouring ground-truth filament;
  * real YOLO OOF proposals (`oof_raw_fold*.pkl`) matched to ground truth at IoU > 0.2, target = that
    ground truth instance;
  * unmatched YOLO proposals as **negatives**: target mask empty, quality = 0 (teaches FP rejection).
  * Other filaments inside the crop are background in the target.
* Loss: BCE + Dice on the mask, smooth L1 on quality.
* Inference: refine every proposal, paste back at 2048, final score = quality (or quality x YOLO conf;
  pick by PQ), then `postprocess.resolve_ranked` and a PQ sweep of the score threshold.
* Stage 3 is accepted when fold-0 OOF PQ beats stage 1 on the same observations.

### Stage 4: topology-aware fusion

File: `fseg/fusion.py`, unit tests in `tests/test_fusion.py`.

* **Merge**: skeletonize each instance, find endpoints and end directions; merge two instances when
  endpoints are within d px, directions differ by less than a degrees, and the mean U-Net probability
  along the connecting segment exceeds t. Merged mask = union (optionally plus the U-Net pixels along
  the gap). Annotators sometimes keep segments separate, so only keep merging if PQ goes up.
* **Recover**: U-Net components above threshold with skeleton length > L that overlap no kept
  instance become new instances, score = mean probability x a factor.
* **Shape filter**: drop instances with low elongation (skeleton length / sqrt(area)) and high
  compactness (sunspots, plage edges), and anything on or outside the limb ring.
* Resolve overlaps last. Sweep d, a, t, L, filter thresholds with PQ on OOF.

### Stage 5: final pipeline, ensembling, efficiency

* `scripts/predict_all.py`: preprocess, YOLO (+TTA), U-Net (+TTA), refiner, fusion, calibrated
  thresholds, `submission.csv`. One command, with timing per stage logged (efficiency is judged).
* Optional: average U-Net folds; YOLO fold ensemble via instance voting (reuse `yolo_infer._vote`).
* Final models trained on all data (`--fold full`) with configs calibrated on fold 0 (or all OOF).
* Update the notebook so it runs the full pipeline on Kaggle; pin versions in `requirements.txt`.

### Report and ablations (4 pages)

Ablation table, all on the same fold-0 OOF observations (PQ, SQ, RQ, one-to-many, many-to-one):
human ceiling; YOLO `largest` labels; YOLO `consensus`; + dark tightening; + dilated neck (p4,p5);
U-Net only; YOLO + refiner; + fusion (full). Figures: pipeline diagram, PQ vs confidence curve,
qualitative overlays (good, typical, failure), distributions of per-filament IoU and Dice.

## 6. Running

* Local / cluster / Kaggle instructions: `README.md`. SLURM: `slurm/env.sh` holds modules, data path,
  `IMGSZ`; resubmit a train job to resume after a time limit.
* Smoke test anything new: `python tests/make_synthetic.py /tmp/synth`, then
  `FSEG_DATA_ROOT=/tmp/synth FSEG_WORK_DIR=/tmp/work python scripts/prepare.py --size 640`.
