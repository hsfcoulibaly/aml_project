# Current best pipeline (pooled 5-fold OOF PQ 0.458, public LB 0.39 to 0.40)

Solar Filament Segmentation Challenge 2026 (Kaggle `filament-segmentation-2026`). Data: 707 GONG H-alpha
full-disk training images (2048 x 2048 grayscale), 1 to 3 independent annotation sets per image (1,154
annotation records, 8,199 filament polygons), 180 test images. Metric: Panoptic Quality,
PQ = sum of matched IoU / (TP + 0.5 FP + 0.5 FN), match when IoU > 0.5, scored against each annotation set.
Code: `Azumah/` in github.com/hsfcoulibaly/aml_project. Everything below runs on Kaggle (2x T4).

## Results so far (fold-0 out-of-fold, same 142 images; local PQ follows the organizers' scorer)

| Variant | PQ | SQ | RQ | Precision | Recall | Public LB |
|---|---|---|---|---|---|---|
| Human agreement (one annotator vs the others) | 0.344 | 0.635 | 0.542 | | | |
| U-Net alone (stage 2) | 0.406 | 0.676 | 0.600 | 0.628 | 0.575 | |
| YOLO11l-seg alone (stage 1) | 0.417 | 0.668 | 0.625 | 0.645 | 0.607 | 0.35 |
| YOLO + refiner (stage 3) | 0.446 | 0.683 | 0.653 | 0.714 | 0.602 | |
| YOLO + refiner + fusion (stage 4, submitted) | 0.447 | 0.683 | 0.654 | 0.721 | 0.599 | **0.40** |
| MeshNet refiner alone (7.5k params, catalyst recipe, D4) | 0.448 | 0.686 | 0.653 | 0.737 | 0.586 | |
| ResNet-34 refiner, 60 epochs (D4) | 0.456 | 0.685 | 0.665 | 0.739 | 0.604 | |
| ResNet-34 60 ep + MeshNet ensemble (D4), YOLO 6-model proposals | **0.457** | 0.688 | 0.665 | 0.703 | 0.630 | 0.39 |
| Same, strict threshold 0.2 | 0.448 | | | 0.765 | | 0.39 |

Provenance of the 0.40 submission: `submission_fused.csv` from the Kaggle run `filament-seg-refine-fuse`
(launched Nov 2, notebook `notebooks/03_refine_fuse.ipynb`, which clones the repo at launch: code commit
`0757dd8`), downloaded and submitted on Nov 10. Upstream runs it reads: `filament-seg-yolo-stage`,
`filament-seg-yolo-folds-1-2`, `filament-seg-yolo-folds-3-4`, `filament-seg-unet-stage`.

YOLO OOF PQ over all 5 folds: 0.417, 0.420, 0.405, 0.427, 0.434 (mean 0.421). Public LB: 0.40 = rank 212 of
1,477; 0.41 to 0.42 is the best honest score on the board (107 teams at exactly 0.55 submit a hard-coded
answer file from a public notebook).

## Pipeline, in run order

### Stage 0: data (`scripts/prepare.py --size 1280 --modes consensus largest`)
* 5 folds grouped by physical image (all annotation sets of an image stay in one fold), stratified by
  observing site x filament-count quartile.
* Consensus targets: instances from different annotation sets are clustered (IoU > 0.2 or > 50% cover);
  a filament is kept if at least half of the image's annotators drew it.
* Preprocessing: fit the solar disk, divide out radial limb darkening, 3 channels
  (raw, limb-flattened, CLAHE), sky set to 0.

### Stage 1: YOLO proposer (`notebooks/01_yolo_stage.ipynb`, `01b_yolo_folds.ipynb`)
```bash
python scripts/train_yolo.py --mode consensus --fold k --model yolo11l-seg.pt --imgsz 1280 --epochs 60 --batch 4
python scripts/predict_yolo.py oof  --weights runs/consensus_fk/weights/best.pt --fold k     # k = 0..4
python scripts/train_yolo.py --mode consensus --fold full ...                                 # test model
python scripts/predict_yolo.py test --weights runs/consensus_ffull/weights/best.pt --config runs/consensus_f0/pq_config.json
```
* YOLO11l-seg (27.6M params), COCO-pretrained, single class, 1280 px input (filament median width ~11 px at
  2048), AdamW lr 1e-3 cosine, 60 epochs, batch 4 on 2x T4 (batch 8 runs out of memory), mask_ratio 2,
  augmentation: 180 degree rotations, both flips, mosaic 0.5, scale 0.25, brightness 0.2.
* Inference: full image at 1280, 4-view flip TTA merged by mask voting, masks upsampled to 2048, greedy
  overlap resolution by score, off-disk filter. Thresholds (conf 0.25, min area 300) tuned for PQ on OOF.
* Output used downstream: all instances with conf >= 0.05 (`oof_raw_fold<k>.pkl`, `test_raw.pkl`).

### Stage 2: dense U-Net (`notebooks/02_unet_stage.ipynb`)
```bash
python scripts/train_unet.py --fold k --epochs 30 --batch 4      # k = 0..4, two folds at a time
python scripts/predict_unet.py oof  --folds 0 1 2 3 4 --sweep-fold 0 --tta flip
python scripts/predict_unet.py test --folds 0 1 2 3 4            # 5-fold average maps
```
* smp U-Net, ConvNeXt-tiny ImageNet encoder + residual ASPP (rates 1, 6, 12, 18 + image pooling) on the
  deepest feature, whole disk resized to 1024.
* 3 outputs: filament (soft target = fraction of annotation sets marking the pixel), centerline (skeletons of
  consensus instances, dilated 2 px), boundary (morphological gradient of consensus instances).
* Loss: 0.5 BCE (pos_weight 10) + 0.5 batch soft Dice + 0.25 clDice on filament; 0.25 centerline BCE
  (pos_weight 10); 0.1 boundary BCE (pos_weight 5). AdamW 3e-4, 1-epoch warmup + cosine, AMP, dihedral +
  on-disk gain/bias augmentation. Checkpoint by validation PQ. 30 epochs, 2 min/epoch; validation PQ was
  still rising at epoch 30 in 4 of 5 folds.
* Used downstream as maps (filament + centerline probabilities at 2048), out-of-fold for all 707 images.

### Stage 3: instance-conditioned refiner (`notebooks/03_refine_fuse.ipynb`)
```bash
python scripts/train_refiner.py --val-fold 0          # folds 1-4, validated on fold 0
python scripts/train_refiner.py --val-fold none       # all folds, used for the test set
python scripts/predict_refiner.py oof --fold 0
python scripts/predict_refiner.py test --config refiner/refiner_v0/pq_config.json
```
* Per proposal: square crop around its box with 1.5x context (min 128 px at 2048), resized to 256.
  Input (5 channels): limb-flattened, CLAHE, proposal mask, U-Net filament prob, U-Net centerline.
* Model: smp U-Net, ResNet-34 ImageNet encoder (24M params), mask output + scalar quality head that predicts
  the IoU of its own mask with the matched ground truth (0 for false positives).
* Training samples, mixed 40/40/20: consensus filaments with a corrupted copy of themselves as the prior
  (4x down-up sampling like YOLO prototypes, erode/dilate, shift, dropped segments, bleed into neighbours);
  real out-of-fold YOLO proposals matched at IoU > 0.2 (target = that filament); unmatched proposals (IoU < 0.1)
  as negatives (empty target, quality 0).
* Loss: BCE + Dice on mask, smooth L1 on quality. AdamW 3e-4, warmup + cosine, 30 epochs (8.8 min/epoch,
  data-loading bound).
* Inference: refine every proposal with conf >= 0.05, hflip TTA, paste back at 2048, score =
  quality x YOLO conf, greedy overlap resolution, keep score >= 0.1 and area >= 300.

### Stage 4: topology-aware fusion (same notebook)
```bash
python scripts/fuse.py oof  --source refiner --fold 0
python scripts/fuse.py test --source refiner --config fusion/refiner_f0/fusion_config.json
```
* Merge two instances when skeleton endpoints face each other (gap <= 120 px, both end directions within
  30 degrees of the gap) and the mean U-Net probability along the gap is >= 0.2; bridge with U-Net pixels.
* Also implemented, rejected by the PQ sweep: recovering U-Net-only components, elongation shape filter.
* Final: score >= 0.1, area >= 300. Gain over stage 3: +0.001.

## Known weaknesses

1. Recall is the weak side (about 0.60 at every stage): about 3 true filaments missed per image.
2. Public LB is about 0.05 below OOF for every model (0.417 -> 0.35, 0.447 -> 0.40). Public notebooks report the
   same gap and attribute it to sparser test labels, which makes every extra prediction cost more.
3. Test proposals come from the single full-data YOLO model, not the 5-fold ensemble.
4. The refiner was data-loading bound (30 epochs only); the U-Net stopped while still improving.
5. Fusion barely helps once the refiner is in: splitting is already down to 1.6%.

## Nov 11 run (`filament-seg-meshnet-ensemble`, notebook 04)

Fold-0 OOF gain +0.010 over the 0.40 submission, but public LB 0.39 for both the best and the strict
threshold. At two decimals on the public subset these differences are noise, so the final pick goes by OOF.

## Nov 12 run (`filament-seg-refiner-5fold`, notebook 05): 5-fold refiner, pooled tuning

ResNet-34 refiner (60 epochs) trained once per held-out fold; every training image refined by a model that
never saw it; ranking, threshold and min area tuned once on all 707 images (conf, thr 0.3, min area 150, D4).

| Fold | n | YOLO | Refiner | Gain |
|---|---|---|---|---|
| 0 | 142 | 0.417 | 0.452 | +0.035 |
| 1 | 142 | 0.420 | 0.458 | +0.038 |
| 2 | 141 | 0.405 | 0.444 | +0.040 |
| 3 | 141 | 0.427 | 0.474 | +0.047 |
| 4 | 141 | 0.434 | 0.462 | +0.028 |
| **Pooled** | **707** | **0.421** | **0.458** | **+0.037** |

* The refiner gain holds on every fold. Fold 0 with pooled settings is 0.452 against 0.456 with fold-0-tuned
  settings: tuning on one fold was about 0.004 optimistic.
* The top of the pooled sweep is flat (many settings within 0.0002), so the threshold choice is not fragile.
* Public LB: 6-refiner vote 0.39; same refiners on single-model proposals 0.39. Every variant from OOF 0.447 to
  0.458 lands on 0.39 or 0.40, so the public board (two decimals, about 90 images) cannot rank them.
  Final pick goes by pooled OOF: `submission_r60x6_pooled.csv` (select manually; auto-select takes the 0.40).

## Nov 13 run (`filament-seg-unet-longer`, notebook 06): 50-epoch U-Net under the same refiner

U-Net warm-started from the 30-epoch weights, 20 more epochs (fresh warmup + cosine, lr 1e-4). Standalone
U-Net val PQ rose on every fold (0.378/0.386/0.380/0.404/0.388 to 0.382/0.386/0.384/0.412/0.397). The run hit
Kaggle's 12 h limit after the OOF step; pooled tuning was done on CPU from its OOF pickles.

| Fold | YOLO | Refiner, 30-ep maps | Refiner, 50-ep maps |
|---|---|---|---|
| 0 | 0.417 | 0.452 | 0.449 |
| 1 | 0.420 | 0.458 | 0.460 |
| 2 | 0.405 | 0.444 | 0.444 |
| 3 | 0.427 | 0.474 | 0.475 |
| 4 | 0.434 | 0.462 | 0.464 |
| **Pooled** | **0.421** | **0.458** | **0.458** |

Pooled 0.4582 against 0.4579: no gain. Better U-Net maps do not reach the final PQ once the refiner is in;
the refiner is the bottleneck, not its U-Net inputs. No test submission made for this variant.

## In progress (Nov 10)

* Refiner crop cache in memory (epochs become GPU-bound), then a 2D MeshNet refiner (13 dilated 3x3 layers,
  rates 1, 3, 5, 7, 13, 19, 31 and back, receptive field 255 = crop size, a few thousand parameters,
  per-channel GroupNorm, GELU) trained with the catalyst recipe (AdamW wd 1e-2, OneCycle to lr 1e-2 with 5%
  warmup, grad clip 1.0, accumulation 2, label smoothing 0.01 + generalized Dice, long training).
* 5-fold + full YOLO ensemble for test proposals, D4 TTA in the refiner, ResNet + MeshNet refiner ensemble.
* Deadline Sat Nov 14, 8:00 pm EST: 4-page report, public repo with pinned requirements and weights, end-to-end notebook.
