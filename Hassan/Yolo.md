# YOLO11-seg Baseline: Solar Filament Segmentation 2026

**Author:** Foungnigue Souleymane Hassan Coulibaly
**Status:** baseline trained, evaluated and submitted to Kaggle (mid-point, Oct 26, 2026)

This folder contains the instance-segmentation baseline for our project. It is meant to be compared directly with Sheikh's U-Net (`Sheikh/`). For that reason it uses **the same train/validation split, the same ground-truth masks and the same Dice/IoU code**.

---

## 1. Why YOLO?

The competition scores **individual filaments** (instances), not just "filament vs. background" pixels. A semantic model like the U-Net produces one binary mask, which then has to be split into separate filaments afterwards. YOLO11-seg predicts **one mask per filament directly**, with a confidence score, so it is the natural off-the-shelf instance baseline.

| | U-Net (Sheikh) | YOLO11s-seg (this folder) |
|---|---|---|
| Task type | Semantic (pixel classification) | Instance (one mask per filament) |
| Output | 1 binary mask per image | N masks + confidence scores |
| Pretrained | No (trained from scratch) | Yes (COCO) |
| Parameters | 1.94M | 10.1M |

---

## 2. Data and split

- **Data:** `MAGFiLO_1.0_Kaggle_2026` (707 training images, 1,154 annotation records, 8,199 filament polygons, 2048×2048 grayscale).
- **Split:** `Sheikh/outputs/threshold_eval/grouped_split.json` (seed 42), grouped by **physical image**, so no image appears in both train and validation.
  - Train: 566 images / 937 annotation records / 6,606 polygons
  - Validation: 141 images / 217 annotation records / 1,593 polygons
- **Multiple annotators:** as in the U-Net pipeline, each annotation record is treated as its own training sample.

### Conversion to YOLO format
Each COCO polygon becomes one line in a YOLO label file:
`0 x1 y1 x2 y2 ...`. There is one class (`filament`) and coordinates are normalised to [0, 1]. The image is linked under the record id, so every annotation record has its own label file. We checked the conversion by turning the labels back into masks and comparing them with the original polygons.

---

## 3. Training setup

| Setting | Value |
|---|---|
| Model | `yolo11s-seg.pt` (Ultralytics 8.4.189, COCO-pretrained) |
| Input size | 1024 × 1024 (higher than the default 640, so thin filaments survive) |
| Epochs / batch | 30 / 8 |
| Optimizer | Ultralytics `auto` |
| Mask prototypes | `mask_ratio=2` (finer masks than the default 4) |
| Augmentation | horizontal + vertical flips, rotation up to 90°, mosaic (p=0.5), brightness jitter; colour jitter off (grayscale) |
| Seed | 42 (`deterministic=True`) |
| Hardware / time | Google Colab, NVIDIA T4, about 57 min |
| Checkpoint | `best.pt` (best validation fitness) |

Notebook: [`yolo_filament_baseline.ipynb`](yolo_filament_baseline.ipynb)

---

## 4. Evaluation protocol (same as the U-Net)

1. **Ground truth:** polygons are rasterised at 2048×2048 and resized to 512×512 with nearest-neighbour, exactly as in `Sheikh/src/prepare_unet_data.py`.
2. **Prediction:** all YOLO instances with confidence ≥ *c* are merged into one binary mask and resized to 512×512.
3. **Metrics:** Dice and IoU per annotation record, using the same formula as `Sheikh/src/evaluate_unet.py`, then averaged over the 217 validation records.
4. **Reported threshold:** *c* = 0.25 (Ultralytics' default). We fixed this before looking at results, so it was not tuned on the validation set.

We checked that the ground-truth foreground pixel counts match the U-Net's evaluation record for record (217/217), so the two models are scored on identical targets.

---

## 5. Results

### 5.1 Main comparison (217 validation records)

| Method | Mean Dice | Median Dice | Mean IoU |
|---|---|---|---|
| Classical threshold (local contrast)* | 0.396 | – | 0.258 |
| U-Net (Sheikh), threshold 0.5 | 0.631 | 0.659 | 0.471 |
| **YOLO11s-seg, conf 0.25** | **0.650** | **0.670** | **0.493** |

\*The classical baseline was evaluated on a 19-record sample only.

![Baseline comparison](figures/baseline_comparison.png)

### 5.2 Paired comparison with the U-Net
- YOLO is better on **129** records, the U-Net on **87**, and they tie on 1.
- Mean Dice difference: **+0.019** (median +0.014). This is statistically significant under a paired **Wilcoxon signed-rank test, p = 0.005**.
- YOLO helps most on **small, thin filaments**:

| Records grouped by filament area | U-Net Dice | YOLO Dice |
|---|---|---|
| Smallest third | 0.584 | 0.613 |
| Middle third | 0.634 | 0.651 |
| Largest third | 0.676 | 0.686 |

- YOLO **over-segments** slightly: its predicted area is 1.26× the true area (median), against 1.06× for the U-Net.

### 5.3 Effect of the confidence threshold

| conf | Mean Dice | Median Dice | Mean IoU | Filaments / image |
|---|---|---|---|---|
| 0.10 | 0.623 | 0.651 | 0.463 | 15.2 |
| **0.25** | **0.650** | **0.670** | **0.493** | **8.2** |
| 0.40 | 0.598 | 0.636 | 0.449 | 5.2 |
| 0.50 | 0.529 | 0.583 | 0.388 | 3.7 |

YOLO is much more sensitive to its threshold than the U-Net, whose Dice stays between 0.629 and 0.632 for thresholds 0.3 to 0.7. Below 0.25 YOLO adds false fragments; above 0.25 it drops real filaments.

### 5.4 Standard YOLO metrics (validation)

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Box | 0.622 | 0.677 | 0.670 | 0.392 |
| Mask | 0.626 | 0.638 | **0.659** | **0.234** |

The large gap between mask mAP50 (0.659) and mask mAP50-95 (0.234) shows that YOLO finds most filaments, but its **boundaries** are not precise enough to reach the strict overlap thresholds (IoU 0.75 to 0.95).

### 5.5 Training curves

![Training curves](figures/yolo_results.png)

All validation metrics were **still increasing at epoch 30**, and the validation losses were still falling, so the model has **not converged**. Longer training should improve it. Unlike the U-Net, which overfit after epoch 7, there is no sign of overfitting yet, probably thanks to augmentation and pretraining.

### 5.6 Qualitative results

![YOLO validation predictions](figures/yolo_validation_predictions.png)
*Green = ground truth, red = prediction, yellow = agreement.*

- **Works well:** large, high-contrast filaments are captured as whole instances, with good shape.
- **Fails on:**
  - thin, faint filaments and the thin ends of long filaments are missed (green-only);
  - a few dark non-filament features are detected (red-only);
  - one long filament is sometimes split into two pieces.

> Note: Ultralytics' `confusion_matrix.png` is computed at a very low confidence, which is why it shows about 33,000 "background → filament" false positives. It does not reflect the conf 0.25 operating point used everywhere else, so we don't use it.

---

## 6. Kaggle submission

Notebook: [`yolo_submission.ipynb`](yolo_submission.ipynb). It loads `best.pt`, predicts the 2048×2048 test images and writes `submission.csv`.

**Format:** one row per predicted filament:
```
filament_id,segmentation_rle
<test_image_name>_1,<COCO compressed RLE of a 2048x2048 mask>
```

**Post-processing:**
1. Keep instances with conf ≥ 0.25, at most 100 per image.
2. Sort by confidence and make the masks **non-overlapping**: a contested pixel goes to the more confident filament.
3. Drop fragments smaller than 50 pixels.

**Result:**

| Metric | Value |
|---|---|
| Predicted filaments | 1,352 |
| Test images with at least one prediction | 178 (7.6 filaments per image on average, at most 20) |
| Median filament size | 1,661 pixels (at 2048×2048) |
| Public leaderboard score (PQ) | **TODO: fill in from Kaggle** |
| Rank (Oct 26, 2026) | about 900 |

The leaderboard uses **Panoptic Quality (PQ)**. A predicted filament only counts if it overlaps a true filament with IoU > 0.5, and PQ combines detection and segmentation quality. It is stricter than Dice, so the score is lower than the 0.650 Dice above. For reference, the human inter-annotator agreement reported in the community is about 0.36–0.39 PQ.

---

## 7. Limitations (be careful when comparing)

- **This comparison is not controlled.** Compared with the U-Net, YOLO has 5× more parameters, COCO pretraining, 4× the input area (1024² vs 512²), augmentation and 3× the epochs. The +0.019 Dice therefore does not show that the instance approach is better. It shows that **resolution, pretraining and augmentation** are worth testing one at a time on the U-Net.
- **The checkpoint was selected on the validation set** (as for the U-Net), so validation scores are slightly optimistic.
- **Dice/IoU is not the competition metric.** We still need to compute PQ on our validation split to choose thresholds that match the leaderboard.
- **Single run, single seed.**

---

## 8. Next steps

| Step | Expected benefit |
|---|---|
| Train YOLO11m-seg for 80 epochs (curves were still rising) | Higher recall and mAP |
| Train and predict at 2048 (native resolution) | Fewer missed thin filaments |
| Implement PQ on the validation split; tune conf, max detections and minimum area for PQ | Better leaderboard score |
| Hybrid: YOLO boxes + U-Net masks inside each box | Combine YOLO's detection with the U-Net's boundaries |
| Re-train the U-Net with the same recipe (1024 px, augmentation) | Fair comparison for the proposed **dilated U-Net** (Varshi) |

---

## 9. How to reproduce

1. Open `yolo_filament_baseline.ipynb` in Google Colab (*Runtime → T4 GPU*) or Kaggle.
2. Run all. On Colab, paste a Kaggle API token when asked; the data and split download automatically.
3. Training outputs are saved as `yolo_*` files (summary JSON, CSVs, figures, `yolo_best.pt`).
4. Open `yolo_submission.ipynb`, upload `yolo_best.pt`, run all, and upload `submission.csv` on the competition page.

## 10. Files

| File | Content |
|---|---|
| `yolo_filament_baseline.ipynb` | Data conversion, training, evaluation |
| `yolo_submission.ipynb` | Test prediction and `submission.csv` |
| `results/yolo_summary.json` | All headline metrics |
| `results/yolo_threshold_evaluation.csv` | Per-record Dice/IoU at each confidence |
| `results/yolo_results.csv` | Per-epoch training log |
| `figures/` | Training curves, PR curve, predictions, comparison chart |
| `yolo_best.pt` | **Not committed** (20 MB); available on request / Google Drive |
