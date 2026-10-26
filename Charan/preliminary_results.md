# Preliminary results: PQ-evaluated U-Net (Charan)

Run 1, 2026-10-26. Code in [`unet/`](unet/README.md), trained with
[`unet/notebooks/kaggle_train_unet.ipynb`](unet/notebooks/kaggle_train_unet.ipynb) on Kaggle (2x T4).
All numbers are on the team's shared validation split (Sheikh's `grouped_split.json`: 566 train / 141 val
physical images, 217 val annotation records). No leaderboard submission yet.

## Setup

| | |
|---|---|
| Model | U-Net (`segmentation_models_pytorch`), ResNet34 encoder, ImageNet-pretrained |
| Input | 1024 x 1024, 3 channels: raw, limb-darkening-corrected, CLAHE; off-disk sky zeroed |
| Loss | 0.5 BCE (pos_weight 10) + 0.5 soft Dice |
| Training | AdamW 3e-4, 1-epoch warmup + cosine, AMP, batch 8, flips/rotations + brightness jitter; best checkpoint by val PQ, early stopping (patience 10) |
| Instances | Upsample to 2048, threshold, keep the solar disk, connected components, drop components below a min area |
| Post-processing | Threshold x min-area grid tuned on val PQ; horizontal/vertical flip TTA |

## Headline numbers

Chosen post-processing: threshold **0.6**, min area **400 px** (at 2048), flip TTA.

| Metric | Value |
|---|---:|
| **PQ (pooled)** | **0.410** |
| PQ (mean per image) | 0.407 |
| SQ (mean IoU of matched filaments) | 0.673 |
| RQ (F1 of filament detection) | 0.608 |
| Precision / recall | 0.631 / 0.588 |
| Predicted / true filaments per image | 7.2 / 7.7 |
| Split rate / merge rate | 6.7% / 1.7% |
| Dice / IoU per record (512 px, baseline protocol) | 0.670 / 0.514 (median Dice 0.686) |

PQ is computed against every annotator's record for an image and averaged; a match needs IoU > 0.5.

## Comparison (same split)

| Model | Dice | IoU | PQ |
|---|---:|---:|---:|
| Threshold baseline (Sheikh) | 0.396* | 0.258* | — |
| U-Net, 512 px (Sheikh) | 0.631 | 0.471 | 0.269** |
| YOLO11s-seg (Hassan) | 0.650 | 0.493 | — |
| **U-Net ResNet34, 1024 px (this run)** | **0.670** | **0.514** | **0.410** |
| One annotator vs. the others | — | 0.445 | 0.344 |

\* 19 records only. \*\* Sheikh's PQ is averaged per record with a separate evaluator (threshold 0.3, min area 50),
so it is close to but not computed identically to ours.

## Training curve

| Epoch | Train loss | Val Dice | Val PQ* |
|---:|---:|---:|---:|
| 1 | 0.773 | 0.403 | 0.033 |
| 5 | 0.299 | 0.610 | 0.277 |
| 10 | 0.191 | 0.652 | 0.353 |
| 15 | 0.177 | 0.655 | 0.377 |
| 21 | 0.171 | 0.664 | 0.387 |
| **29 (best)** | 0.160 | 0.666 | **0.397** |
| 39 (early stop) | 0.155 | 0.669 | 0.395 |

\* In-loop PQ at fixed threshold 0.5 / min area 200, no TTA. Tuning post-processing and adding TTA raise it to 0.410.

About 2 minutes per epoch, 1.6 h total. PQ climbs fast until epoch ~6, then plateaus around 0.39–0.40 from
epoch 21, so longer training alone is unlikely to help.

## Post-processing sweep (top configs)

| Threshold | Min area | PQ | Precision | Recall | Pred / image |
|---:|---:|---:|---:|---:|---:|
| **0.6** | **400** | **0.410** | 0.631 | 0.588 | 7.2 |
| 0.5 | 400 | 0.409 | 0.624 | 0.600 | 7.4 |
| 0.7 | 400 | 0.405 | 0.629 | 0.571 | 7.0 |
| 0.6 | 200 | 0.403 | 0.573 | 0.629 | 8.4 |

The threshold barely matters (0.5–0.7 are within 0.005 PQ). Min area matters more: dropping small components
trades a little recall for much better precision.

## Breakdown

**By observatory site** (val images; PQ mean per image):

| Site | Images | PQ | FP / image | FN / image |
|---|---:|---:|---:|---:|
| Bh | 19 | 0.407 | 2.6 | 3.1 |
| Ch | 27 | 0.381 | 2.8 | 3.7 |
| Lh | 31 | 0.412 | 2.5 | 2.8 |
| Mh | 19 | 0.392 | 3.5 | 3.2 |
| Th | 25 | 0.424 | 2.2 | 3.5 |
| Uh | 20 | 0.426 | 2.4 | 2.7 |

Performance is fairly even across sites. Ch has the most misses and Mh the most false positives.

**By number of filaments in the image:** PQ is 0.39–0.44 across all bins (≤4, 5–8, 9–12, >12 filaments), so
crowded images are not much harder than sparse ones.

**Per image:** median PQ 0.418, interquartile range 0.337–0.490. Two images score 0 (each has a single true
filament that was missed or matched with IoU ≤ 0.5).

## Takeaways

1. **The 1024 px input and PQ-based model selection pay off.** This run has the best Dice of the three models
   on the shared split, and PQ goes from 0.269 to 0.410 over the 512 px U-Net.
2. **It agrees with the annotators more than they agree with each other** (0.410 vs. 0.344). Being scored against
   several annotators rewards predicting the consensus, which the model learns.
3. **Recall is the weak side** (0.59): about 3 true filaments are missed per image.
4. **Over-splitting is the main instance error.** Splits (6.7%) are 4x as frequent as merges (1.7%): long
   filaments break into pieces, and each piece fails the IoU > 0.5 match.

## Next steps

- Join broken pieces of the same filament: morphological closing / gap bridging before connected components, or a
  distance/boundary head with watershed.
- Raise recall: lower threshold combined with gap bridging, a larger encoder, or full-resolution sliding-window inference.
- Submit to the leaderboard to check that the validation PQ carries over to the test set.
- Commit the fp16 weights (49 MB) to the repo; the root `.gitignore` currently blocks `*.pt`.
