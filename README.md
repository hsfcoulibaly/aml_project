# Solar Filament Segmentation (AML course project)

Project for Advanced Machine Learning (CS 8850) with Dr. Sergey Plis.

We are competing in the Kaggle [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(MAGFiLO dataset, GONG H-alpha). The task is to predict every **individual solar filament** as its own mask on
2048 x 2048 full-disk grayscale H-alpha images. It is instance segmentation, not classification or a single
foreground mask.

## Competition at a glance

| | |
|---|---|
| Metric | **Panoptic Quality (PQ)** = sum of matched IoU / (TP + 0.5 FP + 0.5 FN); a match needs IoU > 0.5 |
| Data | 707 training images (1,154 annotation records, 8,199 filaments; many images have several annotators), 180 test images |
| Submission | CSV `filament_id,segmentation_rle`, one row per predicted filament, `filament_id = <image stem>_<k>`, RLE as a pycocotools counts string at 2048 x 2048 |
| Limits | 5 submissions per day, 2 final selections |
| Deadline | **2026-11-15 06:00 UTC** (Nov 15, 1:00 AM EST) |
| Rules | Train on segmentation masks only (spines, boxes and category labels are not allowed as training signal). External and pretrained models are allowed |

Organizer judging is 70% quantitative (PQ, Dice/IoU distributions, split/merge behavior) and 30% qualitative
(pipeline description, morphology, code quality). The public leaderboard top is about 0.70 PQ, and the organizers
consider PQ > 0.35 a strong result.

## Repository layout

Each member owns a folder and builds their own model; models are compared against the U-Net and YOLO baselines.

| Folder | Owner | Contents |
|---|---|---|
| `Varshi/` | Amirthavarshani | Data description, cleaning and EDA notebook (inter-annotator IoU 0.445) |
| `Sheikh/` | Sheikh Tijan Hydara | EDA scripts, classical threshold baseline, 512 px U-Net, PQ evaluation and the first submission file (`Sheikh/src/`, `Sheikh/outputs/`) |
| `Hassan/` | Foungnigue Souleymane Hassan Coulibaly | YOLO11s-seg baseline notebook (Colab) and run report |
| `Azumah/` | Richmond Azumah | `fseg` package: YOLO -> U-Net cascade with a PQ evaluator (in progress) |
| `Charan/` | Charan Tej Peeriga | PQ-evaluated ResNet34 U-Net at 1024 px; code is on branch `charan/unet` (`Charan/unet/`) |
| `figures/`, `Results/`, `EDA/`, `reports/` | shared | Figures, result notes, EDA notes, report drafts |

## Validation split

All baselines below use Sheikh's grouped split (`Sheikh/outputs/threshold_eval/grouped_split.json`): random
80/20 by **physical image**, seed 42, giving 566 train / 141 val images (937 / 217 annotation records).
Splitting by image matters because one image can have several annotation records, which would leak between train and val.

## Results so far (validation)

| Model | Dice | IoU | PQ | Notes |
|---|---:|---:|---:|---|
| Threshold baseline | 0.396 | 0.258 | — | 19 records only |
| U-Net, 512 px (Sheikh) | 0.631 | 0.471 | 0.269 | PQ at threshold 0.3, min area 50 |
| YOLO11s-seg (Hassan) | 0.650 | 0.493 | — | conf 0.25; mask mAP50 0.659 |
| **U-Net ResNet34, 1024 px (Charan)** | **0.670** | **0.514** | **0.410** | threshold 0.6, min area 400, flip TTA |
| Annotator vs. annotator | — | 0.445 | 0.344 | how much human annotators agree with each other |

Dice/IoU are per-record binary scores at 512 px. PQ is instance-level at full resolution. Sheikh's PQ is
averaged per record and Charan's is pooled across annotators for each image, so the two PQ numbers are close
but not computed identically. No leaderboard score has been recorded yet.

## Next steps

- Make leaderboard submissions and settle on one shared PQ evaluator.
- Reduce over-splitting of long filaments (bridge gaps, or add a boundary/distance head with watershed) and raise recall.
- Resolve `.gitignore` blocking `*.pt` / `*.pth`, since the final package needs weights in the repo (fp16, under GitHub's 100 MB limit).
- Merge the model tracks into one reproducible pipeline notebook for the final submission.
