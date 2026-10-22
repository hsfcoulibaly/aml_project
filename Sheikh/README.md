# Solar Filament Segmentation Using Deep Learning

## 1. Project Overview

This project investigates computer vision and deep learning methods for detecting and segmenting solar filaments in grayscale solar images.

The project uses the **Solar Filament Segmentation 2026** dataset. Our initial objective is to build a binary semantic segmentation model that identifies which pixels belong to solar filaments. We began with traditional image-processing techniques and then implemented a U-Net model using PyTorch.

The long-term objective is to improve segmentation quality and develop the instance-level prediction pipeline required for the competition submission.

### Project objectives

- Explore the dataset and understand its annotation structure.
- Convert polygon annotations into binary segmentation masks.
- Establish a traditional image-processing baseline.
- Implement and train a U-Net model.
- Evaluate segmentation performance using Dice and Intersection over Union (IoU).
- Investigate improvements to segmentation quality.
- Develop instance-level filament extraction and competition submission functionality.

## 2. Dataset Overview

The dataset consists of grayscale solar images and polygon annotations describing filament regions.

| Dataset property | Count |
|---|---:|
| Physical training images | 707 |
| Annotation records | 1,154 |
| Polygon annotation objects | 8,199 |
| Training physical images | 566 |
| Validation physical images | 141 |
| Prepared training annotation records | 937 |
| Prepared validation annotation records | 217 |
| Original image resolution | 2048 × 2048 |
| Model input resolution | 512 × 512 |

### Annotation structure

The annotation file, `MAGFiLO_1.0_Annotations_kaggle2026_train.json`, contains separate `images` and `annotations` lists. Each annotation references an image through `image_id`.

Some physical images have multiple annotation records. We preserve these records as separate training examples, with each record's polygons combined into its own binary mask.

To prevent data leakage, the train-validation split is grouped by physical image filename. All annotation records belonging to the same physical image remain in the same split.

### Binary masks

Polygon annotations are converted into binary masks:

- `0`: background
- `1`: filament

Images and masks are resized to 512 × 512. Images are normalized to floating-point values in the range [0, 1], while masks remain binary.

Resizing reduces computational cost but may remove very thin filament structures.

## 3. Project Environment

Development environment:

- macOS
- Python 3.14.4
- PyTorch 2.14.1
- Apple Metal Performance Shaders (MPS)
- Visual Studio Code
- Python virtual environment: `.venv`

Main libraries include PyTorch, NumPy, OpenCV, Matplotlib, and pycocotools.

Activate the environment from the project root:

```bash
source .venv/bin/activate
```

## 4. Work Completed

### Stage 1: Dataset exploration

We inspected the dataset files, image dimensions, annotation categories, polygon structures, and relationships between physical images and annotation records.

This established the distinction between physical images and annotation records, which is important for preparing the training and validation datasets correctly.

### Stage 2: Annotation analysis and visualization

We created scripts to visualize polygon-derived masks and compare multiple annotation records belonging to the same physical image.

This helped verify that the annotations represent filament structures and understand how the different annotation records vary.

### Stage 3: Traditional threshold-based baseline

Before training a neural network, we implemented a traditional image-processing approach using:

- Solar disk detection.
- Local image-contrast estimation.
- Intensity residual thresholding.
- Morphological operations.
- Connected-component filtering.

The initial global thresholding approaches performed poorly. The local-contrast approach improved the results on the sampled images.

An evaluation on a limited sample of 10 physical images and 19 annotation records showed that a minimum connected-component area of 500 pixels was a promising setting among the tested parameters.

This was a preliminary baseline, not a complete evaluation across the entire validation dataset.

### Stage 4: Dataset preparation for U-Net

We developed `src/prepare_unet_data.py` to convert the original dataset into image-mask pairs suitable for PyTorch.

During initial preparation, we discovered that the script incorrectly expected annotations to be nested inside individual image records. In the actual dataset, annotations are stored separately and linked through `image_id`.

We corrected the script to group annotation objects by `image_id`.

The successful preparation run produced:

- 937 training annotation records.
- 217 validation annotation records.
- Zero skipped records.
- Zero empty masks.

A known example was verified: record `040301-20140609195854Bh` has 12 matching annotation objects and 16,917 foreground pixels in the original-resolution mask. Its resized 512 × 512 mask contains 1,054 foreground pixels.

### Stage 5: PyTorch dataset verification

We tested the data loader and confirmed that it produces correctly shaped image and mask batches.

Example output:

```text
Training records: 937
Validation records: 217
Image batch shape: torch.Size([4, 1, 512, 512])
Mask batch shape: torch.Size([4, 1, 512, 512])
Mask unique values: tensor([0., 1.])
PyTorch device: mps
```

The tested batch had a foreground fraction of approximately 0.21%, demonstrating substantial foreground-background imbalance.

### Stage 6: First U-Net implementation

We implemented a compact U-Net using PyTorch.

The model consists of:

- An encoder that extracts increasingly abstract image features.
- A bottleneck that processes deeper features.
- A decoder that restores spatial resolution.
- Skip connections that preserve spatial information.
- A single-channel output representing filament probability logits.

The model contains approximately 1.94 million parameters.

Training configuration:

| Setting | Value |
|---|---|
| Input resolution | 512 × 512 |
| Batch size | 4 |
| Training epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | 0.5 × BCE + 0.5 × Dice loss |
| Device | Apple MPS |
| Prediction threshold | 0.5 |

The combined loss uses binary cross-entropy and Dice loss to optimize pixel classification and region overlap.

### Stage 7: U-Net training

The first training run completed successfully.

| Epoch | Training loss | Validation loss | Validation Dice |
|---:|---:|---:|---:|
| 1 | 0.6391 | 0.5325 | 0.4511 |
| 2 | 0.4706 | 0.4265 | 0.3573 |
| 3 | 0.2896 | 0.2236 | 0.6340 |
| 4 | 0.2206 | 0.2088 | 0.6397 |
| 5 | 0.2049 | 0.2056 | 0.6432 |
| 6 | 0.1995 | 0.2130 | 0.6329 |
| 7 | 0.1938 | 0.1948 | **0.6563** |
| 8 | 0.1950 | 0.1954 | 0.6522 |
| 9 | 0.1922 | 0.2326 | 0.5909 |
| 10 | 0.1858 | 0.2650 | 0.5448 |

The best checkpoint was saved at epoch 7, with a reported validation Dice of 0.6563.

Validation performance declined during the final epochs while training loss continued to decrease, suggesting possible overfitting.

### Stage 8: Prediction visualization

We created `src/visualize_unet_predictions.py` to compare:

1. Original solar images.
2. Ground-truth masks.
3. U-Net predictions.
4. Overlays comparing predictions and ground truth.

Eight sampled validation records had a mean Dice score of 0.6465, with individual scores ranging from 0.4890 to 0.7405.

The visualizations show that the model detects many major filament structures but also misses some regions and predicts extra fragments. Thin filament structures remain challenging.

### Stage 9: Full validation evaluation and threshold testing

We created `src/evaluate_unet.py` to evaluate the saved checkpoint across all 217 validation annotation records at five prediction thresholds.

| Threshold | Mean Dice | Median Dice | Mean IoU |
|---:|---:|---:|---:|
| 0.3 | 0.6289 | 0.6551 | 0.4684 |
| 0.4 | 0.6299 | 0.6572 | 0.4697 |
| 0.5 | 0.6310 | 0.6591 | 0.4709 |
| 0.6 | 0.6316 | 0.6592 | 0.4717 |
| 0.7 | 0.6322 | 0.6581 | 0.4725 |

The best mean Dice and mean IoU among the tested thresholds occurred at 0.7. However, the improvement over the default threshold of 0.5 was small.

The current reference baseline is therefore:

- Mean Dice: **0.6310** at threshold 0.5.
- Median Dice: **0.6591** at threshold 0.5.
- Mean IoU: **0.4709** at threshold 0.5.

The threshold of 0.7 is a candidate for further testing, not a definitively optimal threshold.

Detailed results were saved in `outputs/unet/threshold_evaluation.csv`.

## 5. Current Project Status

**Completed milestone: First U-Net semantic segmentation baseline.**

The project has progressed from dataset exploration and traditional image processing to a trained deep-learning model evaluated across the complete validation annotation-record set.

Key artifacts include:

- `src/prepare_unet_data.py`
- `src/test_unet_dataset.py`
- `src/train_unet.py`
- `src/visualize_unet_predictions.py`
- `src/evaluate_unet.py`
- `outputs/unet/best_unet.pth`
- `outputs/unet/validation_predictions.png`
- `outputs/unet/threshold_evaluation.csv`

## 6. Remaining Work

The next phase will focus on improving performance and preparing the competition pipeline.

1. Experiment with data augmentation and early stopping.
2. Compare new models against the existing baseline using the same validation split and evaluation method.
3. Investigate false positives, missed filaments, and thin-structure preservation.
4. Consider higher-resolution or patch-based training.
5. Investigate loss-function and training-configuration improvements.
6. Develop instance-level filament extraction and prepare the required competition submission format.
7. Document final experiments, limitations, and conclusions.

## 7. Reproducing the Current Baseline

Run these commands from the project root with the virtual environment activated.

Prepare image-mask pairs:

```bash
python3 src/prepare_unet_data.py
```

Test the data loader:

```bash
python3 src/test_unet_dataset.py
```

Train the original U-Net:

```bash
python3 src/train_unet.py
```

Visualize predictions:

```bash
python3 src/visualize_unet_predictions.py
```

Evaluate the saved checkpoint:

```bash
python3 src/evaluate_unet.py
```

The current best checkpoint is `outputs/unet/best_unet.pth`.

## 8. Limitations

- The current model performs binary semantic segmentation, not instance segmentation.
- Multiple annotation records may exist for a physical image; each record is currently used as a separate training example.
- The grouped split prevents physical-image overlap between training and validation.
- Resizing to 512 × 512 may remove very thin filament structures.
- The full validation metrics are record-level metrics and may be affected by differences among annotations for the same physical image.
- The threshold baseline was evaluated on a limited sample and has not yet been compared comprehensively with U-Net.
- The current results do not represent final competition performance. Instance-level extraction and submission validation remain necessary.

## 9. Conclusion

The first phase of the project is complete. We explored the dataset, analyzed and visualized annotations, implemented a traditional image-processing baseline, prepared the dataset, trained a U-Net, and evaluated its predictions across all 217 validation annotation records.

The best training checkpoint was obtained at epoch 7. The full validation evaluation produced a mean Dice of 0.6310 and a mean IoU of 0.4709 at threshold 0.5.

The next phase will investigate improvements to segmentation quality and develop the instance-level pipeline needed for the competition submission.
