"""
Evaluate a trained U-Net using the Panoptic Quality (PQ) metric.

This script:
1. Loads the trained U-Net checkpoint.
2. Uses the existing physical-image-level validation split.
3. Preserves each ground-truth filament as a separate instance.
4. Tests different probability thresholds and minimum component areas.
5. Calculates PQ using the competition evaluator's matching rule.
6. Saves the results to outputs/unet/pq_threshold_sweep.csv.

Run from the project root:

    python src/evaluate_unet_pq.py
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

from train_unet import UNet


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_IMAGE_DIR = PROJECT_ROOT / "data" / "train" / "train_images"

ANNOTATIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "train"
    / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "threshold_eval"
    / "grouped_split.json"
)

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "unet" / "best_unet.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "unet"

RESULTS_PATH = OUTPUT_DIR / "pq_threshold_sweep.csv"

# These should match the image size used to train the U-Net.
MODEL_IMAGE_SIZE = 512

# The competition images and ground-truth masks use this resolution.
ORIGINAL_IMAGE_SIZE = 2048

# Probability thresholds tested by the experiment.
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# Minimum connected-component area at MODEL_IMAGE_SIZE resolution.
MIN_COMPONENT_AREAS = [1, 5, 10, 20, 50]

# The competition counts a match only when IoU is strictly greater
# than this value.
IOU_MATCH_THRESHOLD = 0.5

# Keep this consistent with the U-Net you trained.
BASE_CHANNELS = 16


# ============================================================
# 2. SELECT COMPUTING DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# 3. LOAD MODEL CHECKPOINT
# ============================================================

def load_model():
    """Load the trained U-Net and its saved weights."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Train the U-Net first or check the checkpoint path."
        )

    # The checkpoint was created by our own training script.
    # weights_only=True is appropriate for this state-dictionary checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    base_channels = checkpoint.get("base_channels", BASE_CHANNELS)
    image_size = checkpoint.get("image_size", MODEL_IMAGE_SIZE)

    if image_size != MODEL_IMAGE_SIZE:
        raise ValueError(
            f"Checkpoint image size is {image_size}, but this script "
            f"expects {MODEL_IMAGE_SIZE}. Update MODEL_IMAGE_SIZE if needed."
        )

    model = UNet(base=base_channels).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Saved validation Dice: {checkpoint.get('val_dice', 'unknown')}")
    print(f"Model image size: {image_size}")

    return model


# ============================================================
# 4. LOAD VALIDATION IMAGE SPLIT
# ============================================================


def load_validation_filenames():
    """
    Read validation filenames from the existing grouped split.

    Expected structure:
    {
        "seed": 42,
        "train": {"file_names": [...]},
        "val": {"file_names": [...]}
    }
    """

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Validation split not found: {SPLIT_PATH}"
        )

    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    if "val" not in split_data or "file_names" not in split_data["val"]:
        raise KeyError(
            "Expected validation filenames under "
            "'val' -> 'file_names' in grouped_split.json. "
            f"Available keys: {list(split_data.keys())}"
        )

    filenames = {
        Path(filename).name
        for filename in split_data["val"]["file_names"]
    }

    if not filenames:
        raise ValueError("The validation filename list is empty.")

    print(f"Validation physical images: {len(filenames)}")

    return filenames



# ============================================================
# 5. LOAD ANNOTATIONS AND GROUP THEM BY IMAGE RECORD
# ============================================================

def load_annotation_data(validation_filenames):
    """
    Load the original annotation JSON.

    Important:
    - An image record represents one annotator's annotations
      for a physical image.
    - A physical image can have multiple image records.
    - Every polygon annotation is preserved as an individual
      ground-truth filament instance.
    """

    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Annotation JSON not found: {ANNOTATIONS_PATH}"
        )

    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    images_by_id = {
        image_record["id"]: image_record
        for image_record in data["images"]
    }

    annotations_by_record = {}

    for annotation in data["annotations"]:
        record_id = annotation["image_id"]

        annotations_by_record.setdefault(record_id, []).append(annotation)

    # Keep only annotation records belonging to validation physical images.
    validation_records = {}

    for record_id, image_record in images_by_id.items():
        filename = Path(image_record["file_name"]).name

        if filename not in validation_filenames:
            continue

        validation_records[record_id] = {
            "image_record": image_record,
            "annotations": annotations_by_record.get(record_id, []),
        }

    if not validation_records:
        raise ValueError(
            "No annotation records matched the validation filenames. "
            "Check the split format and the filenames in the JSON."
        )

    physical_images = {
        Path(record["image_record"]["file_name"]).name
        for record in validation_records.values()
    }

    print(f"Validation annotation records: {len(validation_records)}")
    print(f"Matched physical images: {len(physical_images)}")

    return validation_records


# ============================================================
# 6. LOAD AND PREPROCESS ONE IMAGE
# ============================================================

def load_image_tensor(image_path):
    """
    Read a grayscale image and apply the same basic preprocessing
    used during U-Net training.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.resize(
        image,
        (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    image = image.astype(np.float32) / 255.0

    # Shape: (1, 1, 512, 512)
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# 7. PREDICT PROBABILITY MAPS
# ============================================================

def predict_probability_maps(model, validation_records):
    """
    Run inference once per physical validation image.

    Store the probability map in CPU memory so that the same
    predictions can be evaluated at multiple thresholds without
    repeatedly running the model.
    """

    # Several annotation records may refer to the same physical image.
    # Predict once per physical filename.
    unique_filenames = sorted({
        Path(record["image_record"]["file_name"]).name
        for record in validation_records.values()
    })

    probability_maps = {}

    start_time = time.time()

    for index, filename in enumerate(unique_filenames, start=1):
        image_path = TRAIN_IMAGE_DIR / filename

        input_tensor = load_image_tensor(image_path)

        with torch.no_grad():
            output = model(input_tensor)

            # The training model returns logits.
            probabilities = torch.sigmoid(output)

        probability_map = (
            probabilities[0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        probability_maps[filename] = probability_map

        if index % 10 == 0 or index == len(unique_filenames):
            elapsed = time.time() - start_time

            print(
                f"Inference: {index}/{len(unique_filenames)} "
                f"images | elapsed {elapsed:.1f}s"
            )

    return probability_maps


# ============================================================
# 8. CONVERT A GROUND-TRUTH POLYGON TO A BINARY MASK
# ============================================================

def polygon_annotation_to_rle(annotation, height, width):
    """
    Convert one COCO polygon annotation to a compressed RLE mask.

    Each annotation remains a separate instance. If the annotation
    contains multiple polygons, those polygons are merged into
    one instance mask.

    Returns:
        Compressed COCO RLE dictionary.
    """

    segmentation = annotation.get("segmentation", [])

    if not segmentation:
        return None

    # This script expects polygon segmentations, as used by the
    # training annotation file.
    if isinstance(segmentation, dict):
        # Handle an annotation that is already stored as RLE.
        rle = segmentation

        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)

            if isinstance(rle, list):
                rle = mask_utils.merge(rle)

        return rle

    polygon_rles = mask_utils.frPyObjects(
        segmentation,
        height,
        width,
    )

    if not polygon_rles:
        return None

    # A single annotation may contain more than one polygon.
    if isinstance(polygon_rles, list):
        merged_rle = mask_utils.merge(polygon_rles)
    else:
        merged_rle = polygon_rles

    return merged_rle


def build_ground_truth_rles(validation_records):
    """
    Build individual ground-truth RLE masks for every annotation record.

    Output:
        {
            record_id: [rle_1, rle_2, ...],
            ...
        }

    These RLE masks are cached so they are not repeatedly rasterized
    for every threshold/area configuration.
    """

    ground_truth_by_record = {}

    total_instances = 0

    for record_id, record in validation_records.items():
        image_record = record["image_record"]

        width = image_record.get("width", ORIGINAL_IMAGE_SIZE)
        height = image_record.get("height", ORIGINAL_IMAGE_SIZE)

        instance_rles = []

        for annotation in record["annotations"]:
            rle = polygon_annotation_to_rle(
                annotation,
                height,
                width,
            )

            if rle is None:
                continue

            instance_rles.append(rle)

        ground_truth_by_record[record_id] = instance_rles
        total_instances += len(instance_rles)

    print(f"Ground-truth filament instances: {total_instances}")

    return ground_truth_by_record


# ============================================================
# 9. CONVERT A BINARY MASK TO COMPRESSED COCO RLE
# ============================================================

def binary_mask_to_rle(binary_mask):
    """Encode a binary NumPy mask as compressed COCO RLE."""

    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))

    return mask_utils.encode(binary_mask)


# ============================================================
# 10. EXTRACT PREDICTED FILAMENT INSTANCES
# ============================================================

def probability_map_to_instances(
    probability_map,
    threshold,
    min_component_area,
):
    """
    Convert a U-Net probability map into separate filament instances.

    Steps:
    1. Apply the probability threshold at 512x512.
    2. Find connected components.
    3. Remove components smaller than min_component_area.
    4. Resize each retained component to 2048x2048.
    5. Encode each component as a separate COCO RLE.

    Returns:
        List of compressed RLE dictionaries, one per predicted filament.
    """

    binary_mask = (probability_map >= threshold).astype(np.uint8)

    number_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    predicted_instances = []

    # Label 0 represents the background.
    for label_id in range(1, number_of_labels):
        component_area = stats[label_id, cv2.CC_STAT_AREA]

        if component_area < min_component_area:
            continue

        component_mask = (labels == label_id).astype(np.uint8)

        # Return each instance to the original competition resolution.
        full_resolution_mask = cv2.resize(
            component_mask,
            (ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        if not np.any(full_resolution_mask):
            continue

        predicted_instances.append(
            binary_mask_to_rle(full_resolution_mask)
        )

    return predicted_instances


# ============================================================
# 11. CALCULATE PANOPTIC QUALITY FOR ONE RECORD
# ============================================================

def calculate_record_pq(gt_rles, pred_rles):
    """
    Calculate PQ for one annotator/image record.

    This reproduces the competition notebook's matching rule:

    - A pair is a match if IoU > 0.5.
    - Every qualifying pair is counted as a true-positive match.
    - A predicted instance is a false positive if it matches no GT.
    - A GT instance is a false negative if no prediction matches it.

    Note:
    Some standard PQ implementations use one-to-one matching.
    This function intentionally follows the competition evaluator's
    described pair-counting behavior instead.
    """

    number_gt = len(gt_rles)
    number_pred = len(pred_rles)

    # No GT and no predictions: nothing to score.
    if number_gt == 0 and number_pred == 0:
        return 0.0, 0, 0, 0, 0.0

    # There are no GT instances, so every prediction is a false positive.
    if number_gt == 0:
        return 0.0, 0, number_pred, 0, 0.0

    # There are no predictions, so every GT instance is a false negative.
    if number_pred == 0:
        return 0.0, 0, 0, number_gt, 0.0

    # pycocotools.iou expects detections first, ground truths second.
    # Output shape: (number_pred, number_gt).
    crowd_flags = [0] * number_gt

    iou_matrix = mask_utils.iou(
        pred_rles,
        gt_rles,
        crowd_flags,
    )

    # Convert to (number_gt, number_pred) for clearer interpretation.
    iou_matrix = np.asarray(iou_matrix).T

    # The official evaluator uses a strict greater-than comparison.
    matches = iou_matrix > IOU_MATCH_THRESHOLD

    # Count every qualifying pair, matching the organizer's logic.
    true_positive_pairs = int(matches.sum())

    # An instance is unmatched if it has no qualifying pair.
    gt_matched = matches.any(axis=1)
    pred_matched = matches.any(axis=0)

    false_negatives = int((~gt_matched).sum())
    false_positives = int((~pred_matched).sum())

    # Sum the IoUs of all qualifying pairs.
    sum_matched_iou = float(iou_matrix[matches].sum())

    denominator = (
        true_positive_pairs
        + 0.5 * false_positives
        + 0.5 * false_negatives
    )

    pq = (
        sum_matched_iou / denominator
        if denominator > 0
        else 0.0
    )

    return (
        pq,
        true_positive_pairs,
        false_positives,
        false_negatives,
        sum_matched_iou,
    )


# ============================================================
# 12. EVALUATE ONE CONFIGURATION ACROSS THE VALIDATION SET
# ============================================================

def evaluate_configuration(
    probability_maps,
    validation_records,
    ground_truth_by_record,
    threshold,
    min_component_area,
):
    """
    Evaluate one threshold/area configuration.

    Predictions are generated once per physical image. The same
    physical-image predictions are compared independently against
    each annotator's ground-truth record, as in the competition.
    """

    predictions_by_filename = {}

    # Generate the prediction instances for each physical image.
    unique_filenames = sorted({
        Path(record["image_record"]["file_name"]).name
        for record in validation_records.values()
    })

    for filename in unique_filenames:
        predictions_by_filename[filename] = probability_map_to_instances(
            probability_maps[filename],
            threshold,
            min_component_area,
        )

    total_tp_pairs = 0
    total_fp = 0
    total_fn = 0
    total_matched_iou = 0.0

    record_pq_values = []

    for record_id, record in validation_records.items():
        filename = Path(record["image_record"]["file_name"]).name

        gt_rles = ground_truth_by_record[record_id]
        pred_rles = predictions_by_filename[filename]

        (
            record_pq,
            record_tp,
            record_fp,
            record_fn,
            record_matched_iou,
        ) = calculate_record_pq(
            gt_rles,
            pred_rles,
        )

        total_tp_pairs += record_tp
        total_fp += record_fp
        total_fn += record_fn
        total_matched_iou += record_matched_iou

        record_pq_values.append(record_pq)

    denominator = (
        total_tp_pairs
        + 0.5 * total_fp
        + 0.5 * total_fn
    )

    # Aggregate PQ across all annotator/image records.
    overall_pq = (
        total_matched_iou / denominator
        if denominator > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "min_component_area": min_component_area,
        "pq": overall_pq,
        "mean_record_pq": float(np.mean(record_pq_values)),
        "tp_pairs": total_tp_pairs,
        "fp": total_fp,
        "fn": total_fn,
        "matched_iou_sum": total_matched_iou,
        "num_records": len(validation_records),
    }


# ============================================================
# 13. RUN THE PARAMETER SWEEP
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading model...")
    model = load_model()

    print("\nLoading validation split...")
    validation_filenames = load_validation_filenames()

    print("\nLoading annotation records...")
    validation_records = load_annotation_data(validation_filenames)

    print("\nBuilding ground-truth instance masks...")
    ground_truth_by_record = build_ground_truth_rles(validation_records)

    print("\nRunning U-Net inference...")
    probability_maps = predict_probability_maps(
        model,
        validation_records,
    )

    total_configurations = (
        len(THRESHOLDS) * len(MIN_COMPONENT_AREAS)
    )

    print("\nStarting PQ parameter sweep")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Minimum component areas: {MIN_COMPONENT_AREAS}")
    print(f"Configurations: {total_configurations}")
    print(f"IoU match threshold: > {IOU_MATCH_THRESHOLD}")

    all_results = []
    configuration_number = 0

    for threshold in THRESHOLDS:
        for min_area in MIN_COMPONENT_AREAS:
            configuration_number += 1
            start_time = time.time()

            print(
                f"\n[{configuration_number}/{total_configurations}] "
                f"threshold={threshold}, min_area={min_area}"
            )

            result = evaluate_configuration(
                probability_maps=probability_maps,
                validation_records=validation_records,
                ground_truth_by_record=ground_truth_by_record,
                threshold=threshold,
                min_component_area=min_area,
            )

            all_results.append(result)

            elapsed = time.time() - start_time

            print(
                f"PQ={result['pq']:.5f} | "
                f"Mean record PQ={result['mean_record_pq']:.5f} | "
                f"TP pairs={result['tp_pairs']} | "
                f"FP={result['fp']} | "
                f"FN={result['fn']} | "
                f"time={elapsed:.1f}s"
            )

    # Sort from highest to lowest overall PQ.
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(
        by="pq",
        ascending=False,
    ).reset_index(drop=True)

    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n" + "=" * 70)
    print("PQ PARAMETER SWEEP COMPLETE")
    print("=" * 70)

    print(f"Saved results to: {RESULTS_PATH}")

    print("\nTop 10 configurations:")
    print(
        results_df[
            [
                "threshold",
                "min_component_area",
                "pq",
                "mean_record_pq",
                "tp_pairs",
                "fp",
                "fn",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    best_result = results_df.iloc[0]

    print("\nBest configuration:")
    print(f"  Threshold: {best_result['threshold']}")
    print(f"  Minimum component area: {int(best_result['min_component_area'])}")
    print(f"  Overall PQ: {best_result['pq']:.5f}")
    print(f"  Mean record PQ: {best_result['mean_record_pq']:.5f}")

    print(
        "\nImportant: this identifies the best configuration among the "
        "tested values on your validation set. It does not guarantee "
        "that it will be the best configuration on the hidden test set."
    )


if __name__ == "__main__":
    main()