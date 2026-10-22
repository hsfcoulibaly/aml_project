"""
Evaluate minimum-component-area thresholds across a fixed set of physical images.

Design notes
------------
* A "record" is one entry in data["images"]. Several records can describe the
  SAME physical image (same file_name) with different annotations.
* Each record is scored against its OWN ground-truth mask.
* The physical image (file_name) is the independent unit when summarizing, so
  records are first averaged within their image, then mean/std are taken over images.
* The segmentation depends only on pixels, so it is run once per physical image
  and shared by all records of that image.
* Records are grouped by file_name for later train/validation splits.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"
IMAGE_DIR = TRAIN_DIR / "train_images"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "threshold_eval"

# Segmentation parameters (same as the single-image baseline)
BACKGROUND_SIGMA_FRAC = 0.03
K = 1.0
LIMB_ERODE_FRAC = 0.03

# Experiment
MIN_AREAS = [50, 100, 200, 500]

# Fixed evaluation set. The selection is saved to disk the first time and
# reloaded on later runs, so the set never silently changes.
N_IMAGES = 10
REQUIRED_IMAGES = ["20111116063134Lh.jpeg"]
SEED = 42
EVAL_MANIFEST = OUTPUT_DIR / "eval_images.json"

# Grouped train/validation split (over ALL physical images)
VAL_FRACTION = 0.2
SPLIT_PATH = OUTPUT_DIR / "grouped_split.json"


# ----------------------------------------------------------------------------
# Data loading and grouping
# ----------------------------------------------------------------------------
def load_dataset(train_dir):
    json_path = next(train_dir.glob("*.json"))
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def group_records_by_image(data):
    """Map file_name -> list of records describing that physical image."""
    groups = defaultdict(list)
    for record in data["images"]:
        groups[record["file_name"]].append(record)
    for records in groups.values():
        records.sort(key=lambda r: r["id"])
    return dict(sorted(groups.items()))


def group_annotations_by_record(data):
    """Map image_id (record id) -> list of annotations."""
    by_record = defaultdict(list)
    for annotation in data["annotations"]:
        by_record[annotation["image_id"]].append(annotation)
    return by_record


def select_eval_images(groups, manifest_path):
    """Return a fixed list of physical image names (persisted for reproducibility)."""
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            selected = json.load(file)["file_names"]
        missing = [name for name in selected if name not in groups]
        if missing:
            raise ValueError(f"Manifest lists images not in the dataset: {missing}")
        return selected

    for name in REQUIRED_IMAGES:
        if name not in groups:
            raise ValueError(f"Required image not found in dataset: {name}")

    rng = random.Random(SEED)
    pool = [name for name in groups if name not in REQUIRED_IMAGES]
    n_extra = max(0, min(N_IMAGES - len(REQUIRED_IMAGES), len(pool)))
    selected = sorted(REQUIRED_IMAGES + rng.sample(pool, n_extra))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump({"seed": SEED, "file_names": selected}, file, indent=2)
    return selected


def make_grouped_split(groups, val_fraction, seed):
    """Split by physical image so records of one image never straddle the split."""
    names = list(groups)
    random.Random(seed).shuffle(names)
    n_val = max(1, round(val_fraction * len(names)))
    val_names = sorted(names[:n_val])
    train_names = sorted(names[n_val:])

    def record_ids(file_names):
        return [r["id"] for name in file_names for r in groups[name]]

    return {
        "seed": seed,
        "train": {"file_names": train_names, "record_ids": record_ids(train_names)},
        "val": {"file_names": val_names, "record_ids": record_ids(val_names)},
    }


# ----------------------------------------------------------------------------
# Segmentation (runs once per physical image)
# ----------------------------------------------------------------------------
def segment_components(gray):
    """Return (labels, stats) for connected components of the dark-filament candidates."""
    # Solar disk
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n_labels <= 1:
        raise RuntimeError("Could not identify the solar disk.")

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    disk_mask = ((labels == largest) * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    disk_mask = cv2.morphologyEx(disk_mask, cv2.MORPH_CLOSE, kernel)
    if not disk_mask.any():
        raise RuntimeError("The detected solar disk is empty.")

    # Shrink away the dark limb
    disk_radius = np.sqrt((disk_mask > 0).sum() / np.pi)
    erode_px = max(1, int(LIMB_ERODE_FRAC * disk_radius))
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
    )
    inner_disk = cv2.erode(disk_mask, erode_kernel) > 0
    if not inner_disk.any():
        raise RuntimeError("The eroded solar disk is empty.")

    # Local-contrast residual
    sigma = BACKGROUND_SIGMA_FRAC * disk_radius
    gray_f = gray.astype(np.float32)
    weight = inner_disk.astype(np.float32)
    background = cv2.GaussianBlur(gray_f * weight, (0, 0), sigma) / np.maximum(
        cv2.GaussianBlur(weight, (0, 0), sigma), 1e-6
    )
    residual = gray_f - background
    threshold_value = -K * residual[inner_disk].std()

    binary_mask = ((residual < threshold_value) & inner_disk).astype(np.uint8) * 255
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    return labels, stats


def filter_by_area(labels, stats, min_area):
    """Boolean mask of connected components with at least `min_area` pixels."""
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # background
    return keep[labels]


# ----------------------------------------------------------------------------
# Ground truth and metrics
# ----------------------------------------------------------------------------
def decode_ground_truth(record, annotations):
    height, width = record["height"], record["width"]
    truth = np.zeros((height, width), dtype=bool)

    for annotation in annotations:
        segmentation = annotation["segmentation"]

        if isinstance(segmentation, dict) and isinstance(segmentation["counts"], (str, bytes)):
            rle = segmentation  # already compressed RLE
        else:
            rle = mask_utils.frPyObjects(segmentation, height, width)

        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        truth |= decoded > 0

    return truth


def calculate_metrics(candidate, truth):
    """Pixel-level IoU and Dice for two boolean masks."""
    intersection = np.logical_and(candidate, truth).sum()
    union = np.logical_or(candidate, truth).sum()
    total = candidate.sum() + truth.sum()

    iou = intersection / union if union > 0 else 0.0
    dice = 2 * intersection / total if total > 0 else 0.0
    return float(iou), float(dice)


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate_image(file_name, records, annotations_by_record):
    """Score every record of one physical image at every threshold."""
    image = cv2.imread(str(IMAGE_DIR / file_name))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_DIR / file_name}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    labels, stats = segment_components(gray)

    truths = {}
    for record in records:
        if (record["height"], record["width"]) != gray.shape:
            raise ValueError(
                f"Record {record['id']} size does not match image {file_name}: "
                f"{(record['height'], record['width'])} vs {gray.shape}"
            )
        truths[record["id"]] = decode_ground_truth(
            record, annotations_by_record.get(record["id"], [])
        )

    rows = []
    for min_area in MIN_AREAS:
        prediction = filter_by_area(labels, stats, min_area)
        for record in records:
            truth = truths[record["id"]]
            iou, dice = calculate_metrics(prediction, truth)
            rows.append({
                "file_name": file_name,
                "record_id": record["id"],
                "n_annotations": len(annotations_by_record.get(record["id"], [])),
                "gt_pixels": int(truth.sum()),
                "min_area": min_area,
                "pred_pixels": int(prediction.sum()),
                "iou": iou,
                "dice": dice,
            })
    return rows


def sample_std(values):
    """Sample standard deviation (ddof=1); NaN if fewer than two values."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def summarize(rows):
    """
    Per threshold:
      * image-level: average records within each physical image, then mean/std over images
      * record-level: mean/std over all records (for comparison only; records from
        the same image are not independent)
    """
    summary = []
    for min_area in MIN_AREAS:
        subset = [r for r in rows if r["min_area"] == min_area]

        per_image = defaultdict(list)
        for r in subset:
            per_image[r["file_name"]].append(r)

        entry = {
            "min_area": min_area,
            "n_images": len(per_image),
            "n_records": len(subset),
        }
        for metric in ("iou", "dice"):
            image_means = [np.mean([r[metric] for r in recs]) for recs in per_image.values()]
            record_vals = [r[metric] for r in subset]
            entry[f"{metric}_image_mean"] = float(np.mean(image_means))
            entry[f"{metric}_image_std"] = sample_std(image_means)
            entry[f"{metric}_record_mean"] = float(np.mean(record_vals))
            entry[f"{metric}_record_std"] = sample_std(record_vals)
        summary.append(entry)
    return summary


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary):
    print("\nIMAGE-LEVEL SUMMARY (physical image = independent unit)")
    header = (
        f"{'Min area':>9} | {'Images':>6} | {'Records':>7} | "
        f"{'IoU mean':>9} | {'IoU std':>8} | {'Dice mean':>9} | {'Dice std':>8}"
    )
    line = "-" * len(header)
    print(line)
    print(header)
    print(line)
    for s in summary:
        print(
            f"{s['min_area']:>9} | {s['n_images']:>6} | {s['n_records']:>7} | "
            f"{s['iou_image_mean']:>9.4f} | {s['iou_image_std']:>8.4f} | "
            f"{s['dice_image_mean']:>9.4f} | {s['dice_image_std']:>8.4f}"
        )
    print(line)

    print("\nRECORD-LEVEL SUMMARY (for reference; records of one image are correlated)")
    header = (
        f"{'Min area':>9} | {'IoU mean':>9} | {'IoU std':>8} | "
        f"{'Dice mean':>9} | {'Dice std':>8}"
    )
    line = "-" * len(header)
    print(line)
    print(header)
    print(line)
    for s in summary:
        print(
            f"{s['min_area']:>9} | {s['iou_record_mean']:>9.4f} | {s['iou_record_std']:>8.4f} | "
            f"{s['dice_record_mean']:>9.4f} | {s['dice_record_std']:>8.4f}"
        )
    print(line)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset(TRAIN_DIR)
    groups = group_records_by_image(data)
    annotations_by_record = group_annotations_by_record(data)

    print(f"Dataset: {len(data['images'])} records, {len(groups)} physical images")
    multi = {name: len(recs) for name, recs in groups.items() if len(recs) > 1}
    print(f"Physical images with multiple records: {len(multi)}")

    # Fixed evaluation set
    eval_images = select_eval_images(groups, EVAL_MANIFEST)
    print(f"Evaluating {len(eval_images)} physical images "
          f"({sum(len(groups[n]) for n in eval_images)} records)")

    rows = []
    for file_name in eval_images:
        rows.extend(evaluate_image(file_name, groups[file_name], annotations_by_record))

    empty_gt = {(r["file_name"], r["record_id"]) for r in rows if r["gt_pixels"] == 0}
    if empty_gt:
        print(f"WARNING: {len(empty_gt)} record(s) have an empty ground-truth mask "
              f"(they score 0 and pull the means down): {sorted(empty_gt)}")

    summary = summarize(rows)
    print_summary(summary)

    # Grouped split over every physical image, saved for later training
    split = make_grouped_split(groups, VAL_FRACTION, SEED)
    with SPLIT_PATH.open("w", encoding="utf-8") as file:
        json.dump(split, file, indent=2)

    write_csv(OUTPUT_DIR / "per_record_results.csv", rows)
    write_csv(OUTPUT_DIR / "summary.csv", summary)
    print(f"\nSaved results to: {OUTPUT_DIR}")
    print(f"Grouped split: {len(split['train']['file_names'])} train / "
          f"{len(split['val']['file_names'])} val physical images -> {SPLIT_PATH}")


if __name__ == "__main__":
    main()