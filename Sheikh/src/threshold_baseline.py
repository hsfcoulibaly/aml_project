
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils


# 1. Find the project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"
IMAGE_DIR = TRAIN_DIR / "train_images"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "threshold_baseline"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tunable parameters
BACKGROUND_SIGMA_FRAC = 0.03   # blur sigma as a fraction of disk radius (must exceed filament width)
K = 1.0                        # how many std devs darker than local background counts as filament
MIN_COMPONENT_AREA = 500        # drop connected components smaller than this (pixels)
LIMB_ERODE_FRAC = 0.03         # fraction of disk radius to shave off the dark limb


# 2. Load the annotation JSON
json_path = next(TRAIN_DIR.glob("*.json"))

with json_path.open("r", encoding="utf-8") as file:
    data = json.load(file)


# 3. Select one image record to experiment with
record = data["images"][2]
filename = record["file_name"]
image_id = record["id"]

image_path = IMAGE_DIR / filename

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")


# 4. Convert the image from BGR to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 5. Identify the bright solar disk, excluding the dark outer space
disk_threshold, bright_regions = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
)

number_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    bright_regions,
    connectivity=8,
)

# Label 0 is the background, so start at label 1.
if number_of_labels <= 1:
    raise RuntimeError("Could not identify the solar disk.")

largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

disk_mask = np.zeros_like(gray, dtype=np.uint8)
disk_mask[labels == largest_label] = 255

# Fill small holes and gaps in the detected disk
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
disk_mask = cv2.morphologyEx(disk_mask, cv2.MORPH_CLOSE, kernel)

if not np.any(disk_mask):
    raise RuntimeError("The detected solar disk is empty.")


# 5b. Shrink the disk mask to avoid the dark limb and edge artifacts
disk_radius = np.sqrt((disk_mask > 0).sum() / np.pi)   # estimated radius in pixels
erode_px = max(1, int(LIMB_ERODE_FRAC * disk_radius))
erode_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
)
inner_disk = cv2.erode(disk_mask, erode_kernel) > 0

if not inner_disk.any():
    raise RuntimeError("The eroded solar disk is empty.")


# 6. Estimate the slowly varying background with a masked Gaussian blur.
#    Dividing by the blurred mask keeps dark outer space from pulling the estimate down.
sigma = BACKGROUND_SIGMA_FRAC * disk_radius
gray_f = gray.astype(np.float32)
weight = inner_disk.astype(np.float32)

background = cv2.GaussianBlur(gray_f * weight, (0, 0), sigma) / np.maximum(
    cv2.GaussianBlur(weight, (0, 0), sigma), 1e-6
)

# Negative residual = darker than local surroundings
residual = gray_f - background
residual_std = residual[inner_disk].std()

threshold_value = -K * residual_std
binary_mask = np.zeros_like(gray, dtype=np.uint8)
binary_mask[(residual < threshold_value) & inner_disk] = 255

# Light cleanup: remove speckle
binary_mask = cv2.morphologyEx(
    binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
)


# 7. Find connected components and drop tiny ones
number_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary_mask,
    connectivity=8,
)

candidate_mask = np.zeros_like(binary_mask)

for label in range(1, number_of_labels):
    if stats[label, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
        candidate_mask[labels == label] = 255


# 8. Decode the human annotations for this image record
annotations = [
    annotation
    for annotation in data["annotations"]
    if annotation["image_id"] == image_id
]

ground_truth_mask = np.zeros(
    (record["height"], record["width"]),
    dtype=np.uint8,
)

for annotation in annotations:
    segmentation = annotation["segmentation"]

    rle = mask_utils.frPyObjects(
        segmentation,
        record["height"],
        record["width"],
    )

    decoded = mask_utils.decode(rle)

    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)

    ground_truth_mask[decoded > 0] = 1


# 9. Compare different minimum component areas

def calculate_metrics(candidate, truth):
    """Calculate pixel-level IoU and Dice for two boolean masks."""
    intersection = np.logical_and(candidate, truth).sum()
    union = np.logical_or(candidate, truth).sum()
    total = candidate.sum() + truth.sum()

    iou = intersection / union if union > 0 else 0.0
    dice = 2 * intersection / total if total > 0 else 0.0
    return float(iou), float(dice)


def filter_by_area(labels, stats, min_area):
    """Keep connected components with at least `min_area` pixels (boolean mask)."""
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # label 0 is the background
    return keep[labels]  # vectorized lookup, no per-component loop


def print_results_table(results, configured_area):
    header = (
        f"{'Min area':>10} | {'Predicted pixels':>17} | "
        f"{'IoU':>10} | {'Dice':>10}"
    )
    line = "-" * len(header)

    print("\nCOMPONENT-AREA EXPERIMENT")
    print(line)
    print(header)
    print(line)
    for r in results:
        print(
            f"{r['min_area']:>10} | {r['predicted_pixels']:>17} | "
            f"{r['iou']:>10.4f} | {r['dice']:>10.4f}"
        )
    print(line)
    print("Configured minimum area:", configured_area)


ground_truth = ground_truth_mask > 0

# Sweep several minimum component sizes
area_thresholds = [1, 20, 50, 100, 200, 500]

results = []
for min_area in area_thresholds:
    pred = filter_by_area(labels, stats, min_area)
    iou, dice = calculate_metrics(pred, ground_truth)
    results.append({
        "min_area": min_area,
        "predicted_pixels": int(pred.sum()),
        "iou": iou,
        "dice": dice,
    })

print_results_table(results, MIN_COMPONENT_AREA)

# Prediction for the configured parameter, used by the plots and summary below
prediction = filter_by_area(labels, stats, MIN_COMPONENT_AREA)
candidate_mask = prediction.astype(np.uint8) * 255
iou, dice = calculate_metrics(prediction, ground_truth)


# 10. Display the results
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 5, figsize=(22, 5))

axes[0].imshow(image_rgb)
axes[0].set_title("Original image")

axes[1].imshow(inner_disk, cmap="gray")
axes[1].set_title("Inner solar disk (limb removed)")

axes[2].imshow(residual, cmap="gray", vmin=-3 * residual_std, vmax=3 * residual_std)
axes[2].set_title(
    f"Local residual (dark = filament-like)\nthreshold={threshold_value:.1f}"
)

axes[3].imshow(ground_truth_mask, cmap="gray")
axes[3].set_title("Ground-truth filament mask")

axes[4].imshow(image_rgb)
axes[4].imshow(candidate_mask, cmap="Reds", alpha=0.45)
axes[4].set_title(f"Prediction\nIoU={iou:.4f}, Dice={dice:.4f}")

for axis in axes:
    axis.axis("off")

plt.tight_layout()

output_path = OUTPUT_DIR / "local_contrast_baseline.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()


print("Image:", filename)
print("Annotation record:", image_id)
print("Annotations in this record:", len(annotations))
print("Estimated disk radius (px):", round(float(disk_radius), 1))
print("Background sigma (px):", round(float(sigma), 1))
print("Residual std:", round(float(residual_std), 3))
print("Residual threshold:", round(float(threshold_value), 3))
print("Ground-truth foreground pixels:", int(ground_truth.sum()))
print("Predicted foreground pixels:", int(prediction.sum()))
print("Intersection over Union (IoU):", round(float(iou), 4))
print("Dice score:", round(float(dice), 4))
print("Saved visualization to:", output_path)