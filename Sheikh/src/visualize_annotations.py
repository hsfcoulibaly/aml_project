
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

# --------------------------------------------------
# 1. Define paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"
IMAGE_DIR = TRAIN_DIR / "train_images"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

json_files = list(TRAIN_DIR.glob("*.json"))

if not json_files:
    raise FileNotFoundError(
        f"No annotation JSON found in {TRAIN_DIR}"
    )

annotation_path = json_files[0]

# --------------------------------------------------
# 2. Load the COCO annotations
# --------------------------------------------------

coco = COCO(str(annotation_path))

# --------------------------------------------------
# 3. Find image files
# --------------------------------------------------

valid_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

image_paths = [
    path
    for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in valid_extensions
]

# Match by filename stem, so .jpg and .jpeg differences
# do not prevent us from finding the image.
images_by_stem = {}

for path in image_paths:
    images_by_stem.setdefault(path.stem, path)

# Find annotated image records whose image files exist.
candidate_ids = []

for image_id, image_info in coco.imgs.items():
    filename = Path(image_info["file_name"]).stem

    if filename in images_by_stem:
        candidate_ids.append(image_id)

if not candidate_ids:
    raise FileNotFoundError(
        "No annotation filenames matched the training images. "
        "Check the image and JSON folder locations."
    )

# Choose the first available annotated image record.
image_id = candidate_ids[0]
image_info = coco.imgs[image_id]

image_path = images_by_stem[
    Path(image_info["file_name"]).stem
]

print("Selected image:", image_path.name)
print("Annotation image ID:", image_id)

# --------------------------------------------------
# 4. Load the original grayscale image
# --------------------------------------------------

image = np.array(
    Image.open(image_path).convert("L")
)

print("Image shape:", image.shape)
print("Pixel intensity range:", image.min(), image.max())

# --------------------------------------------------
# 5. Retrieve all annotations for this image record
# --------------------------------------------------

annotation_ids = coco.getAnnIds(imgIds=[image_id])
annotations = coco.loadAnns(annotation_ids)

print("Number of filament annotations:", len(annotations))

# --------------------------------------------------
# 6. Convert each annotation into a binary mask
# --------------------------------------------------

individual_masks = []

for annotation in annotations:
    mask = coco.annToMask(annotation)
    individual_masks.append(mask)

# Combine masks only for visualization.
# We preserve individual_masks for instance-level analysis.
if individual_masks:
    combined_mask = np.maximum.reduce(individual_masks)
else:
    combined_mask = np.zeros(image.shape, dtype=np.uint8)

print("Combined mask shape:", combined_mask.shape)
print("Foreground pixels:", int(combined_mask.sum()))

# --------------------------------------------------
# 7. Display the image, mask, and overlay
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original Solar Image")

axes[1].imshow(combined_mask, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Combined Ground-Truth Mask")

axes[2].imshow(image, cmap="gray")

# Show filament pixels in red.
overlay = np.ma.masked_where(
    combined_mask == 0,
    combined_mask
)

axes[2].imshow(
    overlay,
    cmap="autumn",
    alpha=0.65,
    vmin=0,
    vmax=1
)

axes[2].set_title("Ground Truth Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output_path = OUTPUT_DIR / "annotation_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Visualization saved to:", output_path)
