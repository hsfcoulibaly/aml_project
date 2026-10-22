
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

# 1. Set up project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "data" / "train"
IMAGE_DIR = TRAIN_DIR / "train_images"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

json_path = next(TRAIN_DIR.glob("*.json"))

# 2. Load the annotation file
with json_path.open("r", encoding="utf-8") as file:
    data = json.load(file)

coco = COCO(str(json_path))

# 3. Select the solar observation we want to compare
target_filename = "20111116063134Lh"

image_path = next(
    path for path in IMAGE_DIR.iterdir()
    if path.stem == target_filename
)

image = np.array(Image.open(image_path).convert("L"))

# Find all annotation image records for this observation.
records = [
    record for record in data["images"]
    if Path(record["file_name"]).stem == target_filename
]

records.sort(key=lambda record: record["id"])

print("Original image:", image_path.name)
print("Number of annotation records:", len(records))

# 4. Display the original image and each annotation set
fig, axes = plt.subplots(1, len(records) + 1, figsize=(20, 6))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original Image")

for index, record in enumerate(records, start=1):
    image_id = record["id"]

    # Retrieve annotations for this record only.
    annotation_ids = coco.getAnnIds(imgIds=[image_id])
    annotations = coco.loadAnns(annotation_ids)

    # Combine this record's masks for visualization only.
    masks = [
        coco.annToMask(annotation)
        for annotation in annotations
    ]

    if masks:
        combined_mask = np.maximum.reduce(masks)
    else:
        combined_mask = np.zeros(image.shape, dtype=np.uint8)

    axes[index].imshow(image, cmap="gray")
    axes[index].imshow(
        np.ma.masked_where(combined_mask == 0, combined_mask),
        cmap="autumn",
        alpha=0.8,
        vmin=0,
        vmax=1
    )

    short_id = image_id.split("-")[0]

    axes[index].set_title(
        f"Record {short_id}\n"
        f"{len(annotations)} annotations"
    )

    print(
        f"{image_id}: "
        f"{int(combined_mask.sum())} foreground pixels"
    )

for ax in axes:
    ax.axis("off")

plt.tight_layout()

output_path = OUTPUT_DIR / "multiple_annotation_records.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved visualization:", output_path)
