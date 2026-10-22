
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "train" / "train_images"
ANNOTATION_FILE = (
    PROJECT_ROOT
    / "data" / "train"
    / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
)
SPLIT_FILE = (
    PROJECT_ROOT / "outputs" / "threshold_eval" / "grouped_split.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "unet_prepared"
VISUALIZATION_DIR = PROJECT_ROOT / "outputs" / "unet_preparation"

IMAGE_SIZE = 512
NUM_VISUALIZATIONS = 6


def create_binary_mask(annotations, height, width):
    """Create one binary mask from all polygons for one image record."""

    mask = np.zeros((height, width), dtype=np.uint8)

    for annotation in annotations:
        segmentation = annotation.get("segmentation", [])

        if not segmentation:
            continue

        # Convert the polygon coordinates into COCO RLE masks.
        rles = mask_utils.frPyObjects(segmentation, height, width)
        decoded = mask_utils.decode(rles)

        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)

        mask = np.logical_or(mask, decoded).astype(np.uint8)

    return mask


def main():
    print("Loading dataset...")

    with open(ANNOTATION_FILE, "r") as f:
        data = json.load(f)

    with open(SPLIT_FILE, "r") as f:
        split = json.load(f)

    train_files = set(split["train"]["file_names"])
    val_files = set(split["val"]["file_names"])

    assert train_files.isdisjoint(val_files), (
        "A physical image occurs in both training and validation."
    )

    # IMPORTANT FIX:
    # Annotations are stored separately from image records.
    # Link each annotation to its image record using image_id.
    annotations_by_image_id = defaultdict(list)

    for annotation in data["annotations"]:
        annotations_by_image_id[annotation["image_id"]].append(annotation)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

    # Remove old prepared arrays so stale, empty masks cannot be mistaken
    # for the newly generated results.
    for old_file in OUTPUT_DIR.glob("*.npy"):
        old_file.unlink()

    train_count = 0
    val_count = 0
    skipped_count = 0
    empty_mask_count = 0
    saved_examples = []

    print(f"Training physical images: {len(train_files)}")
    print(f"Validation physical images: {len(val_files)}")
    print(f"Total annotation objects: {len(data['annotations'])}")
    print("\nPreparing image-mask pairs...")

    for record in data["images"]:
        filename = record["file_name"]
        record_id = record["id"]

        if filename in train_files:
            split_name = "train"
        elif filename in val_files:
            split_name = "val"
        else:
            skipped_count += 1
            continue

        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            print(f"WARNING: image not found: {image_path}")
            skipped_count += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"WARNING: could not read: {image_path}")
            skipped_count += 1
            continue

        height, width = image.shape

        # Retrieve the annotations linked to THIS record's ID.
        record_annotations = annotations_by_image_id.get(record_id, [])

        mask = create_binary_mask(record_annotations, height, width)

        # Resize image and mask separately.
        image_small = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        mask_small = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        image_small = image_small.astype(np.float32) / 255.0
        mask_small = mask_small.astype(np.uint8)

        foreground_pixels = int(np.count_nonzero(mask_small))

        if foreground_pixels == 0:
            empty_mask_count += 1

        image_output = OUTPUT_DIR / f"{split_name}_{record_id}_image.npy"
        mask_output = OUTPUT_DIR / f"{split_name}_{record_id}_mask.npy"

        np.save(image_output, image_small)
        np.save(mask_output, mask_small)

        if split_name == "train":
            train_count += 1
        else:
            val_count += 1

        if len(saved_examples) < NUM_VISUALIZATIONS:
            saved_examples.append(
                (
                    filename,
                    record_id,
                    image_small,
                    mask_small,
                    len(record_annotations),
                )
            )

        # Print a known example and the first few records for diagnosis.
        if record_id == "040301-20140609195854Bh":
            print("\nKnown example check:")
            print("Record ID:", record_id)
            print("Matched annotations:", len(record_annotations))
            print("Original foreground pixels:", int(np.count_nonzero(mask)))
            print("512x512 foreground pixels:", foreground_pixels)

    print("\nPreparation complete.")
    print(f"Training records saved: {train_count}")
    print(f"Validation records saved: {val_count}")
    print(f"Records skipped: {skipped_count}")
    print(f"Records with empty masks: {empty_mask_count}")
    print(f"Prepared arrays: {OUTPUT_DIR}")

    if not saved_examples:
        print("No examples available to visualize.")
        return

    fig, axes = plt.subplots(
        len(saved_examples),
        3,
        figsize=(12, 4 * len(saved_examples)),
        squeeze=False,
    )

    for row, (filename, record_id, image, mask, n_annotations) in enumerate(
        saved_examples
    ):
        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"Image\n{filename}")

        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(
            f"Binary mask\nAnnotations: {n_annotations}"
        )

        axes[row, 2].imshow(image, cmap="gray", vmin=0, vmax=1)

        # Make background pixels transparent so only filament pixels
        # receive the red overlay.
        visible_mask = np.ma.masked_where(mask == 0, mask)
        axes[row, 2].imshow(
            visible_mask,
            cmap="autumn",
            vmin=0,
            vmax=1,
            alpha=0.8,
        )
        axes[row, 2].set_title(f"Filament overlay\n{record_id}")

        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()

    visualization_path = (
        VISUALIZATION_DIR / "image_mask_examples.png"
    )
    plt.savefig(visualization_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualization saved to: {visualization_path}")


if __name__ == "__main__":
    main()
