"""
Create a Kaggle submission for the Solar Filament Segmentation Challenge 2026.

Model:
    Trained U-Net checkpoint

Configuration selected using validation PQ:
    Probability threshold: 0.3
    Minimum connected-component area: 50 pixels at 512x512

Output:
    outputs/unet/submission.csv

Required CSV columns:
    filament_id
    segmentation_rle

Run from the project root:
    python3 src/create_unet_submission.py
"""

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

TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "test" / "test_images"

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "unet" / "best_unet.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "unet"

OUTPUT_CSV = OUTPUT_DIR / "submission.csv"

# These settings match the U-Net and the best validation PQ result.
IMAGE_SIZE = 512
ORIGINAL_IMAGE_SIZE = 2048

PREDICTION_THRESHOLD = 0.3
MIN_COMPONENT_AREA = 50

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
# 3. LOAD THE TRAINED U-NET
# ============================================================

def load_model():
    """Load the saved U-Net checkpoint."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    base_channels = checkpoint.get("base_channels", BASE_CHANNELS)
    checkpoint_image_size = checkpoint.get("image_size", IMAGE_SIZE)

    if checkpoint_image_size != IMAGE_SIZE:
        raise ValueError(
            f"Checkpoint expects image size {checkpoint_image_size}, "
            f"but this script uses {IMAGE_SIZE}."
        )

    model = UNet(base=base_channels).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(
        "Saved validation Dice:",
        checkpoint.get("val_dice", "unknown"),
    )

    return model


# ============================================================
# 4. LOAD AND PREPROCESS A TEST IMAGE
# ============================================================

def load_image_tensor(image_path):
    """
    Read a grayscale image and apply the same basic preprocessing
    used during U-Net training.
    """

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"Could not read test image:\n{image_path}"
        )

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    image = image.astype(np.float32) / 255.0

    # Add batch and channel dimensions:
    # (512, 512) -> (1, 1, 512, 512)
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# 5. PREDICT A PROBABILITY MAP
# ============================================================

def predict_probability_map(model, image_path):
    """Predict the probability of each pixel belonging to a filament."""

    image_tensor = load_image_tensor(image_path)

    with torch.no_grad():
        logits = model(image_tensor)

        probabilities = torch.sigmoid(logits)

    probability_map = (
        probabilities[0, 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return probability_map


# ============================================================
# 6. EXTRACT INDIVIDUAL FILAMENT INSTANCES
# ============================================================

def predict_filament_instances(probability_map):
    """
    Convert a probability map into separate filament instances.

    Steps:
        1. Apply the prediction threshold.
        2. Find connected components.
        3. Remove components smaller than MIN_COMPONENT_AREA.
        4. Resize each component to the original resolution.

    Returns:
        A list of binary masks, one per predicted filament.
    """

    binary_mask = (
        probability_map >= PREDICTION_THRESHOLD
    ).astype(np.uint8)

    number_of_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    instances = []

    # Label 0 represents the background, so start at label 1.
    for label_id in range(1, number_of_labels):

        component_area = stats[
            label_id,
            cv2.CC_STAT_AREA,
        ]

        # Ignore small predicted regions.
        if component_area < MIN_COMPONENT_AREA:
            continue

        component_mask = (
            labels == label_id
        ).astype(np.uint8)

        # Restore the mask to the original 2048x2048 resolution.
        full_resolution_mask = cv2.resize(
            component_mask,
            (ORIGINAL_IMAGE_SIZE, ORIGINAL_IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        if not np.any(full_resolution_mask):
            continue

        instances.append(full_resolution_mask)

    return instances


# ============================================================
# 7. ENCODE A FILAMENT MASK USING COMPRESSED COCO RLE
# ============================================================

def encode_mask_rle(binary_mask):
    """
    Encode one binary mask using compressed COCO RLE.

    COCO RLE expects a Fortran-contiguous array.

    The encoded counts are bytes, so decode them to ASCII
    for storage in the CSV.
    """

    mask_for_encoding = np.asfortranarray(
        binary_mask.astype(np.uint8)
    )

    encoded = mask_utils.encode(mask_for_encoding)

    return encoded["counts"].decode("ascii")


# ============================================================
# 8. VALIDATE AN RLE ENCODING
# ============================================================

def validate_rle(rle_string):
    """Check that an RLE string decodes to a valid 2048x2048 mask."""

    rle = {
        "size": [
            ORIGINAL_IMAGE_SIZE,
            ORIGINAL_IMAGE_SIZE,
        ],
        "counts": rle_string.encode("ascii"),
    }

    decoded = mask_utils.decode(rle)

    if decoded.shape != (
        ORIGINAL_IMAGE_SIZE,
        ORIGINAL_IMAGE_SIZE,
    ):
        raise ValueError(
            f"Unexpected decoded mask shape: {decoded.shape}"
        )

    if not np.any(decoded):
        raise ValueError("A predicted filament RLE decoded to an empty mask.")


# ============================================================
# 9. CREATE THE SUBMISSION CSV
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TEST_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Test image directory not found:\n{TEST_IMAGE_DIR}"
        )

    test_images = sorted(
        path
        for path in TEST_IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    )

    if not test_images:
        raise FileNotFoundError(
            f"No test images found in:\n{TEST_IMAGE_DIR}"
        )

    print(f"Test image directory: {TEST_IMAGE_DIR}")
    print(f"Number of test images: {len(test_images)}")
    print(f"Prediction threshold: {PREDICTION_THRESHOLD}")
    print(f"Minimum component area: {MIN_COMPONENT_AREA}")

    model = load_model()

    submission_rows = []

    total_instances = 0
    images_without_predictions = []

    start_time = time.time()

    # Keep track of all IDs to prevent duplicates.
    used_filament_ids = set()

    for image_index, image_path in enumerate(test_images, start=1):

        filename_stem = image_path.stem

        # Example:
        # 20111114063134Lh.jpeg
        # becomes:
        # 20111114063134Lh
        probability_map = predict_probability_map(
            model,
            image_path,
        )

        instances = predict_filament_instances(
            probability_map,
        )

        if not instances:
            images_without_predictions.append(filename_stem)

        image_instance_count = 0

        for instance_index, instance_mask in enumerate(
            instances,
            start=1,
        ):
            # Each predicted filament gets a unique ID.
            filament_id = f"{filename_stem}_{instance_index}"

            if filament_id in used_filament_ids:
                raise ValueError(
                    f"Duplicate filament ID detected: {filament_id}"
                )

            used_filament_ids.add(filament_id)

            segmentation_rle = encode_mask_rle(
                instance_mask,
            )

            submission_rows.append(
                {
                    "filament_id": filament_id,
                    "segmentation_rle": segmentation_rle,
                }
            )

            image_instance_count += 1
            total_instances += 1

        if (
            image_index % 10 == 0
            or image_index == len(test_images)
        ):
            elapsed = time.time() - start_time

            print(
                f"Processed {image_index}/{len(test_images)} images | "
                f"Instances in current image: {image_instance_count} | "
                f"Total instances: {total_instances} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    # Construct a DataFrame with exactly the required columns.
    submission_df = pd.DataFrame(
        submission_rows,
        columns=[
            "filament_id",
            "segmentation_rle",
        ],
    )

    # --------------------------------------------------------
    # Validate submission structure.
    # --------------------------------------------------------

    if list(submission_df.columns) != [
        "filament_id",
        "segmentation_rle",
    ]:
        raise ValueError("Submission CSV columns are incorrect.")

    if submission_df["filament_id"].duplicated().any():
        raise ValueError("Duplicate filament IDs found in the CSV.")

    if submission_df["filament_id"].isna().any():
        raise ValueError("Missing filament IDs found in the CSV.")

    if submission_df["segmentation_rle"].isna().any():
        raise ValueError("Missing RLE values found in the CSV.")

    if len(submission_df) > 0:
        if (
            submission_df["segmentation_rle"]
            .astype(str)
            .str.len()
            .eq(0)
            .any()
        ):
            raise ValueError("An empty RLE string was found.")

    # Validate a sample of RLEs by decoding them.
    sample_size = min(10, len(submission_df))

    if sample_size > 0:
        print(f"\nValidating {sample_size} sample RLE masks...")

        for rle_string in submission_df[
            "segmentation_rle"
        ].head(sample_size):
            validate_rle(rle_string)

    # Write the CSV.
    submission_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print("SUBMISSION CSV CREATED")
    print("=" * 65)

    print(f"Output file: {OUTPUT_CSV}")
    print(f"Test images processed: {len(test_images)}")
    print(f"Predicted filament instances: {total_instances}")
    print(
        "Images with no predicted filaments:",
        len(images_without_predictions),
    )
    print(f"CSV rows: {len(submission_df)}")
    print(f"Elapsed time: {elapsed:.1f}s")

    print("\nCSV columns:")
    print(submission_df.columns.tolist())

    print("\nFirst five rows:")
    print(submission_df.head().to_string(index=False))

    if images_without_predictions:
        print("\nImages with no predicted instances:")
        for filename in images_without_predictions[:20]:
            print(f"  {filename}")

        if len(images_without_predictions) > 20:
            print("  ...")

    print("\nIMPORTANT:")
    print("This file is ready for structural inspection.")
    print("Confirm the competition's expected submission format")
    print("before uploading it to Kaggle.")


if __name__ == "__main__":
    main()
    