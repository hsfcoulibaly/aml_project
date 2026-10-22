
from pathlib import Path
import csv

import numpy as np
import torch

from train_unet import UNet, DEVICE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "unet_prepared"
MODEL_PATH = PROJECT_ROOT / "outputs" / "unet" / "best_unet.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "unet"

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def calculate_metrics(prediction, target):
    intersection = np.logical_and(prediction, target).sum()
    pred_pixels = prediction.sum()
    target_pixels = target.sum()
    union = np.logical_or(prediction, target).sum()

    # Empty prediction and empty target count as a perfect match.
    if pred_pixels == 0 and target_pixels == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (
            2.0 * intersection / (pred_pixels + target_pixels)
            if pred_pixels + target_pixels > 0 else 0.0
        )
        iou = intersection / union if union > 0 else 0.0

    return float(dice), float(iou), int(pred_pixels), int(target_pixels)


def main():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = UNet(base=checkpoint["base_channels"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Device:", DEVICE)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint validation Dice:", checkpoint["val_dice"])

    image_paths = sorted(DATA_DIR.glob("val_*_image.npy"))

    if not image_paths:
        raise RuntimeError(f"No validation images found in {DATA_DIR}")

    results = []

    with torch.no_grad():
        for index, image_path in enumerate(image_paths, start=1):
            mask_path = image_path.with_name(
                image_path.name.replace("_image.npy", "_mask.npy")
            )

            image = np.load(image_path).astype(np.float32)
            target = np.load(mask_path).astype(np.uint8)

            tensor = torch.from_numpy(image)[None, None].to(DEVICE)
            logits = model(tensor)
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

            for threshold in THRESHOLDS:
                prediction = probability >= threshold

                dice, iou, pred_pixels, target_pixels = (
                    calculate_metrics(prediction, target)
                )

                results.append({
                    "record": image_path.name,
                    "threshold": threshold,
                    "dice": dice,
                    "iou": iou,
                    "predicted_pixels": pred_pixels,
                    "target_pixels": target_pixels,
                })

            if index % 25 == 0 or index == len(image_paths):
                print(f"Evaluated {index}/{len(image_paths)} records")

    csv_path = OUTPUT_DIR / "threshold_evaluation.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nFull validation results")
    print("-" * 75)
    print(
        f"{'Threshold':>10} {'Mean Dice':>12} {'Median Dice':>13} "
        f"{'Mean IoU':>12} {'Empty preds':>13}"
    )

    for threshold in THRESHOLDS:
        subset = [
            row for row in results
            if row["threshold"] == threshold
        ]

        dice_values = np.array([row["dice"] for row in subset])
        iou_values = np.array([row["iou"] for row in subset])
        empty_predictions = sum(
            row["predicted_pixels"] == 0 for row in subset
        )

        print(
            f"{threshold:>10.1f} "
            f"{dice_values.mean():>12.4f} "
            f"{np.median(dice_values):>13.4f} "
            f"{iou_values.mean():>12.4f} "
            f"{empty_predictions:>13}"
        )

    print("\nSaved detailed results:", csv_path)


if __name__ == "__main__":
    main()
