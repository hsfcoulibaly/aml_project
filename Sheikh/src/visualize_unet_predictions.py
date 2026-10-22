
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
import torch

from train_unet import UNet, DEVICE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "unet_prepared"
MODEL_PATH = PROJECT_ROOT / "outputs" / "unet" / "best_unet.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "unet"

NUM_EXAMPLES = 8
THRESHOLD = 0.5
SEED = 42


def dice_score(prediction, target):
    intersection = np.logical_and(prediction, target).sum()
    denominator = prediction.sum() + target.sum()

    return (
        2.0 * intersection / denominator
        if denominator > 0
        else 1.0
    )


def main():
    random.seed(SEED)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = UNet(base=checkpoint["base_channels"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded best checkpoint from epoch:", checkpoint["epoch"])
    print("Saved validation Dice:", checkpoint["val_dice"])
    print("Prediction threshold:", THRESHOLD)

    paths = sorted(DATA_DIR.glob("val_*_image.npy"))

    if not paths:
        raise RuntimeError("No validation images found.")

    examples = random.sample(paths, min(NUM_EXAMPLES, len(paths)))

    fig, axes = plt.subplots(
        len(examples),
        4,
        figsize=(15, 3.5 * len(examples)),
        squeeze=False,
    )

    scores = []

    for row, image_path in enumerate(examples):
        mask_path = image_path.with_name(
            image_path.name.replace("_image.npy", "_mask.npy")
        )

        image = np.load(image_path).astype(np.float32)
        target = np.load(mask_path).astype(np.uint8)

        input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

        prediction = probability >= THRESHOLD
        score = dice_score(prediction, target)
        scores.append(score)

        # Original image
        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("Solar image")

        # Ground truth
        axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Ground-truth mask")

        # Predicted mask
        axes[row, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Prediction (Dice={score:.3f})")

        # Overlay: green = ground truth, red = prediction.
        overlay = np.zeros((*target.shape, 3), dtype=np.float32)
        overlay[..., 1] = target
        overlay[..., 0] = prediction

        axes[row, 3].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(overlay, alpha=0.75)
        axes[row, 3].set_title("Overlay: green=GT, red=prediction")

        for col in range(4):
            axes[row, col].axis("off")

        print(
            f"{image_path.name}: Dice={score:.4f}, "
            f"GT pixels={int(target.sum())}, "
            f"predicted pixels={int(prediction.sum())}"
        )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "validation_predictions.png"
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nMean Dice across these examples: {np.mean(scores):.4f}")
    print("Saved visualization:", output_path)


if __name__ == "__main__":
    main()
