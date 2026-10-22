
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "unet_prepared"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "unet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-3
BASE_CHANNELS = 16

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Use Apple's Metal GPU when available.
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("Training device:", DEVICE)


# --------------------------------------------------
# 2. Dataset
# --------------------------------------------------

class SolarFilamentDataset(Dataset):
    def __init__(self, split):
        self.image_paths = sorted(
            DATA_DIR.glob(f"{split}_*_image.npy")
        )

        if not self.image_paths:
            raise RuntimeError(f"No {split} data found.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        mask_path = image_path.with_name(
            image_path.name.replace("_image.npy", "_mask.npy")
        )

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


# --------------------------------------------------
# 3. U-Net building blocks
# --------------------------------------------------

class DoubleConv(nn.Module):
    """Two convolution layers with batch normalization and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()

        # Encoder: learn features while reducing resolution.
        self.enc1 = DoubleConv(1, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck: deepest feature representation.
        self.bottleneck = DoubleConv(base * 8, base * 16)

        # Decoder: restore resolution and combine skip connections.
        self.up4 = nn.ConvTranspose2d(
            base * 16, base * 8, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(
            base * 8, base * 4, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(
            base * 4, base * 2, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(
            base * 2, base, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(base * 2, base)

        # One output channel: filament probability logits.
        self.output = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Return logits, not sigmoid probabilities.
        return self.output(d1)


# --------------------------------------------------
# 4. Combined BCE + Dice loss
# --------------------------------------------------

def combined_loss(logits, targets):
    # BCEWithLogitsLoss combines sigmoid and binary cross-entropy
    # in a numerically stable operation.
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets
    )

    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * targets).sum(
        dim=(1, 2, 3)
    )

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (
        (2.0 * intersection + 1e-6)
        / (denominator + 1e-6)
    )

    dice_loss = 1.0 - dice.mean()

    return 0.5 * bce + 0.5 * dice_loss


# --------------------------------------------------
# 5. Evaluation
# --------------------------------------------------

def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    total_intersection = 0.0
    total_predicted = 0.0
    total_actual = 0.0
    batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            loss = combined_loss(logits, masks)

            predictions = torch.sigmoid(logits) >= 0.5

            total_intersection += (
                (predictions & masks.bool()).sum().item()
            )
            total_predicted += predictions.sum().item()
            total_actual += masks.sum().item()

            total_loss += loss.item()
            batches += 1

    dice = (
        2.0 * total_intersection
        / (total_predicted + total_actual + 1e-8)
    )

    average_loss = total_loss / max(batches, 1)

    return average_loss, dice


# --------------------------------------------------
# 6. Training
# --------------------------------------------------

def main():
    train_dataset = SolarFilamentDataset("train")
    val_dataset = SolarFilamentDataset("val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(base=BASE_CHANNELS).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_dice = -1.0
    checkpoint_path = OUTPUT_DIR / "best_unet.pth"

    print(f"Training records: {len(train_dataset)}")
    print(f"Validation records: {len(val_dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nStarting training...\n")

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        for batch_index, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = combined_loss(logits, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss, val_dice = evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        # Keep the model with the best validation Dice.
        if val_dice > best_val_dice:
            best_val_dice = val_dice

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "base_channels": BASE_CHANNELS,
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch + 1,
                    "val_dice": val_dice,
                },
                checkpoint_path,
            )

            print(f"  Saved best model: {checkpoint_path}")

    print("\nTraining finished.")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
