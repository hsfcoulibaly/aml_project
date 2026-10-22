
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "unet_prepared"


class SolarFilamentDataset(Dataset):
    def __init__(self, split):
        self.image_paths = sorted(
            DATA_DIR.glob(f"{split}_*_image.npy")
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No {split} images found in {DATA_DIR}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # The matching mask has the same filename,
        # with "_image.npy" replaced by "_mask.npy".
        mask_path = image_path.with_name(
            image_path.name.replace("_image.npy", "_mask.npy")
        )

        image = np.load(image_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        # Add a channel dimension: (H, W) -> (1, H, W).
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def main():
    train_dataset = SolarFilamentDataset("train")
    val_dataset = SolarFilamentDataset("val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    images, masks = next(iter(train_loader))

    print("Training records:", len(train_dataset))
    print("Validation records:", len(val_dataset))
    print("Image batch shape:", images.shape)
    print("Mask batch shape:", masks.shape)
    print("Image value range:", images.min().item(), images.max().item())
    print("Mask unique values:", torch.unique(masks))
    print("Foreground fraction:", masks.mean().item())
    print("PyTorch device:", "mps" if torch.backends.mps.is_available() else "cpu")


if __name__ == "__main__":
    main()
