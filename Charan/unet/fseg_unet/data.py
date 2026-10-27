"""PyTorch datasets over the prepared work dir."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import IMAGENET_MEAN, IMAGENET_STD, Paths

_MEAN = np.asarray(IMAGENET_MEAN, np.float32)
_STD = np.asarray(IMAGENET_STD, np.float32)


def read_image(paths: Paths, obs: str) -> np.ndarray:
    img = cv2.imread(str(paths.images / f"{obs}.png"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(paths.images / f"{obs}.png")
    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 -> ImageNet-normalized float tensor 3xHxW."""
    x = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class RecordDataset(Dataset):
    """One sample per annotation record: (image, mask). Augments with dihedral transforms + light jitter.

    crop > 0 takes a random crop x crop window (fully convolutional models train on crops, infer on whole images).
    """

    def __init__(self, paths: Paths, manifest, augment: bool, crop: int = 0):
        self.paths = paths
        self.rows = manifest[["observation_id", "record_id"]].to_numpy()
        self.augment = augment
        self.crop = crop

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        obs, rid = self.rows[i]
        img = read_image(self.paths, obs)
        mask = cv2.imread(str(self.paths.masks / f"{rid}.png"), cv2.IMREAD_GRAYSCALE)
        if self.augment:
            # the full disk has no preferred orientation: all 8 dihedral transforms are valid
            k = np.random.randint(4)
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            gain = np.random.uniform(0.85, 1.15)
            bias = np.random.uniform(-12, 12)
            on_disk = img[..., :1] > 0
            img = np.where(on_disk, np.clip(img.astype(np.float32) * gain + bias, 1, 255), 0).astype(np.uint8)
        if 0 < self.crop < img.shape[0]:
            r, c = np.random.randint(img.shape[0] - self.crop + 1, size=2)
            img, mask = img[r:r + self.crop, c:c + self.crop], mask[r:r + self.crop, c:c + self.crop]
        x = to_tensor(np.ascontiguousarray(img))
        y = torch.from_numpy((np.ascontiguousarray(mask) > 127).astype(np.float32))[None]
        return x, y


class ObservationDataset(Dataset):
    """One sample per physical image for inference: (image, observation_id)."""

    def __init__(self, paths: Paths, observation_ids):
        self.paths = paths
        self.ids = list(observation_ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return to_tensor(read_image(self.paths, self.ids[i])), self.ids[i]
