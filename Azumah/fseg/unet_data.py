"""Stage 2 data: per-observation dense targets and the training dataset.

Targets are built once at 2048 px and cached as 3-channel PNGs (`work/unet/targets/<obs>.png`):
    channel 0 filament: fraction of annotation sets marking the pixel (soft multi-annotator target)
    channel 1 centerline: union of per-instance skeletons of consensus masks, dilated by 2 px
    channel 2 boundary: union of per-instance morphological gradients of consensus masks
One sample per physical image (never per annotation record), so annotators do not give the same
image conflicting targets.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from . import rle as R
from .config import IMAGE_SIZE, Paths
from .labels import load_references, load_targets, skeleton
from .unet import normalize

_DISK2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def unet_root(paths: Paths) -> Path:
    return paths.work_dir / "unet"


def dense_targets(sets: list[list[dict]], consensus: list[dict]) -> np.ndarray:
    """(H, W, 3) uint8: filament fraction, centerline, boundary (each scaled to 0..255)."""
    frac = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32)
    for s in sets:
        if s:
            frac += np.any([R.decode(r) for r in s], axis=0)
    frac /= max(len(sets), 1)

    center = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    bound = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    for inst in consensus:
        m = R.decode(inst["rle"])
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        y0, y1, x0, x1 = max(ys.min() - 4, 0), ys.max() + 5, max(xs.min() - 4, 0), xs.max() + 5
        crop = m[y0:y1, x0:x1].astype(np.uint8)
        center[y0:y1, x0:x1] |= skeleton(crop).astype(np.uint8)
        bound[y0:y1, x0:x1] |= cv2.morphologyEx(crop, cv2.MORPH_GRADIENT, _K3)
    center = cv2.dilate(center, _DISK2)
    return np.dstack([np.round(frac * 255), center * 255, bound * 255]).astype(np.uint8)


def build_targets(paths: Paths, observations: list[str], workers: int = 4) -> Path:
    out_dir = unet_root(paths) / "targets"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [o for o in observations if not (out_dir / f"{o}.png").exists()]
    if not todo:
        return out_dir
    refs, cons = load_references(paths), load_targets(paths)["consensus"]

    def one(obs):
        cv2.imwrite(str(out_dir / f"{obs}.png"), dense_targets(refs[obs], cons.get(obs, [])))

    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(one, todo), total=len(todo), desc="U-Net targets"))
    return out_dir


def read_image(paths: Paths, split: str, obs: str, size: int) -> np.ndarray:
    img = cv2.imread(str(paths.preprocessed / split / f"{obs}.jpg"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(paths.preprocessed / split / f"{obs}.jpg")
    return img if size == IMAGE_SIZE else cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def crop_window(tgt: np.ndarray, img: np.ndarray, crop: int, pos_frac: float,
                rng: np.random.Generator) -> tuple[int, int]:
    """Top-left corner of a `crop` px window. With probability `pos_frac` the window is centred
    (with jitter) on a random filament pixel, otherwise on a random on-disk pixel."""
    h, w = tgt.shape[:2]
    step = 8  # pick centres on a coarse grid: cheap, and exact placement does not matter
    src = tgt[::step, ::step, 0] if rng.random() < pos_frac else None
    pts = np.argwhere(src > 0) if src is not None else np.empty((0, 2), int)
    if len(pts) == 0:
        pts = np.argwhere(img[::step, ::step, 0] > 0)
    if len(pts) == 0:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = pts[rng.integers(len(pts))] * step
        jitter = crop // 4
        cy, cx = cy + rng.integers(-jitter, jitter + 1), cx + rng.integers(-jitter, jitter + 1)
    y0 = int(np.clip(cy - crop // 2, 0, h - crop))
    x0 = int(np.clip(cx - crop // 2, 0, w - crop))
    return y0, x0


class FilamentDataset(Dataset):
    """(image, target) per observation at `size` px. Augment: all 8 dihedral transforms + on-disk gain/bias.

    `crop` > 0 trains on random `crop` px windows of the `size` px image instead of the whole disk
    (native resolution: size 2048, crop 1024); `pos_frac` of windows are centred on a filament.
    `repeat` draws that many windows per image per epoch."""

    def __init__(self, paths: Paths, observations: list[str], size: int, augment: bool,
                 crop: int = 0, pos_frac: float = 0.7, repeat: int = 1):
        self.paths, self.obs, self.size, self.augment = paths, list(observations), size, augment
        self.crop, self.pos_frac, self.repeat = (crop if 0 < crop < size else 0), pos_frac, max(1, repeat)
        self.target_dir = unet_root(paths) / "targets"

    def __len__(self):
        return len(self.obs) * self.repeat

    def __getitem__(self, i):
        obs = self.obs[i % len(self.obs)]
        img = read_image(self.paths, "train", obs, self.size)
        tgt = cv2.imread(str(self.target_dir / f"{obs}.png"), cv2.IMREAD_COLOR)
        if self.size != IMAGE_SIZE:
            tgt = cv2.resize(tgt, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.crop:
            # numpy's global RNG is reseeded per DataLoader worker by torch, so draw the seed from it
            rng = np.random.default_rng(np.random.randint(2**31))
            y0, x0 = crop_window(tgt, img, self.crop, self.pos_frac, rng)
            img = img[y0:y0 + self.crop, x0:x0 + self.crop]
            tgt = tgt[y0:y0 + self.crop, x0:x0 + self.crop]
        if self.augment:
            # the full disk has no preferred orientation, so every dihedral transform is label-preserving
            k = np.random.randint(4)
            img, tgt = np.rot90(img, k), np.rot90(tgt, k)
            if np.random.rand() < 0.5:
                img, tgt = img[:, ::-1], tgt[:, ::-1]
            on_disk = img[..., :1] > 0
            gain, bias = np.random.uniform(0.85, 1.15), np.random.uniform(-12, 12)
            img = np.where(on_disk, np.clip(img.astype(np.float32) * gain + bias, 1, 255), 0).astype(np.uint8)
        x = torch.from_numpy(normalize(np.ascontiguousarray(img)))
        y = torch.from_numpy(np.ascontiguousarray(tgt).transpose(2, 0, 1).astype(np.float32) / 255.0)
        return x, y
