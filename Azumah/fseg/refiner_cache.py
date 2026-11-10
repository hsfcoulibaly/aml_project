"""Stage 3 crop cache: read every 2048 px image and U-Net map once, keep all refiner crops in memory.

Reading two 2048 px files per image per epoch made refiner training data-bound (~8.6 min/epoch on Kaggle).
The cache stores, per sample, the four image/map channels and two masks at 256 px:
    kind 0 "gt":       target = consensus instance, aux = other filaments (prior is corrupted on the fly)
    kind 1 "matched":  target = matched consensus instance, aux = the YOLO proposal (prior)
    kind 2 "negative": target empty, aux = the YOLO proposal (prior)
Epochs then only apply random corruption, a small affine jitter and the dihedral transforms.
"""
from __future__ import annotations

import zlib
from multiprocessing import Pool

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

from . import rle as R
from .refiner import CROP, bbox, corrupt_prior, crop_affine, warp
from .refiner_data import MATCH_IOU, NEG_IOU, read_map
from .unet_data import read_image

KINDS = ("gt", "matched", "negative")
_W = {}


def _init(paths, consensus, proposals, map_dir, max_matched, max_neg):
    _W.update(paths=paths, cons=consensus, props=proposals, map_dir=map_dir, max_matched=max_matched,
              max_neg=max_neg)


def _crops_for(obs: str):
    rng = np.random.default_rng(zlib.crc32(obs.encode()))
    image = read_image(_W["paths"], "train", obs, 2048)
    umap = read_map(_W["map_dir"], obs)
    gts = [c["rle"] for c in _W["cons"].get(obs, [])]
    gt_masks = [R.decode(r) for r in gts]
    all_gt = np.any(gt_masks, axis=0) if gt_masks else np.zeros(image.shape[:2], bool)
    props = _W["props"].get(obs, [])
    iou = R.iou_matrix([r for r, _ in props], gts) if props and gts else np.zeros((len(props), len(gts)))

    samples = []  # (kind, M, target_full, aux_full)
    for g, m in enumerate(gt_masks):
        samples.append((0, crop_affine(bbox(m)), m, all_gt & ~m))
    matched = [(i, int(iou[i].argmax())) for i in range(len(props)) if iou.shape[1] and iou[i].max() > MATCH_IOU]
    negative = [i for i in range(len(props)) if (iou[i].max() if iou.shape[1] else 0.0) < NEG_IOU]
    for i, g in [matched[j] for j in rng.permutation(len(matched))[: _W["max_matched"]]]:
        p = R.decode(props[i][0])
        samples.append((1, crop_affine(bbox(p)), gt_masks[g], p))
    for i in [negative[j] for j in rng.permutation(len(negative))[: _W["max_neg"]]]:
        p = R.decode(props[i][0])
        b = bbox(p)
        if b is not None:
            samples.append((2, crop_affine(b), None, p))

    n = len(samples)
    img4 = np.zeros((n, 4, CROP, CROP), np.uint8)
    tgt = np.zeros((n, CROP, CROP), np.uint8)
    aux = np.zeros((n, CROP, CROP), np.uint8)
    kind = np.zeros(n, np.int8)
    for j, (k, M, t, a) in enumerate(samples):
        img4[j, 0], img4[j, 1] = warp(image[:, :, 1], M), warp(image[:, :, 2], M)
        if umap is not None:
            img4[j, 2], img4[j, 3] = warp(umap[:, :, 0], M), warp(umap[:, :, 1], M)
        if t is not None:
            tgt[j] = warp(t.astype(np.uint8), M, nearest=True)
        aux[j] = warp(a.astype(np.uint8), M, nearest=True)
        kind[j] = k
    return obs, img4, tgt, aux, kind


def build_cache(paths, observations, consensus, proposals, map_dir, max_matched=20, max_neg=10, workers=4) -> dict:
    parts = []
    with Pool(workers, initializer=_init, initargs=(paths, consensus, proposals, map_dir, max_matched, max_neg)) as p:
        for obs, img4, tgt, aux, kind in tqdm(p.imap(_crops_for, observations, chunksize=2),
                                              total=len(observations), desc="refiner crop cache"):
            parts.append((obs, img4, tgt, aux, kind))
    cache = {k: np.concatenate([x[i] for x in parts]) for i, k in [(1, "img4"), (2, "tgt"), (3, "aux"), (4, "kind")]}
    cache["obs"] = np.concatenate([np.full(len(q[4]), q[0]) for q in parts])
    return cache


class CachedRefinerDataset(Dataset):
    def __init__(self, cache: dict, augment: bool, seed: int = 0):
        self.c, self.augment, self.seed = cache, augment, seed

    def __len__(self):
        return len(self.c["kind"])

    def __getitem__(self, i):
        rng = np.random.default_rng() if self.augment else np.random.default_rng(self.seed + i)
        img4, t, a, k = self.c["img4"][i], self.c["tgt"][i].astype(bool), self.c["aux"][i].astype(bool), self.c["kind"][i]
        prior = corrupt_prior(t, a, rng) if k == 0 else a
        x = np.stack([img4[0], img4[1], prior.astype(np.uint8) * 255, img4[2], img4[3]])
        y = t[None].astype(np.uint8)
        if self.augment:
            s, ang = rng.uniform(0.9, 1.1), 0.0
            M = cv2.getRotationMatrix2D((CROP / 2, CROP / 2), ang, s)
            M[:, 2] += rng.uniform(-10, 10, 2)
            x = np.stack([cv2.warpAffine(c, M, (CROP, CROP), flags=cv2.INTER_NEAREST if j == 2 else cv2.INTER_LINEAR)
                          for j, c in enumerate(x)])
            y = cv2.warpAffine(y[0], M, (CROP, CROP), flags=cv2.INTER_NEAREST)[None]
            kr = int(rng.integers(4))
            x, y = np.rot90(x, kr, (1, 2)), np.rot90(y, kr, (1, 2))
            if rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def mix_sampler(kind: np.ndarray, mix=(0.4, 0.4, 0.2), num_samples: int = 8000) -> WeightedRandomSampler:
    """Draw kinds in the given proportions regardless of how many of each the cache holds."""
    counts = np.bincount(kind, minlength=3).astype(float)
    per = np.array([m / c if c else 0.0 for m, c in zip(mix, counts)])
    return WeightedRandomSampler(torch.from_numpy(per[kind]).double(), num_samples=num_samples, replacement=True)
