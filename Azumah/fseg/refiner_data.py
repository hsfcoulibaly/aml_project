"""Stage 3 training data: per observation, a mix of refiner samples built on the fly.

Each item is one physical image (its 2048 px image and U-Net map are read once) and yields `k` crops:
* "gt": a consensus instance with a corrupted copy of itself as the prior;
* "matched": a real out-of-fold YOLO proposal with IoU > 0.2 to a consensus instance, target = that instance;
* "negative": a YOLO proposal with IoU < 0.1 to every consensus instance, target empty (quality 0).
Other filaments inside a crop are background in the target. All 8 dihedral transforms are applied.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import rle as R
from .config import Paths
from .refiner import CROP, bbox, corrupt_prior, crop_affine, crop_inputs, warp
from .unet_data import read_image

MATCH_IOU, NEG_IOU = 0.2, 0.1


def load_proposals(paths: Paths, folds, run_prefix: str = "consensus_f", min_score: float = 0.05) -> dict:
    """{obs: [(rle, score)]} from every available out-of-fold YOLO raw file."""
    out = {}
    for k in folds:
        f = paths.work_dir / "runs" / f"{run_prefix}{k}" / f"oof_raw_fold{k}.pkl"
        if not f.exists():
            print(f"[refiner] no YOLO proposals for fold {k} ({f}); using ground-truth priors only there")
            continue
        with open(f, "rb") as fh:
            raw = pickle.load(fh)
        for obs, item in raw.items():
            out[obs] = [(r, s) for r, s in item["instances"] if s >= min_score]
    return out


def read_map(map_dir: Path | None, obs: str) -> np.ndarray | None:
    if map_dir is None:
        return None
    m = cv2.imread(str(map_dir / f"{obs}.png"), cv2.IMREAD_COLOR)
    return m


class RefinerDataset(Dataset):
    def __init__(self, paths: Paths, observations, consensus: dict, proposals: dict, map_dir: Path | None,
                 k: int = 8, mix=(0.4, 0.4, 0.2), augment: bool = True, seed: int = 0):
        self.paths, self.obs, self.cons, self.props = paths, list(observations), consensus, proposals
        self.map_dir, self.k, self.mix, self.augment = map_dir, k, np.asarray(mix, float), augment
        self.seed = seed
        self.missing_maps = 0

    def __len__(self):
        return len(self.obs)

    def _pools(self, obs):
        gts = [c["rle"] for c in self.cons.get(obs, [])]
        props = self.props.get(obs, [])
        matched, negative = [], []
        if props:
            iou = R.iou_matrix([r for r, _ in props], gts) if gts else np.zeros((len(props), 0))
            for i, (r, _) in enumerate(props):
                best = iou[i].max() if iou.shape[1] else 0.0
                if best > MATCH_IOU:
                    matched.append((r, int(iou[i].argmax())))
                elif best < NEG_IOU:
                    negative.append(r)
        return gts, matched, negative

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.augment else self.seed + i)
        obs = self.obs[i]
        image = read_image(self.paths, "train", obs, 2048)
        umap = read_map(self.map_dir, obs)
        gts, matched, negative = self._pools(obs)
        gt_masks = [R.decode(r) for r in gts]
        all_gt = np.any(gt_masks, axis=0) if gt_masks else np.zeros(image.shape[:2], bool)

        pools = {"gt": len(gt_masks), "matched": len(matched), "negative": len(negative)}
        weights = np.array([self.mix[j] if n else 0.0 for j, n in enumerate(pools.values())])
        xs, ys = [], []
        for _ in range(self.k if weights.sum() else 0):
            kind = list(pools)[rng.choice(3, p=weights / weights.sum())]
            if kind == "gt":
                g = int(rng.integers(len(gt_masks)))
                M = crop_affine(bbox(gt_masks[g]), jitter=0.25 if self.augment else 0, rng=rng)
                target = warp(gt_masks[g].astype(np.uint8), M, nearest=True).astype(bool)
                others = warp((all_gt & ~gt_masks[g]).astype(np.uint8), M, nearest=True).astype(bool)
                prior = corrupt_prior(target, others, rng)
            else:
                if kind == "matched":
                    r, g = matched[int(rng.integers(len(matched)))]
                    tgt_full = gt_masks[g]
                else:
                    r, tgt_full = negative[int(rng.integers(len(negative)))], None
                prior_full = R.decode(r)
                b = bbox(prior_full)
                if b is None:
                    continue
                M = crop_affine(b, jitter=0.1 if self.augment else 0, rng=rng)
                prior = warp(prior_full.astype(np.uint8), M, nearest=True).astype(bool)
                target = (warp(tgt_full.astype(np.uint8), M, nearest=True).astype(bool) if tgt_full is not None
                          else np.zeros((CROP, CROP), bool))
            x = crop_inputs(image, umap, prior, M)
            y = target[None].astype(np.uint8)
            if self.augment:
                kr = int(rng.integers(4))
                x, y = np.rot90(x, kr, (1, 2)), np.rot90(y, kr, (1, 2))
                if rng.random() < 0.5:
                    x, y = x[:, :, ::-1], y[:, :, ::-1]
            xs.append(np.ascontiguousarray(x))
            ys.append(np.ascontiguousarray(y))
        if not xs:
            return np.zeros((0, 5, CROP, CROP), np.uint8), np.zeros((0, 1, CROP, CROP), np.uint8)
        return np.stack(xs), np.stack(ys)


def collate(items):
    x = np.concatenate([a for a, _ in items])
    y = np.concatenate([b for _, b in items])
    return torch.from_numpy(x), torch.from_numpy(y)
