"""Model averaging and a stricter, PQ-tuned keep rule.

Pipeline: D4-TTA probability maps per model -> weighted average -> candidate instances -> keep rule.

Candidates are 8-connected components of (prob > t_lo) inside the disk, after an optional morphological
closing that bridges short gaps between fragments of one filament. A candidate is kept when

    peak prob >= t_hi  and  mean prob >= min_mean  and  area >= min_area   (areas at 2048 px)

The single-threshold rule used so far is the special case t_lo = t_hi, min_mean = 0, close = 0.

Every model's validation-to-leaderboard gap has been 0.03 to 0.06 PQ. One explanation (public notebook
"Solar Filament U-Net: OOF 0.45 vs Public 0.39") is that the test labels are sparser: each filament an
annotator would mark in training appears in the test labels with probability r < 1. Matched predictions
then turn into false positives at rate 1 - r, and the expected public score from pooled validation
counts is

    PQ'(r) = r S / (r TP + 0.5 (FP + (1 - r) TP) + 0.5 r FN)        (S = sum of matched IoU)

`fit_r` fits r on submissions whose public score we know, and `sweep` reports PQ and PQ'(r) for every
keep rule, so we can pick a rule for the leaderboard rather than for our own validation labels.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, astuple, dataclass
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import pq
from .data import ObservationDataset
from .postprocess import upsample
from .preprocess import Disk, disk_mask


@dataclass(frozen=True)
class KeepRule:
    t_lo: float
    t_hi: float
    min_mean: float
    min_area: int
    close: int = 0

    @classmethod
    def single(cls, thr: float, min_area: int) -> "KeepRule":
        """The rule every earlier submission used: one threshold plus a minimum area."""
        return cls(thr, thr, 0.0, min_area, 0)


# ---------------------------------------------------------------- inference
def _d4(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = torch.rot90(x, k, dims=(-2, -1))
    return x.flip(-1) if flip else x


def _d4_inv(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = x.flip(-1) if flip else x
    return torch.rot90(x, -k, dims=(-2, -1))


TTA_VIEWS = {
    "none": [(0, False)],
    "flip": [(0, False), (0, True), (2, True)],  # identity, horizontal flip, vertical flip
    "d4": [(k, f) for k in range(4) for f in (False, True)],
}


@torch.no_grad()
def predict_u8(model, paths, observation_ids, device, tta: str = "d4", batch_size: int = 2,
               workers: int = 2) -> dict:
    """{obs: uint8 probability map (p x 255) at the prepared resolution}; logits averaged over TTA views."""
    model.eval()
    views = TTA_VIEWS[tta]
    loader = DataLoader(ObservationDataset(paths, observation_ids), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    out = {}
    for x, ids in tqdm(loader, desc=f"predict ({tta})", leave=False):
        x = x.to(device, non_blocking=True)
        logits = 0
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            for k, f in views:
                logits = logits + _d4_inv(model(_d4(x, k, f)).float(), k, f)
        probs = torch.sigmoid(logits / len(views))[:, 0].cpu().numpy()
        out.update(zip(ids, np.round(probs * 255).astype(np.uint8)))
    return out


def blend(maps: dict, weights: dict, obs: str) -> np.ndarray:
    """Weighted mean probability (float32, 0..1) of the models named in `weights`."""
    total = sum(weights.values())
    return sum(w * maps[name][obs].astype(np.float32) for name, w in weights.items()) / (255.0 * total)


# ---------------------------------------------------------------- candidates
def candidates(prob: np.ndarray, disk: Disk, t_lo: float, close: int, min_area_floor: int,
               full: np.ndarray | None = None, inside: np.ndarray | None = None) -> dict:
    """Components of the full-resolution map: {"rle": [...], "area", "peak", "mean": arrays}.

    `full` / `inside` (the 2048 px map and disk mask) can be passed in to reuse them across thresholds.
    """
    full = upsample(prob) if full is None else full
    inside = disk_mask(full.shape, disk, scale=1.01) if inside is None else inside
    binary = ((full > t_lo) & inside).astype(np.uint8)
    if close > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1, 2 * close + 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) & inside.astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rles, area, peak, mean = [], [], [], []
    for i in range(1, len(stats)):
        x, y, w, h, a = stats[i]
        if a < min_area_floor:
            continue
        m = labels[y:y + h, x:x + w] == i
        vals = full[y:y + h, x:x + w][m]
        full_mask = np.zeros(labels.shape, np.uint8)
        full_mask[y:y + h, x:x + w] = m
        rles.append(pq.encode(full_mask))
        area.append(a)
        peak.append(vals.max())
        mean.append(vals.mean())
    return {"rle": rles, "area": np.asarray(area), "peak": np.asarray(peak), "mean": np.asarray(mean)}


def keep_mask(c: dict, rule: KeepRule) -> np.ndarray:
    return (c["area"] >= rule.min_area) & (c["peak"] >= rule.t_hi) & (c["mean"] >= rule.min_mean)


def apply_rule(c: dict, rule: KeepRule) -> list[dict]:
    return [r for r, k in zip(c["rle"], keep_mask(c, rule)) if k]


# ---------------------------------------------------------------- PQ sweep on cached matches
def _match_cache(c: dict, sets: list) -> list:
    """Per annotation set: (IoU of each candidate's best match if > 0.5 else 0, number of GT filaments)."""
    out = []
    for gts in sets:
        iou = pq.iou_matrix(c["rle"], gts)
        best = iou.max(1) if iou.size else np.zeros(len(c["rle"]))
        out.append((np.where(best > pq.MATCH_IOU, best, 0.0), len(gts)))
    return out


def _image_job(job):
    prob, disk, sets, t_los, closes, floor = job
    full = upsample(prob)
    inside = disk_mask(full.shape, disk, scale=1.01)
    out = {}
    for t_lo, close in itertools.product(t_los, closes):
        c = candidates(prob, disk, t_lo, close, floor, full, inside)
        out[(t_lo, close)] = (c, _match_cache(c, sets))
    return out


def counts(cache: dict, rule: KeepRule) -> dict:
    """Pooled TP / FP / FN / IoU sum; per image the counts are averaged over its annotation sets."""
    tp = fp = fn = s = 0.0
    for per_key in cache.values():
        c, matches = per_key[(rule.t_lo, rule.close)]
        keep = keep_mask(c, rule)
        n_keep = keep.sum()
        for best, n_gt in matches:
            m = best[keep]
            t = (m > 0).sum()
            tp += t / len(matches)
            fp += (n_keep - t) / len(matches)
            fn += (n_gt - t) / len(matches)
            s += m.sum() / len(matches)
    return {"tp": tp, "fp": fp, "fn": fn, "iou_sum": s}


def pq_from_counts(k: dict, r: float = 1.0) -> float:
    denom = r * k["tp"] + 0.5 * (k["fp"] + (1 - r) * k["tp"]) + 0.5 * r * k["fn"]
    return r * k["iou_sum"] / denom if denom > 0 else 0.0


def build_cache(probs: dict, disks: dict, references: dict, t_los, closes, floor: int, workers: int = 4) -> dict:
    """{obs: {(t_lo, close): (candidates, matches)}} for the validation images, in parallel."""
    obs = sorted(probs)
    jobs = [(probs[o], disks[o], references["sets"][o], t_los, closes, floor) for o in obs]
    with Pool(workers) as p:
        results = list(tqdm(p.imap(_image_job, jobs), total=len(jobs), desc="candidates", leave=False))
    return dict(zip(obs, results))


def sweep(cache: dict, rules, r: float) -> pd.DataFrame:
    rows = []
    for rule in rules:
        k = counts(cache, rule)
        n_pred = k["tp"] + k["fp"]
        rows.append({**asdict(rule), "pq": pq_from_counts(k), "pq_lb_est": pq_from_counts(k, r),
                     "precision": k["tp"] / n_pred if n_pred else 0.0,
                     "recall": k["tp"] / (k["tp"] + k["fn"]) if k["tp"] + k["fn"] else 0.0,
                     "pred_per_image": n_pred / len(cache)})
    return pd.DataFrame(rows)


def rule_grid(t_los, closes, t_his, min_means, min_areas) -> list[KeepRule]:
    rules = []
    for t_lo, close, t_hi, mm, a in itertools.product(t_los, closes, t_his, min_means, min_areas):
        if t_hi < t_lo or (0 < mm <= t_lo):  # t_hi below t_lo and min_mean <= t_lo are no-ops: skip duplicates
            continue
        rules.append(KeepRule(t_lo, t_hi, mm, a, close))
    rules += [KeepRule(t, t, 0.0, a, c) for t, c, a in itertools.product(t_los, closes, min_areas)]
    return sorted(set(rules), key=astuple)


def fit_r(known: list[tuple[dict, float]]) -> float:
    """r in [0.3, 1] minimising the squared error between PQ'(r) and known public scores."""
    grid = np.linspace(0.3, 1.0, 141)
    err = [sum((pq_from_counts(k, r) - lb) ** 2 for k, lb in known) for r in grid]
    return float(grid[int(np.argmin(err))])
