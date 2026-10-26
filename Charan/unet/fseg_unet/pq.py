"""Panoptic Quality for single-class filament instances.

    PQ = sum_{(g, p) in TP} IoU(g, p) / (|TP| + 0.5 |FP| + 0.5 |FN|)

A prediction and a ground-truth filament match (TP) when IoU > 0.5; such matches are unique, so no
assignment step is needed. Unmatched predictions are FP and unmatched ground truth is FN.

The hidden test labels are a single annotator's set, so each validation image is scored against
every annotation record of that image and the counts are averaged (the expected leaderboard score).
We report dataset-level PQ (counts pooled over images) and the mean of per-image PQ, because the
exact Kaggle aggregation is not published, plus fragmentation and merge rates:

    split_rate: share of GT filaments covered by >= 2 predictions that each lie mostly inside it
    merge_rate: share of predictions that each mostly contain >= 2 GT filaments
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from .config import FULL_SIZE

MATCH_IOU = 0.5


# ---------------------------------------------------------------- RLE helpers
def encode(mask: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def decode(rle: dict) -> np.ndarray:
    return mask_utils.decode(rle).astype(bool)


def counts_str(rle: dict) -> str:
    c = rle["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else c


def from_counts(counts: str, size: int = FULL_SIZE) -> dict:
    return {"size": [size, size], "counts": counts.encode("ascii")}


def areas(rles: list[dict]) -> np.ndarray:
    return np.asarray(mask_utils.area(rles), np.float64) if rles else np.zeros(0)


def iou_matrix(preds: list[dict], gts: list[dict]) -> np.ndarray:
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)))
    return np.asarray(mask_utils.iou(preds, gts, [0] * len(gts)), np.float64)


# ---------------------------------------------------------------- scoring
def image_stats(preds: list[dict], gts: list[dict], cover: float = 0.5) -> dict:
    iou = iou_matrix(preds, gts)
    matched = iou > MATCH_IOU
    tp = int(matched.sum())
    splits = merges = 0
    if preds and gts:
        pa, ga = areas(preds)[:, None], areas(gts)[None, :]
        inter = iou * (pa + ga) / (1.0 + iou)  # |A n B| recovered from IoU and the two areas
        splits = int(((inter / np.maximum(pa, 1) >= cover).sum(0) >= 2).sum())
        merges = int(((inter / np.maximum(ga, 1) >= cover).sum(1) >= 2).sum())
    return {
        "tp": tp,
        "fp": len(preds) - tp,
        "fn": len(gts) - tp,
        "iou_sum": float(iou[matched].sum()),
        "n_pred": len(preds),
        "n_gt": len(gts),
        "splits": splits,
        "merges": merges,
    }


def _pq(tp, fp, fn, iou_sum) -> float:
    denom = tp + 0.5 * fp + 0.5 * fn
    return iou_sum / denom if denom > 0 else np.nan


def evaluate(predictions: dict, references: dict, observations=None) -> dict:
    """predictions: {obs: [rle]}; references: {obs: [[rle, ...] per annotation record]}."""
    observations = list(references) if observations is None else list(observations)
    rows = []
    for obs in observations:
        preds = predictions.get(obs, [])
        stats = pd.DataFrame([image_stats(preds, gts) for gts in references[obs]]).mean().to_dict()
        stats["observation_id"] = obs
        stats["pq"] = _pq(stats["tp"], stats["fp"], stats["fn"], stats["iou_sum"])
        rows.append(stats)
    df = pd.DataFrame(rows)
    t = df[["tp", "fp", "fn", "iou_sum", "n_pred", "n_gt", "splits", "merges"]].sum()
    return {
        "pq": float(_pq(t.tp, t.fp, t.fn, t.iou_sum)),
        "pq_image_mean": float(df["pq"].mean(skipna=True)),
        "sq": float(t.iou_sum / t.tp) if t.tp else 0.0,
        "rq": float(t.tp / (t.tp + 0.5 * t.fp + 0.5 * t.fn)) if (t.tp + t.fp + t.fn) else 0.0,
        "precision": float(t.tp / t.n_pred) if t.n_pred else 0.0,
        "recall": float(t.tp / t.n_gt) if t.n_gt else 0.0,
        "pred_per_image": float(df["n_pred"].mean()),
        "gt_per_image": float(df["n_gt"].mean()),
        "split_rate": float(t.splits / t.n_gt) if t.n_gt else 0.0,
        "merge_rate": float(t.merges / t.n_pred) if t.n_pred else 0.0,
        "per_image": df,
    }


SUMMARY_KEYS = ["pq", "pq_image_mean", "sq", "rq", "precision", "recall", "pred_per_image", "gt_per_image",
                "split_rate", "merge_rate"]


def summary(result: dict) -> str:
    return " | ".join(f"{k} {result[k]:.4f}" for k in SUMMARY_KEYS)


def annotator_ceiling(references: dict) -> dict:
    """Each annotation record scored against the other records of the same image (images with >= 2)."""
    preds, refs = {}, {}
    for obs, sets in references.items():
        if len(sets) < 2:
            continue
        for i, s in enumerate(sets):
            preds[f"{obs}#{i}"] = s
            refs[f"{obs}#{i}"] = [o for j, o in enumerate(sets) if j != i]
    return evaluate(preds, refs)


def dice_iou(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Binary Dice / IoU with the same convention as Sheikh's evaluate_unet.py (both empty -> 1)."""
    pred, target = pred.astype(bool), target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    s = pred.sum() + target.sum()
    union = np.logical_or(pred, target).sum()
    if s == 0:
        return 1.0, 1.0
    return float(2 * inter / s), float(inter / union) if union else 0.0
