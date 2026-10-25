"""Mask <-> COCO RLE helpers (the competition stores RLE counts only; size is fixed at 2048 x 2048)."""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

from .config import IMAGE_SIZE


def encode(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle


def decode(rle: dict) -> np.ndarray:
    return mask_utils.decode(rle).astype(bool)


def counts_str(rle: dict) -> str:
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def from_counts(counts: str, size: int = IMAGE_SIZE) -> dict:
    return {"size": [size, size], "counts": counts.encode("ascii")}


def area(rles: list[dict]) -> np.ndarray:
    if not rles:
        return np.zeros(0)
    return np.asarray(mask_utils.area(rles), dtype=np.float64)


def iou_matrix(preds: list[dict], gts: list[dict]) -> np.ndarray:
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)))
    return np.asarray(mask_utils.iou(preds, gts, [0] * len(gts)), dtype=np.float64)


def intersection_matrix(preds: list[dict], gts: list[dict]) -> np.ndarray:
    """Pairwise intersection areas, derived from IoU and areas: |A n B| = IoU * (|A| + |B|) / (1 + IoU)."""
    iou = iou_matrix(preds, gts)
    if iou.size == 0:
        return iou
    a, b = area(preds)[:, None], area(gts)[None, :]
    return iou * (a + b) / (1.0 + iou)
