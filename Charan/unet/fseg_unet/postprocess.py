"""Probability map -> filament instances (connected components on the full-resolution grid)."""
from __future__ import annotations

import cv2
import numpy as np

from . import pq
from .config import FULL_SIZE
from .preprocess import Disk, disk_mask


def upsample(prob: np.ndarray, size: int = FULL_SIZE) -> np.ndarray:
    prob = prob.astype(np.float32)
    if prob.shape[0] == size:
        return prob
    return cv2.resize(prob, (size, size), interpolation=cv2.INTER_LINEAR)


def components(prob_full: np.ndarray, thr: float, disk: Disk | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Label map and cv2 stats (row 0 = background) for one threshold, restricted to the disk."""
    binary = (prob_full > thr).astype(np.uint8)
    if disk is not None:
        binary &= disk_mask(binary.shape, disk, scale=1.01).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return labels, stats


def instances(labels: np.ndarray, stats: np.ndarray, min_area: int) -> list[dict]:
    """RLE per connected component with area >= min_area (areas in full-resolution pixels)."""
    out = []
    for i in range(1, len(stats)):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        m = np.zeros(labels.shape, np.uint8)
        m[y : y + h, x : x + w] = labels[y : y + h, x : x + w] == i
        out.append(pq.encode(m))
    return out


def predict_instances(prob: np.ndarray, thr: float, min_area: int, disk: Disk | None = None) -> list[dict]:
    labels, stats = components(upsample(prob), thr, disk)
    return instances(labels, stats, min_area)
