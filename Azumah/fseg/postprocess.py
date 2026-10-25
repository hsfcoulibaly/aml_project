"""Instance post-processing shared by every stage.

PQ assumes non-overlapping segments and charges 0.5 for every false positive, so we
1. drop low-confidence and tiny instances,
2. drop instances that lie mostly off the solar disk,
3. resolve overlaps by giving each pixel to the highest-scoring instance,
4. drop instances that lost most of their area in step 3.

Greedy resolution visits instances in descending score, and a decision never depends on lower-scored
instances. So the result at confidence threshold c is exactly the prefix (score >= c) of the result
at the lowest threshold. `resolve_ranked` computes that once; `resolve` slices it.
"""
from __future__ import annotations

import numpy as np

from . import rle as R
from .config import IMAGE_SIZE


def resolve_ranked(
    instances: list[tuple[dict, float]],
    min_area: int = 150,
    disk: np.ndarray | None = None,
    min_on_disk: float = 0.7,
    min_keep_frac: float = 0.5,
) -> list[tuple[dict, float]]:
    """Greedy overlap resolution over all instances. Returns [(rle, score)] in descending score."""
    if not instances:
        return []
    ranked = sorted(instances, key=lambda x: -x[1])
    occupied = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    out = []
    for rle, score in ranked:
        a = float(R.area([rle])[0])
        if a < min_area:
            continue
        m = R.decode(rle)
        if disk is not None and (m & disk).sum() < min_on_disk * a:
            continue
        m &= ~occupied
        if m.sum() < max(min_area, min_keep_frac * a):
            continue
        occupied |= m
        out.append((R.encode(m), score))
    return out


def by_conf(ranked: list[tuple[dict, float]], conf_thr: float) -> list[dict]:
    return [r for r, s in ranked if s >= conf_thr]


def resolve(instances: list[tuple[dict, float]], conf_thr: float = 0.25, min_area: int = 150,
            disk: np.ndarray | None = None, min_on_disk: float = 0.7, min_keep_frac: float = 0.5) -> list[dict]:
    """Non-overlapping RLEs with score >= conf_thr, sorted by score."""
    kept = [(r, s) for r, s in instances if s >= conf_thr]
    return by_conf(resolve_ranked(kept, min_area, disk, min_on_disk, min_keep_frac), conf_thr)
