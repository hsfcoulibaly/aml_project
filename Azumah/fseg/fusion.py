"""Stage 4: topology-aware fusion of instance proposals with the dense U-Net maps.

PQ punishes a filament split into pieces twice (each piece usually misses IoU > 0.5: one FN plus FPs),
and splitting is the most common instance error of both stages. Fusion therefore:

1. merge: joins two instances whose skeleton endpoints face each other across a short gap
   (distance <= d px, both end directions within `angle` degrees of the gap direction) when the U-Net
   filament probability along the gap is high (mean >= t); the gap is bridged with U-Net pixels;
2. recover: adds U-Net components (prob >= thr) with a skeleton longer than L px that overlap no kept
   instance, scored mean prob x factor;
3. shape filter: drops compact blobs (sunspots, plage edges) with low elongation
   (skeleton length / sqrt(area));
4. resolves overlaps last (`postprocess.resolve_ranked`, which also drops off-disk instances).
Every threshold is calibrated with PQ on out-of-fold data.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize

from . import rle as R
from .config import IMAGE_SIZE

_NB = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


@dataclass
class FusionParams:
    merge_dist: float = 60.0  # 0 disables merging
    merge_angle: float = 35.0
    merge_prob: float = 0.3
    bridge_width: int = 5
    recover_thr: float = 0.0  # 0 disables recovery
    recover_len: float = 80.0
    recover_factor: float = 0.6
    min_elongation: float = 0.0  # 0 disables the shape filter


def _roi(mask: np.ndarray, pad: int = 2):
    ys, xs = np.nonzero(mask)
    y0, x0 = max(ys.min() - pad, 0), max(xs.min() - pad, 0)
    return y0, x0, mask[y0:ys.max() + pad + 1, x0:xs.max() + pad + 1]


def skeleton_info(mask: np.ndarray, back: int = 15) -> tuple[float, list[tuple[np.ndarray, np.ndarray]]]:
    """Skeleton length (px) and endpoints as (position yx, outward unit direction) in full coordinates."""
    if not mask.any():
        return 0.0, []
    y0, x0, crop = _roi(mask)
    sk = skeletonize(crop.astype(bool)).astype(np.uint8)
    pts = np.argwhere(sk)
    if len(pts) == 0:
        return 0.0, []
    n = cv2.filter2D(sk, -1, _NB, borderType=cv2.BORDER_CONSTANT)
    ends = np.argwhere((sk == 1) & (n == 1))
    if len(ends) == 0 and len(pts) <= 2:
        ends = pts[:1]
    out = []
    for e in ends:
        near = pts[np.linalg.norm(pts - e, axis=1) <= back]
        v = e - near.mean(0)
        nv = np.linalg.norm(v)
        out.append((e + (y0, x0), v / nv if nv > 0 else np.zeros(2)))
    return float(len(pts)), out


def _segment_prob(prob: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    n = max(int(np.linalg.norm(q - p)), 2)
    ys = np.linspace(p[0], q[0], n).round().astype(int).clip(0, prob.shape[0] - 1)
    xs = np.linspace(p[1], q[1], n).round().astype(int).clip(0, prob.shape[1] - 1)
    return float(prob[ys, xs].mean())


def _facing(pa, da, pb, db, max_dist, max_angle) -> bool:
    g = pb - pa
    dist = np.linalg.norm(g)
    if dist > max_dist:
        return False
    if dist < 3:  # touching ends: direction is meaningless
        return True
    g = g / dist
    cos = np.cos(np.deg2rad(max_angle))
    return float(da @ g) >= cos and float(db @ -g) >= cos


def merge_fragments(masks: list[np.ndarray], scores: list[float], prob: np.ndarray | None,
                    p: FusionParams) -> tuple[list[np.ndarray], list[float]]:
    if p.merge_dist <= 0 or len(masks) < 2:
        return masks, scores
    info = [skeleton_info(m)[1] for m in masks]
    parent = list(range(len(masks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bridges = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            best = None
            for pa, da in info[i]:
                for pb, db in info[j]:
                    if not _facing(pa, da, pb, db, p.merge_dist, p.merge_angle):
                        continue
                    sp = _segment_prob(prob, pa, pb) if prob is not None else 1.0
                    if sp >= p.merge_prob and (best is None or sp > best[0]):
                        best = (sp, pa, pb)
            if best is not None:
                parent[find(i)] = find(j)
                bridges.append((i, best[1], best[2]))

    groups: dict[int, list[int]] = {}
    for i in range(len(masks)):
        groups.setdefault(find(i), []).append(i)
    out_m, out_s = [], []
    for root, members in groups.items():
        if len(members) == 1:
            out_m.append(masks[members[0]])
            out_s.append(scores[members[0]])
            continue
        m = np.any([masks[k] for k in members], axis=0).astype(np.uint8)
        for i, pa, pb in bridges:
            if find(i) == root:
                line = np.zeros_like(m)
                cv2.line(line, (int(pa[1]), int(pa[0])), (int(pb[1]), int(pb[0])), 1, p.bridge_width)
                if prob is not None:  # bridge only where the U-Net sees filament, plus the line core
                    core = np.zeros_like(m)
                    cv2.line(core, (int(pa[1]), int(pa[0])), (int(pb[1]), int(pb[0])), 1, 1)
                    line = (line & (prob >= p.merge_prob)) | core
                m |= line
        areas = [masks[k].sum() for k in members]
        out_m.append(m.astype(bool))
        out_s.append(float(np.average([scores[k] for k in members], weights=areas)))
    return out_m, out_s


def recover(masks: list[np.ndarray], prob: np.ndarray, disk: np.ndarray | None, p: FusionParams,
            max_overlap: float = 0.1) -> list[tuple[np.ndarray, float]]:
    if p.recover_thr <= 0 or prob is None:
        return []
    covered = np.any(masks, axis=0) if masks else np.zeros(prob.shape, bool)
    binary = (prob >= p.recover_thr).astype(np.uint8)
    if disk is not None:
        binary &= disk.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if max(w, h) < p.recover_len * 0.7:  # cannot have a long enough skeleton
            continue
        m = np.zeros(prob.shape, bool)
        m[y:y + h, x:x + w] = labels[y:y + h, x:x + w] == i
        if (m & covered).sum() > max_overlap * area:
            continue
        length, _ = skeleton_info(m)
        if length >= p.recover_len:
            out.append((m, float(prob[m].mean()) * p.recover_factor))
    return out


def elongation(mask: np.ndarray) -> float:
    area = mask.sum()
    return skeleton_info(mask)[0] / np.sqrt(area) if area else 0.0


def fuse(instances: list[tuple[dict, float]], prob: np.ndarray | None, disk: np.ndarray | None,
         p: FusionParams) -> list[tuple[dict, float]]:
    """[(rle, score)] -> fused [(rle, score)], not yet overlap-resolved or thresholded."""
    masks = [R.decode(r) for r, _ in instances]
    scores = [s for _, s in instances]
    masks, scores = merge_fragments(masks, scores, prob, p)
    extra = recover(masks, prob, disk, p)
    masks += [m for m, _ in extra]
    scores += [s for _, s in extra]
    out = []
    for m, s in zip(masks, scores):
        if p.min_elongation > 0 and elongation(m) < p.min_elongation:
            continue
        out.append((R.encode(m), s))
    return out


def unet_prob(map_png: np.ndarray | None) -> np.ndarray | None:
    """Filament channel of a saved U-Net map (HxWx3 uint8) as float probabilities at 2048 px."""
    if map_png is None:
        return None
    p = map_png[:, :, 0].astype(np.float32) / 255.0
    return p if p.shape[0] == IMAGE_SIZE else cv2.resize(p, (IMAGE_SIZE, IMAGE_SIZE))
