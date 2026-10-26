"""Validation: per-record Dice/IoU (comparable to the baselines) and PQ with a threshold sweep."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from tqdm import tqdm

from . import pq
from .postprocess import components, upsample
from .preprocess import Disk

DICE_SIZE = 512  # Sheikh's evaluate_unet.py and Hassan's YOLO notebook score at 512 x 512


def disks_from_manifest(manifest: pd.DataFrame) -> dict:
    first = manifest.drop_duplicates("observation_id")
    return {o: Disk(cx, cy, r) for o, cx, cy, r in first[["observation_id", "cx", "cy", "r"]].to_numpy()}


def dice_targets(references: dict, observations) -> dict:
    """{record_id: 512 x 512 bool} built like prepare_unet_data.py (full-res raster -> nearest resize)."""
    out = {}
    for obs in observations:
        for rid, rles in zip(references["record_ids"][obs], references["sets"][obs]):
            full = np.zeros((2048, 2048), np.uint8)
            for r in rles:
                full |= mask_utils.decode(r)
            out[rid] = cv2.resize(full, (DICE_SIZE, DICE_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
    return out


def dice_table(probs: dict, targets: dict, references: dict, thr: float = 0.5) -> pd.DataFrame:
    rows = []
    for obs, prob in probs.items():
        small = cv2.resize(prob.astype(np.float32), (DICE_SIZE, DICE_SIZE), interpolation=cv2.INTER_LINEAR) > thr
        for rid in references["record_ids"][obs]:
            d, i = pq.dice_iou(small, targets[rid])
            rows.append({"record_id": rid, "observation_id": obs, "dice": d, "iou": i})
    return pd.DataFrame(rows)


def instances_by_threshold(prob: np.ndarray, disk: Disk, thresholds, min_area_floor: int) -> dict:
    """{thr: (rles, areas)} with every component >= min_area_floor encoded once per threshold."""
    full = upsample(prob)
    out = {}
    for t in thresholds:
        labels, stats = components(full, t, disk)
        rles, areas = [], []
        for i in range(1, len(stats)):
            x, y, w, h, area = stats[i]
            if area < min_area_floor:
                continue
            m = np.zeros(labels.shape, np.uint8)
            m[y : y + h, x : x + w] = labels[y : y + h, x : x + w] == i
            rles.append(pq.encode(m))
            areas.append(area)
        out[t] = (rles, np.asarray(areas))
    return out


def pq_sweep(probs: dict, disks: dict, references: dict, thresholds, min_areas) -> tuple[pd.DataFrame, dict]:
    """Exact PQ for every (threshold, min_area); returns the table and the best config."""
    floor = min(min_areas)
    cache = {obs: instances_by_threshold(p, disks[obs], thresholds, floor)
             for obs, p in tqdm(probs.items(), desc="instances", leave=False)}
    sets = {obs: references["sets"][obs] for obs in probs}
    rows = []
    for t in thresholds:
        for a in min_areas:
            preds = {obs: [r for r, ar in zip(*cache[obs][t]) if ar >= a] for obs in probs}
            res = pq.evaluate(preds, sets)
            rows.append({"threshold": t, "min_area": a, **{k: res[k] for k in pq.SUMMARY_KEYS}})
    table = pd.DataFrame(rows).sort_values("pq", ascending=False).reset_index(drop=True)
    best = {"threshold": float(table.threshold[0]), "min_area": int(table.min_area[0]), "pq": float(table.pq[0])}
    return table, best
