"""Annotation handling: observation-level manifest, folds, multi-annotator consensus, references.

Key facts about MAGFiLO in this competition
* One physical observation (image file) can have 1-3 independent annotation sets. Each set is a
  separate COCO "image" record with its own id (e.g. 050101-..., 050102-..., 050103-...).
* Validation must be split by observation, never by COCO record id, or the same image leaks.
* Annotators often disagree about which filaments exist, so we cluster instances across sets and
  record how many annotators marked each filament ("agreement").

Outputs (written by `build`)
* manifest.csv   one row per observation with its fold
* targets.pkl    {mode: {observation_id: [ {"rle", "agreement", "n_sets"} ]}} for modes
                 "largest" (most complete single set) and "consensus" (clustered across sets)
* references.pkl {observation_id: [[rle, ...] per annotation set]} used by the PQ evaluator
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from sklearn.model_selection import StratifiedKFold

from . import rle as R
from .config import SEED, Paths


def observation_id(file_name: str) -> str:
    return Path(file_name).stem


def ann_to_rle(ann: dict, h: int, w: int) -> dict:
    seg = ann["segmentation"]
    if isinstance(seg, list):
        polys = [p for p in seg if len(p) >= 6]
        rles = mask_utils.frPyObjects(polys, h, w)
        return mask_utils.merge(rles)
    if isinstance(seg["counts"], list):
        return mask_utils.frPyObjects(seg, h, w)
    return seg


def load_sets(ann_path: Path) -> tuple[dict, dict]:
    """Return ({observation_id: file_name}, {observation_id: [[rle,...] per set]}) ."""
    with open(ann_path) as f:
        coco = json.load(f)
    anns_by_record = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_record[ann["image_id"]].append(ann)

    files, sets = {}, defaultdict(list)
    for rec in sorted(coco["images"], key=lambda r: str(r["id"])):
        obs = observation_id(rec["file_name"])
        files[obs] = rec["file_name"]
        h, w = rec["height"], rec["width"]
        rles = [ann_to_rle(a, h, w) for a in anns_by_record.get(rec["id"], [])]
        rles = [r for r in rles if mask_utils.area(r) > 0]
        sets[obs].append(rles)
    return files, dict(sets)


def cluster_across_sets(sets: list[list[dict]], iou_thr: float = 0.2, cover_thr: float = 0.5) -> list[dict]:
    """Group instances that different annotators drew for the same filament.

    Two instances from different sets are linked if IoU >= iou_thr or one covers >= cover_thr of the
    other (annotator drew only a fragment). Connected components of that graph are clusters.
    Cluster mask = pixels marked by at least half of the sets that marked the cluster.
    """
    n_sets = len(sets)
    items = [(s, r) for s, rles in enumerate(sets) for r in rles]
    if not items:
        return []
    if n_sets == 1:
        return [{"rle": r, "agreement": 1.0, "n_sets": 1} for _, r in items]

    rles = [r for _, r in items]
    owner = np.array([s for s, _ in items])
    iou = R.iou_matrix(rles, rles)
    inter = R.intersection_matrix(rles, rles)
    areas = R.area(rles)
    cover = inter / np.minimum(areas[:, None], areas[None, :]).clip(min=1)
    linked = ((iou >= iou_thr) | (cover >= cover_thr)) & (owner[:, None] != owner[None, :])

    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(linked, 1))):
        parent[find(i)] = find(j)

    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)

    out = []
    for members in groups.values():
        member_sets = sorted(set(owner[members].tolist()))
        if len(members) == 1:
            mask_rle = rles[members[0]]
        else:
            votes = None
            for s in member_sets:
                per_set = mask_utils.merge([rles[i] for i in members if owner[i] == s])
                m = mask_utils.decode(per_set).astype(np.uint8)
                votes = m if votes is None else votes + m
            mask = votes * 2 >= len(member_sets)
            mask_rle = R.encode(mask)
        out.append({"rle": mask_rle, "agreement": len(member_sets) / n_sets, "n_sets": n_sets})
    return out


def build(paths: Paths, n_folds: int = 5, min_agreement: float = 0.5) -> pd.DataFrame:
    files, sets = load_sets(paths.ann_path)

    targets = {"largest": {}, "consensus": {}}
    rows = []
    for obs, obs_sets in sets.items():
        largest = max(obs_sets, key=lambda s: (len(s), float(R.area(s).sum()) if s else 0.0))
        targets["largest"][obs] = [{"rle": r, "agreement": 1.0, "n_sets": len(obs_sets)} for r in largest]
        clusters = cluster_across_sets(obs_sets)
        targets["consensus"][obs] = [c for c in clusters if c["agreement"] >= min_agreement - 1e-9]
        rows.append(
            {
                "observation_id": obs,
                "file_name": files[obs],
                "site": obs[-2:],
                "year": int(obs[:4]),
                "n_sets": len(obs_sets),
                "n_largest": len(largest),
                "n_consensus": len(targets["consensus"][obs]),
                "n_clusters": len(clusters),
            }
        )

    manifest = pd.DataFrame(rows).sort_values("observation_id").reset_index(drop=True)
    count_bin = pd.qcut(manifest["n_largest"], 4, labels=False, duplicates="drop").astype(str)
    strata = manifest["site"] + "_" + count_bin
    # fall back to site-only strata if any stratum is smaller than n_folds
    if strata.value_counts().min() < n_folds:
        strata = manifest["site"]
    if strata.value_counts().min() < n_folds:
        strata = pd.Series(["all"] * len(manifest))
    manifest["fold"] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for fold, (_, val_idx) in enumerate(skf.split(manifest, strata)):
        manifest.loc[val_idx, "fold"] = fold

    manifest.to_csv(paths.manifest, index=False)
    with open(paths.targets, "wb") as f:
        pickle.dump(targets, f)
    with open(paths.references, "wb") as f:
        pickle.dump(sets, f)
    return manifest


def load_targets(paths: Paths) -> dict:
    with open(paths.targets, "rb") as f:
        return pickle.load(f)


def load_references(paths: Paths) -> dict:
    with open(paths.references, "rb") as f:
        return pickle.load(f)


def skeleton(mask: np.ndarray) -> np.ndarray:
    """Centerline of a filament mask (used by the U-Net centerline head and the fusion stage)."""
    from skimage.morphology import skeletonize

    return skeletonize(mask.astype(bool))
