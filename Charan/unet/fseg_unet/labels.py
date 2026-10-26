"""Stage 0: preprocessed images, per-record training masks, PQ references, manifest.

Key facts about MAGFiLO in this competition
* One physical image can have 1-3 annotation records (separate COCO "image" entries whose id is
  "<annotator>-<image stem>", e.g. 050101-<stem>, 050102-<stem>). Each record is an independent set
  of filament polygons from one annotator.
* The split is by physical image (Sheikh's grouped_split.json), never by record id.
* Each record is its own training sample (the test labels are one annotator's set), matching the
  protocol of the existing U-Net and YOLO baselines.
* Only the segmentation polygons are used: the rules forbid training on other ground-truth metadata
  (spines, bboxes, category labels).

Outputs in the work dir
* images/<observation_id>.png   size x size x 3 uint8 (raw, limb-flattened, CLAHE), train + test
* masks/<record_id>.png         size x size binary union mask of that record's filaments (0/255)
* references.pkl                {"sets": {obs: [[rle, ...] per record]}, "record_ids": {obs: [id, ...]}}
* manifest.csv                  one row per record (train/val) or per test image, with disk geometry
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from tqdm import tqdm

from .config import FULL_SIZE, Paths
from .preprocess import preprocess

MASK_KEEP = 0.4  # area-downsampled mask value above which a low-res pixel counts as filament


def ann_to_rle(seg, h: int, w: int) -> dict:
    if isinstance(seg, list):
        polys = [p for p in seg if len(p) >= 6]
        return mask_utils.merge(mask_utils.frPyObjects(polys, h, w))
    if isinstance(seg["counts"], list):
        return mask_utils.frPyObjects(seg, h, w)
    return seg


def _job(args):
    obs, img_path, size, out_img, records, mask_dir = args
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(img_path)
    image, disk = preprocess(gray, size)
    cv2.imwrite(str(out_img), image)
    h, w = gray.shape
    rows = []
    for record_id, rles in records:
        full = np.zeros((h, w), np.uint8)
        for r in rles:
            full |= mask_utils.decode(r)
        small = cv2.resize(full.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(mask_dir / f"{record_id}.png"), ((small >= MASK_KEEP) * 255).astype(np.uint8))
        rows.append({"record_id": record_id, "n_instances": len(rles)})
    return obs, (disk.cx, disk.cy, disk.r), rows


def load_records(ann_path: Path) -> dict:
    """{observation_id: [(record_id, file_name, [rle per filament]), ...]}"""
    with open(ann_path) as f:
        coco = json.load(f)
    anns = defaultdict(list)
    for a in coco["annotations"]:
        if not a.get("iscrowd", 0):
            anns[str(a["image_id"])].append(a)
    by_obs = defaultdict(list)
    for rec in sorted(coco["images"], key=lambda r: str(r["id"])):
        h, w = rec.get("height", FULL_SIZE), rec.get("width", FULL_SIZE)
        rles = [ann_to_rle(a["segmentation"], h, w) for a in anns.get(str(rec["id"]), [])]
        by_obs[Path(rec["file_name"]).stem].append((str(rec["id"]), rec["file_name"], rles))
    return by_obs


def build(paths: Paths, size: int, split_path: Path, workers: int = 4, include_test: bool = True) -> pd.DataFrame:
    paths.images.mkdir(parents=True, exist_ok=True)
    paths.masks.mkdir(parents=True, exist_ok=True)
    with open(split_path) as f:
        split = json.load(f)
    val_files = set(split["val"]["file_names"])
    train_files = set(split["train"]["file_names"])

    by_obs = load_records(paths.ann_path)
    jobs, meta = [], {}
    for obs, recs in by_obs.items():
        file_name = recs[0][1]
        if file_name in val_files:
            s = "val"
        elif file_name in train_files:
            s = "train"
        else:
            print(f"[prepare] {file_name} not in the split file -> train")
            s = "train"
        meta[obs] = (file_name, s)
        jobs.append((obs, paths.train_images / file_name, size, paths.images / f"{obs}.png",
                     [(rid, rles) for rid, _, rles in recs], paths.masks))
    test_files = sorted(paths.test_images.glob("*")) if include_test and paths.test_images.exists() else []
    for p in test_files:
        meta[p.stem] = (p.name, "test")
        jobs.append((p.stem, p, size, paths.images / f"{p.stem}.png", [], paths.masks))

    rows = []
    with Pool(workers) as pool:
        for obs, (cx, cy, r), recs in tqdm(pool.imap_unordered(_job, jobs), total=len(jobs), desc="prepare"):
            file_name, s = meta[obs]
            base = {"observation_id": obs, "file_name": file_name, "split": s, "cx": cx, "cy": cy, "r": r}
            rows += [{**base, **rec} for rec in recs] if recs else [{**base, "record_id": "", "n_instances": 0}]

    manifest = pd.DataFrame(rows).sort_values(["split", "observation_id", "record_id"]).reset_index(drop=True)
    manifest.to_csv(paths.manifest, index=False)
    refs = {
        "sets": {obs: [rles for _, _, rles in recs] for obs, recs in by_obs.items()},
        "record_ids": {obs: [rid for rid, _, _ in recs] for obs, recs in by_obs.items()},
    }
    with open(paths.references, "wb") as f:
        pickle.dump(refs, f)
    return manifest


def load_manifest(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.manifest, dtype={"record_id": str, "observation_id": str}, keep_default_na=False)


def load_references(paths: Paths) -> dict:
    with open(paths.references, "rb") as f:
        return pickle.load(f)
