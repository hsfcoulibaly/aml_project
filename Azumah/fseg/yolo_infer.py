"""YOLO inference at full resolution, flip TTA, PQ calibration and submission writing."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import rle as R
from .config import IMAGE_SIZE
from .metrics import evaluate
from .postprocess import by_conf, resolve_ranked
from .preprocess import find_disk, disk_mask

# (forward, inverse) image transforms for TTA; all are self-inverse
TTA = {
    "id": lambda a: a,
    "hflip": lambda a: a[:, ::-1],
    "vflip": lambda a: a[::-1, :],
    "rot180": lambda a: a[::-1, ::-1],
}


def _predict_view(model, img: np.ndarray, imgsz: int, conf: float, iou: float, max_det: int):
    res = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, retina_masks=True, max_det=max_det, verbose=False)[0]
    if res.masks is None or len(res.boxes) == 0:
        return np.zeros((0, *img.shape[:2]), bool), np.zeros(0)
    masks = res.masks.data.cpu().numpy() > 0.5
    scores = res.boxes.conf.cpu().numpy()
    return masks, scores


def _vote(views: list[list[tuple[np.ndarray, float]]], match_iou: float = 0.4, min_frac: float = 0.5):
    """Merge instances across TTA views: cluster by IoU, keep clusters seen in >= min_frac of views,
    mask = pixels predicted by >= half the members, score = summed score / number of views."""
    n_views = len(views)
    if n_views == 1:
        return views[0]
    flat = [(m, s, v) for v, inst in enumerate(views) for m, s in inst]
    if not flat:
        return []
    flat.sort(key=lambda x: -x[1])
    rles = [R.encode(m) for m, _, _ in flat]
    iou = R.iou_matrix(rles, rles)
    used = np.zeros(len(flat), bool)
    out = []
    for i in range(len(flat)):
        if used[i]:
            continue
        members = [j for j in np.nonzero((iou[i] >= match_iou) & ~used)[0]]
        # at most one member per view (highest score first)
        seen, chosen = set(), []
        for j in sorted(members, key=lambda j: -flat[j][1]):
            if flat[j][2] not in seen:
                seen.add(flat[j][2])
                chosen.append(j)
        used[chosen] = True
        if len(chosen) < min_frac * n_views:
            continue
        votes = np.sum([flat[j][0] for j in chosen], axis=0)
        mask = votes * 2 >= len(chosen)
        score = sum(flat[j][1] for j in chosen) / n_views
        out.append((mask, score))
    return out


def predict_image(model, img_full: np.ndarray, imgsz: int, tta: list[str], conf: float = 0.05,
                  iou: float = 0.6, max_det: int = 100) -> list[tuple[dict, float]]:
    """img_full: 2048x2048x3 preprocessed BGR image. Returns [(rle at 2048, score)]."""
    small = cv2.resize(img_full, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    views = []
    for name in tta:
        t = TTA[name]
        masks, scores = _predict_view(model, np.ascontiguousarray(t(small)), imgsz, conf, iou, max_det)
        views.append([(t(m), s) for m, s in zip(masks, scores)])
    merged = _vote(views)
    out = []
    for m, s in merged:
        full = cv2.resize(m.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR) > 0
        out.append((R.encode(full), float(s)))
    return out


DARK_RATIOS = (0.97, 0.93, 0.90, 0.85)  # candidate thresholds on limb-flattened intensity / quiet Sun
_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def dark_tighten(mask: np.ndarray, flat: np.ndarray, ratio: float) -> np.ndarray:
    """Keep the dark core of a coarse mask: pixels darker than `ratio` x quiet Sun, closed to stay
    contiguous. YOLO prototype masks are ~4x coarser than the image, so they bleed into bright
    surroundings; filament material is always darker than the quiet Sun in H-alpha."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask
    y0, y1, x0, x1 = max(ys.min() - 4, 0), ys.max() + 5, max(xs.min() - 4, 0), xs.max() + 5
    m = mask[y0:y1, x0:x1]
    core = m & (flat[y0:y1, x0:x1] < ratio * 127.5)
    core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE, _CLOSE).astype(bool) & m
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = core
    return out


def predict_many(model, image_dir: Path, observations: list[str], imgsz: int, tta: list[str],
                 dark_ratios=DARK_RATIOS) -> dict:
    raw = {}
    for obs in tqdm(observations, desc="YOLO inference"):
        img = cv2.imread(str(image_dir / f"{obs}.jpg"))
        disk = find_disk(img[:, :, 0])
        instances = predict_image(model, img, imgsz, tta)
        flat = img[:, :, 1]
        dark = {r: [] for r in dark_ratios}
        for rle, _ in instances:
            m = R.decode(rle)
            for r in dark_ratios:
                dark[r].append(R.encode(dark_tighten(m, flat, r)))
        raw[obs] = {"instances": instances, "dark": dark, "disk": (disk.cx, disk.cy, disk.r)}
    return raw


def _ranked(raw: dict, min_area: int, dark: float | None, min_on_disk: float = 0.7) -> dict:
    """Greedy-resolved instances at the lowest confidence; any threshold is then a prefix (exact)."""
    from .preprocess import Disk

    out = {}
    for obs, item in raw.items():
        disk = disk_mask((IMAGE_SIZE, IMAGE_SIZE), Disk(*item["disk"]), margin=1.01)
        instances = item["instances"]
        if dark is not None:
            instances = [(r, s) for r, (_, s) in zip(item["dark"][dark], instances)]
        out[obs] = resolve_ranked(instances, min_area=min_area, disk=disk, min_on_disk=min_on_disk)
    return out


def apply(raw: dict, conf_thr: float, min_area: int, dark: float | None = None, min_on_disk: float = 0.7) -> dict:
    ranked = _ranked(raw, min_area, dark, min_on_disk)
    return {obs: by_conf(r, conf_thr) for obs, r in ranked.items()}


def calibrate(raw: dict, references: dict, conf_grid=None, area_grid=None, dark_grid=None) -> tuple[dict, pd.DataFrame]:
    conf_grid = conf_grid or [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    area_grid = area_grid or [50, 150, 300, 500]
    available = set(next(iter(raw.values())).get("dark", {})) if raw else set()
    dark_grid = dark_grid or [None] + sorted(available, reverse=True)
    rows = []
    for d in tqdm(dark_grid, desc="PQ sweep"):
        for a in area_grid:
            ranked = _ranked(raw, a, d)
            for c in conf_grid:
                preds = {obs: by_conf(r, c) for obs, r in ranked.items()}
                res = evaluate(preds, references, observations=list(raw))
                rows.append({"conf_thr": c, "min_area": a, "dark": d, **{k: v for k, v in res.items() if k != "per_image"}})
    table = pd.DataFrame(rows).sort_values("pq", ascending=False).reset_index(drop=True)
    d = table.loc[0, "dark"]
    best = {"conf_thr": float(table.loc[0, "conf_thr"]), "min_area": int(table.loc[0, "min_area"]),
            "dark": None if d is None or pd.isna(d) else float(d)}
    return best, table


def write_submission(predictions: dict, out_csv: Path) -> pd.DataFrame:
    rows = []
    for obs, rles in sorted(predictions.items()):
        for k, r in enumerate(rles, 1):
            rows.append({"filament_id": f"{obs}_{k}", "segmentation_rle": R.counts_str(r)})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(out_csv, index=False)
    return df


def save_json(obj, path: Path):
    Path(path).write_text(json.dumps(obj, indent=2))
