"""Export preprocessed images and YOLO-seg labels.

Layout (images are shared; labels exist per target mode):

    work/preprocessed/{train,test}/<obs>.jpg     full 2048 px, 3-channel preprocessed
    work/yolo/<mode>/images/<obs>.jpg            resized to `size`, symlink-free copies
    work/yolo/<mode>/labels/<obs>.txt            one polygon per filament, single class 0
    work/yolo/<mode>/fold{k}_train.txt / fold{k}_val.txt / full_train.txt
    work/yolo/<mode>/fold{k}.yaml / full.yaml
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import rle as R
from .config import IMAGE_SIZE, Paths
from .preprocess import preprocess


def preprocess_all(paths: Paths, split: str, file_names: list[str], workers: int = 4) -> Path:
    src = paths.train_images if split == "train" else paths.test_images
    dst = paths.preprocessed / split
    dst.mkdir(parents=True, exist_ok=True)

    def one(name):
        out = dst / f"{Path(name).stem}.jpg"
        if out.exists():
            return
        gray = cv2.imread(str(src / name), cv2.IMREAD_GRAYSCALE)
        img, _ = preprocess(gray)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(one, file_names), total=len(file_names), desc=f"Preprocess {split}"))
    return dst


def _merge_segments(contours: list[np.ndarray]) -> np.ndarray:
    if len(contours) == 1:
        return contours[0]
    try:
        from ultralytics.data.converter import merge_multi_segment

        merged = merge_multi_segment([c.reshape(-1).tolist() for c in contours])
        return np.concatenate(merged, axis=0)
    except Exception:
        return np.concatenate(contours, axis=0)


def mask_to_polygon(mask: np.ndarray, eps: float = 0.7) -> np.ndarray | None:
    """Mask -> (N, 2) polygon in pixels; multi-part masks are joined into one polygon."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    parts = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < 4:
            continue
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
        if len(c) >= 3:
            parts.append(c)
    if not parts:
        return None
    return _merge_segments(parts)


def write_labels(instances: list[dict], out_path: Path) -> int:
    lines = []
    for inst in instances:
        poly = mask_to_polygon(R.decode(inst["rle"]))
        if poly is None:
            continue
        poly = np.clip(poly / IMAGE_SIZE, 0, 1)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in poly.reshape(-1)))
    out_path.write_text("\n".join(lines))
    return len(lines)


def export(paths: Paths, manifest: pd.DataFrame, targets: dict, mode: str = "consensus", size: int = 1280) -> Path:
    root = paths.yolo_root / mode
    img_dir, lbl_dir = root / "images", root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for obs in tqdm(manifest["observation_id"], desc=f"YOLO export ({mode}, {size}px)"):
        dst = img_dir / f"{obs}.jpg"
        if not dst.exists():
            img = cv2.imread(str(paths.preprocessed / "train" / f"{obs}.jpg"))
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_labels(targets[mode][obs], lbl_dir / f"{obs}.txt")

    def write_list(name, obs_ids):
        (root / name).write_text("\n".join(str((img_dir / f"{o}.jpg").resolve()) for o in obs_ids))

    def write_yaml(name, train_list, val_list):
        (root / name).write_text(
            f"path: {root.resolve()}\ntrain: {train_list}\nval: {val_list}\nnc: 1\nnames: ['filament']\n"
        )

    for fold in sorted(manifest["fold"].unique()):
        tr = manifest.loc[manifest.fold != fold, "observation_id"]
        va = manifest.loc[manifest.fold == fold, "observation_id"]
        write_list(f"fold{fold}_train.txt", tr)
        write_list(f"fold{fold}_val.txt", va)
        write_yaml(f"fold{fold}.yaml", f"fold{fold}_train.txt", f"fold{fold}_val.txt")
    write_list("full_train.txt", manifest["observation_id"])
    # Ultralytics needs a val split even for the final model; fold 0 val is only used for monitoring.
    write_yaml("full.yaml", "full_train.txt", "fold0_val.txt")
    return root
