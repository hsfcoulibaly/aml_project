"""U-Net-2048 maps -> instance proposals in the stage-1 (YOLO) pickle format, so the stage-3 refiner in
Azumah/ runs on them unchanged.

Proposals are the 8-connected components of the filament probability at several thresholds (a low
threshold gives whole filaments that the high one splits, a high one separates touching filaments). Near
duplicates across thresholds (IoU > 0.9) are dropped. Score = mean probability inside the component.

Writes, under the work dir:
  runs/consensus_f<k>/oof_raw_fold<k>.pkl   {obs: {"instances": [(rle, score)], "dark": None, "disk": (cx, cy, r)}}
  runs/consensus_f<k>/pq_config.json         standalone baseline (the U-Net fold-0 sweep winner)
  runs/unet2048_test/test_raw.pkl            same, from the 5-fold-average test maps

    python Charan/fusion/make_unet_proposals.py --code Azumah --maps work/unet/maps/unet2048
"""
import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

THRESHOLDS = (0.3, 0.5, 0.7)
MIN_AREA, DUP_IOU = 100, 0.9
BASELINE = {"conf_thr": 0.7, "min_area": 400, "dark": None}  # unet2048 fold-0 sweep: thr .5, area 400, mean .7


def components(prob, disk, thr):
    binary = ((prob > thr) & disk).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        m = np.zeros(prob.shape, bool)
        m[y:y + h, x:x + w] = labels[y:y + h, x:x + w] == i
        out.append((m, float(prob[m].mean())))
    return out


def proposals(prob, disk, R):
    cands = [c for t in THRESHOLDS for c in components(prob, disk, t)]
    rles = [R.encode(m) for m, _ in cands]
    keep = []
    if rles:
        iou = R.iou_matrix(rles, rles)
        for i in range(len(rles)):
            if all(iou[i, j] <= DUP_IOU for j in keep):
                keep.append(i)
    return [(rles[i], cands[i][1]) for i in keep]


_G = {}


def _init(code, work, split, map_dir):
    sys.path.insert(0, code)
    from fseg import rle as R
    from fseg.config import get_paths
    from fseg.preprocess import find_disk
    from fseg.unet_data import read_image
    _G.update(R=R, paths=get_paths(work_dir=work), find_disk=find_disk, read_image=read_image, split=split,
              map_dir=Path(map_dir))


def _one(obs):
    img = _G["read_image"](_G["paths"], _G["split"], obs, 2048)
    disk = _G["find_disk"](img[:, :, 0])
    on_disk = img.max(axis=2) > 0  # sky is zero in the preprocessed image
    umap = cv2.imread(str(_G["map_dir"] / f"{obs}.png"), cv2.IMREAD_COLOR)
    prob = umap[:, :, 0].astype(np.float32) / 255.0
    return obs, {"instances": proposals(prob, on_disk, _G["R"]), "dark": None,
                 "disk": (disk.cx, disk.cy, disk.r)}


def build(code, work, split, map_dir, observations, workers):
    with Pool(workers, initializer=_init, initargs=(code, work, split, map_dir)) as p:
        return dict(p.imap(_one, observations, chunksize=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="Azumah/ directory (imported read-only)")
    ap.add_argument("--maps", required=True, help="maps/unet2048 with oof/ and test/")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    code = str(Path(args.code).resolve())
    sys.path.insert(0, code)
    import pandas as pd
    from fseg.config import get_paths

    paths = get_paths(work_dir=args.work_dir)
    work = str(paths.work_dir)
    maps = Path(args.maps)
    manifest = pd.read_csv(paths.manifest)
    for k in sorted(manifest.fold.unique()):
        obs = manifest.loc[manifest.fold == k, "observation_id"].tolist()
        raw = build(code, work, "train", maps / "oof", obs, args.workers)
        d = paths.work_dir / "runs" / f"consensus_f{k}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"oof_raw_fold{k}.pkl", "wb") as f:
            pickle.dump(raw, f)
        (d / "pq_config.json").write_text(json.dumps(BASELINE))
        n = [len(v["instances"]) for v in raw.values()]
        print(f"fold {k}: {len(raw)} images, {np.mean(n):.1f} proposals/image", flush=True)
    obs = sorted(p.stem for p in (paths.preprocessed / "test").glob("*.jpg"))
    raw = build(code, work, "test", maps / "test", obs, args.workers)
    d = paths.work_dir / "runs" / "unet2048_test"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "test_raw.pkl", "wb") as f:
        pickle.dump(raw, f)
    print(f"test: {len(raw)} images, {np.mean([len(v['instances']) for v in raw.values()]):.1f} proposals/image")


if __name__ == "__main__":
    main()
