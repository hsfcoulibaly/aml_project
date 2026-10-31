"""Stage 2 inference: dense maps for later stages, plus the standalone U-Net baseline.

oof:  each fold model predicts its own validation images. Maps are saved as 8-bit 3-channel PNGs
      (filament, centerline, boundary) at 2048 px in work/unet/maps/<name>/oof/<obs>.png. With
      --sweep, standalone U-Net instances (threshold + connected components) are scored with PQ on
      the given fold, and the best config goes to work/unet/runs/<name>_f<k>/pq_config.json.
test: averages every listed fold model per test image -> work/unet/maps/<name>/test/<obs>.png; with
      --config also writes a standalone U-Net submission.

    python scripts/predict_unet.py oof  --folds 0 1 2 3 4 --sweep-fold 0
    python scripts/predict_unet.py test --folds 0 1 2 3 4 --config work/unet/runs/unet_f0/pq_config.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from fseg import labels  # noqa: E402
from fseg.config import IMAGE_SIZE, get_paths  # noqa: E402
from fseg.metrics import evaluate, summary  # noqa: E402
from fseg.postprocess import by_conf  # noqa: E402
from fseg.unet import load_weights, pick_device, predict_full, prob_to_instances  # noqa: E402
from fseg.unet_data import read_image, unet_root  # noqa: E402
from fseg.yolo_infer import write_submission  # noqa: E402


def predict_maps(models, paths, split, obs, out_dir, device, tta):
    out_dir.mkdir(parents=True, exist_ok=True)
    for o in tqdm(obs, desc=f"U-Net {split}"):
        acc = None
        for model, size in models:
            p = predict_full(model, read_image(paths, split, o, size), device, tile=size, tta=tta)
            if size != IMAGE_SIZE:
                p = np.stack([cv2.resize(c, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR) for c in p])
            acc = p if acc is None else acc + p
        acc /= len(models)
        cv2.imwrite(str(out_dir / f"{o}.png"), np.round(acc.transpose(1, 2, 0) * 255).astype(np.uint8))


def load_map(path: Path) -> np.ndarray:
    """(3, H, W) float32 probabilities from a saved map."""
    return cv2.imread(str(path), cv2.IMREAD_COLOR).transpose(2, 0, 1).astype(np.float32) / 255.0


def instances_for(paths, split, map_dir, obs, thr, min_area):
    out = {}
    for o in obs:
        disk = cv2.imread(str(paths.preprocessed / split / f"{o}.jpg"), cv2.IMREAD_GRAYSCALE) > 0
        out[o] = prob_to_instances(load_map(map_dir / f"{o}.png")[0], thr, min_area, disk)
    return out


def sweep(paths, map_dir, obs, refs, thrs, areas, score_grid):
    from fseg import rle as R

    rows = []
    for t in tqdm(thrs, desc="PQ sweep"):
        inst = instances_for(paths, "train", map_dir, obs, t, min(areas))
        areas_of = {o: R.area([r for r, _ in v]) for o, v in inst.items()}
        for a in areas:
            for s in score_grid:  # mean-probability filter (a strict keep rule lifted PQ in public notebooks)
                preds = {o: [r for (r, sc), ar in zip(v, areas_of[o]) if ar >= a and sc >= s]
                         for o, v in inst.items()}
                res = evaluate(preds, refs, observations=obs)
                rows.append({"thr": t, "min_area": a, "min_mean_prob": s,
                             **{k: v for k, v in res.items() if k != "per_image"}})
    return pd.DataFrame(rows).sort_values("pq", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["oof", "test"])
    ap.add_argument("--folds", nargs="+", default=["0"])
    ap.add_argument("--name", default="unet", help="run prefix: weights in work/unet/runs/<name>_f<k>/best.pt")
    ap.add_argument("--tta", default="flip", choices=["none", "flip", "d4"])
    ap.add_argument("--sweep-fold", type=int, default=None, help="oof: score standalone instances on this fold")
    ap.add_argument("--config", default=None, help="test: pq_config.json for a standalone submission")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N images per fold")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    paths = get_paths(args.data_root, args.work_dir)
    device = pick_device(args.device)
    root = unet_root(paths)
    manifest = pd.read_csv(paths.manifest)

    def load(fold):
        model, ckpt = load_weights(root / "runs" / f"{args.name}_f{fold}" / "best.pt", device)
        return model, ckpt["cfg"]["size"]

    if args.split == "oof":
        map_dir = root / "maps" / args.name / "oof"
        for f in args.folds:
            obs = manifest.loc[manifest.fold == int(f), "observation_id"].tolist()[: args.limit]
            predict_maps([load(f)], paths, "train", obs, map_dir, device, args.tta)
        if args.sweep_fold is not None:
            obs = manifest.loc[manifest.fold == args.sweep_fold, "observation_id"].tolist()[: args.limit]
            refs = labels.load_references(paths)
            table = sweep(paths, map_dir, obs, refs, [0.3, 0.4, 0.5, 0.6, 0.7], [100, 200, 400, 600],
                          [0.0, 0.5, 0.6, 0.7])
            run_dir = root / "runs" / f"{args.name}_f{args.sweep_fold}"
            table.to_csv(run_dir / f"pq_sweep_fold{args.sweep_fold}.csv", index=False)
            best = table.iloc[0]
            cfg = {"thr": float(best.thr), "min_area": int(best.min_area), "min_mean_prob": float(best.min_mean_prob),
                   "tta": args.tta}
            (run_dir / "pq_config.json").write_text(json.dumps(cfg, indent=2))
            print(table.head(8).to_string())
            inst = instances_for(paths, "train", map_dir, obs, cfg["thr"], cfg["min_area"])
            res = evaluate({o: by_conf(v, cfg["min_mean_prob"]) for o, v in inst.items()}, refs, observations=obs)
            res["per_image"].to_csv(run_dir / f"oof_per_image_fold{args.sweep_fold}.csv", index=False)
            print(f"U-Net standalone, fold {args.sweep_fold}, {cfg}: {summary(res)}")
    else:
        map_dir = root / "maps" / args.name / "test"
        obs = sorted(p.stem for p in (paths.preprocessed / "test").glob("*.jpg"))[: args.limit]
        predict_maps([load(f) for f in args.folds], paths, "test", obs, map_dir, device, args.tta)
        if args.config:
            cfg = json.loads(Path(args.config).read_text())
            inst = instances_for(paths, "test", map_dir, obs, cfg["thr"], cfg["min_area"])
            preds = {o: by_conf(v, cfg.get("min_mean_prob", 0.0)) for o, v in inst.items()}
            out = Path(args.out) if args.out else root / "submission_unet.csv"
            df = write_submission(preds, out)
            print(f"Wrote {out}: {len(df)} rows for {len(preds)} images")


if __name__ == "__main__":
    main()
