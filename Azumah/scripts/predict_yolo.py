"""Stage 1 evaluation and submission.

Out-of-fold: predict the fold's validation images, sweep thresholds for local PQ, save the best config
and the raw instances (reused by later stages):

    python scripts/predict_yolo.py oof  --weights work/runs/consensus_f0/weights/best.pt --fold 0

Test: predict the 180 test images with a calibrated config and write submission.csv:

    python scripts/predict_yolo.py test --weights work/runs/consensus_ffull/weights/best.pt \
        --config work/runs/consensus_f0/pq_config.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from fseg import labels  # noqa: E402
from fseg.config import get_paths  # noqa: E402
from fseg.metrics import evaluate, summary  # noqa: E402
from fseg.yolo_infer import apply, calibrate, predict_many, save_json, write_submission  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["oof", "test"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tta", default="id,hflip,vflip,rot180", help="comma list from id,hflip,vflip,rot180")
    ap.add_argument("--config", default=None, help="pq_config.json from an oof run (test split)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N images")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    import fseg.dilated  # noqa: F401  (needed to unpickle weights trained with dilated context)
    from ultralytics import YOLO

    paths = get_paths(args.data_root, args.work_dir)
    model = YOLO(args.weights)
    tta = [t for t in args.tta.split(",") if t]
    run_dir = Path(args.weights).resolve().parents[1]

    if args.split == "oof":
        manifest = pd.read_csv(paths.manifest)
        obs = manifest.loc[manifest.fold == args.fold, "observation_id"].tolist()[: args.limit]
        raw = predict_many(model, paths.preprocessed / "train", obs, args.imgsz, tta)
        with open(run_dir / f"oof_raw_fold{args.fold}.pkl", "wb") as f:
            pickle.dump(raw, f)
        refs = labels.load_references(paths)
        best, table = calibrate(raw, refs)
        table.to_csv(run_dir / f"pq_sweep_fold{args.fold}.csv", index=False)
        save_json({**best, "imgsz": args.imgsz, "tta": tta}, run_dir / "pq_config.json")
        print(table.head(10).to_string())
        result = evaluate(apply(raw, best["conf_thr"], best["min_area"], best.get("dark")), refs, observations=obs)
        print(f"Best config {best}\nOOF fold {args.fold}: {summary(result)}")
        result["per_image"].to_csv(run_dir / f"oof_per_image_fold{args.fold}.csv", index=False)
    else:
        cfg = json.loads(Path(args.config).read_text()) if args.config else {"conf_thr": 0.25, "min_area": 150}
        obs = sorted(p.stem for p in (paths.preprocessed / "test").glob("*.jpg"))[: args.limit]
        raw = predict_many(model, paths.preprocessed / "test", obs, args.imgsz, tta)
        with open(run_dir / "test_raw.pkl", "wb") as f:
            pickle.dump(raw, f)
        preds = apply(raw, cfg["conf_thr"], cfg["min_area"], cfg.get("dark"))
        out = Path(args.out) if args.out else (Path("/kaggle/working/submission.csv")
                                               if Path("/kaggle/working").exists() else paths.work_dir / "submission.csv")
        df = write_submission(preds, out)
        n_empty = sum(1 for v in preds.values() if not v)
        print(f"Wrote {out}: {len(df)} rows for {len(preds)} images ({n_empty} with no filament)")


if __name__ == "__main__":
    main()
