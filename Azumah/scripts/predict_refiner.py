"""Stage 3 inference: refine every YOLO proposal, rank by quality, calibrate for PQ.

oof:  proposals from runs/consensus_f<k>/oof_raw_fold<k>.pkl, U-Net maps from unet/maps/<unet>/oof,
      refiner refiner/refiner_v<k>/best.pt. Sweeps the ranking score (quality, quality x YOLO conf, conf),
      score threshold and min area with PQ, prints stage 1 on the same images, saves pq_config.json and
      refined_fold<k>.pkl ({obs: {"instances": [(rle, quality, conf)], "disk": ...}}) for fusion.
test: proposals from runs/<yolo-run>/test_raw.pkl, refiner refiner/refiner_full/best.pt, writes a submission.

    python scripts/predict_refiner.py oof --fold 0
    python scripts/predict_refiner.py test --config work/refiner/refiner_v0/pq_config.json
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from fseg import labels  # noqa: E402
from fseg import rle as R  # noqa: E402
from fseg.config import IMAGE_SIZE, get_paths  # noqa: E402
from fseg.metrics import evaluate, summary  # noqa: E402
from fseg.postprocess import by_conf, resolve_ranked  # noqa: E402
from fseg.preprocess import Disk, disk_mask  # noqa: E402
from fseg.refiner import bbox, crop_affine, crop_inputs, load_refiner, paste, refine  # noqa: E402
from fseg.unet import pick_device  # noqa: E402
from fseg.unet_data import read_image  # noqa: E402
from fseg.yolo_infer import apply as yolo_apply  # noqa: E402
from fseg.yolo_infer import write_submission  # noqa: E402

MODES = {"quality": lambda q, c: q, "quality_x_conf": lambda q, c: q * c, "conf": lambda q, c: c}


def refine_all(model, paths, split, raw, map_dir, device, min_conf, thr):
    out = {}
    for obs, item in tqdm(raw.items(), desc="refine"):
        image = read_image(paths, split, obs, IMAGE_SIZE)
        umap = cv2.imread(str(map_dir / f"{obs}.png"), cv2.IMREAD_COLOR) if map_dir else None
        props = [(r, s) for r, s in item["instances"] if s >= min_conf]
        crops, Ms, confs = [], [], []
        for r, s in props:
            m = R.decode(r)
            b = bbox(m)
            if b is None:
                continue
            M = crop_affine(b)
            crops.append(crop_inputs(image, umap, m, M))
            Ms.append(M)
            confs.append(s)
        probs, quals = refine(model, np.stack(crops), device) if crops else (np.zeros(0), np.zeros(0))
        inst = []
        for p, q, c, M in zip(probs, quals, confs, Ms):
            full = paste(p, M, thr)
            if full.any():
                inst.append((R.encode(full), float(q), float(c)))
        out[obs] = {"instances": inst, "disk": item["disk"]}
    return out


def ranked(refined, mode, min_area):
    res = {}
    for obs, item in refined.items():
        disk = disk_mask((IMAGE_SIZE, IMAGE_SIZE), Disk(*item["disk"]), margin=1.01)
        scored = [(r, MODES[mode](q, c)) for r, q, c in item["instances"]]
        res[obs] = resolve_ranked(scored, min_area=min_area, disk=disk)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["oof", "test"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--refiner", default=None, help="refiner run name (default refiner_v<fold> / refiner_full)")
    ap.add_argument("--unet-name", default="unet")
    ap.add_argument("--yolo-run", default="consensus_ffull", help="test: run dir holding test_raw.pkl")
    ap.add_argument("--min-conf", type=float, default=0.05, help="proposals below this YOLO conf are not refined")
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    paths = get_paths(args.data_root, args.work_dir)
    device = pick_device(args.device)
    name = args.refiner or (f"refiner_v{args.fold}" if args.split == "oof" else "refiner_full")
    run_dir = paths.work_dir / "refiner" / name
    model, _ = load_refiner(run_dir / "best.pt", device)
    sub = "oof" if args.split == "oof" else "test"
    map_dir = paths.work_dir / "unet" / "maps" / args.unet_name / sub
    map_dir = map_dir if map_dir.exists() else None

    if args.split == "oof":
        with open(paths.work_dir / "runs" / f"consensus_f{args.fold}" / f"oof_raw_fold{args.fold}.pkl", "rb") as f:
            raw = pickle.load(f)
        raw = dict(list(raw.items())[: args.limit])
        refined = refine_all(model, paths, "train", raw, map_dir, device, args.min_conf, args.mask_thr)
        with open(run_dir / f"refined_fold{args.fold}.pkl", "wb") as f:
            pickle.dump(refined, f)

        refs, obs = labels.load_references(paths), list(raw)
        base_cfg = json.loads((paths.work_dir / "runs" / f"consensus_f{args.fold}" / "pq_config.json").read_text())
        base = evaluate(yolo_apply(raw, base_cfg["conf_thr"], base_cfg["min_area"], base_cfg.get("dark")), refs, obs)
        rows = []
        for mode in MODES:
            for a in [50, 150, 300, 500]:
                rk = ranked(refined, mode, a)
                for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                    res = evaluate({o: by_conf(v, t) for o, v in rk.items()}, refs, obs)
                    rows.append({"mode": mode, "min_area": a, "thr": t,
                                 **{k: v for k, v in res.items() if k != "per_image"}})
        table = pd.DataFrame(rows).sort_values("pq", ascending=False).reset_index(drop=True)
        table.to_csv(run_dir / f"pq_sweep_fold{args.fold}.csv", index=False)
        best = table.iloc[0]
        cfg = {"mode": best["mode"], "min_area": int(best.min_area), "thr": float(best.thr),
               "min_conf": args.min_conf, "mask_thr": args.mask_thr}
        (run_dir / "pq_config.json").write_text(json.dumps(cfg, indent=2))
        print(table.head(10)[["mode", "min_area", "thr", "pq", "sq", "rq", "precision", "recall"]].to_string())
        res = evaluate({o: by_conf(v, cfg["thr"]) for o, v in ranked(refined, cfg["mode"], cfg["min_area"]).items()},
                       refs, obs)
        res["per_image"].to_csv(run_dir / f"oof_per_image_fold{args.fold}.csv", index=False)
        print(f"Stage 1 (YOLO), fold {args.fold}: {summary(base)}")
        print(f"Stage 3 (refiner), fold {args.fold}, {cfg}: {summary(res)}")
    else:
        cfg = json.loads(Path(args.config).read_text())
        with open(paths.work_dir / "runs" / args.yolo_run / "test_raw.pkl", "rb") as f:
            raw = pickle.load(f)
        raw = dict(list(raw.items())[: args.limit])
        refined = refine_all(model, paths, "test", raw, map_dir, device, cfg["min_conf"], cfg["mask_thr"])
        with open(run_dir / "refined_test.pkl", "wb") as f:
            pickle.dump(refined, f)
        preds = {o: by_conf(v, cfg["thr"]) for o, v in ranked(refined, cfg["mode"], cfg["min_area"]).items()}
        out = Path(args.out) if args.out else paths.work_dir / "submission_refiner.csv"
        df = write_submission(preds, out)
        print(f"Wrote {out}: {len(df)} rows for {len(preds)} images")


if __name__ == "__main__":
    main()
