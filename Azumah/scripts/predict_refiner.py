"""Stage 3 inference: refine every YOLO proposal, rank by quality, calibrate for PQ.

oof:  proposals from runs/consensus_f<k>/oof_raw_fold<k>.pkl, U-Net maps from unet/maps/<unet>/oof,
      refiner refiner/refiner_v<k>/best.pt. Sweeps the ranking score (quality, quality x YOLO conf, conf),
      score threshold and min area with PQ, prints stage 1 on the same images, saves pq_config.json and
      refined_fold<k>.pkl ({obs: {"instances": [(rle, quality, conf)], "disk": ...}}) for fusion.
test: proposals from runs/<yolo-run>/test_raw.pkl, refiner refiner/refiner_full/best.pt, writes a submission.
pool: pools refiner/<tag>_f<k>/refined_fold<k>.pkl over --folds and runs the same sweep on all of them
      (one yardstick: pooled out-of-fold PQ), saves refiner/<tag>_pooled/{pq_sweep_pooled.csv,pq_config.json}.

    python scripts/predict_refiner.py oof --fold 0
    python scripts/predict_refiner.py pool --tag r60 --folds 0 1 2 3 4
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


def refine_all(model, paths, split, raw, map_dir, device, min_conf, thr, tta="hflip"):
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
        probs, quals = refine(model, np.stack(crops), device, tta=tta) if crops else (np.zeros(0), np.zeros(0))
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


def sweep(refined, refs, obs):
    rows = []
    for mode in MODES:
        for a in [50, 150, 300, 500]:
            rk = ranked(refined, mode, a)
            for t in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                res = evaluate({o: by_conf(v, t) for o, v in rk.items()}, refs, obs)
                rows.append({"mode": mode, "min_area": a, "thr": t,
                             **{k: v for k, v in res.items() if k != "per_image"}})
    return pd.DataFrame(rows).sort_values("pq", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["oof", "test", "pool"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="pool only")
    ap.add_argument("--refiner", nargs="+", default=None,
                    help="refiner run name(s), several = ensemble (default refiner_v<fold> / refiner_full)")
    ap.add_argument("--tta", default="hflip", choices=["hflip", "d4"])
    ap.add_argument("--tag", default=None, help="output folder name under work/refiner (default: first refiner)")
    ap.add_argument("--raw", default=None, help="test: proposals pickle (default runs/<yolo-run>/test_raw.pkl)")
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
    if args.split == "pool":
        return pool(paths, args)
    device = pick_device(args.device)
    names = args.refiner or [f"refiner_v{args.fold}" if args.split == "oof" else "refiner_full"]
    models = [load_refiner(paths.work_dir / "refiner" / n / "best.pt", device)[0] for n in names]
    model = models[0] if len(models) == 1 else models
    run_dir = paths.work_dir / "refiner" / (args.tag or names[0])
    run_dir.mkdir(parents=True, exist_ok=True)
    sub = "oof" if args.split == "oof" else "test"
    map_dir = paths.work_dir / "unet" / "maps" / args.unet_name / sub
    map_dir = map_dir if map_dir.exists() else None

    if args.split == "oof":
        with open(paths.work_dir / "runs" / f"consensus_f{args.fold}" / f"oof_raw_fold{args.fold}.pkl", "rb") as f:
            raw = pickle.load(f)
        raw = dict(list(raw.items())[: args.limit])
        refined = refine_all(model, paths, "train", raw, map_dir, device, args.min_conf, args.mask_thr, args.tta)
        with open(run_dir / f"refined_fold{args.fold}.pkl", "wb") as f:
            pickle.dump(refined, f)

        refs, obs = labels.load_references(paths), list(raw)
        base_cfg = json.loads((paths.work_dir / "runs" / f"consensus_f{args.fold}" / "pq_config.json").read_text())
        base = evaluate(yolo_apply(raw, base_cfg["conf_thr"], base_cfg["min_area"], base_cfg.get("dark")), refs, obs)
        table = sweep(refined, refs, obs)
        table.to_csv(run_dir / f"pq_sweep_fold{args.fold}.csv", index=False)
        best = table.iloc[0]
        cfg = {"mode": best["mode"], "min_area": int(best.min_area), "thr": float(best.thr),
               "min_conf": args.min_conf, "mask_thr": args.mask_thr, "tta": args.tta, "refiners": names}
        (run_dir / "pq_config.json").write_text(json.dumps(cfg, indent=2))
        print(table.head(10)[["mode", "min_area", "thr", "pq", "sq", "rq", "precision", "recall"]].to_string())
        res = evaluate({o: by_conf(v, cfg["thr"]) for o, v in ranked(refined, cfg["mode"], cfg["min_area"]).items()},
                       refs, obs)
        res["per_image"].to_csv(run_dir / f"oof_per_image_fold{args.fold}.csv", index=False)
        print(f"Stage 1 (YOLO), fold {args.fold}: {summary(base)}")
        print(f"Stage 3 (refiner), fold {args.fold}, {cfg}: {summary(res)}")
    else:
        cfg = json.loads(Path(args.config).read_text())
        with open(Path(args.raw) if args.raw else paths.work_dir / "runs" / args.yolo_run / "test_raw.pkl", "rb") as f:
            raw = pickle.load(f)
        raw = dict(list(raw.items())[: args.limit])
        refined = refine_all(model, paths, "test", raw, map_dir, device, cfg["min_conf"], cfg["mask_thr"], cfg.get("tta", "hflip"))
        with open(run_dir / "refined_test.pkl", "wb") as f:
            pickle.dump(refined, f)
        preds = {o: by_conf(v, cfg["thr"]) for o, v in ranked(refined, cfg["mode"], cfg["min_area"]).items()}
        out = Path(args.out) if args.out else paths.work_dir / "submission_refiner.csv"
        df = write_submission(preds, out)
        print(f"Wrote {out}: {len(df)} rows for {len(preds)} images")


def pool(paths, args):
    """Pooled out-of-fold PQ over several folds, each refined by the refiner that never saw it."""
    refs, refined, fold_of, base = labels.load_references(paths), {}, {}, {}
    for k in args.folds:
        with open(paths.work_dir / "refiner" / f"{args.tag}_f{k}" / f"refined_fold{k}.pkl", "rb") as f:
            part = pickle.load(f)
        refined.update(part)
        fold_of.update({o: k for o in part})
        with open(paths.work_dir / "runs" / f"consensus_f{k}" / f"oof_raw_fold{k}.pkl", "rb") as f:
            raw = pickle.load(f)
        bc = json.loads((paths.work_dir / "runs" / f"consensus_f{k}" / "pq_config.json").read_text())
        base.update(yolo_apply({o: raw[o] for o in part}, bc["conf_thr"], bc["min_area"], bc.get("dark")))
    obs = list(refined)
    run_dir = paths.work_dir / "refiner" / f"{args.tag}_pooled"
    run_dir.mkdir(parents=True, exist_ok=True)
    table = sweep(refined, refs, obs)
    table.to_csv(run_dir / "pq_sweep_pooled.csv", index=False)
    best = table.iloc[0]
    cfg = {"mode": best["mode"], "min_area": int(best.min_area), "thr": float(best.thr),
           "min_conf": args.min_conf, "mask_thr": args.mask_thr, "tta": args.tta, "folds": args.folds}
    (run_dir / "pq_config.json").write_text(json.dumps(cfg, indent=2))
    print(table.head(10)[["mode", "min_area", "thr", "pq", "sq", "rq", "precision", "recall"]].to_string())
    preds = {o: by_conf(v, cfg["thr"]) for o, v in ranked(refined, cfg["mode"], cfg["min_area"]).items()}
    rows = []
    for k in args.folds:
        ok = [o for o in obs if fold_of[o] == k]
        rows.append({"fold": k, "n": len(ok), "yolo": evaluate(base, refs, ok)["pq"], "refiner": evaluate(preds, refs, ok)["pq"]})
    rows.append({"fold": "pooled", "n": len(obs), "yolo": evaluate(base, refs, obs)["pq"], "refiner": evaluate(preds, refs, obs)["pq"]})
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(run_dir / "per_fold.csv", index=False)
    print(per_fold.round(4).to_string(index=False))
    print(f"pooled config: {cfg}")


if __name__ == "__main__":
    main()
