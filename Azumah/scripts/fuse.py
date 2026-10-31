"""Stage 4: fuse instance proposals with U-Net maps, calibrated for PQ.

Proposals come from stage 1 (`--source yolo`: runs/consensus_f<k>/oof_raw_fold<k>.pkl, score = conf) or
stage 3 (`--source refiner`: refiner/<name>/refined_fold<k>.pkl, scored with that run's pq_config mode).
Pipeline per image: overlap-resolve all proposals, keep score >= --pre-thr, fuse (merge, recover,
shape filter), overlap-resolve again, then threshold. The sweep is coordinate-wise (merge, then
recover, then shape), each step keeping the best PQ so far, so it stays affordable on CPU.

    python scripts/fuse.py oof --source yolo --fold 0
    python scripts/fuse.py test --source refiner --config work/fusion/refiner_f0/fusion_config.json
"""
import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fseg import labels  # noqa: E402
from fseg.config import IMAGE_SIZE, get_paths  # noqa: E402
from fseg.fusion import FusionParams, fuse, unet_prob  # noqa: E402
from fseg.metrics import evaluate, summary  # noqa: E402
from fseg.postprocess import by_conf, resolve_ranked  # noqa: E402
from fseg.preprocess import Disk, disk_mask  # noqa: E402
from fseg.yolo_infer import write_submission  # noqa: E402

MODES = {"quality": lambda q, c: q, "quality_x_conf": lambda q, c: q * c, "conf": lambda q, c: c}
_CTX = {}


def load_source(paths, source, split, fold, refiner_name):
    if source == "yolo":
        f = (paths.work_dir / "runs" / f"consensus_f{fold}" / f"oof_raw_fold{fold}.pkl" if split == "oof"
             else paths.work_dir / "runs" / "consensus_ffull" / "test_raw.pkl")
        with open(f, "rb") as fh:
            raw = pickle.load(fh)
        return {o: {"instances": it["instances"], "disk": it["disk"]} for o, it in raw.items()}
    run = paths.work_dir / "refiner" / refiner_name
    with open(run / (f"refined_fold{fold}.pkl" if split == "oof" else "refined_test.pkl"), "rb") as fh:
        refined = pickle.load(fh)
    mode = json.loads((run / "pq_config.json").read_text())["mode"]
    return {o: {"instances": [(r, MODES[mode](q, c)) for r, q, c in it["instances"]], "disk": it["disk"]}
            for o, it in refined.items()}


def _init(data, map_dir, pre_thr, base_area):
    _CTX.update(data=data, map_dir=map_dir, pre_thr=pre_thr, base_area=base_area)


def _fuse_one(args):
    obs, params = args
    item = _CTX["data"][obs]
    disk = disk_mask((IMAGE_SIZE, IMAGE_SIZE), Disk(*item["disk"]), margin=1.01)
    base = [(r, s) for r, s in resolve_ranked(item["instances"], min_area=_CTX["base_area"], disk=disk)
            if s >= _CTX["pre_thr"]]
    m = cv2.imread(str(_CTX["map_dir"] / f"{obs}.png"), cv2.IMREAD_COLOR) if _CTX["map_dir"] else None
    fused = fuse(base, unet_prob(m), disk, params)
    return obs, resolve_ranked(fused, min_area=_CTX["base_area"], disk=disk)


def run_config(pool, obs, params):
    return dict(pool.map(_fuse_one, [(o, params) for o in obs], chunksize=4))


def score(ranked, refs, obs, thrs, areas):
    from fseg import rle as R

    best = None
    for a in areas:
        for t in thrs:
            preds = {o: [r for r, s in v if s >= t and R.area([r])[0] >= a] for o, v in ranked.items()}
            res = evaluate(preds, refs, observations=obs)
            if best is None or res["pq"] > best[0]["pq"]:
                best = (res, t, a)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["oof", "test"])
    ap.add_argument("--source", choices=["yolo", "refiner"], default="refiner")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--refiner", default=None, help="refiner run (default refiner_v<fold> / refiner_full)")
    ap.add_argument("--unet-name", default="unet")
    ap.add_argument("--pre-thr", type=float, default=0.1, help="proposals below this score are not fused")
    ap.add_argument("--base-area", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default=None, help="test: fusion_config.json from an oof run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    paths = get_paths(args.data_root, args.work_dir)
    refiner_name = args.refiner or (f"refiner_v{args.fold}" if args.split == "oof" else "refiner_full")
    data = load_source(paths, args.source, args.split, args.fold, refiner_name)
    data = dict(list(data.items())[: args.limit])
    obs = list(data)
    map_dir = paths.work_dir / "unet" / "maps" / args.unet_name / ("oof" if args.split == "oof" else "test")
    map_dir = map_dir if map_dir.exists() else None
    if map_dir is None:
        print("[fusion] no U-Net maps: merging cannot check the gap and recovery is off")
    out_dir = paths.work_dir / "fusion" / f"{args.source}_f{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(data, map_dir, args.pre_thr, args.base_area)) as pool:
        if args.split == "test":
            cfg = json.loads(Path(args.config).read_text())
            ranked = run_config(pool, obs, FusionParams(**cfg["params"]))
            preds = {o: [r for r, s in v if s >= cfg["thr"]] for o, v in ranked.items()}
            from fseg import rle as R

            preds = {o: [r for r in v if R.area([r])[0] >= cfg["min_area"]] for o, v in preds.items()}
            out = Path(args.out) if args.out else paths.work_dir / "submission_fused.csv"
            df = write_submission(preds, out)
            print(f"Wrote {out}: {len(df)} rows for {len(preds)} images")
            return

        refs = labels.load_references(paths)
        thrs, areas = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6], [50, 150, 300, 500]
        off = FusionParams(merge_dist=0, recover_thr=0, min_elongation=0)
        rows = []

        def trial(step, params):
            res, t, a = score(run_config(pool, obs, params), refs, obs, thrs, areas)
            rows.append({"step": step, **asdict(params), "thr": t, "min_area": a,
                         **{k: v for k, v in res.items() if k != "per_image"}})
            print(f"{step:8s} pq {res['pq']:.4f} split {res['one_to_many_rate']:.3f} merge "
                  f"{res['many_to_one_rate']:.3f} | {params}", flush=True)
            return res["pq"], params, res, t, a

        best = trial("baseline", off)
        for d in [40, 80, 120]:
            for ang in [30, 45]:
                for pr in [0.2, 0.4]:
                    best = max(best, trial("merge", replace(best[1], merge_dist=d, merge_angle=ang, merge_prob=pr)),
                               key=lambda x: x[0])
        if map_dir is not None:
            for thr in [0.6, 0.8]:
                for ln in [100, 200]:
                    for fac in [0.4, 0.7]:
                        best = max(best, trial("recover", replace(best[1], recover_thr=thr, recover_len=ln,
                                                                  recover_factor=fac)), key=lambda x: x[0])
        for e in [2.0, 3.0, 4.0]:
            best = max(best, trial("shape", replace(best[1], min_elongation=e)), key=lambda x: x[0])

        pd.DataFrame(rows).sort_values("pq", ascending=False).to_csv(out_dir / f"fusion_sweep_fold{args.fold}.csv",
                                                                    index=False)
        pq, params, res, t, a = best
        cfg = {"params": asdict(params), "thr": t, "min_area": a, "pre_thr": args.pre_thr, "source": args.source}
        (out_dir / "fusion_config.json").write_text(json.dumps(cfg, indent=2))
        res["per_image"].to_csv(out_dir / f"oof_per_image_fold{args.fold}.csv", index=False)
        print(f"Baseline ({args.source}, no fusion): {summary(rows[0])}")
        print(f"Fused, fold {args.fold}: {summary(res)}\nconfig {cfg}")


if __name__ == "__main__":
    main()
