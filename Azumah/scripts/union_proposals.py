"""Stage 3b: refine U-Net proposals alongside YOLO proposals, then merge both for PQ.

The refiner only ever saw YOLO proposals, so filaments YOLO misses stay missed (recall ~0.62). Here the
full-disk U-Net map is also cut into instances (connected components at several thresholds, score = mean
probability), each is refined by the same refiner, and the two refined sets are merged by greedy overlap
resolution (a duplicate loses when more than half of it is already covered). U-Net scores are scaled by a
source weight w; w = 0 reproduces the YOLO-only result.

    python scripts/union_proposals.py refine oof  --fold k --refiner resnet60_v<k> --tag uprop_f<k>
    python scripts/union_proposals.py refine test --refiner resnet60_v0 ... resnet60_full --tag uprop_test
    python scripts/union_proposals.py sweep --yolo-tag r60 --unet-tag uprop        # pooled 707-image OOF
    python scripts/union_proposals.py submit --yolo-test refiner/test_r60x6/refined_test.pkl \\
        --unet-test refiner/uprop_test/refined_unet_test.pkl --config refiner/union_pooled/pq_config.json
"""
import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fseg import labels  # noqa: E402
from fseg.config import IMAGE_SIZE, get_paths  # noqa: E402
from fseg.metrics import evaluate  # noqa: E402
from fseg.postprocess import by_conf, resolve_ranked  # noqa: E402
from fseg.preprocess import Disk, disk_mask  # noqa: E402
from fseg.refiner import load_refiner  # noqa: E402
from fseg.unet import pick_device, prob_to_instances  # noqa: E402
from fseg.yolo_infer import write_submission  # noqa: E402
from predict_refiner import refine_all  # noqa: E402

WEIGHTS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
THRS = [0.05, 0.1, 0.15, 0.2, 0.3]
AREAS = [50, 150, 300]


def unet_proposals(map_dir, obs_disks, thrs, min_area):
    """{obs: {"instances": [(rle, mean prob)], "disk": disk}} from the filament channel of the U-Net map."""
    out = {}
    for obs, d in obs_disks.items():
        m = cv2.imread(str(map_dir / f"{obs}.png"), cv2.IMREAD_COLOR)
        prob = m[..., 0].astype(np.float32) / 255.0  # written and read back as (filament, centerline, boundary)
        if prob.shape[0] != IMAGE_SIZE:
            prob = cv2.resize(prob, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        disk = disk_mask((IMAGE_SIZE, IMAGE_SIZE), Disk(*d), margin=1.0)
        inst = [p for t in thrs for p in prob_to_instances(prob, t, min_area, disk)]
        out[obs] = {"instances": inst, "disk": d}
    return out


def merged(yolo, unet, w, min_area):
    """Greedy overlap resolution over refined YOLO and U-Net instances; score = quality x conf (x w for U-Net)."""
    res = {}
    for obs, item in yolo.items():
        disk = disk_mask((IMAGE_SIZE, IMAGE_SIZE), Disk(*item["disk"]), margin=1.01)
        scored = [(r, q * c) for r, q, c in item["instances"]]
        if w > 0 and obs in unet:
            scored += [(r, w * q * c) for r, q, c in unet[obs]["instances"]]
        res[obs] = resolve_ranked(scored, min_area=min_area, disk=disk)
    return res


def refine_cmd(paths, args):
    device = pick_device(args.device)
    models = [load_refiner(paths.work_dir / "refiner" / n / "best.pt", device)[0] for n in args.refiner]
    model = models[0] if len(models) == 1 else models
    sub = "oof" if args.split == "oof" else "test"
    map_dir = paths.work_dir / "unet" / "maps" / args.unet_name / sub
    if args.split == "oof":
        raw_path = paths.work_dir / "runs" / f"consensus_f{args.fold}" / f"oof_raw_fold{args.fold}.pkl"
    else:
        raw_path = Path(args.raw) if args.raw else paths.work_dir / "runs" / "ensemble" / "test_raw.pkl"
    with open(raw_path, "rb") as f:
        disks = {o: v["disk"] for o, v in pickle.load(f).items()}
    disks = dict(list(disks.items())[: args.limit])
    props = unet_proposals(map_dir, disks, args.thrs, args.min_area)
    n = np.mean([len(v["instances"]) for v in props.values()])
    print(f"[union] {len(props)} images, {n:.1f} U-Net proposals per image (thresholds {args.thrs})")
    refined = refine_all(model, paths, "train" if args.split == "oof" else "test", props, map_dir, device,
                         0.0, 0.5, args.tta)
    run_dir = paths.work_dir / "refiner" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    name = f"refined_unet_fold{args.fold}.pkl" if args.split == "oof" else "refined_unet_test.pkl"
    with open(run_dir / name, "wb") as f:
        pickle.dump(refined, f)
    print(f"[union] wrote {run_dir / name}")


_S = {}


def _job(wa):
    w, a = wa
    rk = merged(_S["yolo"], _S["unet"], w, a)
    rows = []
    for t in THRS:
        res = evaluate({o: by_conf(v, t) for o, v in rk.items()}, _S["refs"], _S["obs"])
        rows.append({"w": w, "min_area": a, "thr": t, **{k: v for k, v in res.items() if k != "per_image"}})
    return rows


def sweep_cmd(paths, args):
    yolo, unet, fold_of = {}, {}, {}
    for k in args.folds:
        with open(paths.work_dir / "refiner" / f"{args.yolo_tag}_f{k}" / f"refined_fold{k}.pkl", "rb") as f:
            part = pickle.load(f)
        yolo.update(part)
        fold_of.update({o: k for o in part})
        with open(paths.work_dir / "refiner" / f"{args.unet_tag}_f{k}" / f"refined_unet_fold{k}.pkl", "rb") as f:
            unet.update(pickle.load(f))
    _S.update(yolo=yolo, unet=unet, refs=labels.load_references(paths), obs=list(yolo))
    combos = [(w, a) for w in WEIGHTS for a in AREAS]
    with Pool(args.workers) as p:
        rows = [r for part in p.map(_job, combos) for r in part]
    table = pd.DataFrame(rows).sort_values("pq", ascending=False).reset_index(drop=True)
    out = paths.work_dir / "refiner" / "union_pooled"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "pq_sweep_pooled.csv", index=False)
    b = table.iloc[0]
    cfg = {"w": float(b.w), "min_area": int(b.min_area), "thr": float(b.thr)}
    (out / "pq_config.json").write_text(json.dumps(cfg, indent=2))
    cols = ["w", "min_area", "thr", "pq", "sq", "rq", "precision", "recall"]
    print(table.head(8)[cols].round(4).to_string())
    print("best YOLO-only (w = 0):")
    print(table[table.w == 0].head(1)[cols].round(4).to_string())
    preds = {o: by_conf(v, cfg["thr"]) for o, v in merged(yolo, unet, cfg["w"], cfg["min_area"]).items()}
    base = table[table.w == 0].iloc[0]
    base_preds = {o: by_conf(v, base.thr) for o, v in merged(yolo, unet, 0.0, int(base.min_area)).items()}
    rows = []
    for k in list(args.folds) + ["pooled"]:
        ok = _S["obs"] if k == "pooled" else [o for o in _S["obs"] if fold_of[o] == k]
        rows.append({"fold": k, "yolo_only": evaluate(base_preds, _S["refs"], ok)["pq"],
                     "union": evaluate(preds, _S["refs"], ok)["pq"]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("config:", cfg)


def submit_cmd(paths, args):
    cfg = json.loads(Path(args.config).read_text())
    with open(args.yolo_test, "rb") as f:
        yolo = pickle.load(f)
    with open(args.unet_test, "rb") as f:
        unet = pickle.load(f)
    preds = {o: by_conf(v, cfg["thr"]) for o, v in merged(yolo, unet, cfg["w"], cfg["min_area"]).items()}
    df = write_submission(preds, Path(args.out))
    print(f"Wrote {args.out}: {len(df)} rows for {len(preds)} images ({cfg})")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("refine")
    r.add_argument("split", choices=["oof", "test"])
    r.add_argument("--fold", type=int, default=0)
    r.add_argument("--refiner", nargs="+", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--unet-name", default="unet")
    r.add_argument("--thrs", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    r.add_argument("--min-area", type=int, default=150)
    r.add_argument("--tta", default="d4", choices=["hflip", "d4"])
    r.add_argument("--raw", default=None, help="test: proposals pickle that holds the disks")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--device", default=None)
    s = sp.add_parser("sweep")
    s.add_argument("--yolo-tag", default="r60")
    s.add_argument("--unet-tag", default="uprop")
    s.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    s.add_argument("--workers", type=int, default=8)
    u = sp.add_parser("submit")
    u.add_argument("--yolo-test", required=True)
    u.add_argument("--unet-test", required=True)
    u.add_argument("--config", required=True)
    u.add_argument("--out", required=True)
    for p in (r, s, u):
        p.add_argument("--data-root", default=None)
        p.add_argument("--work-dir", default=None)
    args = ap.parse_args()
    paths = get_paths(args.data_root, args.work_dir)
    {"refine": refine_cmd, "sweep": sweep_cmd, "submit": submit_cmd}[args.cmd](paths, args)


if __name__ == "__main__":
    main()
