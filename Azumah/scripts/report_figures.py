"""Report numbers and figures from the pooled 707-image out-of-fold predictions.

Stages (each fold scored by models that never saw it, settings tuned once on all 707 images):
YOLO alone, YOLO + refiner, Charan's U-Net-2048 + refiner, union of both (final). Also the human
agreement ceiling. Writes report/stages.csv, report/match_stats.csv and three figures.

    python scripts/report_figures.py --yolo-tag r60 --unet-tag c2r --out report
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
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fseg import labels  # noqa: E402
from fseg import rle as R  # noqa: E402
from fseg.config import get_paths  # noqa: E402
from fseg.metrics import MATCH_IOU, annotator_ceiling, evaluate  # noqa: E402
from fseg.postprocess import by_conf  # noqa: E402
from fseg.yolo_infer import apply as yolo_apply  # noqa: E402
from predict_refiner import ranked  # noqa: E402
from union_proposals import merged  # noqa: E402

_S = {}


def matched_ious(preds, refs, obs):
    """IoU of every matched (prediction, GT) pair and the best IoU of every GT filament, over all sets."""
    m, best = [], []
    for o in obs:
        for gts in refs[o]:
            p = preds.get(o, [])
            if not gts:
                continue
            if not p:
                best += [0.0] * len(gts)
                continue
            iou = R.iou_matrix(p, gts)
            m += iou[iou > MATCH_IOU].tolist()
            best += iou.max(0).tolist()
    return np.array(m), np.array(best)


def _build(name):
    yolo_ref, unet_ref, base = _S["yolo_ref"], _S["unet_ref"], _S["base_raw"]
    rc, uc, mc = _S["cfgs"]
    if name == "YOLO11l-seg":
        out = {}
        for k, (raw, bc) in base.items():
            out.update(yolo_apply(raw, bc["conf_thr"], bc["min_area"], bc.get("dark")))
        return out
    if name == "YOLO + refiner":
        return {o: by_conf(v, rc["thr"]) for o, v in ranked(yolo_ref, rc["mode"], rc["min_area"]).items()}
    if name == "U-Net-2048 + refiner":
        return {o: by_conf(v, uc["thr"]) for o, v in ranked(unet_ref, uc["mode"], uc["min_area"]).items()}
    return {o: by_conf(v, mc["thr"]) for o, v in merged(yolo_ref, unet_ref, mc["w"], mc["min_area"]).items()}


def _score(name):
    preds = _build(name)
    res = evaluate(preds, _S["refs"], _S["obs"])
    m, best = matched_ious(preds, _S["refs"], _S["obs"])
    print(f"[report] {name}: pq {res['pq']:.4f}", flush=True)
    return name, preds, res, m, best


STAGES = ["YOLO11l-seg", "YOLO + refiner", "U-Net-2048 + refiner", "Union (final)"]


def load_inputs(paths, args):
    yolo_ref, unet_ref, base, fold_of = {}, {}, {}, {}
    for k in args.folds:
        with open(paths.work_dir / "refiner" / f"{args.yolo_tag}_f{k}" / f"refined_fold{k}.pkl", "rb") as f:
            part = pickle.load(f)
        yolo_ref.update(part)
        fold_of.update({o: k for o in part})
        with open(paths.work_dir / "refiner" / f"{args.unet_tag}_f{k}" / f"refined_unet_fold{k}.pkl", "rb") as f:
            unet_ref.update(pickle.load(f))
        run = paths.work_dir / "runs" / f"consensus_f{k}"
        with open(run / f"oof_raw_fold{k}.pkl", "rb") as f:
            raw = pickle.load(f)
        base[k] = ({o: raw[o] for o in part}, json.loads((run / "pq_config.json").read_text()))
    cfgs = tuple(json.loads(Path(p).read_text()) for p in (args.refiner_config, args.unet_config, args.union_config))
    _S.update(yolo_ref=yolo_ref, unet_ref=unet_ref, base_raw=base, cfgs=cfgs)
    return fold_of


def overlay(paths, obs, preds, refs, ax, title):
    img = cv2.imread(str(paths.preprocessed / "train" / f"{obs}.jpg"), cv2.IMREAD_COLOR)[..., 1]
    small = 1024
    rgb = np.repeat(cv2.resize(img, (small, small), interpolation=cv2.INTER_AREA)[..., None], 3, 2)
    rgb = (rgb * 0.85).astype(np.uint8)
    gt = np.zeros((small, small), np.uint8)
    for r in refs[obs][0]:
        gt |= cv2.resize(R.decode(r).astype(np.uint8), (small, small), interpolation=cv2.INTER_NEAREST)
    pr = np.zeros((small, small), np.uint8)
    for r in preds.get(obs, []):
        pr |= cv2.resize(R.decode(r).astype(np.uint8), (small, small), interpolation=cv2.INTER_NEAREST)
    rgb[pr > 0] = (0.45 * rgb[pr > 0] + 0.55 * np.array([230, 80, 40])).astype(np.uint8)
    cnt, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cnt, -1, (40, 220, 90), 2)
    ys, xs = np.nonzero(gt | pr)
    if len(ys):  # square crop around all filaments
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        h = min(max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + 40, small // 2)
        cy, cx = int(np.clip(cy, h, small - h)), int(np.clip(cx, h, small - h))
        rgb = rgb[cy - h:cy + h, cx - h:cx + h]
    ax.imshow(rgb)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-tag", default="r60")
    ap.add_argument("--unet-tag", default="c2r")
    ap.add_argument("--refiner-config", required=True)
    ap.add_argument("--unet-config", required=True)
    ap.add_argument("--union-config", required=True)
    ap.add_argument("--union-sweep", required=True, help="pq_sweep_pooled.csv of union_proposals.py sweep")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="report")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()
    paths = get_paths(args.data_root, args.work_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    refs = labels.load_references(paths)
    fold_of = load_inputs(paths, args)
    obs = list(_S["yolo_ref"])
    _S.update(refs=refs, obs=obs)
    with Pool(args.workers) as p:
        scored = p.map(_score, STAGES, chunksize=1)
    stages = {name: preds for name, preds, *_ in scored}
    with open(out / "oof_stages.pkl", "wb") as f:
        pickle.dump(stages, f)

    rows, dist = [], {}
    human = annotator_ceiling({o: refs[o] for o in obs})
    print("[report] human ceiling done", flush=True)
    rows.append({"stage": "Human agreement", **{k: v for k, v in human.items() if k != "per_image"}})
    for name, _, res, m, best in scored:
        rows.append({"stage": name, **{k: v for k, v in res.items() if k != "per_image"}})
        dist[name] = (m, best, res["per_image"])
    cols = ["stage", "pq", "sq", "rq", "precision", "recall", "pred_per_image", "gt_per_image",
            "one_to_many_rate", "many_to_one_rate"]
    table = pd.DataFrame(rows)[cols]
    table.to_csv(out / "stages.csv", index=False)
    print(table.round(4).to_string(index=False))

    ms = []
    for name, (m, best, _) in dist.items():
        dice = 2 * m / (1 + m)
        ms.append({"stage": name, "matched": len(m), "iou_median": np.median(m), "iou_q25": np.quantile(m, .25),
                   "iou_q75": np.quantile(m, .75), "dice_median": np.median(dice),
                   "gt_best_iou_median": np.median(best), "gt_missed_share": float((best <= MATCH_IOU).mean())})
    ms = pd.DataFrame(ms)
    ms.to_csv(out / "match_stats.csv", index=False)
    print(ms.round(4).to_string(index=False))

    per_fold = []
    for k in args.folds:
        ok = [o for o in obs if fold_of[o] == k]
        per_fold.append({"fold": k, "n": len(ok),
                         **{n: evaluate(stages[n], refs, ok)["pq"] for n in stages}})
    pf = pd.DataFrame(per_fold)
    pf.to_csv(out / "per_fold.csv", index=False)
    print(pf.round(4).to_string(index=False))

    # Figure 1: distributions of matched IoU / Dice and best IoU per GT filament
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1))
    bins = np.linspace(0, 1, 41)
    for name, color in [("YOLO11l-seg", "#9aa5b1"), ("Union (final)", "#d1495b")]:
        m, best, _ = dist[name]
        axs[0].hist(m, bins=bins, histtype="step", density=True, color=color, lw=1.4, label=name)
        axs[1].hist(2 * m / (1 + m), bins=bins, histtype="step", density=True, color=color, lw=1.4)
        axs[2].hist(best, bins=bins, histtype="step", density=True, color=color, lw=1.4)
    for ax, t in zip(axs, ["IoU of matched pairs", "Dice of matched pairs", "Best IoU per GT filament"]):
        ax.set_title(t, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
    axs[2].axvline(MATCH_IOU, color="k", lw=0.6, ls=":")
    axs[0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig_distributions.pdf")
    fig.savefig(out / "fig_distributions.png", dpi=200)

    # Figure 2: pooled PQ against score threshold for each U-Net weight
    sw = pd.read_csv(args.union_sweep)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for w, g in sw.groupby("w"):
        best = g.groupby("thr").pq.max()
        ax.plot(best.index, best.values, marker="o", ms=3, lw=1.2, label=f"w = {w:g}")
    ax.set_xlabel("score threshold", fontsize=8)
    ax.set_ylabel("pooled OOF PQ", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_threshold.pdf")
    fig.savefig(out / "fig_threshold.png", dpi=200)

    # Figure 3: qualitative overlays (good, typical, failure) by per-image PQ of the final model
    pi = dist["Union (final)"][2].dropna(subset=["pq"]).sort_values("pq").reset_index(drop=True)
    pi = pi[pi.n_gt >= 4].reset_index(drop=True)
    picks = [("good", pi.iloc[int(0.95 * (len(pi) - 1))]), ("typical", pi.iloc[(len(pi) - 1) // 2]),
             ("failure", pi.iloc[int(0.05 * (len(pi) - 1))])]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (kind, row) in zip(axs, picks):
        overlay(paths, row.observation_id, stages["Union (final)"], refs, ax,
                f"{kind}: {row.observation_id}, PQ {row.pq:.2f}")
    fig.tight_layout()
    fig.savefig(out / "fig_qualitative.pdf", dpi=200)
    fig.savefig(out / "fig_qualitative.png", dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
