"""Before/after figure for the refiner: YOLO proposal, refined mask and annotation on OOF crops.

Rows: two proposals the refiner turns from a miss (IoU < 0.5) into a match, and one confident
false positive that the quality head scores near zero. One-column width for the report.

    FSEG_WORK_DIR=... python scripts/make_refiner_strip.py --out report/.../fig_refiner_col.pdf
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fseg import labels  # noqa: E402
from fseg import rle as R  # noqa: E402
from fseg.config import get_paths  # noqa: E402

RED, GREEN = np.array([230, 80, 40]), (40, 220, 90)


def pairs(paths, folds, tag, min_conf):
    """(obs, proposal rle, conf, refined rle, predicted quality) for every refined OOF proposal."""
    for k in folds:
        with open(paths.work_dir / "refiner" / f"{tag}_f{k}" / f"refined_fold{k}.pkl", "rb") as f:
            ref = pickle.load(f)
        with open(paths.work_dir / "runs" / f"consensus_f{k}" / f"oof_raw_fold{k}.pkl", "rb") as f:
            raw = pickle.load(f)
        for obs, item in ref.items():
            props = [(r, s) for r, s in raw[obs]["instances"] if s >= min_conf]
            i = 0
            for r, q, c in item["instances"]:  # refined list keeps proposal order, empty outputs dropped
                while i < len(props) and abs(props[i][1] - c) > 1e-6:
                    i += 1
                if i == len(props):
                    break
                yield obs, props[i][0], c, r, q
                i += 1


def spread(p, r, g):
    """Box area of the proposal relative to the box of refined mask and annotation (1 = compact)."""
    def box(rles):
        m = np.zeros_like(R.decode(rles[0]), dtype=bool)
        for x in rles:
            m |= R.decode(x).astype(bool)
        ys, xs = np.nonzero(m)
        return (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    return box([p, r, g]) / box([r, g])


def candidates(paths, refs, args):
    rows = []
    for obs, p, c, r, q in pairs(paths, args.folds, args.tag, args.min_conf):
        gts = [g for s in refs[obs] for g in s]
        if not gts:
            continue
        ip, ir = R.iou_matrix([p], gts)[0], R.iou_matrix([r], gts)[0]
        j = int(ir.argmax())
        rows.append(dict(obs=obs, p=p, r=r, c=c, q=q, gt=gts[j], iou_p=float(ip[j]), iou_r=float(ir[j]),
                         best_p=float(ip.max()), area=int(R.area([r])[0])))
    return rows


def pick(rows):
    fixes = [x for x in rows if x["iou_p"] < 0.5 and x["iou_r"] > 0.7 and abs(x["q"] - x["iou_r"]) < 0.1
             and x["area"] > 2000]
    fixes = [x for x in fixes if spread(x["p"], x["r"], x["gt"]) < 2.0]
    fixes.sort(key=lambda x: x["iou_r"] - x["iou_p"], reverse=True)
    chosen, seen = [], set()
    for x in fixes:
        if x["obs"] not in seen:
            chosen.append(x)
            seen.add(x["obs"])
        if len(chosen) == 2:
            break
    fps = [x for x in rows if x["best_p"] < 0.05 and x["c"] >= 0.5 and x["q"] < 0.15 and x["area"] > 800
           and x["obs"] not in seen]
    fps.sort(key=lambda x: x["c"] - x["q"], reverse=True)
    return chosen + fps[:1]


def crop(paths, x, size=160):
    img = cv2.imread(str(paths.preprocessed / "train" / f"{x['obs']}.jpg"), cv2.IMREAD_COLOR)[..., 1]
    masks = [R.decode(x[k]).astype(np.uint8) for k in ("p", "r", "gt")]
    ys, xs = np.nonzero(masks[0] | masks[1] | (masks[2] if x["best_p"] >= 0.05 else 0))
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    h = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.75) + 12
    h = min(h, img.shape[0] // 2)
    cy, cx = int(np.clip(cy, h, img.shape[0] - h)), int(np.clip(cx, h, img.shape[1] - h))
    sl = (slice(cy - h, cy + h), slice(cx - h, cx + h))
    g = cv2.resize(img[sl], (size, size), interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
    ms = [cv2.resize(m[sl], (size, size), interpolation=cv2.INTER_NEAREST) for m in masks]
    return g, ms


def panel(ax, g, m, outline=None, color=RED):
    rgb = np.repeat(g[..., None], 3, 2).astype(np.float32)
    if m.any():
        rgb[m > 0] = 0.45 * rgb[m > 0] + 0.55 * np.asarray(color)
    rgb = rgb.astype(np.uint8)
    if outline is not None and outline.any():
        cnt, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cnt, -1, GREEN, 1)
    ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r60")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--min-conf", type=float, default=0.05)
    ap.add_argument("--out", default="fig_refiner_col.pdf")
    args = ap.parse_args()
    paths = get_paths()
    refs = labels.load_references(paths)
    chosen = pick(candidates(paths, refs, args))
    plt.rcParams.update({"font.size": 6.5})
    fig, axs = plt.subplots(len(chosen), 3, figsize=(3.33, 1.12 * len(chosen) + 0.15))
    for row, x in zip(np.atleast_2d(axs), chosen):
        g, (mp, mr, mg) = crop(paths, x)
        fp = x["best_p"] < 0.05
        panel(row[0], g, mp, None if fp else mg)
        panel(row[1], g, mr, None if fp else mg)
        panel(row[2], g, mg if not fp else 0 * mg, None, color=np.array(GREEN))
        row[0].set_xlabel(f"conf {x['c']:.2f}" + ("" if fp else f", IoU {x['iou_p']:.2f}"), labelpad=1.5)
        row[1].set_xlabel(f"pred. IoU {x['q']:.2f}" + ("" if fp else f", IoU {x['iou_r']:.2f}"), labelpad=1.5)
        row[2].set_xlabel("none: dropped" if fp else "", labelpad=1.5)
        print(x["obs"], {k: round(x[k], 3) for k in ("c", "q", "iou_p", "iou_r")})
    for ax, t in zip(np.atleast_2d(axs)[0], ["YOLO proposal", "refined", "annotation"]):
        ax.set_title(t, fontsize=7.5, pad=2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.04, wspace=0.04, hspace=0.32)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out.rsplit(".", 1)[0] + ".png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
