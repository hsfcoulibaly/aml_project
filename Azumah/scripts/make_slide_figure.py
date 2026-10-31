"""Qualitative figure for the slides: solar image, ground truth, YOLO instances, overlay.

Uses out-of-fold images only (the fold the weights never trained on) and picks rows spread across
the per-image PQ range, so the figure shows typical results rather than the best ones.

    python scripts/make_slide_figure.py --weights work/runs/consensus_f0/weights/best.pt \
        --per-image work/runs/consensus_f0/oof_per_image_fold0.csv --out yolo_predictions.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fseg import labels  # noqa: E402
from fseg import rle as R  # noqa: E402
from fseg.config import get_paths  # noqa: E402
from fseg.yolo_infer import apply, predict_many  # noqa: E402

SHOW = 512  # display resolution per panel


def shrink(mask: np.ndarray) -> np.ndarray:
    """Downsample with max-pooling so thin filaments survive at slide size."""
    k = mask.shape[0] // SHOW
    return mask[: SHOW * k, : SHOW * k].reshape(SHOW, k, SHOW, k).max(axis=(1, 3))


def pick_rows(per_image: pd.DataFrame, n: int) -> list[str]:
    df = per_image.dropna(subset=["pq"])
    df = df[df.n_gt >= 2].sort_values("pq").reset_index(drop=True)
    idx = np.linspace(0, len(df) - 1, n + 2)[1:-1].round().astype(int)  # skip the extremes
    return df.loc[idx[::-1], "observation_id"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--per-image", required=True, help="oof_per_image_fold*.csv from predict_yolo.py oof")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-area", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default="yolo_predictions.png")
    args = ap.parse_args()

    import fseg.dilated  # noqa: F401  (needed to unpickle weights trained with dilated context)
    from ultralytics import YOLO

    paths = get_paths()
    per_image = pd.read_csv(args.per_image)
    pq = dict(zip(per_image.observation_id, per_image.pq))
    obs = pick_rows(per_image, args.rows)

    raw = predict_many(YOLO(args.weights), paths.preprocessed / "train", obs, args.imgsz,
                       ["id", "hflip", "vflip", "rot180"])
    preds = apply(raw, args.conf, args.min_area)
    targets = labels.load_targets(paths)["consensus"]

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(obs), 4, figsize=(8, 2.05 * len(obs)), squeeze=False)
    titles = ["Solar image", "Ground truth (consensus)", None, "Overlay: green=GT, red=YOLO, yellow=both"]
    for row, o in enumerate(obs):
        img = cv2.imread(str(paths.preprocessed / "train" / f"{o}.jpg"))[:, :, 0]
        img = cv2.resize(img, (SHOW, SHOW), interpolation=cv2.INTER_AREA)
        gt = shrink(np.any([R.decode(t["rle"]) for t in targets[o]], axis=0)) if targets[o] else np.zeros((SHOW, SHOW), bool)

        inst = np.zeros((SHOW, SHOW, 3))
        pred = np.zeros((SHOW, SHOW), bool)
        for r in preds[o]:
            m = shrink(R.decode(r)).astype(bool)
            inst[m] = rng.uniform(0.35, 1.0, 3)  # one colour per filament instance
            pred |= m

        overlay = np.dstack([img / 255.0] * 3) * 0.6
        overlay[gt & ~pred] = (0, 1, 0)
        overlay[pred & ~gt] = (1, 0, 0)
        overlay[pred & gt] = (1, 1, 0)

        panels = [(img, "gray"), (gt, "gray"), (inst, None), (overlay, None)]
        for col, (data, cmap) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(data, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            title = titles[col] or f"YOLO instances (PQ={pq[o]:.3f}, n={len(preds[o])})"
            ax.set_title(title, fontsize=6.5, pad=2)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.2)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}: {obs}")


if __name__ == "__main__":
    main()
