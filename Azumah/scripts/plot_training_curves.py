"""Validation curves for the slides from an Ultralytics results.csv.

Left: validation mask mAP50 and mAP50-95 per epoch. Right: train vs validation segmentation loss
(overfitting check). Two panels rather than one twin-axis chart, since the scales differ.

    python scripts/plot_training_curves.py work/runs/consensus_f0/results.csv --out figures/yolo_val_curves.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DARK, LIGHT = "#2648a8", "#2f9ad9"  # validated pair (lightness, chroma, CVD, contrast)
INK, MUTED, GRID = "#1a1a2e", "#5b6070", "#e3e5ea"


def style(ax, ylabel):
    ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def label_end(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_csv")
    ap.add_argument("--out", default="yolo_val_curves.png")
    args = ap.parse_args()

    df = pd.read_csv(args.results_csv)
    df.columns = df.columns.str.strip()
    e = df["epoch"]
    m50, m5095 = df["metrics/mAP50(M)"], df["metrics/mAP50-95(M)"]
    best = m5095.idxmax()

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))

    a.plot(e, m50, color=DARK, linewidth=2, marker="o", markersize=3, label="mask mAP50")
    a.plot(e, m5095, color=LIGHT, linewidth=2, marker="o", markersize=3, label="mask mAP50-95")
    a.scatter([e[best]], [m5095[best]], s=60, facecolor="white", edgecolor=LIGHT, linewidth=2, zorder=3)
    a.annotate(f"best epoch {e[best]}", (e[best], m5095[best]), xytext=(0, 12), textcoords="offset points",
               ha="center", fontsize=8, color=INK)
    label_end(a, e.iloc[-1], m50.iloc[-1], f"{m50.iloc[-1]:.2f}")
    label_end(a, e.iloc[-1], m5095.iloc[-1], f"{m5095.iloc[-1]:.2f}")
    a.set_ylim(0, max(0.7, m50.max() + 0.05))
    a.set_xlim(0, e.iloc[-1] + 5)
    a.legend(frameon=False, fontsize=8, loc="center right", labelcolor=INK)
    a.set_title("Validation mask mAP levels off near epoch 60", fontsize=10, color=INK, loc="left")
    style(a, "validation score")

    tr, va = df["train/seg_loss"], df["val/seg_loss"]
    b.plot(e, tr, color=DARK, linewidth=2, label="train seg loss")
    b.plot(e, va, color=LIGHT, linewidth=2, label="validation seg loss")
    # the lines end on top of each other, so the legend names them instead of end labels
    b.set_ylim(0, va.iloc[2:].max() * 1.15)  # epoch-1/2 validation spike would flatten the rest
    b.set_xlim(0, e.iloc[-1] + 2)
    b.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    b.set_title("Segmentation loss: no overfitting gap", fontsize=10, color=INK, loc="left")
    style(b, "loss")

    fig.tight_layout(w_pad=2.5)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"wrote {args.out}; best mAP50-95 {m5095[best]:.3f} at epoch {e[best]}")


if __name__ == "__main__":
    main()
