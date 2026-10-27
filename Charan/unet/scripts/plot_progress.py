"""Plot PQ progress split into SQ x RQ (validation, per epoch) for one or more runs, next to the team's
public-leaderboard history.

    python scripts/plot_progress.py --out figures/pq_progress.png \
        --run "U-Net r34 run 1 (image split)" runs/run1/history.csv 0.410 0.673 0.608 \
        --run "MeshNet 0.1M (7-day split)" runs/mesh/history.csv 0.372 0.660 0.564

Each --run is LABEL HISTORY PQ SQ RQ, where PQ/SQ/RQ are the tuned val scores after the post-processing sweep.
PQ = SQ x RQ: SQ is the mean IoU of matched filaments (boundary quality), RQ is the detection F1 at IoU > 0.5.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PQ, SQ, RQ = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# Public leaderboard (Kaggle, team Epoch-5), in submission order
LEADERBOARD = [
    ("2026-10-26 05:20", "Sheikh\nU-Net baseline", 0.23),
    ("2026-10-26 16:06", "Hassan\nYOLO11s-seg", 0.28),
    ("2026-10-26 17:25", "Richmond\nYOLO11l-seg", 0.35),
    ("2026-10-26 17:46", "Charan\nU-Net r34", 0.36),
    ("2026-10-27 16:12", "Charan\nU-Net 7-day", 0.35),
    ("2026-10-27 17:47", "Charan\nMeshNet", 0.33),
]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def plot_run(a, label, history, final):
    h = pd.read_csv(history).dropna(subset=["val_pq"])
    last = h.epoch.max()
    for col, color, name in [("val_sq", SQ, "SQ (segmentation quality)"), ("val_rq", RQ, "RQ (recognition quality)"),
                             ("val_pq", PQ, "PQ = SQ × RQ")]:
        a.plot(h.epoch, h[col], color=color, lw=2.4 if col == "val_pq" else 2, label=name)
    pad = last * 0.18
    for v, color in zip(final, [PQ, SQ, RQ]):
        a.scatter([last + pad * 0.45], [v], s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        a.text(last + pad * 0.6, v, f"{v:.3f}", color=INK2, va="center")
    a.text(last + pad * 0.55, 0.97, "tuned", ha="center", va="top", color=INK2, fontsize=8)
    a.set_xlim(0, last + pad * 1.4)
    a.set_ylim(0, 1)
    a.set_xlabel("epoch")
    a.set_title(label, fontsize=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=5, action="append", required=True, metavar=("LABEL", "HISTORY", "PQ", "SQ", "RQ"),
                    help="one training run: label, history.csv from train.py, tuned val PQ SQ RQ")
    ap.add_argument("--out", default="figures/pq_progress.png")
    args = ap.parse_args()

    n = len(args.run)
    fig, ax = plt.subplots(1, n + 1, figsize=(5 * n + 6, 5.4), gridspec_kw={"width_ratios": [1] * n + [1.25]})
    for i, (label, history, *final) in enumerate(args.run):
        plot_run(ax[i], label, history, [float(v) for v in final])
        if i:
            ax[i].tick_params(labelleft=False)
    ax[0].set_ylabel("validation score (in-loop thr 0.5 / min area 200)")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=3, bbox_to_anchor=(0.01, 0.995))

    b = ax[n]
    x = range(len(LEADERBOARD))
    scores = [s for _, _, s in LEADERBOARD]
    b.plot(x, scores, color=PQ, lw=2, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2)
    for i, (_, who, s) in enumerate(LEADERBOARD):
        b.text(i, s + 0.02, f"{s:.2f}", ha="center", color=INK2)
    b.set_xticks(list(x), [w for _, w, _ in LEADERBOARD], fontsize=9)
    b.set_ylim(0, 1)
    b.set_ylabel("public leaderboard PQ")
    b.set_title("Team public leaderboard (PQ only)")
    b.grid(axis="x", visible=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
