"""Pipeline diagram for the report (vector PDF, full text width, no page margins).

    python scripts/make_pipeline_figure.py --image <preprocessed train or test jpg> --out pipeline_figure.pdf
"""
import argparse

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK = "#1f2a37"
COLORS = {  # (fill, edge)
    "in": ("#eef1f5", "#8a96a8"),
    "s1": ("#e3ecfa", "#2f5fb3"),
    "s2": ("#e2f3ea", "#2e8b57"),
    "s3": ("#efe7f8", "#6a3fa0"),
    "s4": ("#e1f1f6", "#1f7a99"),
    "m": ("#fdeee2", "#d2691e"),
    "out": ("#e9f5e9", "#3a8a3a"),
}


def box(ax, x, y, w, h, key, title, lines, title_size=7.2, size=6.1, band=3.6):
    fill, edge = COLORS[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fill, ec=edge, lw=1.0))
    ax.add_patch(FancyBboxPatch((x, y + h - band), w, band, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=edge, ec=edge, lw=1.0))
    ax.text(x + w / 2, y + h - band / 2, title, ha="center", va="center", color="white",
            fontsize=title_size, fontweight="bold")
    ax.text(x + 0.9, y + h - band - 0.9, "\n".join(lines), ha="left", va="top", color=INK, fontsize=size,
            linespacing=1.25)


def arrow(ax, p, q, label=None, label_xy=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.9, color=INK,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text(*label_xy, label, fontsize=5.6, color=INK, ha="center", va="center", style="italic")


def column_figure(args):
    """One-column version (3.33 in wide), flowing top to bottom."""
    fig = plt.figure(figsize=(3.33, 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 45)
    ax.axis("off")
    ts, fs = 6.0, 5.2

    fill, edge = COLORS["in"]
    ax.add_patch(FancyBboxPatch((0.3, 38.6), 49.4, 6.2, boxstyle="round,pad=0,rounding_size=1", fc=fill, ec=edge,
                                lw=0.9))
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)[..., 1]
    ys, xs = (img > 20).nonzero()
    cy, cx, r = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2, (ys.max() - ys.min()) // 2
    img = cv2.resize(img[cy - r:cy + r, cx - r:cx + r], (200, 200), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[:200, :200]
    rgba = np.dstack([img, img, img, np.full_like(img, 255)])
    rgba[(yy - 100) ** 2 + (xx - 100) ** 2 > 94 ** 2, 3] = 0
    ax.imshow(rgba, extent=(1.5, 7.0, 39.1, 44.3), zorder=3)
    ax.text(9, 43.1, "2048$^2$ H$\\alpha$ image", fontsize=ts, fontweight="bold", color=INK, va="center")
    ax.text(9, 40.4, "disk fit, limb flattening, CLAHE $\\rightarrow$ 3 channels", fontsize=fs, color=INK,
            va="center")

    def b(x, y, w, h, key, title, lines):
        box(ax, x, y, w, h, key, title, lines, title_size=ts, size=fs, band=3.0)

    # the box helper uses a 3.6-unit title band; shrink it for this smaller canvas
    b(0.3, 26.6, 15.8, 9.4, "s1", "1: YOLO11l-seg", ["1280 px, 4 flips", "6-model vote", "conf $\\geq$ 0.05"])
    b(17.1, 26.6, 15.8, 9.4, "s2", "2: ConvNeXt U-Net", ["ASPP, 1024 px", "filament, center-", "line, boundary"])
    b(33.9, 26.6, 15.8, 9.4, "s4", "4a: U-Net-2048", ["native res.,", "1024 tiles; comp.", "p > .3/.5/.7"])
    b(0.3, 13.6, 32.6, 10.4, "s3", "3: refiner (ResNet-34 U-Net)",
      ["256 px crop, 5 ch: image, CLAHE,", "proposal, filament + centerline maps", "out: mask + predicted IoU",
       "6 refiners, D4 views"])
    b(33.9, 13.6, 15.8, 10.4, "s3", "refiner", ["same recipe on", "U-Net-2048 maps", "6 refiners, D4"])
    b(0.3, 4.1, 49.4, 7.0, "m", "4b: merge", [
        "score = quality $\\times$ conf (U-Net-2048 $\\times$ 0.25); greedy overlap",
        "resolution; keep score $\\geq$ 0.1, area $\\geq$ 150 px"])
    fill, edge = COLORS["out"]
    ax.add_patch(FancyBboxPatch((12, 0.2), 26, 3, boxstyle="round,pad=0,rounding_size=1", fc=fill, ec=edge,
                                lw=0.9))
    ax.text(25, 1.7, "disjoint instances (COCO RLE)", ha="center", va="center", fontsize=fs, fontweight="bold",
            color=INK)

    for x in (8.2, 25, 41.8):
        arrow(ax, (x, 38.6), (x, 36))
    arrow(ax, (8.2, 26.6), (8.2, 24))
    arrow(ax, (25, 26.6), (25, 24))
    ax.text(26.2, 25.3, "maps", fontsize=5.0, style="italic", color=INK, va="center")
    arrow(ax, (41.8, 26.6), (41.8, 24))
    arrow(ax, (16.6, 13.6), (16.6, 11.1))
    arrow(ax, (41.8, 13.6), (41.8, 11.1))
    arrow(ax, (25, 4.1), (25, 3.2))
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out.rsplit(".", 1)[0] + ".png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="pipeline_figure.pdf")
    ap.add_argument("--column", action="store_true", help="one-column layout, top to bottom")
    args = ap.parse_args()
    if args.column:
        return column_figure(args)

    fig = plt.figure(figsize=(7.16, 2.35))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 34)
    ax.axis("off")

    # input with a real (preprocessed, limb-flattened) solar disk
    fill, edge = COLORS["in"]
    ax.add_patch(FancyBboxPatch((0.5, 3), 13.5, 28, boxstyle="round,pad=0,rounding_size=1.2", fc=fill, ec=edge,
                                lw=1.0))
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)[..., 1]
    ys, xs = (img > 20).nonzero()  # sky is zero after preprocessing: crop to the disk
    cy, cx, r = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2, (ys.max() - ys.min()) // 2
    img = cv2.resize(img[cy - r:cy + r, cx - r:cx + r], (300, 300), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[:300, :300]
    rgba = np.dstack([img, img, img, np.full_like(img, 255)])
    rgba[(yy - 150) ** 2 + (xx - 150) ** 2 > 141 ** 2, 3] = 0  # drop the limb ring, transparent outside
    ax.imshow(rgba, extent=(2.2, 12.3, 17.5, 27.6), zorder=3)
    ax.text(7.25, 15.2, "2048$^2$ H$\\alpha$ image", ha="center", va="top", fontsize=6.6, fontweight="bold",
            color=INK)
    ax.text(7.25, 12.6, "disk fit, limb\nflattening, CLAHE\n$\\rightarrow$ 3 channels", ha="center", va="top",
            fontsize=5.9, color=INK, linespacing=1.25)

    box(ax, 18, 23, 22.5, 10.5, "s1", "Stage 1: YOLO11l-seg",
        ["1280 px, 4 flip views", "6-model vote (5 folds + full)", "proposals, conf $\\geq$ 0.05"])
    box(ax, 18, 12, 22.5, 10, "s2", "Stage 2: ConvNeXt U-Net",
        ["ASPP, whole disk at 1024 px", "maps: filament, centerline,", "boundary"])
    box(ax, 18, 0.5, 22.5, 10.5, "s4", "Stage 4a: U-Net-2048",
        ["native resolution, 1024 tiles", "components at p > 0.3/0.5/0.7", "(second proposal source)"])
    box(ax, 47, 20, 24, 13.5, "s3", "Stage 3: refiner",
        ["ResNet-34 U-Net, 256 px crop", "5 ch: image, CLAHE, proposal,", "U-Net filament + centerline",
         "out: mask + predicted IoU", "6 refiners, D4 views"])
    box(ax, 47, 0.5, 24, 10.5, "s3", "Refiner (same recipe)",
        ["guided by U-Net-2048 maps", "out: mask + predicted IoU", "6 refiners, D4 views"])
    box(ax, 76, 8, 18, 19, "m", "Stage 4b: merge",
        ["score = quality $\\times$ conf", "U-Net-2048 score", "$\\quad\\times\\,w$ = 0.25", "greedy overlap",
         "resolution", "keep score $\\geq$ 0.1,", "area $\\geq$ 150 px"])
    fill, edge = COLORS["out"]
    ax.add_patch(FancyBboxPatch((96.5, 11.5), 8.3, 11, boxstyle="round,pad=0,rounding_size=1.2", fc=fill, ec=edge,
                                lw=1.0))
    ax.text(100.65, 17, "disjoint\ninstances\n(COCO\nRLE)", ha="center", va="center", fontsize=6.0, color=INK,
            fontweight="bold", linespacing=1.2)

    # arrows: input to the three models
    ax.plot([14, 15.8, 15.8], [17, 17, 5.75], color=INK, lw=0.9)
    ax.plot([15.8, 15.8], [17, 28.25], color=INK, lw=0.9)
    arrow(ax, (15.8, 28.25), (18, 28.25))
    arrow(ax, (14, 17), (18, 17))
    arrow(ax, (15.8, 5.75), (18, 5.75))
    # proposals and maps into the refiners
    arrow(ax, (40.5, 28.25), (47, 28.25), "proposals", (43.75, 29.6))
    ax.plot([40.5, 43.75, 43.75], [17, 17, 23], color=INK, lw=0.9)
    arrow(ax, (43.75, 23), (47, 23))
    ax.text(45.6, 18.6, "maps", fontsize=5.6, color=INK, ha="center", va="center", style="italic")
    arrow(ax, (40.5, 5.75), (47, 5.75), "proposals\n+ maps", (43.75, 8.4))
    # refined sets into the merge, then out
    ax.plot([71, 73.5, 73.5], [26.75, 26.75, 21], color=INK, lw=0.9)
    arrow(ax, (73.5, 21), (76, 21))
    ax.plot([71, 73.5, 73.5], [5.75, 5.75, 13], color=INK, lw=0.9)
    arrow(ax, (73.5, 13), (76, 13))
    arrow(ax, (94, 17), (96.5, 17))

    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out.rsplit(".", 1)[0] + ".png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
