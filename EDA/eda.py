"""Exploratory data analysis of the MAGFiLO 1.0 Kaggle 2026 data. Writes figures/ and stats.json next to this file.

    python EDA/eda.py --data /path/to/MAGFiLO_1.0_Kaggle_2026

Needs numpy, pandas, matplotlib, opencv-python-headless.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
ANN_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
FULL = 2048
SMALL = 512  # resolution for intensity and disk statistics

# Colors: reference categorical palette, slot 1 = train, slot 2 = test; recessive greys for structure
TRAIN, TEST, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 10, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 110,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def obs_date(file_name: str) -> date:
    return date(int(file_name[:4]), int(file_name[4:6]), int(file_name[6:8]))


def poly(seg) -> np.ndarray:
    return np.asarray(seg[0], dtype=np.float32).reshape(-1, 2)


def spine_length(spine) -> float:
    p = np.asarray(spine, dtype=np.float64).reshape(-1, 2)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def label_mask(polys, size) -> np.ndarray:
    """Instance-label raster (0 = background, i = i-th polygon) at `size`; later polygons win overlaps."""
    m = np.zeros((size, size), np.int32)
    s = size / FULL
    for i, p in enumerate(polys, 1):
        cv2.fillPoly(m, [np.round(p * s).astype(np.int32)], i)
    return m


def disk_stats(path: str) -> dict:
    """Disk centre/radius (Otsu + largest component at SMALL px) and intensity statistics."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (SMALL, SMALL), interpolation=cv2.INTER_AREA)
    _, th = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, cen = cv2.connectedComponentsWithStats(th)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    disk = lab == k
    cx, cy = cen[k]
    r_otsu = float(np.sqrt(disk.sum() / np.pi))
    # Some images have a grey ring outside the limb that Otsu merges into the disk, so the limb itself is
    # found with a Hough circle fit (radius 800-980 px at 2048); Otsu is kept only to flag those images
    circ = cv2.HoughCircles(cv2.GaussianBlur(small, (5, 5), 1.5), cv2.HOUGH_GRADIENT, dp=1, minDist=SMALL,
                            param1=60, param2=30, minRadius=200, maxRadius=245)
    if circ is not None:
        cx, cy, r = (float(v) for v in circ[0, 0])
    else:
        r = r_otsu
    yy, xx = np.mgrid[:SMALL, :SMALL]
    rr = np.hypot(xx - cx, yy - cy) / r
    on = rr < 0.98
    bins = np.linspace(0, 1, 21)
    prof = [float(small[(rr >= a) & (rr < b)].mean()) for a, b in zip(bins[:-1], bins[1:])]
    hist = np.bincount(small[on].ravel(), minlength=256).astype(np.int64)
    s = FULL / SMALL
    return {"file_name": Path(path).name, "cx": cx * s, "cy": cy * s, "r": r * s, "r_otsu": r_otsu * s,
            "hough_found": circ is not None, "ring": bool(r_otsu > 1.05 * r),
            "annulus_mean": float(small[(rr > 1.02) & (rr < 1.10)].mean()),
            "mean_on_disk": float(small[on].mean()), "std_on_disk": float(small[on].std()),
            "mean_off_disk": float(small[rr > 1.05].mean()), "profile": prof, "hist": hist.tolist()}


def record_pair_stats(args):
    """Pairwise agreement between two annotation records of one image at 1024 px."""
    polys_a, polys_b = args
    size = 1024
    la, lb = label_mask(polys_a, size), label_mask(polys_b, size)
    fa, fb = la > 0, lb > 0
    union = (fa | fb).sum()
    pix_iou = float((fa & fb).sum() / union) if union else 1.0
    na, nb = len(polys_a), len(polys_b)
    # Instance IoU matrix from the joint histogram of the two label rasters
    joint = np.bincount((la.ravel() * (nb + 1) + lb.ravel()), minlength=(na + 1) * (nb + 1)).reshape(na + 1, nb + 1)
    inter = joint[1:, 1:].astype(np.float64)
    area_a, area_b = joint[1:, :].sum(1), joint[:, 1:].sum(0)
    iou = inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1)
    matched = int((iou > 0.5).sum())  # IoU > 0.5 matches are unique
    return {"pixel_iou": pix_iou, "n_a": na, "n_b": nb, "matched": matched,
            "match_iou_mean": float(iou[iou > 0.5].mean()) if matched else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="MAGFiLO_1.0_Kaggle_2026 folder (contains train/ and test/)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = Path(args.data)
    FIG.mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    coco = json.loads((root / "train" / ANN_NAME).read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    recs = pd.DataFrame(coco["images"]).rename(columns={"id": "record_id"})
    recs["annotator"] = recs.record_id.str.split("-").str[0]
    anns = pd.DataFrame(coco["annotations"])
    anns["category"] = anns.category_id.map(cats)
    anns = anns.merge(recs[["record_id", "file_name", "annotator"]], left_on="image_id", right_on="record_id")
    anns["poly"] = anns.segmentation.map(poly)
    anns["n_vertices"] = anns.poly.map(len)
    anns["spine_len"] = anns.spine.map(spine_length)
    anns["bbox_w"] = anns.bbox.str[2]
    anns["bbox_h"] = anns.bbox.str[3]
    anns["cx"] = anns.poly.map(lambda p: float(p[:, 0].mean()))
    anns["cy"] = anns.poly.map(lambda p: float(p[:, 1].mean()))
    anns["width"] = anns.area / anns.spine_len.where(anns.spine_len > 0)

    train_files = sorted((root / "train" / "train_images").glob("*.jpeg"))
    test_files = sorted((root / "test" / "test_images").glob("*.jpeg"))
    imgs = pd.DataFrame({"file_name": [p.name for p in train_files + test_files],
                         "set": ["train"] * len(train_files) + ["test"] * len(test_files)})
    imgs["date"] = imgs.file_name.map(obs_date)
    imgs["year"] = imgs.date.map(lambda d: d.year)
    imgs["station"] = imgs.file_name.str[14]
    imgs["hour"] = imgs.file_name.str[8:10].astype(int)

    per_rec = anns.groupby("record_id").size()
    recs["n_filaments"] = recs.record_id.map(per_rec).fillna(0).astype(int)
    recs["year"] = recs.file_name.map(lambda f: obs_date(f).year)
    recs["station"] = recs.file_name.str[14]
    rec_per_img = recs.groupby("file_name").size()

    # ---------------- inventory ----------------
    stats["inventory"] = {
        "train_images": len(train_files), "test_images": len(test_files),
        "annotation_records": len(recs), "filament_polygons": len(anns),
        "annotators": int(recs.annotator.nunique()),
        "image_size": sorted({(int(h), int(w)) for h, w in zip(recs.height, recs.width)}),
        "train_images_without_record": int(len(set(p.name for p in train_files) - set(recs.file_name))),
        "records_with_zero_filaments": int((recs.n_filaments == 0).sum()),
        "polygon_parts": dict(Counter(len(s) for s in anns.segmentation)),
        "iscrowd": dict(Counter(anns.iscrowd.astype(int))),
        "categories": {cats[k]: int(v) for k, v in anns.category_id.value_counts().sort_index().items()},
        "date_range_train": [str(imgs[imgs.set == "train"].date.min()), str(imgs[imgs.set == "train"].date.max())],
        "date_range_test": [str(imgs[imgs.set == "test"].date.min()), str(imgs[imgs.set == "test"].date.max())],
    }
    stats["records_per_image"] = {int(k): int(v) for k, v in rec_per_img.value_counts().sort_index().items()}

    # ---------------- time ----------------
    yr = imgs.groupby(["year", "set"]).size().unstack(fill_value=0)[["train", "test"]]
    stats["images_per_year"] = {int(y): {"train": int(r.train), "test": int(r.test)} for y, r in yr.iterrows()}
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(yr))
    ax.bar(x - 0.2, yr.train, 0.38, color=TRAIN, label="train", edgecolor=SURFACE, linewidth=1)
    ax.bar(x + 0.2, yr.test, 0.38, color=TEST, label="test", edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, yr.index)
    ax.set_ylabel("images")
    ax.set_title("Images per year, train vs test")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save(fig, "01_images_per_year.png")

    fil_year = recs.groupby("year").n_filaments.agg(["median", "mean", "count"])
    fil_year["q25"] = recs.groupby("year").n_filaments.quantile(0.25)
    fil_year["q75"] = recs.groupby("year").n_filaments.quantile(0.75)
    stats["filaments_per_record_by_year"] = {int(y): {k: float(v) for k, v in r.items()} for y, r in fil_year.iterrows()}
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(fil_year.index, fil_year.q25, fil_year.q75, color=TRAIN, alpha=0.18, linewidth=0, label="25th-75th percentile")
    ax.plot(fil_year.index, fil_year["median"], color=TRAIN, lw=2, marker="o", ms=5, label="median")
    ax.set_xticks(fil_year.index)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("filaments per record")
    ax.set_title("Filaments per annotation record, by year (train)")
    ax.legend(loc="upper right")
    save(fig, "02_filaments_per_record_by_year.png")

    days_train = imgs[imgs.set == "train"].groupby("date").size()
    stats["train_images_per_day"] = {int(k): int(v) for k, v in days_train.value_counts().sort_index().items()}
    tr_dates = np.array(sorted(days_train.index), dtype="datetime64[D]")

    def gap(d):
        d = np.datetime64(d, "D")
        i = np.searchsorted(tr_dates, d)
        c = [abs(int((tr_dates[j] - d).astype(int))) for j in (i - 1, i) if 0 <= j < len(tr_dates)]
        return min(c)

    test_gap = imgs[imgs.set == "test"].date.map(gap)
    uniq = sorted(days_train.index)
    consec = np.array([(b - a).days for a, b in zip(uniq[:-1], uniq[1:])])
    stats["test_gap_to_nearest_train_day"] = {int(k): int(v) for k, v in test_gap.value_counts().sort_index().items()}
    stats["train_gap_between_consecutive_days"] = {"median": float(np.median(consec)), "max": int(consec.max()),
                                                   "share_1_day": float((consec == 1).mean())}
    fig, ax = plt.subplots(figsize=(9, 4))
    g = test_gap.clip(upper=15).value_counts().sort_index()
    ax.bar(g.index, g.values, 0.8, color=TEST, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(range(0, 16), [str(i) for i in range(15)] + ["15+"])
    ax.set_xlabel("days from test image to the nearest training-image date")
    ax.set_ylabel("test images")
    ax.set_title("How close each test image is in time to the training set")
    ax.grid(axis="x", visible=False)
    save(fig, "03_test_gap_to_train.png")

    # ---------------- stations ----------------
    st = imgs.groupby(["station", "set"]).size().unstack(fill_value=0)[["train", "test"]]
    stats["images_per_station"] = {s: {"train": int(r.train), "test": int(r.test)} for s, r in st.iterrows()}
    stats["hour_utc_by_station"] = {s: [int(v) for v in sorted(g.hour.unique())] for s, g in imgs.groupby("station")}
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(st))
    ax.bar(x - 0.2, st.train, 0.38, color=TRAIN, label="train", edgecolor=SURFACE, linewidth=1)
    ax.bar(x + 0.2, st.test, 0.38, color=TEST, label="test", edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x, [f"{s}" for s in st.index])
    ax.set_xlabel("GONG station code (15th character of the file name)")
    ax.set_ylabel("images")
    ax.set_title("Images per observing station")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    save(fig, "04_images_per_station.png")

    # ---------------- records and annotators ----------------
    rpi = rec_per_img.value_counts().sort_index()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 2.4]})
    ax[0].bar(rpi.index, rpi.values, 0.7, color=TRAIN, edgecolor=SURFACE)
    for k, v in rpi.items():
        ax[0].text(k, v, f"{v}", ha="center", va="bottom", color=INK2, fontsize=9)
    ax[0].set_xticks(rpi.index)
    ax[0].set_xlabel("annotation records on the image")
    ax[0].set_ylabel("training images")
    ax[0].set_title("Records per image")
    ax[0].grid(axis="x", visible=False)
    by_ann = recs.annotator.value_counts()
    ax[1].bar(range(len(by_ann)), by_ann.values, 0.75, color=TRAIN, edgecolor=SURFACE)
    ax[1].set_xticks(range(len(by_ann)), by_ann.index, rotation=90, fontsize=8)
    ax[1].set_xlabel("annotator id (record-id prefix)")
    ax[1].set_ylabel("records")
    ax[1].set_title(f"Records per annotator ({len(by_ann)} annotators)")
    ax[1].grid(axis="x", visible=False)
    save(fig, "05_records_and_annotators.png")
    stats["records_per_annotator"] = {"min": int(by_ann.min()), "median": float(by_ann.median()), "max": int(by_ann.max())}
    fpa = recs.groupby("annotator").n_filaments.median()
    stats["median_filaments_per_record_by_annotator"] = {"min": float(fpa.min()), "max": float(fpa.max())}

    # ---------------- filaments per record ----------------
    n = recs.n_filaments
    stats["filaments_per_record"] = {"min": int(n.min()), "median": float(n.median()), "mean": float(n.mean()),
                                     "max": int(n.max()), "q25": float(n.quantile(.25)), "q75": float(n.quantile(.75))}
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(n, bins=np.arange(0, n.max() + 2) - 0.5, color=TRAIN, edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(n.median(), color=INK, lw=1, ls="--")
    ax.text(0.97, 0.92, f"median {n.median():.0f}", transform=ax.transAxes, ha="right", color=INK2)
    ax.set_xlabel("filaments in the record")
    ax.set_ylabel("records")
    ax.set_title("Filaments per annotation record")
    ax.grid(axis="x", visible=False)
    save(fig, "06_filaments_per_record.png")

    # ---------------- filament size and shape ----------------
    q = lambda s: {"min": float(s.min()), "q05": float(s.quantile(.05)), "q25": float(s.quantile(.25)),
                   "median": float(s.median()), "q75": float(s.quantile(.75)), "q95": float(s.quantile(.95)),
                   "max": float(s.max())}
    stats["filament_area_px_at_2048"] = q(anns.area)
    stats["spine_length_px"] = q(anns.spine_len)
    stats["width_px_area_over_spine"] = q(anns.width.dropna())
    stats["vertices_per_polygon"] = q(anns.n_vertices)
    stats["share_area_below"] = {str(t): float((anns.area < t).mean()) for t in (100, 200, 400, 800)}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col, lab in [(ax[0], "area", "area (px at 2048)"), (ax[1], "spine_len", "spine length (px)"),
                        (ax[2], "width", "mean width = area / spine length (px)")]:
        v = anns[col].dropna()
        v = v[v > 0]
        a.hist(v, bins=np.logspace(np.log10(v.min()), np.log10(v.max()), 50), color=TRAIN, edgecolor=SURFACE, linewidth=0.6)
        a.set_xscale("log")
        a.axvline(v.median(), color=INK, lw=1, ls="--")
        a.text(0.97, 0.92, f"median {v.median():,.0f}" if v.median() > 20 else f"median {v.median():.1f}",
               transform=a.transAxes, ha="right", color=INK2)
        a.set_xlabel(lab)
        a.set_ylabel("filaments")
        a.grid(axis="x", visible=False)
    ax[0].set_title("Filament area")
    ax[1].set_title("Filament length")
    ax[2].set_title("Filament width")
    save(fig, "07_filament_size.png")

    stats["per_category"] = {c: {"count": int(len(g)), "median_area": float(g.area.median()),
                                 "median_spine_len": float(g.spine_len.median())} for c, g in anns.groupby("category")}
    cc = anns.category.value_counts()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(cc.index[::-1], cc.values[::-1], 0.6, color=TRAIN, edgecolor=SURFACE)
    for i, v in enumerate(cc.values[::-1]):
        ax.text(v + 40, i, f"{v:,} ({v / len(anns):.0%})", va="center", color=INK2)
    ax.set_xlim(0, cc.max() * 1.25)
    ax.set_xlabel("filament polygons")
    ax.set_title("Polygon category labels (chirality)")
    ax.grid(axis="y", visible=False)
    save(fig, "08_categories.png")

    # ---------------- disk geometry and intensity ----------------
    with ProcessPoolExecutor(args.workers) as ex:
        disk = pd.DataFrame(list(ex.map(disk_stats, [str(p) for p in train_files + test_files], chunksize=8)))
    disk = disk.merge(imgs, on="file_name")
    stats["disk"] = {"radius_px": q(disk.r), "center_x_px": q(disk.cx), "center_y_px": q(disk.cy),
                     "radius_by_station_median": {s: float(g.r.median()) for s, g in disk.groupby("station")},
                     "hough_fit_failed": int((~disk.hough_found).sum()),
                     "otsu_radius_px": q(disk.r_otsu),
                     "ring_images": {"total": int(disk.ring.sum()),
                                     "by_set": {s: int(g.ring.sum()) for s, g in disk.groupby("set")},
                                     "by_station": {s: int(g.ring.sum()) for s, g in disk.groupby("station")},
                                     "by_year": {int(y): int(g.ring.sum()) for y, g in disk.groupby("year")}},
                     "annulus_1.02_1.10_mean": q(disk.annulus_mean),
                     "annulus_share_above": {str(t): float((disk.annulus_mean > t).mean()) for t in (5, 20, 40, 60)}}
    ring_eg = [disk.loc[disk.annulus_mean.idxmax()], disk.loc[disk.annulus_mean.idxmin()]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 6))
    for a, row, t in zip(ax, ring_eg, ["brightest ring outside the limb", "darkest sky outside the limb"]):
        p = root / ("train/train_images" if row.set == "train" else "test/test_images") / row.file_name
        a.imshow(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), cmap="gray")
        a.add_patch(plt.Circle((row.cx, row.cy), row.r, fill=False, color=TEST, lw=1.2))
        a.add_patch(plt.Circle((row.cx, row.cy), row.r_otsu, fill=False, color=TRAIN, lw=1.2, ls="--"))
        a.set_title(f"{row.file_name}: {t}", fontsize=10, pad=8)
        a.axis("off")
    ax[1].plot([], [], color=TEST, lw=1.5, label="limb (Hough fit)")
    ax[1].plot([], [], color=TRAIN, lw=1.5, ls="--", label="Otsu foreground radius")
    ax[1].legend(loc="lower right", fontsize=8, labelcolor="white", facecolor="black", framealpha=0.6, frameon=True)
    save(fig, "15_ring_example.png")
    stats["intensity"] = {s: {"mean_on_disk": q(g.mean_on_disk), "std_on_disk_median": float(g.std_on_disk.median()),
                              "mean_off_disk_median": float(g.mean_off_disk.median())} for s, g in disk.groupby("set")}
    stats["intensity_by_station_median"] = {s: float(g.mean_on_disk.median()) for s, g in disk.groupby("station")}

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for s, c, w in [("train", TRAIN, 4), ("test", TEST, 1.5)]:
        h = np.sum(np.stack(disk[disk.set == s]["hist"].values), 0).astype(float)
        ax[0].plot(np.arange(256), h / h.sum(), color=c, lw=w, label=s)
    ax[0].set_xlabel("pixel value on the disk (0-255)")
    ax[0].set_ylabel("share of on-disk pixels")
    ax[0].set_title("On-disk intensity distribution")
    ax[0].legend(loc="upper right")
    rb = np.linspace(0.025, 0.975, 20)
    stations = sorted(disk.station.unique())
    for s in stations:
        p = np.median(np.stack(disk[disk.station == s]["profile"].values), 0)
        ax[1].plot(rb, p, lw=1.6, color=INK2, alpha=0.35)
    allp = np.median(np.stack(disk["profile"].values), 0)
    ax[1].plot(rb, allp, lw=2.2, color=TRAIN, label="all images")
    ax[1].plot([], [], lw=1.6, color=INK2, alpha=0.5, label="one line per station")
    ax[1].set_xlabel("distance from disk centre / disk radius")
    ax[1].set_ylabel("median pixel value")
    ax[1].set_title("Radial intensity profile (limb darkening)")
    ax[1].legend(loc="lower left")
    save(fig, "09_intensity.png")
    stats["radial_profile_all"] = {"r_0.025": float(allp[0]), "r_0.975": float(allp[-1])}

    fig, ax = plt.subplots(figsize=(8, 4))
    data = [disk[disk.station == s].mean_on_disk.values for s in stations]
    bp = ax.boxplot(data, tick_labels=stations, widths=0.5, patch_artist=True, medianprops={"color": INK, "lw": 1.5},
                    whiskerprops={"color": INK2}, capprops={"color": INK2}, flierprops={"marker": "o", "ms": 3, "mec": INK2})
    for b in bp["boxes"]:
        b.set(facecolor="#cde2fb", edgecolor=TRAIN)
    ax.set_xlabel("station")
    ax.set_ylabel("mean on-disk pixel value")
    ax.set_title("Image brightness by station (train + test)")
    ax.grid(axis="x", visible=False)
    save(fig, "10_brightness_by_station.png")

    # ---------------- where filaments are ----------------
    dk = disk.set_index("file_name")
    anns["rel_x"] = (anns.cx - anns.file_name.map(dk.cx)) / anns.file_name.map(dk.r)
    anns["rel_y"] = (anns.cy - anns.file_name.map(dk.cy)) / anns.file_name.map(dk.r)
    anns["rel_r"] = np.hypot(anns.rel_x, anns.rel_y)
    stats["filament_position"] = {"rel_r": q(anns.rel_r), "share_outside_disk": float((anns.rel_r > 1).mean()),
                                  "share_upper_half": float((anns.rel_y < 0).mean()),
                                  "share_abs_y_below_0.1": float((anns.rel_y.abs() < 0.1).mean()),
                                  "share_abs_y_0.2_to_0.6": float(anns.rel_y.abs().between(0.2, 0.6).mean())}
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    h, xe, ye = np.histogram2d(anns.rel_x, anns.rel_y, bins=60, range=[[-1.1, 1.1], [-1.1, 1.1]])
    im = ax[0].imshow(h.T, extent=[-1.1, 1.1, 1.1, -1.1], cmap=cmap, interpolation="nearest")
    ax[0].add_patch(plt.Circle((0, 0), 1, fill=False, color=INK2, lw=1))
    ax[0].set_aspect("equal")
    ax[0].grid(False)
    ax[0].set_xlabel("x / disk radius")
    ax[0].set_ylabel("y / disk radius (image rows, top = -1)")
    ax[0].set_title("Filament centroids on the disk")
    fig.colorbar(im, ax=ax[0], fraction=0.046, label="filaments per cell")
    ax[1].hist(anns.rel_y, bins=np.linspace(-1.05, 1.05, 43), orientation="horizontal", color=TRAIN, edgecolor=SURFACE)
    ax[1].invert_yaxis()
    ax[1].set_ylabel("y / disk radius (top = -1)")
    ax[1].set_xlabel("filaments")
    ax[1].set_title("Vertical position of centroids")
    ax[1].grid(axis="y", visible=False)
    save(fig, "11_filament_positions.png")

    # foreground fraction per record (exact polygon raster at 2048 is slow; 1024 is used)
    def fg_frac(g):
        return float((label_mask(list(g.poly), 1024) > 0).mean())
    fg = anns.groupby("record_id").apply(fg_frac, include_groups=False)
    fg_disk = fg / recs.set_index("record_id").file_name.map(dk.r).reindex(fg.index).map(lambda r: np.pi * r * r / FULL ** 2)
    stats["foreground_fraction"] = {"of_image": q(fg), "of_disk": q(fg_disk)}
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(fg * 100, bins=50, color=TRAIN, edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(fg.median() * 100, color=INK, lw=1, ls="--")
    ax.text(0.97, 0.92, f"median {fg.median() * 100:.2f}%", transform=ax.transAxes, ha="right", color=INK2)
    ax.set_xlabel("filament pixels as % of the whole image (per record, 1024 px raster)")
    ax.set_ylabel("records")
    ax.set_title("Foreground fraction per annotation record")
    ax.grid(axis="x", visible=False)
    save(fig, "12_foreground_fraction.png")

    # ---------------- inter-annotator agreement ----------------
    polys = anns.groupby("record_id").poly.apply(list)
    jobs, meta = [], []
    for f, g in recs.groupby("file_name"):
        ids = [r for r in g.record_id if r in polys.index]
        for a, b in itertools.combinations(ids, 2):
            jobs.append((polys[a], polys[b]))
            meta.append(f)
    with ProcessPoolExecutor(args.workers) as ex:
        pairs = pd.DataFrame(list(ex.map(record_pair_stats, jobs, chunksize=4)))
    pairs["file_name"] = meta
    pairs["count_ratio"] = pairs[["n_a", "n_b"]].min(axis=1) / pairs[["n_a", "n_b"]].max(axis=1)
    pairs["match_f1"] = 2 * pairs.matched / (pairs.n_a + pairs.n_b)
    pairs["pq"] = pairs.match_iou_mean.fillna(0) * pairs.match_f1
    stats["inter_annotator"] = {"pairs": len(pairs), "images": int(pairs.file_name.nunique()),
                                "pixel_iou": q(pairs.pixel_iou), "pixel_iou_mean": float(pairs.pixel_iou.mean()),
                                "count_ratio": q(pairs.count_ratio), "instance_f1_iou_gt_0.5": q(pairs.match_f1),
                                "matched_iou_mean": float(pairs.match_iou_mean.mean()),
                                "pairwise_pq": q(pairs.pq), "pairwise_pq_mean": float(pairs.pq.mean())}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col, lab, title in [(ax[0], "pixel_iou", "pixel IoU of the two records", "Pixel agreement"),
                               (ax[1], "match_f1", "share of filaments matched at IoU > 0.5 (F1)", "Instance agreement"),
                               (ax[2], "pq", "PQ of one record scored against the other", "Pairwise PQ")]:
        a.hist(pairs[col], bins=np.linspace(0, 1, 26), color=TRAIN, edgecolor=SURFACE, linewidth=0.8)
        a.axvline(pairs[col].mean(), color=INK, lw=1, ls="--")
        a.text(0.97, 0.92, f"mean {pairs[col].mean():.3f}", transform=a.transAxes, ha="right", color=INK2)
        a.set_xlabel(lab)
        a.set_ylabel("record pairs")
        a.set_title(title)
        a.grid(axis="x", visible=False)
    save(fig, "13_inter_annotator.png")

    # ---------------- gallery ----------------
    multi = rec_per_img[rec_per_img >= 2].index
    picks = [multi[0], multi[len(multi) // 2], rec_per_img[rec_per_img == 1].index[len(rec_per_img) // 3]]
    fig, ax = plt.subplots(len(picks), 2, figsize=(11, 5.5 * len(picks)))
    colors = [TRAIN, TEST, THIRD, "#eda100"]
    for i, f in enumerate(picks):
        img = cv2.imread(str(root / "train" / "train_images" / f), cv2.IMREAD_GRAYSCALE)
        ax[i, 0].imshow(img, cmap="gray")
        ax[i, 0].set_title(f"{f}", fontsize=10)
        ax[i, 1].imshow(img, cmap="gray")
        rids = list(recs[recs.file_name == f].record_id)
        for j, rid in enumerate(rids):
            for p in polys.get(rid, []):
                ax[i, 1].fill(p[:, 0], p[:, 1], facecolor="none", edgecolor=colors[j % 4], lw=1.0)
            ax[i, 1].plot([], [], color=colors[j % 4], label=f"annotator {rid.split('-')[0]}")
        ax[i, 1].legend(loc="lower right", fontsize=8, labelcolor="white", facecolor="black", framealpha=0.6, frameon=True)
        ax[i, 1].set_title(f"{len(rids)} annotation record(s)", fontsize=10)
        for a in ax[i]:
            a.axis("off")
    save(fig, "14_examples.png")

    (OUT / "stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
