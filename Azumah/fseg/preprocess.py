"""Solar-disk preprocessing.

Filaments are dark features on the disk. Two non-filament dark sources dominate false positives:
the limb (intensity falls toward the disk edge, "limb darkening") and the sky outside the disk.
We locate the disk, divide by a robust radial intensity profile, and build a 3-channel image:

    channel 0: raw intensity (keeps what the pretrained backbone expects)
    channel 1: limb-flattened intensity (filament contrast is now uniform across the disk)
    channel 2: CLAHE of channel 1 (boosts faint material and barbs)

Pixels off the disk are set to 0 in every channel.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Disk:
    cx: float
    cy: float
    r: float


def find_disk(gray: np.ndarray) -> Disk:
    """Fit the solar disk as the largest bright connected region."""
    small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), 1)
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu sits between sky and mean disk brightness and clips the darkened limb; the sky in GONG
    # images is near zero, so a fraction of Otsu keeps the whole disk.
    _, binary = cv2.threshold(blur, max(0.3 * otsu, 8), 255, cv2.THRESH_BINARY)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    if n <= 1:
        h, w = gray.shape
        return Disk(w / 2, h / 2, min(h, w) * 0.45)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == k).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(region)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, -1)  # fill dark holes
    m = cv2.moments(filled, binaryImage=True)
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    r = np.sqrt(m["m00"] / np.pi)  # area-equivalent radius, robust to a ragged limb
    scale = gray.shape[1] / 512.0
    return Disk(cx * scale, cy * scale, r * scale)


def radial_profile(gray: np.ndarray, disk: Disk, bins: int = 256, stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Median intensity vs normalized radius. Medians ignore the small filament fraction per ring."""
    sub = gray[::stride, ::stride].astype(np.float32)
    yy, xx = np.mgrid[0 : gray.shape[0] : stride, 0 : gray.shape[1] : stride]
    rho = np.hypot(xx - disk.cx, yy - disk.cy) / disk.r
    inside = rho < 1.0
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(rho[inside], edges) - 1, 0, bins - 1)
    vals = sub[inside]
    order = np.argsort(idx, kind="stable")
    idx, vals = idx[order], vals[order]
    splits = np.searchsorted(idx, np.arange(bins))
    profile = np.full(bins, np.nan, dtype=np.float32)
    for b in range(bins):
        lo = splits[b]
        hi = splits[b + 1] if b + 1 < bins else len(idx)
        if hi > lo:
            profile[b] = np.median(vals[lo:hi])
    centers = (edges[:-1] + edges[1:]) / 2
    good = ~np.isnan(profile)
    profile = np.interp(centers, centers[good], profile[good]).astype(np.float32)
    # light smoothing
    kernel = np.ones(5, dtype=np.float32) / 5
    profile = np.convolve(np.pad(profile, 2, mode="edge"), kernel, mode="valid")
    return centers, np.maximum(profile, 1.0)


def preprocess(gray: np.ndarray, clahe_clip: float = 2.0) -> tuple[np.ndarray, Disk]:
    """Return an HxWx3 uint8 image (raw, limb-flattened, CLAHE) and the fitted disk."""
    disk = find_disk(gray)
    centers, profile = radial_profile(gray, disk)
    yy, xx = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]].astype(np.float32)
    rho = np.hypot(xx - disk.cx, yy - disk.cy) / disk.r
    on_disk = rho < 1.0
    expected = np.interp(np.clip(rho, 0, 1), centers, profile).astype(np.float32)
    ratio = gray.astype(np.float32) / expected
    flat = np.clip(ratio * 127.5, 0, 255).astype(np.uint8)  # ratio 1.0 (quiet Sun) -> 127
    flat[~on_disk] = 0
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(16, 16)).apply(flat)
    clahe[~on_disk] = 0
    raw = gray.copy()
    raw[~on_disk] = 0
    return np.dstack([raw, flat, clahe]), disk


def disk_mask(shape: tuple[int, int], disk: Disk, margin: float = 1.0) -> np.ndarray:
    yy, xx = np.ogrid[0 : shape[0], 0 : shape[1]]
    return (xx - disk.cx) ** 2 + (yy - disk.cy) ** 2 < (disk.r * margin) ** 2
