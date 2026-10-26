"""Solar-disk preprocessing.

Filaments are dark, thin features on the disk. The two big sources of false dark pixels are the
limb (intensity falls toward the disk edge) and the sky outside the disk. We fit the disk, divide
out a radial intensity profile, and stack three channels for an ImageNet-pretrained encoder:

    channel 0: raw intensity
    channel 1: limb-flattened intensity (uniform filament contrast across the disk)
    channel 2: CLAHE of channel 1 (boosts faint material and barbs)

Pixels off the disk are 0 in every channel.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

N_RADIAL_BINS = 128


@dataclass
class Disk:
    cx: float
    cy: float
    r: float


def find_disk(gray: np.ndarray) -> Disk:
    """Fit the solar disk as the largest bright connected region (centroid + area-equivalent radius)."""
    h, w = gray.shape
    small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(cv2.GaussianBlur(small, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return Disk(w / 2, h / 2, min(h, w) * 0.45)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    # convex hull fills dark filaments and sunspots inside the disk before measuring its area
    disk = (labels == k).astype(np.uint8)
    contours, _ = cv2.findContours(disk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), r_enc = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    area = cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))
    r = 0.5 * (np.sqrt(area / np.pi) + r_enc)
    sx, sy = w / 512, h / 512
    return Disk(cx * sx, cy * sy, r * (sx + sy) / 2)


def radius_map(shape: tuple[int, int], disk: Disk) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    return np.sqrt((xx - disk.cx) ** 2 + (yy - disk.cy) ** 2) / disk.r


def disk_mask(shape: tuple[int, int], disk: Disk, scale: float = 1.0) -> np.ndarray:
    return radius_map(shape, disk) <= scale


def limb_flatten(gray: np.ndarray, disk: Disk) -> np.ndarray:
    """Divide by the median intensity at each normalized radius; returns float32, ~1.0 on quiet Sun."""
    img = gray.astype(np.float32)
    rho = radius_map(gray.shape, disk)
    inside = rho <= 1.0
    # the profile is estimated on a 4x-downsampled copy (fast, and plenty of pixels per bin)
    f = 4
    small = cv2.resize(img, (img.shape[1] // f, img.shape[0] // f), interpolation=cv2.INTER_AREA)
    rho_s = radius_map(small.shape, Disk(disk.cx / f, disk.cy / f, disk.r / f))
    sel_s = rho_s <= 1.0
    bins = np.minimum((rho_s[sel_s] * N_RADIAL_BINS).astype(np.int32), N_RADIAL_BINS - 1)
    vals = small[sel_s]
    profile = np.zeros(N_RADIAL_BINS, np.float32)
    for b in range(N_RADIAL_BINS):
        sel = vals[bins == b]
        profile[b] = np.median(sel) if sel.size else np.nan
    # fill empty bins and smooth so filaments do not imprint on the profile
    idx = np.arange(N_RADIAL_BINS)
    good = np.isfinite(profile) & (profile > 0)
    profile = np.interp(idx, idx[good], profile[good]).astype(np.float32)
    profile = np.convolve(np.pad(profile, 3, mode="edge"), np.ones(7) / 7, mode="valid").astype(np.float32)
    lookup = np.interp(np.clip(rho, 0, 1) * N_RADIAL_BINS, idx + 0.5, profile)
    flat = img / np.maximum(lookup, 1.0)
    flat[~inside] = 0
    return flat


def preprocess(gray: np.ndarray, size: int) -> tuple[np.ndarray, Disk]:
    """gray: uint8 HxW at full resolution -> uint8 size x size x 3, and the disk in full-res pixels."""
    disk = find_disk(gray)
    inside = disk_mask(gray.shape, disk)
    flat = limb_flatten(gray, disk)
    flat_u8 = np.clip(flat * 160.0, 0, 255).astype(np.uint8)  # quiet Sun ~160, filaments darker
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(flat_u8)
    raw = gray.copy()
    for ch in (raw, flat_u8, clahe):
        ch[~inside] = 0
    stacked = np.dstack([raw, flat_u8, clahe])
    if size != gray.shape[0]:
        stacked = cv2.resize(stacked, (size, size), interpolation=cv2.INTER_AREA)
    return stacked, disk
