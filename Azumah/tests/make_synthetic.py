"""Generate a tiny MAGFiLO-shaped dataset (limb-darkened disk, dark curved filaments, 1-3 annotation
sets per image) so the whole pipeline can be smoke-tested without the competition data."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

N_TRAIN, N_TEST, S = 20, 4, 2048
SITES = ["Bh", "Ch", "Lh", "Mh", "Th", "Uh"]


def make_image(rng):
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    cx, cy, r = S / 2 + rng.uniform(-20, 20), S / 2 + rng.uniform(-20, 20), 900
    rho = np.hypot(xx - cx, yy - cy) / r
    disk = rho < 1
    mu = np.sqrt(np.clip(1 - rho**2, 0, 1))
    img = np.where(disk, 180 * (0.4 + 0.6 * mu), 3).astype(np.float32)
    polys = []
    for _ in range(rng.integers(3, 9)):
        ang0 = rng.uniform(0, 2 * np.pi)
        rad0 = rng.uniform(0, 0.8) * r
        px, py = cx + rad0 * np.cos(ang0), cy + rad0 * np.sin(ang0)
        theta, length, width = rng.uniform(0, np.pi), rng.uniform(80, 300), rng.uniform(6, 14)
        t = np.linspace(-0.5, 0.5, 40)
        curve = 0.15 * length * np.sin(t * np.pi * rng.uniform(0.5, 2))
        xs = px + t * length * np.cos(theta) - curve * np.sin(theta)
        ys = py + t * length * np.sin(theta) + curve * np.cos(theta)
        nx, ny = -np.sin(theta), np.cos(theta)
        w = width * (1 - 0.6 * np.abs(t) * 2)
        top = np.stack([xs + nx * w / 2, ys + ny * w / 2], 1)
        bot = np.stack([xs - nx * w / 2, ys - ny * w / 2], 1)[::-1]
        poly = np.concatenate([top, bot]).astype(np.float32)
        m = np.zeros((S, S), np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 1)
        img[m.astype(bool) & disk] *= 0.55
        polys.append(poly)
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), polys


def main(root):
    root = Path(root)
    tr, te = root / "train" / "train_images", root / "test" / "test_images"
    tr.mkdir(parents=True, exist_ok=True)
    te.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    images, anns = [], []
    for i in range(N_TRAIN):
        img, polys = make_image(rng)
        stem = f"2015{(i % 12) + 1:02d}{i + 1:02d}120000{SITES[i % 6]}"
        cv2.imwrite(str(tr / f"{stem}.jpeg"), img)
        n_sets = 1 + (i % 3)
        for s in range(n_sets):
            rid = f"0{i:02d}0{s + 1}-{stem}"
            images.append({"id": rid, "file_name": f"{stem}.jpeg", "height": S, "width": S})
            for k, p in enumerate(polys):
                if s > 0 and rng.random() < 0.25:  # other annotators skip some filaments
                    continue
                q = p + rng.normal(0, 1.5, p.shape)
                anns.append({"id": f"{rid}-{k}", "image_id": rid, "category_id": int(rng.integers(1, 4)),
                             "segmentation": [q.reshape(-1).round(2).tolist()], "iscrowd": 0,
                             "area": float(cv2.contourArea(q.astype(np.float32))), "bbox": [0, 0, 0, 0]})
    for i in range(N_TEST):
        img, _ = make_image(rng)
        cv2.imwrite(str(te / f"2016010{i + 1}120000{SITES[i]}.jpeg"), img)
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "Left"}, {"id": 2, "name": "Right"},
                           {"id": 3, "name": "Unidentifiable"}, {"id": 4, "name": "Ambiguous"}]}
    (root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(json.dumps(coco))
    print(f"synthetic data: {len(images)} records, {len(anns)} annotations -> {root}")


if __name__ == "__main__":
    main(sys.argv[1])
