"""Tiny fake MAGFiLO folder (disk + dark elongated filaments) for an end-to-end CPU smoke run.

    python tests/make_synthetic.py /tmp/synth
    UNET_DATA_ROOT=/tmp/synth UNET_WORK_DIR=/tmp/work python scripts/prepare.py --size 256 \
        --split /tmp/synth/split.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

N_TRAIN, N_TEST, S = 10, 3, 2048
SITES = ["Bh", "Ch", "Lh", "Mh", "Th", "Uh"]


def solar_image(rng):
    img = np.zeros((S, S), np.uint8)
    cx, cy, r = S // 2 + rng.integers(-20, 20), S // 2 + rng.integers(-20, 20), 860
    yy, xx = np.mgrid[0:S, 0:S]
    rho = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    disk = rho <= 1
    img[disk] = (170 * np.sqrt(np.clip(1 - 0.6 * rho[disk] ** 2, 0, 1))).astype(np.uint8)
    polys = []
    for _ in range(rng.integers(3, 7)):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0, 0.7) * r
        c = (int(cx + rad * np.cos(ang)), int(cy + rad * np.sin(ang)))
        axes = (int(rng.uniform(40, 120)), int(rng.uniform(6, 14)))
        pts = cv2.ellipse2Poly(c, axes, int(rng.uniform(0, 180)), 0, 360, 15)
        cv2.fillPoly(img, [pts], 60)
        polys.append(pts)
    noise = rng.normal(0, 4, img.shape)
    img = np.where(disk, np.clip(img + noise, 0, 255), 0).astype(np.uint8)
    return img, polys


def main(root: Path):
    rng = np.random.default_rng(0)
    (root / "train" / "train_images").mkdir(parents=True, exist_ok=True)
    (root / "test" / "test_images").mkdir(parents=True, exist_ok=True)
    images, anns, files = [], [], []
    for i in range(N_TRAIN):
        stem = f"2015{i + 1:02d}01120000{SITES[i % 6]}"
        img, polys = solar_image(rng)
        cv2.imwrite(str(root / "train" / "train_images" / f"{stem}.jpeg"), img)
        files.append(f"{stem}.jpeg")
        for a in range(1 + i % 2):  # every other image gets a second annotator who skips one filament
            rid = f"0101{a + 1:02d}-{stem}"
            images.append({"id": rid, "width": S, "height": S, "file_name": f"{stem}.jpeg"})
            for k, p in enumerate(polys[: len(polys) - a]):
                flat = p.reshape(-1).astype(float).tolist()
                anns.append({"id": f"{rid}-{k}", "image_id": rid, "category_id": 1, "segmentation": [flat],
                             "area": float(cv2.contourArea(p)), "spine": [], "bbox": list(cv2.boundingRect(p)),
                             "iscrowd": 0})
    coco = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "Left"}]}
    (root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(json.dumps(coco))
    for i in range(N_TEST):
        img, _ = solar_image(rng)
        cv2.imwrite(str(root / "test" / "test_images" / f"2016{i + 1:02d}01120000Bh.jpeg"), img)
    split = {"seed": 0, "train": {"file_names": files[:7]}, "val": {"file_names": files[7:]}}
    (root / "split.json").write_text(json.dumps(split))
    print(f"wrote {N_TRAIN} train ({len(images)} records), {N_TEST} test images to {root}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
