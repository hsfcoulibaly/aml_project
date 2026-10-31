"""Write YOLO-shaped proposal files for a synthetic work dir, so stages 3+ can be smoke-tested without
training YOLO: each consensus filament becomes a coarse, shifted copy (some are missed), plus random
false positives. Writes runs/consensus_f<k>/oof_raw_fold<k>.pkl + pq_config.json and
runs/consensus_ffull/test_raw.pkl.

    FSEG_DATA_ROOT=<synth> FSEG_WORK_DIR=<work> python tests/make_fake_proposals.py
"""
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fseg import labels  # noqa: E402
from fseg import rle as R  # noqa: E402
from fseg.config import IMAGE_SIZE, get_paths  # noqa: E402
from fseg.preprocess import find_disk  # noqa: E402


def coarse(mask, rng):
    small = cv2.resize(mask.astype(np.float32), (IMAGE_SIZE // 4, IMAGE_SIZE // 4), interpolation=cv2.INTER_AREA)
    m = cv2.resize(small, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR) > 0.4
    return np.roll(m, (int(rng.integers(-6, 7)), int(rng.integers(-6, 7))), axis=(0, 1))


def fake(obs, consensus, image_dir, rng):
    disk = find_disk(cv2.imread(str(image_dir / f"{obs}.jpg"))[:, :, 0])
    inst = []
    for c in consensus.get(obs, []):
        if rng.random() < 0.15:  # missed
            continue
        inst.append((R.encode(coarse(R.decode(c["rle"]), rng)), float(rng.uniform(0.3, 0.95))))
    for _ in range(int(rng.integers(1, 4))):  # false positives on the disk
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        a = rng.uniform(0, 2 * np.pi)
        cx, cy = int(disk.cx + 0.6 * disk.r * np.cos(a)), int(disk.cy + 0.6 * disk.r * np.sin(a))
        cv2.ellipse(m, (cx, cy), (int(rng.integers(30, 90)), 8), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        inst.append((R.encode(m.astype(bool)), float(rng.uniform(0.05, 0.5))))
    return {"instances": inst, "dark": {}, "disk": (disk.cx, disk.cy, disk.r)}


def main():
    paths = get_paths()
    rng = np.random.default_rng(0)
    manifest = pd.read_csv(paths.manifest)
    cons = labels.load_targets(paths)["consensus"]
    for k in sorted(manifest.fold.unique()):
        obs = manifest.loc[manifest.fold == k, "observation_id"]
        raw = {o: fake(o, cons, paths.preprocessed / "train", rng) for o in obs}
        d = paths.work_dir / "runs" / f"consensus_f{k}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"oof_raw_fold{k}.pkl", "wb") as f:
            pickle.dump(raw, f)
        (d / "pq_config.json").write_text(json.dumps({"conf_thr": 0.25, "min_area": 150, "dark": None}))
    test = sorted(p.stem for p in (paths.preprocessed / "test").glob("*.jpg"))
    d = paths.work_dir / "runs" / "consensus_ffull"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "test_raw.pkl", "wb") as f:  # no labels on test: random blobs only
        pickle.dump({o: fake(o, {}, paths.preprocessed / "test", rng) for o in test}, f)
    print(f"fake proposals for {len(manifest)} train and {len(test)} test images -> {paths.work_dir / 'runs'}")


if __name__ == "__main__":
    main()
