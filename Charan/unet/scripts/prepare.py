"""Stage 0: preprocess train + test images, rasterize per-record masks, build PQ references.

    python scripts/prepare.py --size 1024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fseg_unet import labels, pq  # noqa: E402
from fseg_unet.config import DEFAULT_SPLIT, get_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--split", default=str(DEFAULT_SPLIT), help="grouped split JSON (by physical image)")
    ap.add_argument("--size", type=int, default=1024, help="training resolution")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args()

    paths = get_paths(args.data_root, args.work_dir)
    manifest = labels.build(paths, args.size, Path(args.split), args.workers, include_test=not args.no_test)
    for s, g in manifest.groupby("split"):
        print(f"{s:>5}: {g.observation_id.nunique()} images, {len(g)} rows, {int(g.n_instances.sum())} filaments")

    refs = labels.load_references(paths)
    val_obs = manifest.loc[manifest.split == "val", "observation_id"].unique()
    ceiling = pq.annotator_ceiling({o: refs["sets"][o] for o in val_obs})
    print("annotator ceiling (val, one annotator vs the others):", pq.summary(ceiling))
    print("work dir:", paths.work_dir.resolve())


if __name__ == "__main__":
    main()
