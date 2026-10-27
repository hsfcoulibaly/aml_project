"""Build a time-blocked train/val split: whole 7-day blocks go to one side, stratified by year.

Filaments last days to weeks and the Sun rotates ~13 deg/day, so images a day apart show nearly the same
filaments. A split by physical image (Sheikh's) puts 20 val images on the same day as a training image and
45 more one day away. Grouping by 7-day block removes that leakage; stratifying by year keeps every phase of
the solar cycle (2011-2022) in both train and val, which matches the test set (random over the same years).

    python scripts/make_split.py                       # -> splits/week_split.json
    python scripts/make_split.py --block-days 27       # one solar rotation per block
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fseg_unet.config import DEFAULT_SPLIT, SEED  # noqa: E402

ORIGIN = date(2011, 1, 1)


def obs_date(file_name: str) -> date:
    # GONG names start with YYYYMMDDhhmmss, e.g. 20110120105534Ch.jpeg
    return date(int(file_name[:4]), int(file_name[4:6]), int(file_name[6:8]))


def block_split(file_names: list[str], block_days: int, val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    blocks = defaultdict(list)
    for f in file_names:
        blocks[(obs_date(f) - ORIGIN).days // block_days].append(f)
    by_year = defaultdict(list)
    for b in sorted(blocks):
        by_year[obs_date(blocks[b][0]).year].append(b)

    rng = random.Random(seed)
    val_blocks = set()
    for year in sorted(by_year):
        bs = by_year[year][:]
        rng.shuffle(bs)
        target, taken = val_frac * sum(len(blocks[b]) for b in bs), 0
        for b in bs:
            if taken >= target:
                break
            val_blocks.add(b)
            taken += len(blocks[b])
    train = sorted(f for b in blocks if b not in val_blocks for f in blocks[b])
    val = sorted(f for b in val_blocks for f in blocks[b])
    return train, val


def nearest_gaps(val: list[str], train: list[str]) -> list[int]:
    days = sorted({obs_date(f) for f in train})
    gaps = []
    for f in val:
        d, i = obs_date(f), bisect.bisect_left(days, obs_date(f))
        gaps.append(min(abs((days[j] - d).days) for j in (i - 1, i) if 0 <= j < len(days)))
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-from", default=str(DEFAULT_SPLIT), help="any split JSON listing all training images")
    ap.add_argument("--block-days", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "splits" / "week_split.json"))
    args = ap.parse_args()

    with open(args.images_from) as f:
        src = json.load(f)
    file_names = sorted(set(src["train"]["file_names"]) | set(src["val"]["file_names"]))
    train, val = block_split(file_names, args.block_days, args.val_frac, args.seed)

    gaps = nearest_gaps(val, train)
    print(f"train {len(train)} / val {len(val)} images ({len(val) / len(file_names):.1%} val)")
    print(f"val gap to nearest train day: same day {gaps.count(0)}, <=1 d {sum(g <= 1 for g in gaps)}, "
          f"median {sorted(gaps)[len(gaps) // 2]} d")
    tr_y, va_y = Counter(f[:4] for f in train), Counter(f[:4] for f in val)
    print("per year (train/val):", ", ".join(f"{y} {tr_y[y]}/{va_y[y]}" for y in sorted(tr_y | va_y)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"seed": args.seed, "block_days": args.block_days, "val_frac": args.val_frac,
                   "train": {"file_names": train}, "val": {"file_names": val}}, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
