"""Stage 0: build manifest/folds/targets, preprocess every image, export YOLO datasets.

    python scripts/prepare.py --size 1280 --modes consensus largest
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fseg import labels, metrics, yolo_data  # noqa: E402
from fseg.config import get_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--size", type=int, default=1280, help="YOLO training resolution")
    ap.add_argument("--modes", nargs="+", default=["consensus", "largest"])
    ap.add_argument("--min-agreement", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    paths = get_paths(args.data_root, args.work_dir)
    manifest = labels.build(paths, n_folds=args.folds, min_agreement=args.min_agreement)
    print(f"Observations: {len(manifest)} | sets per observation: {manifest.n_sets.value_counts().to_dict()}")
    print(f"Filaments per image: largest {manifest.n_largest.mean():.2f}, consensus {manifest.n_consensus.mean():.2f}")
    print(manifest.groupby("fold").size().rename("observations_per_fold").to_string())

    ceiling = metrics.annotator_ceiling(labels.load_references(paths))
    print(f"Human agreement ceiling (one annotator vs others): {metrics.summary(ceiling)}")

    yolo_data.preprocess_all(paths, "train", manifest["file_name"].tolist(), workers=args.workers)
    test_files = sorted(p.name for p in paths.test_images.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    yolo_data.preprocess_all(paths, "test", test_files, workers=args.workers)

    targets = labels.load_targets(paths)
    for mode in args.modes:
        root = yolo_data.export(paths, manifest, targets, mode=mode, size=args.size)
        print(f"YOLO dataset ({mode}) -> {root}")


if __name__ == "__main__":
    main()
