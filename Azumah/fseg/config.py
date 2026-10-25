"""Paths and global constants.

Paths are discovered automatically on Kaggle. Elsewhere, set FSEG_DATA_ROOT to the folder that
contains `train/` and `test/` (the MAGFiLO_1.0_Kaggle_2026 directory) and FSEG_WORK_DIR to a
writable folder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ANN_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_SIZE = 2048  # every GONG H-alpha image in the competition is 2048 x 2048
SEED = 2026


def _find_data_root() -> Path:
    env = os.environ.get("FSEG_DATA_ROOT")
    if env:
        return Path(env)
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hit = next(kaggle.rglob(ANN_NAME), None)
        if hit is not None:
            return hit.parent.parent
    raise FileNotFoundError(
        "Could not locate the competition data. Set FSEG_DATA_ROOT to the MAGFiLO_1.0_Kaggle_2026 folder."
    )


def _default_work_dir() -> Path:
    env = os.environ.get("FSEG_WORK_DIR")
    if env:
        return Path(env)
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/fseg")
    return Path("work")


@dataclass
class Paths:
    data_root: Path
    work_dir: Path

    @property
    def ann_path(self) -> Path:
        return self.data_root / "train" / ANN_NAME

    @property
    def train_images(self) -> Path:
        return self.data_root / "train" / "train_images"

    @property
    def test_images(self) -> Path:
        return self.data_root / "test" / "test_images"

    @property
    def manifest(self) -> Path:
        return self.work_dir / "manifest.csv"

    @property
    def targets(self) -> Path:
        return self.work_dir / "targets.pkl"

    @property
    def references(self) -> Path:
        return self.work_dir / "references.pkl"

    @property
    def preprocessed(self) -> Path:
        return self.work_dir / "preprocessed"

    @property
    def yolo_root(self) -> Path:
        return self.work_dir / "yolo"


def get_paths(data_root: str | Path | None = None, work_dir: str | Path | None = None) -> Paths:
    paths = Paths(
        data_root=Path(data_root) if data_root else _find_data_root(),
        work_dir=Path(work_dir) if work_dir else _default_work_dir(),
    )
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    return paths
