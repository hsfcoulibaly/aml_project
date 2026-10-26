"""Paths and constants.

On Kaggle the competition data is found automatically under /kaggle/input. Elsewhere set
UNET_DATA_ROOT to the folder that contains `train/` and `test/` (MAGFiLO_1.0_Kaggle_2026) and
optionally UNET_WORK_DIR to a writable folder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ANN_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
FULL_SIZE = 2048  # every GONG H-alpha image in the competition is 2048 x 2048
SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[3]
# Sheikh's grouped split (by physical image, seed 42): the split the U-Net and YOLO baselines use
DEFAULT_SPLIT = REPO_ROOT / "Sheikh" / "outputs" / "threshold_eval" / "grouped_split.json"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _find_data_root() -> Path:
    env = os.environ.get("UNET_DATA_ROOT")
    if env:
        return Path(env)
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hit = next(kaggle.rglob(ANN_NAME), None)
        if hit is not None:
            return hit.parent.parent
    raise FileNotFoundError("Competition data not found. Set UNET_DATA_ROOT to the MAGFiLO_1.0_Kaggle_2026 folder.")


def _default_work_dir() -> Path:
    env = os.environ.get("UNET_WORK_DIR")
    if env:
        return Path(env)
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/unet")
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
    def references(self) -> Path:
        return self.work_dir / "references.pkl"

    @property
    def images(self) -> Path:
        return self.work_dir / "images"

    @property
    def masks(self) -> Path:
        return self.work_dir / "masks"

    @property
    def runs(self) -> Path:
        return self.work_dir / "runs"


def get_paths(data_root=None, work_dir=None) -> Paths:
    paths = Paths(
        data_root=Path(data_root) if data_root else _find_data_root(),
        work_dir=Path(work_dir) if work_dir else _default_work_dir(),
    )
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    return paths
