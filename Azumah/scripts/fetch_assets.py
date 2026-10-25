"""Download everything Ultralytics would otherwise fetch at training time.

Cluster compute nodes usually have no internet. Run this once on the LOGIN node; files are cached in
the Ultralytics weights/config folders under your home directory, which compute nodes share.

    python scripts/fetch_assets.py yolo11l-seg.pt yolo11m-seg.pt
"""
import sys

from ultralytics import YOLO
from ultralytics.utils import WEIGHTS_DIR
from ultralytics.utils.checks import check_font

models = sys.argv[1:] or ["yolo11l-seg.pt"]
for name in models + ["yolo26n.pt"]:  # yolo26n.pt is used by the automatic mixed-precision check
    YOLO(WEIGHTS_DIR / name)
    print(f"cached {WEIGHTS_DIR / name}")
check_font("Arial.ttf")  # used for training plots
print("assets ready")
