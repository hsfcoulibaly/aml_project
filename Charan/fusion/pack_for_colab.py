"""Kaggle CPU script: bundle everything the refiner stage needs into one tar for Colab.

Input: the output of charanpeeriga/filament-seg-unet2048 (prepared data, labels, U-Net-2048 maps).
Builds the U-Net proposals (make_unet_proposals.py), then writes /kaggle/working/fusion_pack.tar with
manifest.csv, targets.pkl, references.pkl, preprocessed/, unet/maps/unet2048/ and runs/ (proposals).
"""
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

T0 = time.time()
REPO = Path("/tmp/aml_project")
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "charan/fusion-self",
                    "https://github.com/hsfcoulibaly/aml_project.git", str(REPO)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"], check=True)

inp = Path("/kaggle/input")
base = next(p.parent for p in inp.rglob("manifest.csv") if (p.parent / "references.pkl").exists())
maps = next(p for p in inp.rglob("maps/unet2048") if (p / "oof").exists() and (p / "test").exists())
WORK = Path("/tmp/fseg")
(WORK / "unet" / "maps").mkdir(parents=True, exist_ok=True)
for name in ["manifest.csv", "targets.pkl", "references.pkl", "preprocessed"]:
    if not (WORK / name).exists():
        (WORK / name).symlink_to(base / name)
if not (WORK / "unet" / "maps" / "unet2048").exists():
    (WORK / "unet" / "maps" / "unet2048").symlink_to(maps)
os.environ["FSEG_WORK_DIR"] = str(WORK)
print("base", base, "| maps", maps, flush=True)

subprocess.run([sys.executable, str(REPO / "Charan/fusion/make_unet_proposals.py"), "--code", str(REPO / "Azumah"),
                "--maps", str(maps), "--work-dir", str(WORK), "--workers", str(os.cpu_count() or 4)], check=True)
print(f"proposals done, {(time.time() - T0) / 60:.1f} min", flush=True)

out = Path("/kaggle/working/fusion_pack.tar")
with tarfile.open(out, "w") as tar:  # jpg/png are already compressed
    for name in ["manifest.csv", "targets.pkl", "references.pkl", "preprocessed", "runs"]:
        tar.add(str((WORK / name).resolve()), arcname=f"fseg/{name}")
    tar.add(str(maps.resolve()), arcname="fseg/unet/maps/unet2048")
print(f"wrote {out} ({out.stat().st_size / 2**30:.2f} GB), {(time.time() - T0) / 60:.1f} min")
