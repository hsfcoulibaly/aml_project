"""Stage 1: train the YOLO-seg proposer on one fold (development) or all data (final).

    python scripts/train_yolo.py --fold 0 --model yolo11l-seg.pt --imgsz 1280 --epochs 60
    python scripts/train_yolo.py --fold 0 --dilated p4,p5 --name dil_p45      # dilation ablation
    python scripts/train_yolo.py --fold full --epochs 60                        # final model
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# multi-GPU training relaunches Python; make `fseg` importable there too
os.environ["PYTHONPATH"] = f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

from fseg.config import SEED, get_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--mode", default="consensus", choices=["consensus", "largest"])
    ap.add_argument("--fold", default="0", help="fold index or 'full'")
    ap.add_argument("--model", default="yolo11l-seg.pt", help="weights (.pt) or architecture (.yaml)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8, help="total across GPUs")
    ap.add_argument("--device", default=None, help="'0,1' for both Kaggle T4s, 'cpu' for smoke tests")
    ap.add_argument("--dilated", default="", help="'' (off), 'p4,p5' or 'p3,p4,p5'")
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--fraction", type=float, default=1.0, help="use a fraction of training data (debug)")
    ap.add_argument("--resume", action="store_true",
                    help="continue work/runs/<name>/weights/last.pt if it exists (same --name/--mode/--fold/--dilated)")
    args = ap.parse_args()

    os.environ["FSEG_DILATED"] = args.dilated
    from ultralytics import YOLO

    from fseg.dilated import DilatedSegmentationTrainer

    paths = get_paths(args.data_root, args.work_dir)
    data_yaml = paths.yolo_root / args.mode / ("full.yaml" if args.fold == "full" else f"fold{args.fold}.yaml")
    if args.device is None:
        import torch

        n = torch.cuda.device_count()
        if n:
            args.device = ",".join(str(i) for i in range(n))
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            args.device = "mps"  # Apple Silicon GPU
        else:
            args.device = "cpu"
    print(f"[fseg] training on device: {args.device}")
    name = args.name or f"{args.mode}_f{args.fold}" + (f"_dil{args.dilated.replace(',', '')}" if args.dilated else "")
    last = paths.work_dir / "runs" / name / "weights" / "last.pt"

    if args.resume:
        # Continue an interrupted run (e.g. a SLURM job that hit its time limit). All settings,
        # including the epoch count, come from the checkpoint.
        if not last.exists():
            print(f"[fseg] no checkpoint at {last}; starting a new run instead")
        else:
            from fseg.dilated import levels_in_checkpoint

            model = YOLO(str(last))
            os.environ["FSEG_DILATED"] = ",".join(levels_in_checkpoint(model.model))
            print(f"[fseg] resuming {last} (dilated levels: {os.environ['FSEG_DILATED'] or 'none'})")
            try:
                model.train(trainer=DilatedSegmentationTrainer, resume=True, device=args.device)
            except AssertionError as e:  # Ultralytics raises this when the run already finished
                print(f"[fseg] nothing to resume: {e}")
            print(f"Weights: {last.parent / 'best.pt'}")
            return

    model = YOLO(args.model)
    model.train(
        trainer=DilatedSegmentationTrainer,
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(paths.work_dir / "runs"),
        name=name,
        exist_ok=True,
        seed=SEED,
        workers=args.workers,
        patience=args.patience,
        fraction=args.fraction,
        single_cls=True,
        optimizer="AdamW",
        lr0=1e-3,
        cos_lr=True,
        warmup_epochs=3,
        # the Sun has no preferred orientation: full rotations and both flips are label-preserving
        degrees=180.0,
        fliplr=0.5,
        flipud=0.5,
        translate=0.05,
        scale=0.25,
        shear=0.0,
        perspective=0.0,
        mosaic=0.5,
        close_mosaic=10,
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        mask_ratio=2,  # finer mask supervision for thin structures (default 4)
        overlap_mask=True,
        max_det=100,
        plots=True,
        amp=args.device not in ("cpu", "mps"),
    )
    print(f"Weights: {paths.work_dir / 'runs' / name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
