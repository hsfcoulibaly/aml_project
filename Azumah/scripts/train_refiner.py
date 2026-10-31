"""Stage 3: train the instance-conditioned refiner.

Trains on every fold except --val-fold, using out-of-fold YOLO proposals (`runs/consensus_f<k>/
oof_raw_fold<k>.pkl`) and out-of-fold U-Net maps (`unet/maps/<unet-name>/oof/`). The checkpoint with
the lowest validation loss on the held-out fold is kept; PQ is measured by predict_refiner.py.

    python scripts/train_refiner.py                      # validate on fold 0, train on folds 1-4
    python scripts/train_refiner.py --val-fold none      # final refiner on all folds
"""
import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fseg import labels  # noqa: E402
from fseg.config import SEED, get_paths  # noqa: E402
from fseg.refiner import Refiner, RefinerLossWeights, refiner_loss, save_refiner  # noqa: E402
from fseg.refiner_data import RefinerDataset, collate, load_proposals  # noqa: E402
from fseg.unet import pick_device  # noqa: E402


def to_input(x: torch.Tensor, device) -> torch.Tensor:
    return ((x.to(device, non_blocking=True).float() / 255.0) - 0.5) / 0.25


def run_epoch(model, loader, device, opt, sched, scaler, weights, train: bool, max_batches: int = 0):
    model.train(train)
    tot, n, parts_sum = 0.0, 0, {}
    use_amp = device.type == "cuda"
    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            if len(x) == 0:
                continue
            x, y = to_input(x, device), y.to(device).float()
            with torch.autocast(device.type, enabled=use_amp):
                mask, q = model(x)
            loss, parts = refiner_loss(mask, q, y, weights)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            tot += float(loss)
            n += 1
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
    return tot / max(n, 1), {k: v / max(n, 1) for k, v in parts_sum.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--val-fold", default="0", help="held-out fold, or 'none' to train on everything")
    ap.add_argument("--name", default=None)
    ap.add_argument("--unet-name", default="unet", help="U-Net maps in work/unet/maps/<name>/oof (missing -> zeros)")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--images", type=int, default=4, help="images per batch")
    ap.add_argument("--k", type=int, default=8, help="crops per image (batch = images x k)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=0, help="debug: cap batches per epoch")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    paths = get_paths(args.data_root, args.work_dir)
    manifest = pd.read_csv(paths.manifest)
    folds = sorted(manifest.fold.unique())
    val_fold = None if args.val_fold == "none" else int(args.val_fold)
    train_obs = manifest.loc[manifest.fold != val_fold, "observation_id"].tolist()
    val_obs = manifest.loc[manifest.fold == (0 if val_fold is None else val_fold), "observation_id"].tolist()

    consensus = labels.load_targets(paths)["consensus"]
    proposals = load_proposals(paths, folds)
    map_dir = paths.work_dir / "unet" / "maps" / args.unet_name / "oof"
    if not map_dir.exists():
        print(f"[refiner] no U-Net maps in {map_dir}: those channels are zeros")
        map_dir = None
    n_prop = sum(len(v) for o, v in proposals.items() if o in set(train_obs))
    print(f"[refiner] {len(train_obs)} train / {len(val_obs)} val images, {n_prop} train proposals")

    name = args.name or ("refiner_full" if val_fold is None else f"refiner_v{val_fold}")
    run_dir = paths.work_dir / "refiner" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    cfg = {"encoder": args.encoder, "unet_name": args.unet_name}
    weights = RefinerLossWeights()
    (run_dir / "config.json").write_text(json.dumps({**cfg, **vars(args), "loss": asdict(weights)}, indent=2))

    def loader(obs, augment):
        ds = RefinerDataset(paths, obs, consensus, proposals, map_dir, k=args.k, augment=augment)
        return DataLoader(ds, batch_size=args.images, shuffle=augment, num_workers=args.workers, collate_fn=collate,
                          persistent_workers=args.workers > 0)

    train_dl, val_dl = loader(train_obs, True), loader(val_obs, False)
    model = Refiner(args.encoder, None if args.no_pretrained else "imagenet").to(device)
    steps = min(len(train_dl), args.max_batches or len(train_dl))
    total, warmup = args.epochs * steps, steps

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start, best, stale, history = 1, float("inf"), 0, []
    state_path = run_dir / "last_state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start, best, stale, history = st["epoch"] + 1, st["best"], st["stale"], st["history"]

    for epoch in range(start, args.epochs + 1):
        t0 = time.time()
        tr, trp = run_epoch(model, train_dl, device, opt, sched, scaler, weights, True, args.max_batches)
        va, vap = run_epoch(model, val_dl, device, opt, sched, scaler, weights, False, args.max_batches)
        row = {"epoch": epoch, "train_loss": tr, **{f"train_{k}": v for k, v in trp.items()},
               "val_loss": va, **{f"val_{k}": v for k, v in vap.items()}, "epoch_s": time.time() - t0}
        if va < best:
            best, stale = va, 0
            save_refiner(model, run_dir / "best.pt", cfg, {"epoch": epoch, "val_loss": va, "val_fold": val_fold})
        else:
            stale += 1
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "stale": stale,
                    "history": history}, state_path)
        if stale >= args.patience:
            print(f"[refiner] early stop after {epoch} epochs")
            break
    print(f"[refiner] best val loss {best:.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
