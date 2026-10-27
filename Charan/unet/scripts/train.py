"""Stage 1: train the U-Net (or MeshNet); keep the checkpoint with the best validation PQ.

    python scripts/train.py --epochs 40 --batch 8          # uses every visible GPU
    python scripts/train.py --resume                        # continue an interrupted run
    python scripts/train.py --arch meshnet --channels 32 --crop 512 --batch 16 --lr 3e-3 \
        --weight-decay 1e-2 --warmup-frac 0.05 --grad-clip 1.0 --epochs 100   # dilated MeshNet recipe

Writes to <work>/runs/<name>/: best.pt (fp16 weights), last_state.pt (resume), history.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fseg_unet import evaluate as E  # noqa: E402
from fseg_unet import labels  # noqa: E402
from fseg_unet.config import SEED, get_paths  # noqa: E402
from fseg_unet.data import RecordDataset, read_image  # noqa: E402
from fseg_unet.meshnet import receptive_field  # noqa: E402
from fseg_unet.model import BCEDiceLoss, build_model, predict_probs, save_weights, unwrap  # noqa: E402


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int):
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--name", default="unet_r34")
    ap.add_argument("--arch", choices=["unet", "meshnet"], default="unet")
    ap.add_argument("--encoder", default="resnet34", help="U-Net encoder")
    ap.add_argument("--channels", type=int, default=32, help="MeshNet width")
    ap.add_argument("--crop", type=int, default=0, help="random training crop size in px (0 = whole image)")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (offline smoke tests)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8, help="total across GPUs")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.0, help="warmup as a share of all steps (0 = one epoch)")
    ap.add_argument("--grad-clip", type=float, default=0.0, help="max gradient norm (0 = off)")
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pq-every", type=int, default=1, help="compute val PQ every N epochs")
    ap.add_argument("--pq-threshold", type=float, default=0.5)
    ap.add_argument("--pq-min-area", type=int, default=200, help="min filament area, full-res pixels")
    ap.add_argument("--patience", type=int, default=10, help="epochs without PQ improvement before stopping")
    ap.add_argument("--max-train-batches", type=int, default=0, help="debug: cap batches per epoch")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    seed_everything(SEED)
    paths = get_paths(args.data_root, args.work_dir)
    run_dir = paths.runs / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    manifest = labels.load_manifest(paths)
    refs = labels.load_references(paths)
    train_m = manifest[manifest.split == "train"]
    val_obs = sorted(manifest.loc[manifest.split == "val", "observation_id"].unique())
    disks = E.disks_from_manifest(manifest)
    size = read_image(paths, val_obs[0]).shape[0]
    print(f"train records {len(train_m)} | val images {len(val_obs)} | input {size}px")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.encoder, pretrained=not args.no_pretrained, arch=args.arch,
                        channels=args.channels).to(device)
    if args.arch == "meshnet":
        print(f"MeshNet dilations {model.dilations} | receptive field {receptive_field(model.dilations)} px")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = nn.DataParallel(model)
    print(f"device {device} x{max(n_gpu, 1)} | params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    loader = DataLoader(RecordDataset(paths, train_m, augment=True, crop=args.crop), batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True,
                        worker_init_fn=worker_init, persistent_workers=args.workers > 0)
    steps_per_epoch = min(len(loader), args.max_train_batches or len(loader))
    total = args.epochs * steps_per_epoch
    warmup = max(1, int(args.warmup_frac * total)) if args.warmup_frac > 0 else steps_per_epoch

    def lr_lambda(step):  # linear warmup, then cosine down to 1% of the peak
        if step < warmup:
            return (step + 1) / warmup
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    criterion = BCEDiceLoss(pos_weight=args.pos_weight).to(device)
    if args.arch == "meshnet":  # decay conv weights only, not GroupNorm gains/biases or conv biases
        decay = [p for p in model.parameters() if p.ndim > 1]
        no_decay = [p for p in model.parameters() if p.ndim <= 1]
        params = [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    else:
        params = model.parameters()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch, best_pq, stale, history = 1, -1.0, 0, []
    state_path = run_dir / "last_state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        unwrap(model).load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, best_pq, stale, history = st["epoch"] + 1, st["best_pq"], st["stale"], st["history"]
        print(f"resumed from epoch {st['epoch']} (best PQ {best_pq:.4f})")

    dice_gt = E.dice_targets(refs, val_obs)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        running, n = 0.0, 0
        for step, (x, y) in enumerate(loader):
            if step >= steps_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * x.size(0)
            n += x.size(0)

        probs = predict_probs(model, paths, val_obs, device, batch_size=max(2, args.batch // 2),
                              workers=min(2, args.workers))
        dice = E.dice_table(probs, dice_gt, refs, thr=0.5)
        row = {"epoch": epoch, "train_loss": running / max(n, 1), "val_dice": dice.dice.mean(),
               "val_dice_median": dice.dice.median(), "val_iou": dice.iou.mean(), "lr": opt.param_groups[0]["lr"]}
        if epoch % args.pq_every == 0 or epoch == args.epochs:
            table, _ = E.pq_sweep(probs, disks, refs, [args.pq_threshold], [args.pq_min_area])
            row.update({f"val_{k}": table.loc[0, k] for k in ["pq", "sq", "rq", "precision", "recall",
                                                               "split_rate", "merge_rate", "pred_per_image"]})
        row["time_s"] = time.time() - t0
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        msg = " | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items())
        if "val_pq" in row and row["val_pq"] > best_pq:
            best_pq, stale = row["val_pq"], 0
            save_weights(model, run_dir / "best.pt", {"arch": args.arch, "encoder": args.encoder,
                                                      "channels": args.channels, "size": size, "epoch": epoch,
                                                      "metrics": {k: float(v) for k, v in row.items()}})
            msg += "  <- best"
        elif "val_pq" in row:
            stale += args.pq_every
        print(msg, flush=True)

        torch.save({"model": unwrap(model).state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_pq": best_pq, "stale": stale,
                    "history": history}, state_path)
        if stale >= args.patience:
            print(f"early stop: no PQ improvement for {stale} epochs")
            break

    print(f"best val PQ {best_pq:.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
