"""Stage 2: train the dense U-Net (filament, centerline, boundary) on one fold.

The checkpoint with the best validation PQ (standalone U-Net instances, fixed threshold and min area)
is kept as fp16 `best.pt`; `last_state.pt` holds everything needed to resume.

    python scripts/train_unet.py --fold 0                       # one fold, every visible GPU
    CUDA_VISIBLE_DEVICES=1 python scripts/train_unet.py --fold 1 # one fold per GPU in parallel
    python scripts/train_unet.py --fold full --epochs 30         # final model on all images
"""
import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fseg import labels  # noqa: E402
from fseg.config import IMAGE_SIZE, SEED, get_paths  # noqa: E402
from fseg.metrics import evaluate  # noqa: E402
from fseg.unet import (LossWeights, build_model, pick_device, predict_full, prob_to_instances,  # noqa: E402
                       save_weights, unet_loss)
from fseg.unet_data import FilamentDataset, build_targets, read_image, unet_root  # noqa: E402


def val_pq(model, paths, obs, refs, size, device, thr, min_area):
    model.eval()
    preds = {}
    for o in obs:
        prob = predict_full(model, read_image(paths, "train", o, size), device, tile=size, tta="none")[0]
        if size != IMAGE_SIZE:
            prob = cv2.resize(prob, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        disk = cv2.imread(str(paths.preprocessed / "train" / f"{o}.jpg"), cv2.IMREAD_GRAYSCALE) > 0
        preds[o] = [r for r, _ in prob_to_instances(prob, thr, min_area, disk)]
    return evaluate(preds, refs, observations=obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--fold", default="0", help="fold index or 'full'")
    ap.add_argument("--name", default=None)
    ap.add_argument("--encoder", default="tu-convnext_tiny")
    ap.add_argument("--no-pretrained", action="store_true", help="random encoder init (offline smoke tests)")
    ap.add_argument("--no-aspp", action="store_true")
    ap.add_argument("--size", type=int, default=1024, help="training/inference resolution of the full disk")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=10, help="epochs without val PQ gain before stopping")
    ap.add_argument("--pq-every", type=int, default=1)
    ap.add_argument("--pq-threshold", type=float, default=0.5)
    ap.add_argument("--pq-min-area", type=int, default=200, help="full-resolution pixels")
    ap.add_argument("--val-limit", type=int, default=0, help="score val PQ on the first N images only (0 = all)")
    ap.add_argument("--max-batches", type=int, default=0, help="debug: cap batches per epoch")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    paths = get_paths(args.data_root, args.work_dir)
    manifest = pd.read_csv(paths.manifest)
    build_targets(paths, manifest.observation_id.tolist(), workers=args.workers)
    if args.fold == "full":
        train_obs, val_obs = manifest.observation_id.tolist(), manifest.loc[manifest.fold == 0, "observation_id"].tolist()
    else:
        f = int(args.fold)
        train_obs = manifest.loc[manifest.fold != f, "observation_id"].tolist()
        val_obs = manifest.loc[manifest.fold == f, "observation_id"].tolist()
    pq_obs = val_obs[: args.val_limit] if args.val_limit else val_obs
    refs = labels.load_references(paths)

    name = args.name or f"unet_f{args.fold}"
    run_dir = unet_root(paths) / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    cfg = {"encoder": args.encoder, "encoder_weights": None if args.no_pretrained else "imagenet",
           "aspp": not args.no_aspp, "size": args.size}
    weights = LossWeights()
    (run_dir / "config.json").write_text(json.dumps({**cfg, **vars(args), "loss": asdict(weights)}, indent=2))
    print(f"[unet] fold {args.fold}: {len(train_obs)} train / {len(val_obs)} val images, device {device}")

    model = build_model(cfg).to(device)
    core = model
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    loader = DataLoader(FilamentDataset(paths, train_obs, args.size, augment=True), batch_size=args.batch,
                        shuffle=True, num_workers=args.workers, drop_last=len(train_obs) > args.batch,
                        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    steps = min(len(loader), args.max_batches or len(loader))
    total, warmup = args.epochs * steps, steps

    def lr_lambda(step):  # 1 epoch linear warmup, then cosine to 1% of the peak
        if step < warmup:
            return (step + 1) / warmup
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start, best, stale, history = 1, -1.0, 0, []
    state_path = run_dir / "last_state.pt"
    if args.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        core.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start, best, stale, history = st["epoch"] + 1, st["best"], st["stale"], st["history"]
        print(f"[unet] resumed after epoch {st['epoch']} (best val PQ {best:.4f})")

    for epoch in range(start, args.epochs + 1):
        model.train()
        t0, run, parts_sum = time.time(), 0.0, {}
        for i, (x, y) in enumerate(loader):
            if i >= steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(x)
            loss, parts = unet_loss(logits, y, weights)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss)
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
        row = {"epoch": epoch, "train_loss": run / steps, **{k: v / steps for k, v in parts_sum.items()},
               "lr": opt.param_groups[0]["lr"], "epoch_s": time.time() - t0}
        if epoch % args.pq_every == 0 or epoch == args.epochs:
            res = val_pq(core, paths, pq_obs, refs, args.size, device, args.pq_threshold, args.pq_min_area)
            row.update({f"val_{k}": res[k] for k in ("pq", "sq", "rq", "precision", "recall", "one_to_many_rate")})
            if res["pq"] > best:
                best, stale = res["pq"], 0
                save_weights(core, run_dir / "best.pt", cfg, {"epoch": epoch, "val_pq": best, "fold": args.fold})
            else:
                stale += args.pq_every
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        torch.save({"model": core.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "stale": stale,
                    "history": history}, state_path)
        if stale >= args.patience:
            print(f"[unet] early stop: no val PQ gain for {stale} epochs")
            break
    print(f"[unet] best val PQ {best:.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
