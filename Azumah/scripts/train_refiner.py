"""Stage 3: train the instance-conditioned refiner.

Trains on every fold except --val-fold, using out-of-fold YOLO proposals (`runs/consensus_f<k>/
oof_raw_fold<k>.pkl`) and out-of-fold U-Net maps (`unet/maps/<unet-name>/oof/`). All crops are cached in
memory once (fseg.refiner_cache), so epochs are GPU-bound. The checkpoint with the lowest validation loss
on the held-out fold is kept; PQ is measured by predict_refiner.py.

Architectures: `unet` (smp U-Net, ResNet-34 encoder, ~24M parameters) or `meshnet` (2D MeshNet with the
hdc_deep dilation schedule, a few thousand parameters). `--recipe catalyst` follows the training recipe of
neuroneural/catalyst_example for small models: AdamW (weight decay 1e-2), OneCycle with 5% warmup to a high
peak LR, gradient clipping at 1.0, gradient accumulation, label smoothing 0.01 and generalized Dice.

    python scripts/train_refiner.py                                         # ResNet-34 U-Net, fold 0 held out
    python scripts/train_refiner.py --arch meshnet --channels 8 --recipe catalyst --lr 0.01 --epochs 150
    python scripts/train_refiner.py --val-fold none                         # final refiner on all folds
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
from fseg.refiner import RefinerLossWeights, make_refiner, refiner_loss, save_refiner  # noqa: E402
from fseg.refiner_cache import CachedRefinerDataset, build_cache, mix_sampler  # noqa: E402
from fseg.refiner_data import load_proposals  # noqa: E402
from fseg.unet import pick_device  # noqa: E402


def to_input(x: torch.Tensor, device) -> torch.Tensor:
    return ((x.to(device, non_blocking=True).float() / 255.0) - 0.5) / 0.25


def run_epoch(model, loader, device, opt, sched, scaler, weights, train, accum=1, clip=0.0, max_batches=0):
    model.train(train)
    tot, n, parts_sum = 0.0, 0, {}
    use_amp = device.type == "cuda"
    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y = to_input(x, device), y.to(device).float()
            with torch.autocast(device.type, enabled=use_amp):
                mask, q = model(x)
            loss, parts = refiner_loss(mask, q, y, weights)
            if train:
                scaler.scale(loss / accum).backward()
                if (i + 1) % accum == 0:
                    if clip:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
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
    ap.add_argument("--arch", default="unet", choices=["unet", "meshnet"])
    ap.add_argument("--encoder", default="resnet34", help="unet only")
    ap.add_argument("--channels", type=int, default=8, help="meshnet width")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--recipe", default="default", choices=["default", "catalyst"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples", type=int, default=8000, help="crops drawn per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=None, help="gradient accumulation (catalyst default 2)")
    ap.add_argument("--lr", type=float, default=None, help="default 3e-4; catalyst default 1e-2 (OneCycle peak)")
    ap.add_argument("--max-matched", type=int, default=20)
    ap.add_argument("--max-neg", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--patience", type=int, default=0, help="0 = no early stopping")
    ap.add_argument("--max-batches", type=int, default=0, help="debug: cap batches per epoch")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    catalyst = args.recipe == "catalyst"
    lr = args.lr or (1e-2 if catalyst else 3e-4)
    accum = args.accum or (2 if catalyst else 1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    paths = get_paths(args.data_root, args.work_dir)
    manifest = pd.read_csv(paths.manifest)
    val_fold = None if args.val_fold == "none" else int(args.val_fold)
    train_obs = manifest.loc[manifest.fold != val_fold, "observation_id"].tolist()
    val_obs = manifest.loc[manifest.fold == (0 if val_fold is None else val_fold), "observation_id"].tolist()

    consensus = labels.load_targets(paths)["consensus"]
    proposals = load_proposals(paths, sorted(manifest.fold.unique()))
    map_dir = paths.work_dir / "unet" / "maps" / args.unet_name / "oof"
    if not map_dir.exists():
        print(f"[refiner] no U-Net maps in {map_dir}: those channels are zeros")
        map_dir = None
    t0 = time.time()
    tr_cache = build_cache(paths, train_obs, consensus, proposals, map_dir, args.max_matched, args.max_neg, args.workers)
    va_cache = build_cache(paths, val_obs, consensus, proposals, map_dir, args.max_matched, args.max_neg, args.workers)
    counts = np.bincount(tr_cache["kind"], minlength=3)
    print(f"[refiner] cache: {len(train_obs)} train images -> {counts.tolist()} gt/matched/negative crops, "
          f"{len(va_cache['kind'])} val crops, {tr_cache['img4'].nbytes / 2**30:.1f} GB, {time.time() - t0:.0f} s")

    name = args.name or ("refiner_full" if val_fold is None else f"refiner_v{val_fold}")
    run_dir = paths.work_dir / "refiner" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    cfg = {"arch": args.arch, "encoder": args.encoder, "channels": args.channels, "unet_name": args.unet_name}
    weights = (RefinerLossWeights(bce=0.5, dice=0.5, label_smoothing=0.01, generalized_dice=True) if catalyst
               else RefinerLossWeights())
    model = make_refiner(cfg, pretrained=not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    (run_dir / "config.json").write_text(json.dumps({**cfg, **vars(args), "lr": lr, "accum": accum,
                                                     "params": n_params, "loss": asdict(weights)}, indent=2))
    print(f"[refiner] {args.arch}: {n_params:,} parameters, recipe {args.recipe}, lr {lr}, accum {accum}")

    train_dl = DataLoader(CachedRefinerDataset(tr_cache, augment=True), batch_size=args.batch,
                          sampler=mix_sampler(tr_cache["kind"], num_samples=args.samples), num_workers=args.workers,
                          drop_last=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(CachedRefinerDataset(va_cache, augment=False), batch_size=args.batch, shuffle=False,
                        num_workers=args.workers)
    steps = math.ceil(min(len(train_dl), args.max_batches or len(train_dl)) / accum)
    if catalyst:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=args.epochs * steps + 1,
                                                    pct_start=0.05)
    else:
        total, warmup = args.epochs * steps, steps
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                                  0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warmup) /
                                                                                     max(1, total - warmup))))
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
        tr, trp = run_epoch(model, train_dl, device, opt, sched, scaler, weights, True, accum,
                            1.0 if catalyst else 0.0, args.max_batches)
        va, vap = run_epoch(model, val_dl, device, opt, sched, scaler, weights, False, max_batches=args.max_batches)
        row = {"epoch": epoch, "train_loss": tr, **{f"train_{k}": v for k, v in trp.items()},
               "val_loss": va, **{f"val_{k}": v for k, v in vap.items()}, "lr": opt.param_groups[0]["lr"],
               "epoch_s": time.time() - t0}
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
        if args.patience and stale >= args.patience:
            print(f"[refiner] early stop after {epoch} epochs")
            break
    print(f"[refiner] best val loss {best:.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
