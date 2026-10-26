"""Stage 2: tune post-processing on validation PQ, then predict the test set.

    python scripts/infer.py sweep --weights work/runs/unet_r34/best.pt [--tta]
    python scripts/infer.py test  --weights work/runs/unet_r34/best.pt --config work/runs/unet_r34/pq_config.json

`sweep` writes pq_sweep.csv, pq_config.json, val_per_image.csv, val_dice_per_record.csv next to the weights.
`test` writes submission.csv (filament_id, segmentation_rle) in the competition format.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fseg_unet import evaluate as E  # noqa: E402
from fseg_unet import labels, pq  # noqa: E402
from fseg_unet.config import get_paths  # noqa: E402
from fseg_unet.model import load_weights, predict_probs  # noqa: E402
from fseg_unet.postprocess import predict_instances  # noqa: E402

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_AREAS = [25, 100, 200, 400, 800, 1600]


def sweep(args, paths, manifest, refs, model, device):
    run_dir = Path(args.weights).parent
    val_obs = sorted(manifest.loc[manifest.split == "val", "observation_id"].unique())
    probs = predict_probs(model, paths, val_obs, device, args.batch, args.workers, tta=args.tta)
    disks = E.disks_from_manifest(manifest)

    table, best = E.pq_sweep(probs, disks, refs, THRESHOLDS, MIN_AREAS)
    table.to_csv(run_dir / "pq_sweep.csv", index=False)
    print(table.head(10).to_string(index=False))

    preds = {o: predict_instances(probs[o], best["threshold"], best["min_area"], disks[o]) for o in val_obs}
    res = pq.evaluate(preds, {o: refs["sets"][o] for o in val_obs})
    res["per_image"].to_csv(run_dir / "val_per_image.csv", index=False)
    dice = E.dice_table(probs, E.dice_targets(refs, val_obs), refs, thr=best["threshold"])
    dice.to_csv(run_dir / "val_dice_per_record.csv", index=False)
    ceiling = pq.annotator_ceiling({o: refs["sets"][o] for o in val_obs})

    config = {**best, "tta": args.tta, "val": {k: res[k] for k in pq.SUMMARY_KEYS},
              "val_dice_mean": float(dice.dice.mean()), "val_dice_median": float(dice.dice.median()),
              "val_iou_mean": float(dice.iou.mean()), "annotator_ceiling_pq": ceiling["pq"]}
    (run_dir / "pq_config.json").write_text(json.dumps(config, indent=2))
    print("\nbest:", pq.summary(res))
    print(f"Dice@512 mean {dice.dice.mean():.4f} median {dice.dice.median():.4f} | IoU mean {dice.iou.mean():.4f}")
    print(f"annotator ceiling PQ {ceiling['pq']:.4f}")
    print("config ->", run_dir / "pq_config.json")


def test(args, paths, manifest, model, device):
    cfg = json.loads(Path(args.config).read_text())
    tta = cfg.get("tta", False) if args.tta is None else args.tta
    test_obs = sorted(manifest.loc[manifest.split == "test", "observation_id"].unique())
    probs = predict_probs(model, paths, test_obs, device, args.batch, args.workers, tta=tta)
    disks = E.disks_from_manifest(manifest)
    rows = []
    for obs in test_obs:
        for k, rle in enumerate(predict_instances(probs[obs], cfg["threshold"], cfg["min_area"], disks[obs]), 1):
            rows.append({"filament_id": f"{obs}_{k}", "segmentation_rle": pq.counts_str(rle)})
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    assert sub.filament_id.is_unique
    out = Path(args.out)
    sub.to_csv(out, index=False)
    n_img = sub.filament_id.str.rsplit("_", n=1).str[0].nunique()
    print(f"{len(sub)} filaments over {n_img}/{len(test_obs)} test images -> {out.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "test"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", help="pq_config.json from sweep (test mode)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--tta", action=argparse.BooleanOptionalAction, default=None,
                    help="flip TTA (sweep default off; test default: whatever the sweep used)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    paths = get_paths(args.data_root, args.work_dir)
    manifest = labels.load_manifest(paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_weights(Path(args.weights), device)
    if args.mode == "sweep":
        args.tta = bool(args.tta)
        sweep(args, paths, manifest, labels.load_references(paths), model, device)
    else:
        if not args.config:
            ap.error("test mode needs --config")
        test(args, paths, manifest, model, device)


if __name__ == "__main__":
    main()
