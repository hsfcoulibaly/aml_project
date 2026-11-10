"""Average trained models, tune a stricter keep rule on validation PQ, write test submissions.

    python scripts/ensemble.py \
        --model unet_week=.../best.pt --model meshnet_week=.../best.pt --model unet_r1=.../best.pt \
        --leaky unet_r1 \
        --calib unet_week:0.7:400:0.35 --calib meshnet_week:0.7:400:0.33 \
        --blend ens2=unet_week:1,meshnet_week:1 --blend ens3=unet_week:1,meshnet_week:1,unet_r1:1 \
        --rules-from ens3=ens2 --out-dir /kaggle/working/ensemble

Needs a work dir prepared with the validation split the honest models were trained against
(`prepare.py --split splits/week_split.json`). A model listed in --leaky was trained on some of those
validation images, so blends that contain it are not swept; they reuse the rules chosen for the honest
blend named in --rules-from.

Outputs in --out-dir: sweep_<blend>.csv, summary.json (r, chosen rules, full validation metrics), and
submission_<blend>_<pq|lb>.csv: "pq" is the rule with the best validation PQ, "lb" the rule with the
best estimated public score PQ'(r) (see fseg_unet/ensemble.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fseg_unet import ensemble as ens  # noqa: E402
from fseg_unet import evaluate as E  # noqa: E402
from fseg_unet import labels, pq  # noqa: E402
from fseg_unet.config import get_paths  # noqa: E402
from fseg_unet.model import load_weights  # noqa: E402

T_LOS = [0.4, 0.5, 0.6, 0.7, 0.8]
CLOSES = [0, 6]  # closing radius in 2048 px (bridges gaps of up to ~12 px)
T_HIS = [0.7, 0.8, 0.9, 0.95]
MIN_MEANS = [0.0, 0.6, 0.7, 0.8]
MIN_AREAS = [200, 400, 800, 1200, 1600, 2400, 3200]


def parse_weights(spec: str) -> dict:
    return {name: float(w) for name, w in (p.split(":") for p in spec.split(","))}


def write_submission(path: Path, test_obs, prob_of, disks, rule: ens.KeepRule) -> int:
    rows = []
    for obs in test_obs:
        c = ens.candidates(prob_of(obs), disks[obs], rule.t_lo, rule.close, rule.min_area)
        for k, rle in enumerate(ens.apply_rule(c, rule), 1):
            rows.append({"filament_id": f"{obs}_{k}", "segmentation_rle": pq.counts_str(rle)})
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    assert sub.filament_id.is_unique
    sub.to_csv(path, index=False)
    n_img = sub.filament_id.str.rsplit("_", n=1).str[0].nunique()
    print(f"  {path.name}: {len(sub)} filaments on {n_img}/{len(test_obs)} test images")
    return len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=path/to/best.pt")
    ap.add_argument("--leaky", action="append", default=[], help="model trained on part of this val split")
    ap.add_argument("--calib", action="append", default=[], help="NAME:THR:MIN_AREA:PUBLIC_PQ of a past submission")
    ap.add_argument("--blend", action="append", default=[], help="LABEL=name:w,name:w")
    ap.add_argument("--rules-from", action="append", default=[], help="LEAKY_BLEND=HONEST_BLEND")
    ap.add_argument("--tta", default="d4", choices=list(ens.TTA_VIEWS))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--out-dir", default="ensemble_out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = get_paths(args.data_root, args.work_dir)
    manifest = labels.load_manifest(paths)
    refs = labels.load_references(paths)
    disks = E.disks_from_manifest(manifest)
    val_obs = sorted(manifest.loc[manifest.split == "val", "observation_id"].unique())
    test_obs = sorted(manifest.loc[manifest.split == "test", "observation_id"].unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. probability maps (uint8) for every model on val + test
    maps = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        model, meta = load_weights(Path(path), device)
        print(f"{name}: {meta.get('arch', 'unet')} from {path}")
        maps[name] = ens.predict_u8(model, paths, val_obs + test_obs, device, args.tta, args.batch, args.workers)
        del model
        torch.cuda.empty_cache()

    blends = {label: parse_weights(w) for label, w in (b.split("=", 1) for b in args.blend)}
    calib = [(n, float(t), int(a), float(lb)) for n, t, a, lb in (c.split(":") for c in args.calib)]
    for n, *_ in calib:  # calibration needs each calibrated model on its own
        blends.setdefault(n, {n: 1.0})
    rules_from = dict(r.split("=", 1) for r in args.rules_from)
    leaky = set(args.leaky)
    honest = {b: w for b, w in blends.items() if not leaky & set(w)}
    for b in set(blends) - set(honest):
        if b not in rules_from:
            ap.error(f"blend {b} contains a leaky model; give --rules-from {b}=<honest blend>")

    # 2. candidate instances + GT matches on val for every honest blend
    caches = {}
    for b, w in honest.items():
        print(f"\nbuilding val candidates for {b} = {w}")
        probs = {o: ens.blend(maps, w, o) for o in val_obs}
        caches[b] = ens.build_cache(probs, disks, refs, T_LOS, CLOSES, min(MIN_AREAS), args.workers)
        del probs

    # 3. fit the test-label sparsity r on past submissions
    known = [(ens.counts(caches[n], ens.KeepRule.single(t, a)), lb) for n, t, a, lb in calib]
    r = ens.fit_r(known) if known else 1.0
    print(f"\nfitted r = {r:.3f}")
    for (n, t, a, lb), (k, _) in zip(calib, known):
        print(f"  {n} thr {t} min_area {a}: val PQ {ens.pq_from_counts(k):.4f}, "
              f"PQ'(r) {ens.pq_from_counts(k, r):.4f}, public {lb:.2f}")

    # 4. sweep keep rules
    rules = ens.rule_grid(T_LOS, CLOSES, T_HIS, MIN_MEANS, MIN_AREAS)
    chosen, summary = {}, {"r": r, "tta": args.tta, "blends": {}}
    for b in honest:
        table = ens.sweep(caches[b], rules, r)
        table.to_csv(out / f"sweep_{b}.csv", index=False)
        pick = {}
        for col, tag in [("pq", "pq"), ("pq_lb_est", "lb")]:
            row = table.sort_values(col, ascending=False).iloc[0]
            pick[tag] = ens.KeepRule(float(row.t_lo), float(row.t_hi), float(row.min_mean), int(row.min_area),
                                     int(row.close))
        chosen[b] = pick
        print(f"\n{b}: top rules by estimated public PQ")
        print(table.sort_values("pq_lb_est", ascending=False).head(8).to_string(index=False))
        blend_summary = {}
        for tag, rule in pick.items():
            preds = {o: ens.apply_rule(caches[b][o][(rule.t_lo, rule.close)][0], rule) for o in val_obs}
            res = pq.evaluate(preds, {o: refs["sets"][o] for o in val_obs})
            k = ens.counts(caches[b], rule)
            blend_summary[tag] = {"rule": asdict(rule), "val": {kk: res[kk] for kk in pq.SUMMARY_KEYS},
                                  "pq_lb_est": ens.pq_from_counts(k, r)}
            print(f"  [{tag}] {rule}\n       {pq.summary(res)} | PQ'(r) {ens.pq_from_counts(k, r):.4f}")
        summary["blends"][b] = {"weights": blends[b], **blend_summary}
    for b, src in rules_from.items():
        chosen[b] = chosen[src]
        summary["blends"][b] = {"weights": blends[b], "rules_from": src}
    del caches

    # 5. test submissions
    print("\nsubmissions")
    for b, pick in chosen.items():
        for tag, rule in pick.items():
            n = write_submission(out / f"submission_{b}_{tag}.csv", test_obs,
                                 lambda o, w=blends[b]: ens.blend(maps, w, o), disks, rule)
            summary["blends"][b].setdefault("test_rows", {})[tag] = n
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("summary ->", out / "summary.json")


if __name__ == "__main__":
    main()
