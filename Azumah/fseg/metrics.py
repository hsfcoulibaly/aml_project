"""Local Panoptic Quality, following the organizers' definition.

    PQ = sum_{(y, y_hat) in TP} IoU(y, y_hat) / (|TP| + 0.5 |FP| + 0.5 |FN|)

A prediction and a ground-truth segment form a TP when IoU > 0.5 (such matches are unique, so no
assignment problem). Unmatched predictions are FP, unmatched ground truth is FN.

Because the hidden test labels are one annotator's set, we score each validation observation
against every available annotation set and average, which estimates the expected leaderboard PQ.
We report both the dataset-level PQ (sums pooled over images, the standard definition) and the mean
of per-image PQ, since the exact Kaggle aggregation is not published.

Fragmentation and over-merging are reported as well, because the final rubric weighs them:
* one_to_many: share of GT filaments split across >= 2 predictions (each lying mostly inside it)
* many_to_one: share of predictions that swallow >= 2 GT filaments (each mostly inside it)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import rle as R

MATCH_IOU = 0.5


def image_stats(preds: list[dict], gts: list[dict], cover: float = 0.5) -> dict:
    p, t = len(preds), len(gts)
    iou = R.iou_matrix(preds, gts)
    matched = iou > MATCH_IOU
    tp = int(matched.sum())
    iou_sum = float(iou[matched].sum())

    one_to_many = many_to_one = 0
    if p and t:
        inter = R.intersection_matrix(preds, gts)
        pa, ga = R.area(preds), R.area(gts)
        pred_inside_gt = inter / pa[:, None].clip(min=1) >= cover  # pred mostly inside gt
        gt_inside_pred = inter / ga[None, :].clip(min=1) >= cover  # gt mostly inside pred
        one_to_many = int((pred_inside_gt.sum(0) >= 2).sum())
        many_to_one = int((gt_inside_pred.sum(1) >= 2).sum())

    return {
        "tp": tp,
        "fp": p - tp,
        "fn": t - tp,
        "iou_sum": iou_sum,
        "n_pred": p,
        "n_gt": t,
        "one_to_many": one_to_many,
        "many_to_one": many_to_one,
    }


def _pq(row) -> float:
    denom = row["tp"] + 0.5 * row["fp"] + 0.5 * row["fn"]
    return row["iou_sum"] / denom if denom > 0 else np.nan


def evaluate(predictions: dict, references: dict, observations=None) -> dict:
    """predictions: {obs: [rle]}, references: {obs: [[rle] per annotation set]}."""
    observations = list(references) if observations is None else list(observations)
    per_image = []
    for obs in observations:
        preds = predictions.get(obs, [])
        stats = [image_stats(preds, gts) for gts in references[obs]]
        avg = pd.DataFrame(stats).mean().to_dict()  # expectation over annotators
        avg["observation_id"] = obs
        per_image.append(avg)
    df = pd.DataFrame(per_image)
    df["pq"] = df.apply(_pq, axis=1)

    tot = df[["tp", "fp", "fn", "iou_sum", "n_pred", "n_gt", "one_to_many", "many_to_one"]].sum()
    pq = _pq(tot)
    sq = tot["iou_sum"] / tot["tp"] if tot["tp"] else 0.0
    rq = tot["tp"] / (tot["tp"] + 0.5 * tot["fp"] + 0.5 * tot["fn"]) if (tot["tp"] + tot["fp"] + tot["fn"]) else 0.0
    return {
        "pq": float(pq),
        "pq_image_mean": float(df["pq"].mean(skipna=True)),
        "sq": float(sq),
        "rq": float(rq),
        "precision": float(tot["tp"] / tot["n_pred"]) if tot["n_pred"] else 0.0,
        "recall": float(tot["tp"] / tot["n_gt"]) if tot["n_gt"] else 0.0,
        "pred_per_image": float(df["n_pred"].mean()),
        "gt_per_image": float(df["n_gt"].mean()),
        "one_to_many_rate": float(tot["one_to_many"] / tot["n_gt"]) if tot["n_gt"] else 0.0,
        "many_to_one_rate": float(tot["many_to_one"] / tot["n_pred"]) if tot["n_pred"] else 0.0,
        "per_image": df,
    }


def summary(result: dict) -> str:
    keys = ["pq", "pq_image_mean", "sq", "rq", "precision", "recall", "pred_per_image", "gt_per_image",
            "one_to_many_rate", "many_to_one_rate"]
    return " | ".join(f"{k} {result[k]:.4f}" for k in keys)


def annotator_ceiling(references: dict) -> dict:
    """PQ of one annotator scored against the others, on observations with >= 2 sets.

    This is the human-agreement ceiling to report next to model scores.
    """
    preds, refs = {}, {}
    for obs, sets in references.items():
        if len(sets) < 2:
            continue
        for i, s in enumerate(sets):
            key = f"{obs}#{i}"
            preds[key] = s
            refs[key] = [o for j, o in enumerate(sets) if j != i]
    return evaluate(preds, refs)
