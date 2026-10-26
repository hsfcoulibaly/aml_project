"""PQ unit tests: python tests/test_pq.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fseg_unet import pq  # noqa: E402
from fseg_unet.postprocess import predict_instances  # noqa: E402

S = 64


def box(y0, y1, x0, x1):
    m = np.zeros((S, S), np.uint8)
    m[y0:y1, x0:x1] = 1
    return pq.encode(m)


def test_perfect_match():
    g = [box(0, 10, 0, 10), box(20, 30, 20, 40)]
    r = pq.evaluate({"a": g}, {"a": [g]})
    assert abs(r["pq"] - 1.0) < 1e-9 and r["precision"] == 1 and r["recall"] == 1


def test_empty_prediction():
    r = pq.evaluate({}, {"a": [[box(0, 10, 0, 10)]]})
    assert r["pq"] == 0 and r["recall"] == 0


def test_partial_and_fp():
    gt = [box(0, 10, 0, 10)]
    pred = [box(0, 10, 0, 8), box(40, 50, 40, 50)]  # IoU 0.8 match + one FP
    r = pq.evaluate({"a": pred}, {"a": [gt]})
    assert abs(r["pq"] - 0.8 / (1 + 0.5)) < 1e-9


def test_iou_must_exceed_half():
    gt = [box(0, 10, 0, 10)]
    pred = [box(0, 10, 0, 5)]  # IoU exactly 0.5 -> not a match
    r = pq.evaluate({"a": pred}, {"a": [gt]})
    assert r["pq"] == 0


def test_average_over_annotators():
    a1 = [box(0, 10, 0, 10)]
    a2 = [box(0, 10, 0, 10), box(30, 40, 30, 40)]
    r = pq.evaluate({"a": a1}, {"a": [a1, a2]})
    # set 1: PQ 1 (tp 1); set 2: tp 1, fn 1 -> pooled: iou 2, tp 2, fn 1 -> 2 / 2.5
    assert abs(r["pq"] - 2 / 2.5) < 1e-9


def test_split_and_merge_rates():
    gt = [box(0, 10, 0, 40)]
    pred = [box(0, 10, 0, 18), box(0, 10, 22, 40)]  # one GT split into two
    r = pq.evaluate({"a": pred}, {"a": [gt]})
    assert r["split_rate"] == 1.0 and r["merge_rate"] == 0.0
    r = pq.evaluate({"a": [box(0, 10, 0, 40)]}, {"a": [[box(0, 10, 0, 18), box(0, 10, 22, 40)]]})
    assert r["merge_rate"] == 1.0


def test_postprocess_components():
    prob = np.zeros((128, 128), np.float32)
    prob[10:20, 10:60] = 0.9
    prob[80:82, 80:82] = 0.9  # speck: ~1,000 px once upsampled 16x to 2048, removed by min_area
    inst = predict_instances(prob, thr=0.5, min_area=5000)
    assert len(inst) == 1 and inst[0]["size"] == [2048, 2048]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
