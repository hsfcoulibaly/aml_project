"""Unit tests for the PQ evaluator on hand-constructed cases."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fseg import rle as R  # noqa: E402
from fseg.metrics import evaluate, image_stats  # noqa: E402


def box(x0, y0, x1, y1, s=2048):
    m = np.zeros((s, s), bool)
    m[y0:y1, x0:x1] = True
    return R.encode(m)


def test_perfect():
    g = [box(0, 0, 100, 10), box(200, 200, 260, 220)]
    res = evaluate({"a": g}, {"a": [g]})
    assert abs(res["pq"] - 1.0) < 1e-9


def test_fp_fn():
    g = [box(0, 0, 100, 10), box(200, 200, 260, 220)]
    p = [box(0, 0, 100, 10), box(500, 500, 520, 520)]  # 1 TP (IoU 1), 1 FP, 1 FN
    res = evaluate({"a": p}, {"a": [g]})
    assert abs(res["pq"] - 1.0 / (1 + 0.5 + 0.5)) < 1e-9


def test_threshold_is_strict():
    g = [box(0, 0, 100, 10)]
    p = [box(0, 0, 50, 10)]  # IoU exactly 0.5 -> not a match
    st = image_stats(p, g)
    assert st["tp"] == 0 and st["fp"] == 1 and st["fn"] == 1


def test_fragmentation_counted():
    g = [box(0, 0, 100, 10)]
    p = [box(0, 0, 40, 10), box(60, 0, 100, 10)]
    st = image_stats(p, g)
    assert st["one_to_many"] == 1 and st["tp"] == 0
    st2 = image_stats([box(0, 0, 300, 10)], [box(0, 0, 100, 10), box(150, 0, 300, 10)])
    assert st2["many_to_one"] == 1


def test_average_over_annotators():
    g1 = [box(0, 0, 100, 10)]
    g2 = [box(0, 0, 100, 10), box(300, 300, 340, 320)]
    res = evaluate({"a": g1}, {"a": [g1, g2]})
    # set1: tp1 iou1; set2: tp1 fn1 -> pooled tp=1, fn=0.5 -> PQ = 1 / (1 + 0.25)
    assert abs(res["pq"] - 1 / 1.25) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
