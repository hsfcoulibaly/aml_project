"""Unit tests for Stage 2 targets, losses and instance extraction."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fseg import rle as R  # noqa: E402
from fseg.unet import cldice_loss, gaussian_window, prob_to_instances, soft_skeleton  # noqa: E402
from fseg.unet_data import dense_targets  # noqa: E402


def box(x0, y0, x1, y1, s=2048):
    m = np.zeros((s, s), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_soft_target_is_annotator_fraction():
    a, b = R.encode(box(100, 100, 300, 120)), R.encode(box(500, 500, 600, 520))
    t = dense_targets([[a, b], [a]], [{"rle": a}, {"rle": b}])
    assert t[110, 200, 0] == 255  # both annotators
    assert t[510, 550, 0] == 128  # one of two
    assert t[0, 0].sum() == 0


def test_centerline_and_boundary_follow_consensus():
    a = R.encode(box(100, 100, 300, 120))
    t = dense_targets([[a]], [{"rle": a}])
    assert t[110, 200, 1] == 255  # skeleton runs along the middle row
    assert t[100, 200, 2] == 255 and t[110, 200, 2] == 0  # boundary on the edge, not inside


def test_soft_skeleton_and_cldice():
    x = torch.zeros(1, 1, 64, 64)
    x[0, 0, 30:35, 5:60] = 1.0
    sk = soft_skeleton(x, iters=5)
    assert sk[0, 0, 32, 30] > 0.5 and sk[0, 0, 10, 30] == 0
    assert float(cldice_loss(x, x, iters=5)) < 1e-3
    assert float(cldice_loss(torch.zeros_like(x), x, iters=5)) > 0.9


def test_gaussian_window_peaks_in_centre():
    w = gaussian_window(64)
    assert w.max() == w[31:33, 31:33].max() and w[0, 0] < 0.2


def test_prob_to_instances_threshold_area_score():
    prob = np.zeros((2048, 2048), np.float32)
    prob[100:120, 100:300] = 0.9  # area 4000
    prob[500:505, 500:505] = 0.9  # area 25, dropped by min_area
    prob[900:920, 900:1000] = 0.4  # below threshold
    out = prob_to_instances(prob, thr=0.5, min_area=100)
    assert len(out) == 1 and abs(out[0][1] - 0.9) < 1e-6 and R.area([out[0][0]])[0] == 4000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
