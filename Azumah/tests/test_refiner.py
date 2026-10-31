"""Unit tests for the refiner's crop geometry, prior corruption and loss."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fseg.refiner import CROP, RefinerLossWeights, bbox, corrupt_prior, crop_affine, paste, refiner_loss, warp  # noqa: E402


def box(x0, y0, x1, y1, s=2048):
    m = np.zeros((s, s), bool)
    m[y0:y1, x0:x1] = True
    return m


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_crop_then_paste_round_trip():
    m = box(400, 700, 700, 760)
    M = crop_affine(bbox(m))
    c = warp(m.astype(np.float32), M)
    assert c.shape == (CROP, CROP) and c.max() == 1
    assert iou(paste(c, M), m) > 0.95


def test_crop_has_context_and_min_side():
    M = crop_affine((1000, 1000, 1010, 1004))  # tiny filament -> 128 px window
    assert abs(M[0, 0] - CROP / 128) < 1e-6
    M = crop_affine((0, 0, 400, 100))  # 1.5x context around the long side
    assert abs(M[0, 0] - CROP / 600) < 1e-6


def test_paste_clips_at_image_border():
    m = box(0, 0, 40, 20)
    M = crop_affine(bbox(m))
    out = paste(warp(m.astype(np.float32), M), M)
    assert out.shape == (2048, 2048) and iou(out, m) > 0.8


def test_corrupt_prior_stays_near_truth():
    rng = np.random.default_rng(0)
    t = np.zeros((CROP, CROP), bool)
    t[120:136, 40:216] = True
    ious = [iou(corrupt_prior(t, np.zeros_like(t), rng), t) for _ in range(50)]
    assert 0.2 < np.median(ious) < 0.95  # corrupted, but still a prior for the same filament


def test_quality_target_is_zero_for_negatives():
    logits = torch.full((2, 1, 8, 8), 5.0)  # predicts filament everywhere
    target = torch.zeros(2, 1, 8, 8)
    target[0, 0, :4] = 1  # sample 0 positive (IoU 0.5), sample 1 negative
    q_logits = torch.tensor([0.0, 0.0])  # predicted quality 0.5 for both
    _, parts = refiner_loss(logits, q_logits, target, RefinerLossWeights())
    # smooth L1 (beta 0.1): |0.5 - 0.5| = 0 for sample 0, |0.5 - 0| = 0.5 -> 0.45 for sample 1
    assert abs(parts["quality"] - 0.225) < 1e-4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
