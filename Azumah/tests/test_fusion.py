"""Unit tests for stage-4 fusion: fragment merging, recovery and the shape filter."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fseg import rle as R  # noqa: E402
from fseg.fusion import FusionParams, elongation, fuse, merge_fragments, recover, skeleton_info  # noqa: E402

S = 2048


def bar(x0, y0, x1, y1, w=8):
    m = np.zeros((S, S), np.uint8)
    cv2.line(m, (x0, y0), (x1, y1), 1, w)
    return m.astype(bool)


def test_endpoints_point_outward():
    _, ends = skeleton_info(bar(100, 500, 300, 500))
    assert len(ends) == 2
    left, right = sorted(ends, key=lambda e: e[0][1])
    assert left[1][1] < -0.9 and right[1][1] > 0.9  # unit vectors pointing away from the bar


def test_collinear_fragments_merge_across_supported_gap():
    a, b = bar(100, 500, 300, 500), bar(340, 500, 540, 500)
    prob = (bar(100, 500, 540, 500, w=10)).astype(np.float32) * 0.8
    m, s = merge_fragments([a, b], [0.9, 0.5], prob, FusionParams(merge_dist=60))
    assert len(m) == 1 and m[0][500, 320]  # gap bridged
    assert abs(s[0] - 0.7) < 1e-6  # area-weighted score


def test_no_merge_without_unet_support_or_when_perpendicular():
    a, b = bar(100, 500, 300, 500), bar(340, 500, 540, 500)
    m, _ = merge_fragments([a, b], [0.9, 0.5], np.zeros((S, S), np.float32), FusionParams())
    assert len(m) == 2
    c = bar(330, 520, 330, 700)  # starts near a's end but runs perpendicular to it
    prob = np.ones((S, S), np.float32)
    m, _ = merge_fragments([a, c], [0.9, 0.5], prob, FusionParams(merge_angle=30))
    assert len(m) == 2


def test_recover_adds_only_long_uncovered_components():
    prob = np.zeros((S, S), np.float32)
    prob[bar(100, 100, 400, 100)] = 0.9  # long, uncovered -> recovered
    prob[bar(100, 900, 130, 900)] = 0.9  # too short
    prob[bar(100, 1500, 400, 1500)] = 0.9  # already covered by an instance
    got = recover([bar(100, 1500, 400, 1500)], prob, None, FusionParams(recover_thr=0.5, recover_len=150))
    assert len(got) == 1 and got[0][0][100, 250] and abs(got[0][1] - 0.9 * 0.6) < 1e-5


def test_shape_filter_drops_blobs_keeps_filaments():
    blob = np.zeros((S, S), np.uint8)
    cv2.circle(blob, (800, 800), 30, 1, -1)
    fil = bar(100, 300, 500, 340)
    assert elongation(fil) > 3 > elongation(blob.astype(bool))
    out = fuse([(R.encode(blob.astype(bool)), 0.9), (R.encode(fil), 0.8)], None, None,
               FusionParams(merge_dist=0, min_elongation=3))
    assert len(out) == 1 and out[0][1] == 0.8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
