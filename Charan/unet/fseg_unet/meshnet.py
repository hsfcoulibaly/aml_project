"""2D MeshNet: a plain stack of dilated 3x3 convolutions at full resolution (no downsampling, no skips).

Adapted from the neuroneural/catalyst_example `modelAE_hdc_deep.json` (3D, 256^3 brains): same coprime dilation
ramp 1,3,5,7,13,19,31,19,13,7,5,3,1 + a 1x1 head, so the receptive field is 1 + 2 * 127 = 255 px. With no gaps in
the dilation spectrum (all odd, gcd 1, a dilation-1 layer present) the aggregate receptive field has no gridding
holes. A deep stack this narrow has no skip connections to carry gradients, so every block uses per-channel
GroupNorm with a learned affine (use_affine=True in the original recipe), GELU and Kaiming init.
"""
from __future__ import annotations

import json
from pathlib import Path

from torch import nn

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "meshnet_hdc_deep.json"


def load_dilations(config: Path = DEFAULT_CONFIG) -> list[int]:
    layers = json.loads(Path(config).read_text())["layers"]
    return [int(layer["dilation"]) for layer in layers if layer["kernel_size"] == 3]


def receptive_field(dilations) -> int:
    return 1 + 2 * sum(dilations)


class MeshNet2d(nn.Module):
    def __init__(self, in_channels: int = 3, channels: int = 32, n_classes: int = 1, dilations=None):
        super().__init__()
        dilations = list(dilations or load_dilations())
        blocks, c_in = [], in_channels
        for d in dilations:
            blocks += [nn.Conv2d(c_in, channels, 3, padding=d, dilation=d),
                       nn.GroupNorm(channels, channels, affine=True),  # one group per channel
                       nn.GELU()]
            c_in = channels
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv2d(channels, n_classes, 1)
        self.dilations = dilations
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.body(x))
