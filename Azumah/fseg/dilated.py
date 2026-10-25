"""Zero-initialized dilated residual branches for the YOLO neck (the group's dilation idea, done safely).

Changing the dilation of existing pretrained convolutions changes what those weights compute. Instead
we wrap selected neck blocks:

    y = block(x)
    y = y + Proj( concat( DWConv3x3_d1(y), DWConv3x3_d2(y), DWConv3x3_d5(y) ) )

`Proj` is zero-initialized, so at step 0 the network is exactly the pretrained YOLO and training
decides how much long-range context to add. Rates (1, 2, 5) follow hybrid dilated convolution to
avoid gridding. Depthwise convolutions keep the cost small.

Which layers: by default the P4 and P5 inputs of the Segment head (strides 16 and 32), where a larger
receptive field helps join long filaments. Control with env var FSEG_DILATED = "p4,p5" | "p3,p4,p5" | "".
The env var is used (instead of a trainer argument) because Ultralytics rejects unknown arguments and
re-launches a fresh process for multi-GPU training; environment variables survive that relaunch.
"""
from __future__ import annotations

import os

import torch
from torch import nn


class DilatedContext(nn.Module):
    def __init__(self, channels: int, rates=(1, 2, 5)):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=r, dilation=r, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
            for r in rates
        )
        self.proj = nn.Conv2d(channels * len(rates), channels, 1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, y):
        return y + self.proj(torch.cat([b(y) for b in self.branches], 1))


class WithDilatedContext(nn.Module):
    """Wraps an Ultralytics layer and keeps the routing attributes its graph executor needs."""

    def __init__(self, block: nn.Module, channels: int, rates=(1, 2, 5)):
        super().__init__()
        self.block = block
        self.context = DilatedContext(channels, rates)
        for attr in ("f", "i", "type", "np"):
            if hasattr(block, attr):
                setattr(self, attr, getattr(block, attr))
        self.type = f"WithDilatedContext({getattr(block, 'type', block.__class__.__name__)})"

    def forward(self, x):
        return self.context(self.block(x))


def _head_in_channels(head: nn.Module, k: int) -> int:
    """Channels of the k-th feature map entering the Detect/Segment head (exact, from its first conv)."""
    first = head.cv2[k][0]
    conv = getattr(first, "conv", first)
    return conv.in_channels


def inject(model: nn.Module, levels: list[str], rates=(1, 2, 5), verbose: bool = True) -> list[int]:
    """Wrap the layers feeding the Segment/Detect head at the requested pyramid levels."""
    layers = model.model  # nn.Sequential of Ultralytics layers
    head = layers[-1]
    feeders = list(head.f)  # e.g. [16, 19, 22] -> P3, P4, P5
    level_index = {f"p{3 + k}": (k, idx) for k, idx in enumerate(feeders)}
    wrapped = []
    for lvl in levels:
        k, idx = level_index[lvl]
        block = layers[idx]
        if isinstance(block, WithDilatedContext):
            continue
        layers[idx] = WithDilatedContext(block, _head_in_channels(head, k), rates)
        wrapped.append(idx)
    if verbose and wrapped:
        n = sum(p.numel() for i in wrapped for p in layers[i].context.parameters())
        print(f"[fseg] dilated context added to layers {wrapped} ({n / 1e6:.2f}M params, zero-init)")
    return wrapped


def levels_in_checkpoint(model: nn.Module) -> list[str]:
    """Pyramid levels that carry dilated context in an already-built model (used when resuming)."""
    layers, head = model.model, model.model[-1]
    return [f"p{3 + k}" for k, idx in enumerate(head.f) if isinstance(layers[idx], WithDilatedContext)]


def levels_from_env() -> list[str]:
    raw = os.environ.get("FSEG_DILATED", "").strip().lower()
    return [s.strip() for s in raw.split(",") if s.strip()]


def make_trainer():
    """Segmentation trainer that injects dilated context after the pretrained model is built."""
    from ultralytics.models.yolo.segment import SegmentationTrainer

    class DilatedSegmentationTrainer(SegmentationTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            # 1st load fills every layer whose name is unchanged (all layers of a plain pretrained model)
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            levels = levels_from_env()
            if levels:
                inject(model, levels, verbose=verbose)
                # 2nd load fills the wrapped layers when `weights` already has them (resuming a dilated run);
                # for a plain pretrained model it matches nothing new, so it is harmless
                if weights is not None:
                    model.load(weights)
            return model

    return DilatedSegmentationTrainer


DilatedSegmentationTrainer = None
try:  # importable at module level so Ultralytics' multi-GPU relaunch can `from fseg.dilated import ...`
    DilatedSegmentationTrainer = make_trainer()
    DilatedSegmentationTrainer.__module__ = __name__
    DilatedSegmentationTrainer.__qualname__ = "DilatedSegmentationTrainer"
except Exception:  # ultralytics not installed
    pass
