"""U-Net, loss, inference and checkpoint helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ObservationDataset


def build_model(encoder: str = "resnet34", pretrained: bool = True) -> nn.Module:
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
    )


class BCEDiceLoss(nn.Module):
    """w * BCE(pos_weight) + (1 - w) * soft Dice over the whole batch (foreground is ~0.4% of pixels)."""

    def __init__(self, pos_weight: float = 10.0, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, target):
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        p = torch.sigmoid(logits)
        inter = (p * target).sum()
        dice = 1 - (2 * inter + self.smooth) / (p.sum() + target.sum() + self.smooth)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


@torch.no_grad()
def predict_probs(model, paths, observation_ids, device, batch_size=4, workers=2, tta=False) -> dict:
    """{observation_id: float16 probability map at the prepared resolution}."""
    model.eval()
    loader = DataLoader(ObservationDataset(paths, observation_ids), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    out = {}
    for x, ids in tqdm(loader, desc="predict", leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x).float()
            if tta:
                logits = logits + model(x.flip(-1)).float().flip(-1) + model(x.flip(-2)).float().flip(-2)
                logits = logits / 3
        probs = torch.sigmoid(logits)[:, 0].cpu().numpy().astype(np.float16)
        out.update(zip(ids, probs))
    return out


def save_weights(model, path: Path, meta: dict, half: bool = True) -> None:
    """Weights-only checkpoint (fp16 keeps a ResNet34 U-Net under GitHub's 100 MB file limit)."""
    state = {k: (v.half() if half and v.is_floating_point() else v) for k, v in unwrap(model).state_dict().items()}
    torch.save({"state_dict": state, "meta": meta}, path)


def load_weights(path: Path, device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    model = build_model(meta["encoder"], pretrained=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()})
    return model.to(device), meta
