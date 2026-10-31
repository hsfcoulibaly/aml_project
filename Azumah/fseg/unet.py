"""Stage 2: dense U-Net with ASPP that predicts filament probability, centerline and boundary maps.

Channels (all sigmoid outputs):
    0 filament: soft target = fraction of annotation sets that mark the pixel
    1 centerline: per-instance skeletons of consensus masks, dilated by 2 px
    2 boundary: morphological gradient of consensus instance masks

The filament channel is trained with BCE + soft Dice + clDice (Shit et al., CVPR 2021), which rewards
connected, thin structures; centerline and boundary use positively weighted BCE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
CHANNELS = ("filament", "centerline", "boundary")


# --------------------------------------------------------------------------- model


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling: parallel dilated 3x3 convs plus image pooling, projected back."""

    def __init__(self, ch: int, rates=(1, 6, 12, 18), mid: int = 256):
        super().__init__()

        def branch(k, d):
            return nn.Sequential(nn.Conv2d(ch, mid, k, padding=0 if k == 1 else d, dilation=d, bias=False),
                                 nn.BatchNorm2d(mid), nn.ReLU(inplace=True))

        self.branches = nn.ModuleList([branch(1 if r == 1 else 3, r) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, mid, 1, bias=False), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(mid * (len(rates) + 1), ch, 1, bias=False), nn.BatchNorm2d(ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        outs.append(F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False))
        return x + self.project(torch.cat(outs, 1))  # residual: starts close to the plain U-Net


class FilamentUNet(nn.Module):
    def __init__(self, encoder: str = "tu-convnext_tiny", encoder_weights: str | None = "imagenet",
                 aspp: bool = True):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet(encoder, encoder_weights=encoder_weights, in_channels=3, classes=len(CHANNELS))
        self.aspp = ASPP(self.net.encoder.out_channels[-1]) if aspp else nn.Identity()

    def forward(self, x):
        features = list(self.net.encoder(x))
        features[-1] = self.aspp(features[-1])
        return self.net.segmentation_head(self.net.decoder(features))


def build_model(cfg: dict) -> FilamentUNet:
    return FilamentUNet(cfg["encoder"], cfg.get("encoder_weights"), cfg.get("aspp", True))


# --------------------------------------------------------------------------- losses


def soft_erode(x):
    return -torch.max(F.max_pool2d(-x, (3, 1), 1, (1, 0)), F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x, iters: int = 10):
    skel = F.relu(x - soft_dilate(soft_erode(x)))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_dilate(soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_dice_loss(p, t, eps: float = 1.0):
    """Dice over the whole batch: foreground is ~0.4% of pixels, so per-sample Dice is noisy."""
    return 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)


def cldice_loss(p, t, iters: int = 10, eps: float = 1.0):
    sp, st = soft_skeleton(p, iters), soft_skeleton(t, iters)
    tprec = ((sp * t).sum() + eps) / (sp.sum() + eps)
    tsens = ((st * p).sum() + eps) / (st.sum() + eps)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


@dataclass
class LossWeights:
    bce: float = 0.5
    dice: float = 0.5
    cldice: float = 0.25
    centerline: float = 0.25
    boundary: float = 0.1
    filament_pos_weight: float = 10.0  # Charan's U-Net (val PQ 0.410) used BCE pos_weight 10
    centerline_pos_weight: float = 10.0
    boundary_pos_weight: float = 5.0
    skeleton_iters: int = 10


def unet_loss(logits: torch.Tensor, target: torch.Tensor, w: LossWeights) -> tuple[torch.Tensor, dict]:
    logits, target = logits.float(), target.float()
    lf, tf = logits[:, :1], target[:, :1]
    pf = torch.sigmoid(lf)
    parts = {
        "bce": F.binary_cross_entropy_with_logits(
            lf, tf, pos_weight=torch.tensor(w.filament_pos_weight, device=logits.device)),
        "dice": soft_dice_loss(pf, tf),
        "cldice": cldice_loss(pf, tf, w.skeleton_iters),
        "centerline": F.binary_cross_entropy_with_logits(
            logits[:, 1:2], target[:, 1:2], pos_weight=torch.tensor(w.centerline_pos_weight, device=logits.device)),
        "boundary": F.binary_cross_entropy_with_logits(
            logits[:, 2:3], target[:, 2:3], pos_weight=torch.tensor(w.boundary_pos_weight, device=logits.device)),
    }
    total = (w.bce * parts["bce"] + w.dice * parts["dice"] + w.cldice * parts["cldice"]
             + w.centerline * parts["centerline"] + w.boundary * parts["boundary"])
    return total, {k: float(v) for k, v in parts.items()}


# --------------------------------------------------------------------------- inference


def normalize(img_u8: np.ndarray) -> np.ndarray:
    """HWC uint8 preprocessed image -> CHW float32, ImageNet-normalized (sky stays at the mean)."""
    x = (img_u8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def gaussian_window(size: int, sigma_frac: float = 0.25) -> np.ndarray:
    g = np.exp(-0.5 * ((np.arange(size) - (size - 1) / 2) / (sigma_frac * size)) ** 2)
    w = np.outer(g, g)
    return (w / w.max()).astype(np.float32)


def _starts(n: int, tile: int, stride: int) -> list[int]:
    if n <= tile:
        return [0]
    s = list(range(0, n - tile + 1, stride))
    if s[-1] != n - tile:
        s.append(n - tile)
    return s


TTA_VIEWS = {
    "none": [(0, False)],
    "flip": [(0, False), (0, True), (2, False), (2, True)],  # id, hflip, rot180, vflip
    "d4": [(k, f) for k in range(4) for f in (False, True)],
}


def _view(x, k, flip):
    x = torch.rot90(x, k, (-2, -1))
    return torch.flip(x, (-1,)) if flip else x


def _unview(x, k, flip):
    x = torch.flip(x, (-1,)) if flip else x
    return torch.rot90(x, -k, (-2, -1))


@torch.no_grad()
def predict_full(model: nn.Module, img_u8: np.ndarray, device, tile: int = 1024, stride: int = 768,
                 tta: str = "flip", amp: bool = True, batch: int = 4) -> np.ndarray:
    """Sliding-window probabilities for one HxWx3 uint8 image -> (3, H, W) float32 in [0, 1]."""
    x = torch.from_numpy(normalize(img_u8)).to(device)
    _, h, w = x.shape
    win = torch.from_numpy(gaussian_window(min(tile, h, w))).to(device)
    acc = torch.zeros((len(CHANNELS), h, w), device=device)
    norm = torch.zeros((h, w), device=device)
    t = min(tile, h, w)
    coords = [(y, xx) for y in _starts(h, t, stride) for xx in _starts(w, t, stride)]
    use_amp = amp and device.type == "cuda"
    for k, flip in TTA_VIEWS[tta]:
        for i in range(0, len(coords), batch):
            chunk = coords[i:i + batch]
            crops = torch.stack([_view(x[:, y:y + t, xx:xx + t], k, flip) for y, xx in chunk])
            with torch.autocast(device.type, enabled=use_amp):
                prob = torch.sigmoid(model(crops).float())
            for (y, xx), p in zip(chunk, prob):
                acc[:, y:y + t, xx:xx + t] += _unview(p, k, flip) * win
                norm[y:y + t, xx:xx + t] += win
    out = acc / norm.clamp(min=1e-6)
    out[:, (img_u8.max(axis=2) == 0)] = 0  # sky is zero in the preprocessed image
    return out.cpu().numpy()


def prob_to_instances(prob: np.ndarray, thr: float, min_area: int, disk: np.ndarray | None = None) -> list[tuple[dict, float]]:
    """Standalone U-Net instances: threshold, 8-connected components, score = mean probability."""
    import cv2

    from . import rle as R

    binary = (prob > thr).astype(np.uint8)
    if disk is not None:
        binary &= disk.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        m = np.zeros(labels.shape, bool)
        m[y:y + h, x:x + w] = labels[y:y + h, x:x + w] == i
        out.append((R.encode(m), float(prob[m].mean())))
    return out


def save_weights(model: nn.Module, path, cfg: dict, meta: dict) -> None:
    """fp16 weights-only checkpoint; small enough to commit (the organizers require weights in the repo)."""
    state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "cfg": cfg, "meta": meta}, path)


def load_weights(path, device) -> tuple[FilamentUNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model({**ckpt["cfg"], "encoder_weights": None})
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()})
    return model.to(device).eval(), ckpt


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
