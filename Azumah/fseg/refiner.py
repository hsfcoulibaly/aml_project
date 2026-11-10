"""Stage 3: instance-conditioned refiner.

For every proposal (a YOLO instance) the refiner sees a square crop around it and redraws the mask of
that one filament at crop resolution, plus a scalar quality: the predicted IoU of its own mask with
the matched ground truth (0 for a proposal that matches nothing). Quality is the final ranking score,
so it also learns to reject false positives.

Input channels (5): limb-flattened, CLAHE, proposal prior mask, U-Net filament prob, U-Net centerline.
One affine transform per crop maps 2048-px coordinates to the crop and back, used both to extract
inputs and to paste the refined mask into the full image.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import rle as R
from .config import IMAGE_SIZE

CROP = 256
IN_CHANNELS = 5


# --------------------------------------------------------------------------- geometry


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_affine(box, context: float = 1.5, min_side: float = 128, out: int = CROP,
                jitter: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """2x3 matrix mapping full-image pixels into an `out` x `out` square around `box`."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0, 1) * context
    if jitter and rng is not None:  # training: vary context and position a little
        side *= rng.uniform(1 - jitter, 1 + jitter)
        cx += rng.uniform(-jitter, jitter) * side / 2
        cy += rng.uniform(-jitter, jitter) * side / 2
    side = max(side, min_side)
    s = out / side
    return np.array([[s, 0, out / 2 - s * cx], [0, s, out / 2 - s * cy]], np.float32)


def warp(img: np.ndarray, M: np.ndarray, out: int = CROP, nearest: bool = False) -> np.ndarray:
    return cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def paste(prob: np.ndarray, M: np.ndarray, thr: float = 0.5, size: int = IMAGE_SIZE) -> np.ndarray:
    """Crop probability -> full-size boolean mask (only the crop's footprint is warped)."""
    Minv = cv2.invertAffineTransform(M)
    corners = np.array([[0, 0, 1], [prob.shape[1], 0, 1], [0, prob.shape[0], 1], [prob.shape[1], prob.shape[0], 1]],
                       np.float32) @ Minv.T
    x0, y0 = np.floor(corners.min(0)).astype(int).clip(0, size)
    x1, y1 = np.ceil(corners.max(0)).astype(int).clip(0, size)
    full = np.zeros((size, size), bool)
    if x1 <= x0 or y1 <= y0:
        return full
    Mroi = Minv.copy()
    Mroi[:, 2] -= (x0, y0)
    roi = cv2.warpAffine(prob.astype(np.float32), Mroi, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    full[y0:y1, x0:x1] = roi > thr
    return full


def crop_inputs(image: np.ndarray, unet_map: np.ndarray | None, prior: np.ndarray, M: np.ndarray) -> np.ndarray:
    """(5, CROP, CROP) uint8: flattened, CLAHE, prior, U-Net filament, U-Net centerline.

    image: preprocessed HxWx3 uint8 (raw, flattened, CLAHE); unet_map: HxWx3 uint8 (filament,
    centerline, boundary) or None (zeros); prior: HxW bool at full resolution, or already a crop.
    """
    flat, clahe = warp(image[:, :, 1], M), warp(image[:, :, 2], M)
    p = prior.astype(np.uint8) * 255
    p = p if p.shape == (CROP, CROP) else warp(p, M)
    if unet_map is None:
        uf = uc = np.zeros((CROP, CROP), np.uint8)
    else:
        uf, uc = warp(unet_map[:, :, 0], M), warp(unet_map[:, :, 1], M)
    return np.stack([flat, clahe, p, uf, uc])


# --------------------------------------------------------------------------- prior corruption


def corrupt_prior(mask: np.ndarray, others: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Turn a true crop mask into a YOLO-like proposal: coarse, shifted, wrong extent, bleeding."""
    m = mask.astype(np.uint8)
    if rng.random() < 0.7:  # YOLO prototypes are ~4x coarser than the image
        f = rng.choice([2, 4, 6])
        small = cv2.resize(m.astype(np.float32), (CROP // f, CROP // f), interpolation=cv2.INTER_AREA)
        m = (cv2.resize(small, (CROP, CROP), interpolation=cv2.INTER_LINEAR) > rng.uniform(0.3, 0.6)).astype(np.uint8)
    if rng.random() < 0.5:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(rng.integers(1, 4)) + 1,) * 2)
        m = cv2.dilate(m, k) if rng.random() < 0.6 else cv2.erode(m, k)
    if rng.random() < 0.5:
        m = np.roll(m, (int(rng.integers(-8, 9)), int(rng.integers(-8, 9))), axis=(0, 1))
    if rng.random() < 0.3 and m.any():  # drop a segment
        ys, xs = np.nonzero(m)
        i = rng.integers(len(ys))
        cv2.circle(m, (int(xs[i]), int(ys[i])), int(rng.integers(6, 20)), 0, -1)
    if rng.random() < 0.2 and others.any():  # bleed into a neighbouring filament
        ys, xs = np.nonzero(others)
        i = rng.integers(len(ys))
        blob = np.zeros_like(m)
        cv2.circle(blob, (int(xs[i]), int(ys[i])), int(rng.integers(8, 25)), 1, -1)
        m |= (blob & others.astype(np.uint8))
    return m.astype(bool)


# --------------------------------------------------------------------------- model and loss


class Refiner(nn.Module):
    def __init__(self, encoder: str = "resnet34", encoder_weights: str | None = "imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet(encoder, encoder_weights=encoder_weights, in_channels=IN_CHANNELS, classes=1,
                            aux_params={"classes": 1, "dropout": 0.2})

    def forward(self, x):
        mask, quality = self.net(x)
        return mask, quality[:, 0]


# Dilation ramp of modelAE_hdc_deep.json (neuroneural/catalyst_example, refiner-implementations):
# odd, coprime, symmetric; receptive field 1 + 2 * sum = 255, matched to the 256 px crop.
HDC_DEEP = (1, 3, 5, 7, 13, 19, 31, 19, 13, 7, 5, 3, 1)


class MeshNetRefiner(nn.Module):
    """2D MeshNet (Fedorov et al.) with the hdc_deep schedule: a plain stack of dilated 3x3 convs, no skips,
    no downsampling. Following meshnet_gn.py: per-channel GroupNorm with affine, GELU, no conv bias, Kaiming
    (fan_out) init, final 1x1 conv with bias. The quality head pools the last hidden features."""

    def __init__(self, channels: int = 8, dilations=HDC_DEEP):
        super().__init__()
        layers, c_in = [], IN_CHANNELS
        for d in dilations:
            layers += [nn.Conv2d(c_in, channels, 3, padding=d, dilation=d, bias=False),
                       nn.GroupNorm(channels, channels, affine=True), nn.GELU()]
            c_in = channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(channels, 1, 1)
        self.quality = nn.Linear(2 * channels + 1, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.body(x)
        mask = self.head(h)
        pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3)), torch.sigmoid(mask).mean((2, 3))], 1)
        return mask, self.quality(pooled)[:, 0]


def make_refiner(cfg: dict, pretrained: bool = False) -> nn.Module:
    if cfg.get("arch", "unet") == "meshnet":
        return MeshNetRefiner(cfg.get("channels", 8))
    return Refiner(cfg["encoder"], "imagenet" if pretrained else None)


def to_tensor(crops: np.ndarray) -> torch.Tensor:
    """(N, 5, H, W) uint8 -> float in roughly [-2, 2]."""
    return (torch.from_numpy(crops).float() / 255.0 - 0.5) / 0.25


@dataclass
class RefinerLossWeights:
    bce: float = 1.0
    dice: float = 1.0
    quality: float = 1.0
    label_smoothing: float = 0.0  # catalyst recipe: 0.01
    generalized_dice: bool = False  # catalyst recipe: True (fg and bg weighted by inverse volume)


def _generalized_dice(p, t, eps: float = 1e-6):
    """Per-sample two-class generalized Dice (Crum et al.): each class weighted by 1 / volume^2."""
    p2, t2 = torch.cat([p, 1 - p], 1).flatten(2), torch.cat([t, 1 - t], 1).flatten(2)
    w = 1.0 / (t2.sum(2) ** 2 + eps)
    num = (w * (p2 * t2).sum(2)).sum(1)
    den = (w * (p2 + t2).sum(2)).sum(1)
    return 1 - (2 * num + eps) / (den + eps)


def refiner_loss(mask_logits, q_logits, target, w: RefinerLossWeights) -> tuple[torch.Tensor, dict]:
    mask_logits, q_logits, target = mask_logits.float(), q_logits.float(), target.float()
    prob = torch.sigmoid(mask_logits)
    smooth = target * (1 - w.label_smoothing) + 0.5 * w.label_smoothing
    bce = F.binary_cross_entropy_with_logits(mask_logits, smooth)
    pos = target.flatten(1).sum(1) > 0
    if pos.any() and w.generalized_dice:
        dice = _generalized_dice(prob[pos], target[pos]).mean()
    elif pos.any():
        p, t = prob[pos], target[pos]
        dice = 1 - ((2 * (p * t).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)).mean()
    else:
        dice = prob.sum() * 0
    with torch.no_grad():  # quality target: IoU of the refiner's own mask with its target (0 for negatives)
        hard = prob > 0.5
        inter = (hard & (target > 0.5)).flatten(1).sum(1).float()
        union = (hard | (target > 0.5)).flatten(1).sum(1).float()
        q_target = torch.where(pos, inter / union.clamp(min=1), torch.zeros_like(inter))
    q = F.smooth_l1_loss(torch.sigmoid(q_logits), q_target, beta=0.1)
    total = w.bce * bce + w.dice * dice + w.quality * q
    return total, {"bce": float(bce), "dice": float(dice), "quality": float(q)}


REFINE_TTA = {"hflip": [(0, False), (0, True)], "d4": [(k, f) for k in range(4) for f in (False, True)]}


@torch.no_grad()
def refine(model, crops: np.ndarray, device, batch: int = 64, tta: str = "hflip") -> tuple[np.ndarray, np.ndarray]:
    """(N, 5, H, W) uint8 crops -> (N, H, W) probabilities and (N,) quality.

    `model` may be a list (ensemble); masks and quality are averaged over models and TTA views."""
    models = model if isinstance(model, (list, tuple)) else [model]
    views = REFINE_TTA[tta]
    probs, quals = [], []
    use_amp = device.type == "cuda"
    for i in range(0, len(crops), batch):
        x = to_tensor(crops[i:i + batch]).to(device)
        m_acc, q_acc = 0, 0
        for mdl in models:
            for k, f in views:
                xv = torch.rot90(x, k, (-2, -1))
                xv = torch.flip(xv, (-1,)) if f else xv
                with torch.autocast(device.type, enabled=use_amp):
                    mv, qv = mdl(xv)
                mv = torch.sigmoid(mv.float())
                mv = torch.flip(mv, (-1,)) if f else mv
                m_acc = m_acc + torch.rot90(mv, -k, (-2, -1))
                q_acc = q_acc + torch.sigmoid(qv.float())
        n = len(models) * len(views)
        probs.append((m_acc / n)[:, 0].cpu().numpy())
        quals.append((q_acc / n).cpu().numpy())
    if not probs:
        return np.zeros((0, CROP, CROP), np.float32), np.zeros(0, np.float32)
    return np.concatenate(probs), np.concatenate(quals)


def save_refiner(model: Refiner, path, cfg: dict, meta: dict) -> None:
    state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "cfg": cfg, "meta": meta}, path)


def load_refiner(path, device) -> tuple[Refiner, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = make_refiner(ckpt["cfg"])
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()})
    return model.to(device).eval(), ckpt


def decode_list(rles: list[dict]) -> list[np.ndarray]:
    return [R.decode(r) for r in rles]
