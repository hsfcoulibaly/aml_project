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


def to_tensor(crops: np.ndarray) -> torch.Tensor:
    """(N, 5, H, W) uint8 -> float in roughly [-2, 2]."""
    return (torch.from_numpy(crops).float() / 255.0 - 0.5) / 0.25


@dataclass
class RefinerLossWeights:
    bce: float = 1.0
    dice: float = 1.0
    quality: float = 1.0


def refiner_loss(mask_logits, q_logits, target, w: RefinerLossWeights) -> tuple[torch.Tensor, dict]:
    mask_logits, q_logits, target = mask_logits.float(), q_logits.float(), target.float()
    prob = torch.sigmoid(mask_logits)
    bce = F.binary_cross_entropy_with_logits(mask_logits, target)
    pos = target.flatten(1).sum(1) > 0
    if pos.any():
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


@torch.no_grad()
def refine(model: Refiner, crops: np.ndarray, device, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """(N, 5, H, W) uint8 crops -> (N, H, W) probabilities and (N,) quality, with hflip TTA."""
    probs, quals = [], []
    use_amp = device.type == "cuda"
    for i in range(0, len(crops), batch):
        x = to_tensor(crops[i:i + batch]).to(device)
        with torch.autocast(device.type, enabled=use_amp):
            m1, q1 = model(x)
            m2, q2 = model(torch.flip(x, (-1,)))
        m = (torch.sigmoid(m1.float()) + torch.sigmoid(torch.flip(m2.float(), (-1,)))) / 2
        q = (torch.sigmoid(q1.float()) + torch.sigmoid(q2.float())) / 2
        probs.append(m[:, 0].cpu().numpy())
        quals.append(q.cpu().numpy())
    if not probs:
        return np.zeros((0, CROP, CROP), np.float32), np.zeros(0, np.float32)
    return np.concatenate(probs), np.concatenate(quals)


def save_refiner(model: Refiner, path, cfg: dict, meta: dict) -> None:
    state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "cfg": cfg, "meta": meta}, path)


def load_refiner(path, device) -> tuple[Refiner, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Refiner(ckpt["cfg"]["encoder"], encoder_weights=None)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()})
    return model.to(device).eval(), ckpt


def decode_list(rles: list[dict]) -> list[np.ndarray]:
    return [R.decode(r) for r in rles]
