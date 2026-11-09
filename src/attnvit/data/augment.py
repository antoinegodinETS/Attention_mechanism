"""The one augmentation recipe shared by every mechanism.

Per image, on uint8 ``(3, H, W)`` tensors in the DataLoader workers: random crop after
zero-padding by ``crop_padding * res`` pixels (4 px at 32 px, 20 px at 160 px), horizontal
flip, then RandAugment (``num_ops`` ops at ``magnitude``). Images stay uint8 until they reach
the device, where :class:`Normalize` turns them into normalised floats.

Per batch, on the device: :class:`MixupCutmix` mixes each image with its mirror in the batch
(DeiT-style mixup or cutmix) and returns soft targets that already include label smoothing.
Evaluation uses no augmentation: only :class:`Normalize`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torchvision.transforms import v2


def build_train_transform(res: int, aug) -> v2.Compose | None:
    """Per-image training transform for ``res`` px images (``aug`` is an ``AugmentConfig``)."""
    ops = []
    pad = int(round(aug.crop_padding * res))
    if pad > 0:
        ops.append(v2.RandomCrop(res, padding=pad))
    if aug.hflip:
        ops.append(v2.RandomHorizontalFlip())
    if aug.randaugment_ops > 0:
        ops.append(v2.RandAugment(num_ops=aug.randaugment_ops, magnitude=aug.randaugment_magnitude))
    return v2.Compose(ops) if ops else None


class Normalize:
    """uint8 ``(B, 3, H, W)`` on any device -> float32, scaled to [0, 1], minus mean, over std.

    Unlike :class:`attnvit.data.Normalize` (per image, statistics copied on each call), the
    statistics live on ``device`` once, so the training step issues no host-to-device copy.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float], device=None):
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div_(255)
        return (x - self.mean) / self.std


def smooth_one_hot(y: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    off = smoothing / num_classes
    t = torch.full((y.shape[0], num_classes), off, device=y.device, dtype=torch.float32)
    return t.scatter_(1, y.view(-1, 1), 1.0 - smoothing + off)


class MixupCutmix:
    """Batch-level mixup / cutmix with label smoothing; returns ``(images, soft targets)``.

    Each image is mixed with the one at the mirrored batch position (``x.flip(0)``), as in
    timm. With probability ``prob`` a batch is mixed; it then uses cutmix with probability
    ``switch_prob`` when both are enabled. The random draws come from a private numpy
    generator, reseeded per epoch by the training loop, so they do not depend on the device.
    """

    def __init__(self, num_classes: int, *, mixup_alpha: float = 0.8, cutmix_alpha: float = 1.0,
                 prob: float = 1.0, switch_prob: float = 0.5, label_smoothing: float = 0.1, seed: int = 0):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.reseed(seed)

    @classmethod
    def from_config(cls, aug, num_classes: int, seed: int = 0) -> "MixupCutmix":
        return cls(num_classes, mixup_alpha=aug.mixup_alpha, cutmix_alpha=aug.cutmix_alpha,
                   prob=aug.mix_prob, switch_prob=aug.switch_prob,
                   label_smoothing=aug.label_smoothing, seed=seed)

    @property
    def active(self) -> bool:
        return self.prob > 0 and (self.mixup_alpha > 0 or self.cutmix_alpha > 0)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = smooth_one_hot(y, self.num_classes, self.label_smoothing)
        if not self.active or self.rng.random() >= self.prob:
            return x, target
        if self.cutmix_alpha > 0 and (self.mixup_alpha <= 0 or self.rng.random() < self.switch_prob):
            x, lam = self._cutmix(x)
        else:
            lam = float(self.rng.beta(self.mixup_alpha, self.mixup_alpha))
            x = x * lam + x.flip(0) * (1.0 - lam)
        return x, target * lam + target.flip(0) * (1.0 - lam)

    def _cutmix(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        lam = float(self.rng.beta(self.cutmix_alpha, self.cutmix_alpha))
        H, W = x.shape[-2:]
        cut = np.sqrt(1.0 - lam)
        ch, cw = int(H * cut), int(W * cut)
        cy, cx = int(self.rng.integers(H)), int(self.rng.integers(W))
        y1, y2 = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
        x1, x2 = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
        out = x.clone()
        out[..., y1:y2, x1:x2] = x.flip(0)[..., y1:y2, x1:x2]
        lam = 1.0 - float((y2 - y1) * (x2 - x1)) / (H * W)  # the area actually pasted
        return out, lam
