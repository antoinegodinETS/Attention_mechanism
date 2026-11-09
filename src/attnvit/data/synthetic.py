"""Random uint8 images with random labels, for tests, smoke runs and timing.

Matches the item format of :func:`attnvit.data.get_dataset`: ``(uint8 tensor (3, res, res),
int label)``, plus ``classes``, ``mean`` and ``std`` attributes. Every split holds the same
images (``seed`` fixes them), so validation accuracy on ``synthetic`` measures memorisation:
that is what the overfit test checks.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticImages(Dataset):
    def __init__(self, n: int = 256, res: int = 32, num_classes: int = 10, *, seed: int = 0,
                 split: str = "train", transform=None):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randint(0, 256, (n, 3, res, res), dtype=torch.uint8, generator=g)
        self.labels = torch.randint(0, num_classes, (n,), generator=g)
        self.name = "synthetic"
        self.split = split
        self.classes = [f"class{i}" for i in range(num_classes)]
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.289, 0.289, 0.289]  # uniform on [0, 1]: 1 / sqrt(12)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = self.images[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])
