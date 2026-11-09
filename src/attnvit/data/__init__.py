"""Datasets for the SYS819 study behind one entry point, :func:`get_dataset`.

=============  ==========  ===============================================================
name           res (px)    splits (train / val / test, trainval = train + val)
=============  ==========  ===============================================================
``cifar10``    32          45,000 / 5,000 / 10,000
``cifar10_hr`` 32, 160     45,000 / 5,000 / 10,000 (ImageNet-21K, built on demand)
``imagenette`` 32, 160     ~8,522 / ~947 / 3,925 (imagenette2-160; test = official val)
=============  ==========  ===============================================================

Datasets live under ``$SYS819_DATA`` (default ``~/data/sys819``), never in the repository.
Items are ``(uint8 tensor (3, res, res), int label)``; pass ``transform=ds.normalizer()``
(or any callable) to get normalised floats.
"""

from __future__ import annotations

from .common import (
    CIFAR10_CLASSES,
    DATA_ENV,
    SPLITS,
    ImageArrayDataset,
    Normalize,
    data_root,
    splits_dir,
)

DATASETS = ("cifar10", "cifar10_hr", "imagenette")


def get_dataset(
    name: str,
    res: int,
    split: str,
    *,
    root=None,
    splits=None,
    transform=None,
    in_memory: bool | None = None,
) -> ImageArrayDataset:
    """Return split ``split`` of dataset ``name`` at ``res`` px.

    ``root`` overrides ``$SYS819_DATA``; ``splits`` overrides the committed ``data_splits/``
    directory (tests use both). ``in_memory`` defaults to True at 32 px and False (memmap)
    at 160 px; CIFAR-10 is always in memory.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if name == "cifar10":
        from .cifar10 import get_cifar10

        return get_cifar10(split, res, root=root, splits=splits, transform=transform)
    if name == "cifar10_hr":
        from .cifar10_hr import get_cifar10_hr

        return get_cifar10_hr(split, res, root=root, splits=splits, transform=transform, in_memory=in_memory)
    if name == "imagenette":
        from .imagenette import get_imagenette

        return get_imagenette(split, res, root=root, splits=splits, transform=transform, in_memory=in_memory)
    raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")


__all__ = [
    "CIFAR10_CLASSES",
    "DATASETS",
    "DATA_ENV",
    "SPLITS",
    "ImageArrayDataset",
    "Normalize",
    "data_root",
    "get_dataset",
    "splits_dir",
]
