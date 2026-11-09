"""CIFAR-10 at 32 px, held in memory as uint8, with the committed 45k/5k train/val split.

Raw data: torchvision's ``cifar-10-batches-py`` under ``<data root>/cifar10``.
Split file: ``data_splits/cifar10_val_idx.json`` (made by ``scripts/make_cifar10_split.py``),
the 5,000 validation indices into torchvision's 50,000-image training set, 500 per class.

Splits: ``train`` 45,000, ``val`` 5,000, ``trainval`` 50,000 and ``test`` 10,000.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from .common import (
    CIFAR10_CLASSES,
    ImageArrayDataset,
    data_root,
    read_json,
    require_file,
    sha256_bytes,
    splits_dir,
    stratified_split,
)

NAME = "cifar10"
RESOLUTIONS = (32,)
SPLIT_FILE = "cifar10_val_idx.json"
SPLIT_SEED = 0
VAL_FRACTION = 0.1
# Per-channel statistics of the 50k training images, the usual published values.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def raw_dir(root=None) -> Path:
    return data_root(root) / "cifar10"


@functools.lru_cache(maxsize=4)
def _load(raw: str, train: bool) -> tuple[np.ndarray, np.ndarray]:
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(raw, train=train, download=False)
    return np.ascontiguousarray(ds.data), np.asarray(ds.targets, dtype=np.int64)


def load_arrays(train: bool, root=None) -> tuple[np.ndarray, np.ndarray]:
    """(images (N, 32, 32, 3) uint8, labels (N,) int64) of the official train or test set."""
    raw = raw_dir(root)
    require_file(
        raw / "cifar-10-batches-py",
        f"Download CIFAR-10 with: python scripts/make_cifar10_split.py --download (data root {data_root(root)}).",
    )
    return _load(str(raw), train)


def labels_digest(labels: np.ndarray) -> str:
    return sha256_bytes(np.asarray(labels, dtype=np.int64).tobytes())


def make_split(labels: np.ndarray, seed: int = SPLIT_SEED, fraction: float = VAL_FRACTION) -> dict:
    """The split-file content for the given training labels (deterministic)."""
    val = stratified_split(labels, fraction, seed)
    per_class = np.bincount(labels[val], minlength=len(CIFAR10_CLASSES)).tolist()
    return {
        "dataset": NAME,
        "description": "Validation indices into torchvision CIFAR10(train=True) (data_batch_1..5 order); "
        "train = the other 45,000.",
        "generated_by": "scripts/make_cifar10_split.py",
        "method": "per class c: numpy default_rng([seed, c]).permutation(indices of c)[:round(0.1 * n_c)]",
        "seed": seed,
        "val_fraction": fraction,
        "n_source": len(labels),
        "n_val": len(val),
        "val_per_class": per_class,
        "labels_sha256": labels_digest(labels),
        "val_idx": val.tolist(),
    }


def load_split(splits=None) -> dict:
    path = require_file(
        splits_dir(splits) / SPLIT_FILE, "It is committed in the repository; regenerate only with make_cifar10_split.py."
    )
    return read_json(path)


def split_indices(split: str, labels: np.ndarray, splits=None) -> np.ndarray | None:
    """Row indices of ``split`` in the official train set (None for the whole of it)."""
    spec = load_split(splits)
    if spec["labels_sha256"] != labels_digest(labels):
        raise ValueError(
            f"{SPLIT_FILE} was made for a different CIFAR-10 training set (labels hash mismatch); "
            "check the downloaded data."
        )
    val = np.asarray(spec["val_idx"], dtype=np.int64)
    if split == "val":
        return val
    if split == "train":
        mask = np.ones(len(labels), dtype=bool)
        mask[val] = False
        return np.flatnonzero(mask)
    return None


def get_cifar10(split: str, res: int = 32, *, root=None, splits=None, transform=None, in_memory=True):
    if res not in RESOLUTIONS:
        raise ValueError(f"cifar10 exists only at {RESOLUTIONS} px, not {res}")
    if split == "test":
        images, labels = load_arrays(False, root)
        indices = None
    else:
        images, labels = load_arrays(True, root)
        indices = split_indices(split, labels, splits)
    return ImageArrayDataset(
        images,
        labels,
        indices,
        CIFAR10_CLASSES,
        name=NAME,
        split=split,
        mean=MEAN,
        std=STD,
        transform=transform,
        in_memory=True,
    )
