"""CIFAR-10-HR: the ten CIFAR-10 classes from ImageNet-21K at 160 px, plus a 32 px copy.

Built by ``scripts/build_cifar10_hr.py`` into ``<data root>/cifar10_hr`` (memmapped
``images_160.npy`` about 4.6 GB, ``images_32.npy`` about 184 MB). Per class: 4,500 train,
500 val, 1,000 test. The committed ``data_splits/cifar10_hr_manifest.csv.gz`` records, for
every array row, its split, label, synset and source file; the loader checks that the arrays
under the data root were written with exactly that manifest.
"""

from __future__ import annotations

import gzip
import io

import pandas as pd

from .common import (
    CIFAR10_CLASSES,
    ImageArrayDataset,
    open_built,
    require_file,
    sha256_bytes,
    split_rows,
    splits_dir,
)

NAME = "cifar10_hr"
RESOLUTIONS = (32, 160)
MANIFEST_FILE = "cifar10_hr_manifest.csv.gz"
SYNSET_FILE = "cifar10_hr_synsets.csv"


def read_manifest_bytes(splits=None) -> bytes:
    path = require_file(
        splits_dir(splits) / MANIFEST_FILE,
        "It is written by scripts/build_cifar10_hr.py and committed once CIFAR-10-HR is built.",
    )
    return gzip.decompress(path.read_bytes())


def get_cifar10_hr(split: str, res: int = 160, *, root=None, splits=None, transform=None, in_memory=None):
    if res not in RESOLUTIONS:
        raise ValueError(f"cifar10_hr exists at {RESOLUTIONS} px, not {res}")
    images, labels, info = open_built(NAME, res, root, "Build it with scripts/build_cifar10_hr.py (see README).")
    data = read_manifest_bytes(splits)
    if sha256_bytes(data) != info["manifest_sha256"]:
        raise ValueError(
            f"{images.parent} was not built with the committed data_splits/{MANIFEST_FILE} "
            "(manifest checksum mismatch)."
        )
    manifest = pd.read_csv(io.BytesIO(data))
    rows = split_rows(manifest["split"].to_numpy(), split)
    return ImageArrayDataset(
        images,
        labels,
        rows,
        CIFAR10_CLASSES,
        name=NAME,
        split=split,
        mean=info["stats"][str(res)]["mean"],
        std=info["stats"][str(res)]["std"],
        transform=transform,
        in_memory=(res <= 32) if in_memory is None else in_memory,
    )
