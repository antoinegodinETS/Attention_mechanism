"""Imagenette-160 (fastai ``imagenette2-160``), center-cropped to 160x160, plus a 32 px copy.

Built by ``scripts/build_imagenette.py`` into ``<data root>/imagenette``. The official
``val`` folder (3,925 images) is our ``test`` split; our ``val`` is a seeded stratified 10 %
of the official ``train`` folder (9,469 images), listed in the committed
``data_splits/imagenette_val.json``; ``train`` is the rest and ``trainval`` all of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import ImageArrayDataset, open_built, read_json, require_file, split_rows, splits_dir

NAME = "imagenette"
RESOLUTIONS = (32, 160)
SPLIT_FILE = "imagenette_val.json"
SPLIT_SEED = 0
VAL_FRACTION = 0.1
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
WNIDS = (
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028079",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03445777",
    "n03888257",
)
CLASSES = (
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
)


def load_split(splits=None) -> dict:
    path = require_file(splits_dir(splits) / SPLIT_FILE, "It is committed; build_imagenette.py creates it once.")
    return read_json(path)


def get_imagenette(split: str, res: int = 160, *, root=None, splits=None, transform=None, in_memory=None):
    if res not in RESOLUTIONS:
        raise ValueError(f"imagenette exists at {RESOLUTIONS} px, not {res}")
    images, labels, info = open_built(NAME, res, root, "Build it with: python scripts/build_imagenette.py")
    manifest = pd.read_csv(images.parent / "manifest.csv")
    committed_val = set(load_split(splits)["val_files"])
    built_val = set(manifest.loc[manifest["split"] == "val", "file"])
    if built_val != committed_val:
        raise ValueError(
            f"{images.parent} was built with a different validation split than data_splits/{SPLIT_FILE}; "
            "rebuild with scripts/build_imagenette.py --force."
        )
    rows = split_rows(manifest["split"].to_numpy(), split)
    return ImageArrayDataset(
        images,
        labels,
        rows,
        CLASSES,
        name=NAME,
        split=split,
        mean=info["stats"][str(res)]["mean"],
        std=info["stats"][str(res)]["std"],
        transform=transform,
        in_memory=(res <= 32) if in_memory is None else in_memory,
    )


def stratified_val_files(files: list[str], labels: np.ndarray, seed: int = SPLIT_SEED) -> list[str]:
    """Seeded stratified 10 % of the official train files (sorted relative paths)."""
    from .common import stratified_split

    order = np.argsort(files)
    files_sorted = [files[i] for i in order]
    idx = stratified_split(np.asarray(labels)[order], VAL_FRACTION, seed)
    return [files_sorted[i] for i in idx]
