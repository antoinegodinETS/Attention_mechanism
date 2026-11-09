"""Shared pieces of the dataset loaders: data root, split files and an array-backed dataset.

Every dataset is stored as one uint8 array of shape (N, H, W, 3) (in memory or as a
``.npy`` memmap) plus int64 labels. :class:`ImageArrayDataset` selects the rows of one
split and returns ``(image, label)`` with ``image`` a ``torch.uint8`` tensor of shape
(3, H, W). Augmentation belongs to phase 2; :class:`Normalize` is the only transform here.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_ENV = "SYS819_DATA"
DEFAULT_DATA_ROOT = "~/data/sys819"
SPLITS = ("train", "val", "trainval", "test")

# src/attnvit/data/common.py -> repository root is three levels above the package dir.
REPO_ROOT = Path(__file__).resolve().parents[3]

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def data_root(root: str | os.PathLike | None = None) -> Path:
    """Return the dataset root: ``root`` if given, else ``$SYS819_DATA``, else ``~/data/sys819``."""
    if root is None:
        root = os.environ.get(DATA_ENV) or DEFAULT_DATA_ROOT
    return Path(root).expanduser()


def splits_dir(path: str | os.PathLike | None = None) -> Path:
    """Return the directory holding the committed split files (``<repo>/data_splits`` by default)."""
    return Path(path).expanduser() if path is not None else REPO_ROOT / "data_splits"


def stratified_split(labels: Sequence[int], fraction: float, seed: int) -> np.ndarray:
    """Pick ``round(fraction * n_c)`` indices from each class ``c``, seeded, and return them sorted.

    Classes are visited in increasing label order, each with its own generator
    ``default_rng([seed, c])``, so the result does not depend on the other classes.
    """
    labels = np.asarray(labels)
    picked = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        n = int(round(fraction * len(idx)))
        rng = np.random.default_rng([seed, int(c)])
        picked.append(rng.permutation(idx)[:n])
    return np.sort(np.concatenate(picked)).astype(np.int64)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | os.PathLike, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    tmp.replace(path)


def require_file(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    return path


class Normalize:
    """uint8 (C, H, W) or (N, C, H, W) tensor -> float32 scaled to [0, 1], minus mean, over std."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().div_(255)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean.flatten().tolist()}, std={self.std.flatten().tolist()})"


def channel_stats(images: np.ndarray, max_elems: int = 1 << 24) -> tuple[list[float], list[float]]:
    """Per-channel mean and std, in [0, 1] units, of a (N, H, W, 3) uint8 array (memmaps welcome).

    Exact integer sums over chunks of at most ``max_elems`` values, so memory stays bounded
    (about 64 MB of uint32 temporaries by default) whatever the array size.
    """
    s = np.zeros(3, dtype=np.uint64)
    s2 = np.zeros(3, dtype=np.uint64)
    n = 0
    chunk = max(1, max_elems // int(np.prod(images.shape[1:])))
    for i in range(0, len(images), chunk):
        x = np.asarray(images[i : i + chunk]).astype(np.uint32)
        s += x.sum(axis=(0, 1, 2), dtype=np.uint64)
        s2 += (x * x).sum(axis=(0, 1, 2), dtype=np.uint64)
        n += x.shape[0] * x.shape[1] * x.shape[2]
    mean = s.astype(np.float64) / n
    std = np.sqrt(np.maximum(s2.astype(np.float64) / n - mean * mean, 0.0))
    mean, std = mean / 255.0, std / 255.0
    return [round(float(v), 4) for v in mean], [round(float(v), 4) for v in std]


class ImageArrayDataset(Dataset):
    """Rows ``indices`` of a (N, H, W, 3) uint8 image array with int64 labels.

    ``images`` is either an in-memory array or the path of a ``.npy`` file opened as a
    read-only memmap. A memmap is reopened lazily after pickling, so DataLoader workers
    started with ``spawn`` do not copy the array. With ``in_memory=True`` the selected rows
    are read into RAM once.
    """

    def __init__(
        self,
        images: np.ndarray | str | os.PathLike,
        labels: np.ndarray,
        indices: np.ndarray | None,
        classes: Sequence[str],
        *,
        name: str,
        split: str,
        mean: Sequence[float],
        std: Sequence[float],
        transform: Callable | None = None,
        in_memory: bool = True,
    ):
        self.name = name
        self.split = split
        self.classes = list(classes)
        self.mean = list(mean)
        self.std = list(std)
        self.transform = transform
        self._path: Path | None = None
        self._array: np.ndarray | None = None
        self._rows: slice | np.ndarray

        if isinstance(images, (str, os.PathLike)):
            self._path = Path(images)
            full = np.load(self._path, mmap_mode="r")
        else:
            full = images
        n_total = len(full)
        if len(labels) != n_total:
            raise ValueError(f"{name}: {n_total} images but {len(labels)} labels")
        indices = np.arange(n_total) if indices is None else np.asarray(indices, dtype=np.int64)
        if len(indices) and (indices.min() < 0 or indices.max() >= n_total):
            raise IndexError(f"{name}/{split}: split index out of range for {n_total} images")
        self.indices = indices
        self.labels = np.ascontiguousarray(np.asarray(labels, dtype=np.int64)[indices])

        contiguous = len(indices) > 0 and np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices)))
        self._rows = slice(int(indices[0]), int(indices[0]) + len(indices)) if contiguous else indices
        if in_memory or self._path is None:
            self._array = np.ascontiguousarray(full[self._rows])
            self._rows = slice(0, len(indices))
            self._path = None
        else:
            self._array = full

    # -- array access ---------------------------------------------------------------
    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self._path, mmap_mode="r")
        return self._array

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._path is not None:
            state["_array"] = None
        return state

    @property
    def images(self) -> np.ndarray:
        """The split's images as a (n, H, W, 3) uint8 array (a memmap view when not in memory)."""
        return self.array[self._rows]

    @property
    def resolution(self) -> int:
        return int(self.array.shape[1])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"index {i} out of range for {n} images")
        row = self._rows.start + i if isinstance(self._rows, slice) else int(self._rows[i])
        img = torch.from_numpy(np.ascontiguousarray(self.array[row].transpose(2, 0, 1)))
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])

    def tensors(self, device: str | torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Whole split as (uint8 (n, 3, H, W), int64 (n,)) tensors, optionally moved to ``device``.

        Lets a training loop keep a small dataset resident on the GPU and index batches
        directly. At 160 px this reads the split into RAM (about 3.5 GB for HR train).
        """
        x = torch.from_numpy(np.ascontiguousarray(self.images)).permute(0, 3, 1, 2).contiguous()
        y = torch.from_numpy(self.labels.copy())
        if device is not None:
            x, y = x.to(device), y.to(device)
        return x, y

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=len(self.classes))

    def normalizer(self) -> Normalize:
        return Normalize(self.mean, self.std)

    def __repr__(self) -> str:
        where = "memmap" if self._path is not None else "memory"
        return (
            f"{type(self).__name__}(name={self.name!r}, split={self.split!r}, n={len(self)}, "
            f"res={self.resolution}, storage={where})"
        )


# -- datasets built by the scripts (Imagenette-160, CIFAR-10-HR) --------------------------
# Layout under <data root>/<name>/:
#   images_160.npy, images_32.npy   uint8 (N, R, R, 3), rows ordered train, val, test
#   labels.npy                      int64 (N,)
#   manifest.csv                    one row per image: index, split, label, ... (source file)
#   build_info.json                 parameters, counts, channel statistics, checksums


def built_dir(name: str, root=None) -> Path:
    return data_root(root) / name


def open_built(name: str, res: int, root=None, build_hint: str = "") -> tuple[Path, np.ndarray, dict]:
    """(path of the images_<res>.npy array, labels, build_info) of a built dataset."""
    d = built_dir(name, root)
    hint = build_hint or f"Build it first (see README, 'Datasets'); data root is {data_root(root)}."
    images = require_file(d / f"images_{res}.npy", hint)
    labels = np.load(require_file(d / "labels.npy", hint))
    info = read_json(require_file(d / "build_info.json", hint))
    return images, labels, info


def split_rows(split_column: np.ndarray, split: str) -> np.ndarray:
    """Row indices of ``split`` given each row's base split ("train", "val" or "test")."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    wanted = ("train", "val") if split == "trainval" else (split,)
    return np.flatnonzero(np.isin(split_column, wanted)).astype(np.int64)
