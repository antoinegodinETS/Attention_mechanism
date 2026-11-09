"""Deterministic image operations shared by the dataset build scripts (Pillow and numpy only)."""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
from PIL import Image

Status = Literal["ok", "small", "corrupt"]


def resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    """Resize so the short side is ``size`` (Lanczos, antialiased), then center-crop to ``size``²."""
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) != size:
        scale = size / min(w, h)
        nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def load_crop(source: str | bytes, min_side: int, size: int) -> tuple[Status, np.ndarray | None, tuple[int, int]]:
    """Open an image file (path or raw bytes), filter on the short side and return a crop.

    Returns ``(status, array, (width, height))``: ``"small"`` when the short side is under
    ``min_side`` (decided from the header, no decode), ``"corrupt"`` when Pillow cannot open
    or fully decode the file, else ``"ok"`` with a (size, size, 3) uint8 array.
    """
    try:
        fp = io.BytesIO(source) if isinstance(source, bytes) else source
        with Image.open(fp) as img:
            wh = img.size
            if min(wh) < min_side:
                return "small", None, wh
            img.load()
            crop = resize_center_crop(img, size)
        return "ok", np.asarray(crop, dtype=np.uint8), wh
    except Exception:  # noqa: BLE001 - any decoder error means the file is unusable
        return "corrupt", None, (0, 0)


def downscale(images: np.ndarray, size: int) -> np.ndarray:
    """Box-filter a (N, S, S, 3) uint8 batch down to (N, size, size, 3) (area average, as ImageNet32)."""
    out = np.empty((len(images), size, size, 3), dtype=np.uint8)
    for i, a in enumerate(images):
        out[i] = np.asarray(Image.fromarray(np.asarray(a)).resize((size, size), Image.Resampling.BOX))
    return out


def phash64(image: np.ndarray) -> int:
    """64-bit perceptual hash (imagehash.phash, 8x8 DCT) of a (H, W, 3) uint8 image, as an int."""
    import imagehash  # optional "data" extra, only needed by the builders

    h = imagehash.phash(Image.fromarray(np.asarray(image)))
    return int(str(h), 16)


_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Number of set bits of each uint64 element."""
    x = np.asarray(x, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).astype(np.uint8)
    return _POPCOUNT8[x.view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1, dtype=np.uint8)


def near_pairs(a: np.ndarray, b: np.ndarray, threshold: int, max_elems: int = 1 << 23) -> np.ndarray:
    """All (i, j) with Hamming(a[i], b[j]) <= threshold, for uint64 hash arrays; shape (k, 2).

    Works on blocks of at most ``max_elems`` distances (64 MB of uint64 by default).
    """
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    chunk = max(1, max_elems // max(1, len(b)))
    found = []
    for s in range(0, len(a), chunk):
        d = popcount64(a[s : s + chunk, None] ^ b[None, :])
        i, j = np.nonzero(d <= threshold)
        if len(i):
            found.append(np.stack([i + s, j], axis=1))
    return np.concatenate(found) if found else np.zeros((0, 2), dtype=np.int64)


def load_candidate(task: tuple[str, int, int, int, int]) -> tuple[Status, np.ndarray | None, tuple[int, int]]:
    """Multiprocessing worker: ``(location, offset, nbytes, min_side, size)`` -> :func:`load_crop`.

    ``offset < 0`` means ``location`` is an image file; otherwise the image is the
    ``nbytes`` bytes at ``offset`` inside the uncompressed tar file ``location``.
    """
    loc, offset, nbytes, min_side, size = task
    if offset < 0:
        return load_crop(loc, min_side, size)
    try:
        with open(loc, "rb") as f:
            f.seek(offset)
            data = f.read(nbytes)
    except OSError:
        return "corrupt", None, (0, 0)
    return load_crop(data, min_side, size)
