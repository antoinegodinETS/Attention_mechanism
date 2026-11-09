"""Quality-assurance outputs written by the dataset scripts.

For one dataset this writes, all small enough to commit:

- ``<results_dir>/<name>_counts.csv``: images per class and split;
- ``<results_dir>/<name>_qa.json``: the balance check per split plus any extra build facts;
- ``<figures_dir>/<name>_grid.png``: a seeded sample grid, one row per class (train split);
- ``<figures_dir>/<name>_hist.png``: class histogram per split and the pixel-intensity
  histogram per split (on a seeded subsample).
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from .common import write_json

QA_SPLITS = ("train", "val", "test")


def class_counts(labels: np.ndarray, splits: np.ndarray, n_classes: int) -> dict[str, list[int]]:
    """{split: per-class counts} for the splits present, in QA_SPLITS order."""
    out = {}
    for s in QA_SPLITS:
        mask = splits == s
        if mask.any():
            out[s] = np.bincount(labels[mask], minlength=n_classes).astype(int).tolist()
    return out


def balance_report(counts: dict[str, list[int]], tolerance: float) -> dict[str, dict]:
    """Per split: min, max, max/min ratio and whether the ratio stays within ``1 + tolerance``."""
    rep = {}
    for s, c in counts.items():
        lo, hi = min(c), max(c)
        ratio = hi / lo if lo else float("inf")
        rep[s] = {"min": lo, "max": hi, "ratio": round(ratio, 4), "balanced": bool(ratio <= 1 + tolerance)}
    return rep


def sample_grid(
    images: np.ndarray,
    labels: np.ndarray,
    rows_mask: np.ndarray,
    classes: Sequence[str],
    path: Path,
    per_class: int = 8,
    tile: int = 48,
    seed: int = 0,
) -> list[int]:
    """Save a (classes x per_class) grid of seeded random images; return the rows used."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n_cls = len(classes)
    canvas = np.full((n_cls * tile, per_class * tile, 3), 255, dtype=np.uint8)
    used = []
    for c in range(n_cls):
        idx = np.flatnonzero((labels == c) & rows_mask)
        pick = np.sort(rng.choice(idx, size=min(per_class, len(idx)), replace=False)) if len(idx) else []
        for k, r in enumerate(pick):
            t = Image.fromarray(np.asarray(images[r])).resize((tile, tile), Image.Resampling.BOX)
            canvas[c * tile : (c + 1) * tile, k * tile : (k + 1) * tile] = np.asarray(t)
            used.append(int(r))
    fig_w = per_class * tile / 100 + 1.4
    fig, ax = plt.subplots(figsize=(fig_w, n_cls * tile / 100 + 0.3), dpi=100)
    ax.imshow(canvas, interpolation="nearest")
    ax.set_yticks([c * tile + tile / 2 for c in range(n_cls)], [str(x) for x in classes], fontsize=7)
    ax.set_xticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return used


def histograms(
    images: np.ndarray,
    splits: np.ndarray,
    counts: dict[str, list[int]],
    classes: Sequence[str],
    path: Path,
    max_images: int = 2000,
    seed: int = 0,
) -> None:
    """Class histogram per split (left) and grey-level intensity histogram per split (right)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2), dpi=100)
    x = np.arange(len(classes))
    width = 0.8 / max(1, len(counts))
    for k, (s, c) in enumerate(counts.items()):
        a1.bar(x + k * width - 0.4 + width / 2, c, width=width, label=s)
    a1.set_xticks(x, [str(c) for c in classes], rotation=45, ha="right", fontsize=7)
    a1.set_ylabel("images")
    a1.set_title("images per class", fontsize=9)
    a1.set_ylim(0, 1.2 * max(max(c) for c in counts.values()))
    a1.legend(fontsize=7, ncols=len(counts), loc="upper center")

    rng = np.random.default_rng(seed)
    for s in counts:
        rows = np.flatnonzero(splits == s)
        rows = np.sort(rng.choice(rows, size=min(max_images, len(rows)), replace=False))
        hist = np.zeros(256, dtype=np.float64)
        for k in range(0, len(rows), 128):
            grey = np.asarray(images[rows[k : k + 128]], dtype=np.uint16).sum(axis=-1) // 3
            hist += np.bincount(grey.ravel(), minlength=256)
        a2.plot(np.arange(256), hist / hist.sum(), label=s, linewidth=1)
    a2.set_xlabel("grey level (RGB mean)")
    a2.set_title(f"pixel intensity (<= {max_images} images per split)", fontsize=9)
    a2.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_qa(
    name: str,
    images: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    classes: Sequence[str],
    results_dir: Path,
    figures_dir: Path,
    *,
    tolerance: float = 0.0,
    extra: dict | None = None,
    seed: int = 0,
) -> dict:
    """Write the counts table, QA JSON, sample grid and histograms; return the QA dict.

    ``images`` (N, H, W, 3), ``labels`` (N,) and ``splits`` (N,) of strings in
    {"train", "val", "test"} describe every image once (trainval is train + val).
    """
    labels = np.asarray(labels)
    splits = np.asarray(splits)
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    counts = class_counts(labels, splits, len(classes))

    with open(results_dir / f"{name}_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "class", *counts.keys(), "total"])
        for c, cname in enumerate(classes):
            row = [counts[s][c] for s in counts]
            w.writerow([c, cname, *row, sum(row)])
        totals = [sum(counts[s]) for s in counts]
        w.writerow(["", "all", *totals, sum(totals)])

    grid_rows = sample_grid(images, labels, splits == "train", classes, figures_dir / f"{name}_grid.png", seed=seed)
    histograms(images, splits, counts, classes, figures_dir / f"{name}_hist.png", seed=seed)

    balance = balance_report(counts, tolerance)
    qa = {
        "dataset": name,
        "n_images": len(labels),
        "resolution": int(images.shape[1]),
        "classes": list(classes),
        "counts": counts,
        "balance_tolerance": tolerance,
        "balance": balance,
        "balanced": all(v["balanced"] for v in balance.values()),
        "grid_rows": grid_rows,
        **(extra or {}),
    }
    write_json(results_dir / f"{name}_qa.json", qa)
    return qa
