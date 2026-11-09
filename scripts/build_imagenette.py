"""Build Imagenette-160 (the CIFAR-10-HR fallback) as 160 px and 32 px uint8 memmaps.

Reads the official fastai ``imagenette2-160.tgz`` (downloaded with ``--download``), center-
crops every image to 160x160 (Lanczos resize first if the short side is not 160), writes a
32 px box-filtered copy, a manifest and build info under ``$SYS819_DATA/imagenette``, and
the QA outputs under ``results/data_qa`` and ``figures/data_qa``.

The validation split (seeded stratified 10 % of the official train folder) is the committed
``data_splits/imagenette_val.json``: created on the first build, verified on every later one.

Usage:
  python scripts/build_imagenette.py --download
  python scripts/build_imagenette.py --force     # rebuild the arrays even if present
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from attnvit.data import imagenette as inet
from attnvit.data.common import (
    REPO_ROOT,
    channel_stats,
    data_root,
    read_json,
    sha256_bytes,
    sha256_file,
    splits_dir,
    write_json,
)
from attnvit.data.imageops import downscale, load_crop
from attnvit.data.qa import write_qa

SIZE = 160
SMALL = 32


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="data root (default: $SYS819_DATA or ~/data/sys819)")
    p.add_argument("--splits", help="split-file directory (default: <repo>/data_splits)")
    p.add_argument("--archive", help="path of imagenette2-160.tgz (default: <root>/imagenette/raw/)")
    p.add_argument("--download", action="store_true", help="download the archive if it is missing")
    p.add_argument("--force", action="store_true", help="rebuild arrays that already exist")
    p.add_argument("--force-split", action="store_true", help="overwrite a committed split that differs")
    p.add_argument("--results-dir", default=str(REPO_ROOT / "results" / "data_qa"))
    p.add_argument("--figures-dir", default=str(REPO_ROOT / "figures" / "data_qa"))
    return p.parse_args(argv)


def read_archive(archive: Path) -> list[tuple[str, str, int, bytes]]:
    """[(official split, relative file, label, jpeg bytes)] for every image, sorted by file."""
    wnid_to_label = {w: i for i, w in enumerate(inet.WNIDS)}
    items = []
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            parts = m.name.split("/")
            if not m.isfile() or len(parts) != 4 or not parts[3].endswith(".JPEG"):
                continue
            _, official, wnid, _ = parts
            rel = "/".join(parts[1:])
            items.append((official, rel, wnid_to_label[wnid], tar.extractfile(m).read()))
    items.sort(key=lambda t: t[1])
    return items


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    out = data_root(args.root) / inet.NAME
    archive = Path(args.archive) if args.archive else out / "raw" / "imagenette2-160.tgz"
    if not archive.exists():
        if not args.download:
            print(f"FAIL  {archive} missing; rerun with --download")
            return 1
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"get   {inet.URL}")
        urllib.request.urlretrieve(inet.URL, archive)
    archive_sha = sha256_file(archive)
    print(f"archive {archive} sha256 {archive_sha}")

    items = read_archive(archive)
    train_items = [it for it in items if it[0] == "train"]
    test_items = [it for it in items if it[0] == "val"]
    print(f"read  {len(train_items)} official train, {len(test_items)} official val images")

    # Validation split: create once, verify afterwards.
    expected_val = inet.stratified_val_files([it[1] for it in train_items], np.array([it[2] for it in train_items]))
    val_labels = np.array([inet.WNIDS.index(f.split("/")[1]) for f in expected_val])
    split_spec = {
        "dataset": inet.NAME,
        "description": "Validation files: a stratified 10 % of the official imagenette2-160 train folder; "
        "train = the other official train files; test = the official val folder.",
        "generated_by": "scripts/build_imagenette.py",
        "method": "files sorted; per class c: numpy default_rng([seed, c]).permutation(rows of c)[:round(0.1 * n_c)]",
        "archive": "imagenette2-160.tgz",
        "archive_url": inet.URL,
        "archive_sha256": archive_sha,
        "seed": inet.SPLIT_SEED,
        "val_fraction": inet.VAL_FRACTION,
        "n_official_train": len(train_items),
        "n_val": len(expected_val),
        "val_per_class": np.bincount(val_labels, minlength=10).tolist(),
        "val_files": expected_val,
    }
    split_path = splits_dir(args.splits) / inet.SPLIT_FILE
    if split_path.exists() and not args.force_split:
        current = read_json(split_path)
        if current["val_files"] != expected_val:
            print(f"FAIL  {split_path} differs from the split recomputed from the archive (--force-split overwrites)")
            return 1
        if current.get("archive_sha256") != archive_sha:
            print(f"warn  archive sha256 differs from the one recorded in {split_path}; the file lists still match")
        print(f"ok    {split_path} matches: {len(expected_val)} val files")
    else:
        write_json(split_path, split_spec)
        print(f"wrote {split_path}: {len(expected_val)} val files, per class {split_spec['val_per_class']}")

    val_set = set(expected_val)
    ordered = (
        [("train", it) for it in train_items if it[1] not in val_set]
        + [("val", it) for it in train_items if it[1] in val_set]
        + [("test", it) for it in test_items]
    )
    n = len(ordered)
    p160, p32 = out / f"images_{SIZE}.npy", out / f"images_{SMALL}.npy"
    if p160.exists() and p32.exists() and (out / "build_info.json").exists() and not args.force:
        print(f"skip  arrays exist in {out} (use --force to rebuild)")
    else:
        out.mkdir(parents=True, exist_ok=True)
        x160 = open_memmap(p160.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8, shape=(n, SIZE, SIZE, 3))
        labels = np.empty(n, dtype=np.int64)
        rows = []
        not_160 = 0
        for i, (split, (official, rel, label, data)) in enumerate(ordered):
            status, arr, (w, h) = load_crop(data, 1, SIZE)
            if status != "ok":
                print(f"FAIL  {rel}: {status}")
                return 1
            not_160 += min(w, h) != SIZE
            x160[i] = arr
            labels[i] = label
            rows.append([i, split, label, inet.WNIDS[label], rel, official, w, h])
        x160.flush()
        x32 = open_memmap(p32.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8, shape=(n, SMALL, SMALL, 3))
        for s in range(0, n, 1024):
            x32[s : s + 1024] = downscale(x160[s : s + 1024], SMALL)
        x32.flush()
        np.save(out / "labels.npy", labels)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["index", "split", "label", "wnid", "file", "official_split", "width", "height"])
        w.writerows(rows)
        manifest = buf.getvalue().encode()
        (out / "manifest.csv").write_bytes(manifest)
        train_rows = slice(0, sum(1 for s, _ in ordered if s == "train"))
        stats = {}
        for res, arr in ((SIZE, x160), (SMALL, x32)):
            mean, std = channel_stats(arr[train_rows])
            stats[str(res)] = {"mean": mean, "std": std}
        del x160, x32
        p160.with_suffix(".tmp.npy").replace(p160)
        p32.with_suffix(".tmp.npy").replace(p32)
        info = {
            "dataset": inet.NAME,
            "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "archive": str(archive),
            "archive_sha256": archive_sha,
            "n_images": n,
            "counts": {s: sum(1 for x, _ in ordered if x == s) for s in ("train", "val", "test")},
            "source_short_side_not_160": int(not_160),
            "crop": f"Lanczos resize of the short side to {SIZE} when needed, then center crop {SIZE}x{SIZE}",
            "downscale": f"{SIZE} -> {SMALL} px with a box filter",
            "stats": stats,
            "stats_note": "per-channel mean/std in [0, 1] units over the train split",
            "manifest_sha256": sha256_bytes(manifest),
        }
        write_json(out / "build_info.json", info)
        print(f"wrote {p160} {p32} ({n} images) in {time.time() - t0:.0f} s; stats {stats}")

    info = read_json(out / "build_info.json")
    x160 = np.load(p160, mmap_mode="r")
    labels = np.load(out / "labels.npy")
    split_col = np.array([s for s, _ in ordered])
    qa = write_qa(
        inet.NAME,
        x160,
        labels,
        split_col,
        inet.CLASSES,
        Path(args.results_dir),
        Path(args.figures_dir),
        tolerance=0.2,
        extra={
            "note": "Imagenette is not exactly balanced; the check allows max/min <= 1.2 per split.",
            "split_file": f"data_splits/{inet.SPLIT_FILE}",
            "archive_sha256": info["archive_sha256"],
            "source_short_side_not_160": info["source_short_side_not_160"],
            "stats": info["stats"],
        },
    )
    print(f"qa    counts {qa['counts']}")
    print(f"qa    balance {qa['balance']} -> {args.results_dir}, {args.figures_dir}")
    print(f"done  in {time.time() - t0:.0f} s")
    return 0 if qa["balanced"] else 1


if __name__ == "__main__":
    sys.exit(main())
