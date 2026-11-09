"""Create or verify the committed CIFAR-10 45k/5k train/val split, and write CIFAR-10 QA outputs.

The split file ``data_splits/cifar10_val_idx.json`` is created once and then only verified:
running this script again recomputes the split from the downloaded labels and fails if the
committed file differs. ``--force`` is the only way to overwrite it.

Usage:
  python scripts/make_cifar10_split.py --download        # fetch CIFAR-10 into $SYS819_DATA/cifar10
  python scripts/make_cifar10_split.py                   # create the split file if absent, else verify it
  python scripts/make_cifar10_split.py --qa              # also write results/ and figures/ QA outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from attnvit.data import cifar10
from attnvit.data.common import CIFAR10_CLASSES, REPO_ROOT, data_root, read_json, splits_dir, write_json


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="data root (default: $SYS819_DATA or ~/data/sys819)")
    p.add_argument("--splits", help="split-file directory (default: <repo>/data_splits)")
    p.add_argument("--download", action="store_true", help="download CIFAR-10 with torchvision first")
    p.add_argument("--force", action="store_true", help="overwrite an existing split file that differs")
    p.add_argument("--qa", action="store_true", help="write QA outputs")
    p.add_argument("--results-dir", default=str(REPO_ROOT / "results" / "data_qa"))
    p.add_argument("--figures-dir", default=str(REPO_ROOT / "figures" / "data_qa"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    raw = cifar10.raw_dir(args.root)
    if args.download:
        from torchvision.datasets import CIFAR10

        raw.mkdir(parents=True, exist_ok=True)
        for train in (True, False):
            CIFAR10(str(raw), train=train, download=True)
    print(f"data root   {data_root(args.root)}")

    images, labels = cifar10.load_arrays(True, args.root)
    expected = cifar10.make_split(labels)
    path = splits_dir(args.splits) / cifar10.SPLIT_FILE
    if path.exists() and not args.force:
        current = read_json(path)
        if current != expected:
            print(f"FAIL  {path} differs from the split recomputed from the data (use --force to overwrite)")
            return 1
        print(f"ok    {path} matches the data: {current['n_val']} val, per class {current['val_per_class']}")
    else:
        write_json(path, expected)
        print(f"wrote {path}: {expected['n_val']} val, per class {expected['val_per_class']}")

    if args.qa:
        from attnvit.data.qa import write_qa

        test_images, test_labels = cifar10.load_arrays(False, args.root)
        split_col = np.full(len(labels), "train", dtype=object)
        split_col[expected["val_idx"]] = "val"
        all_images = np.concatenate([images, test_images])
        all_labels = np.concatenate([labels, test_labels])
        all_splits = np.concatenate([split_col, np.full(len(test_labels), "test", dtype=object)])
        qa = write_qa(
            cifar10.NAME,
            all_images,
            all_labels,
            all_splits.astype(str),
            CIFAR10_CLASSES,
            Path(args.results_dir),
            Path(args.figures_dir),
            extra={"split_file": f"data_splits/{cifar10.SPLIT_FILE}", "mean": cifar10.MEAN, "std": cifar10.STD},
        )
        print(f"qa    counts {qa['counts']} balanced={qa['balanced']} -> {args.results_dir}, {args.figures_dir}")
        if not qa["balanced"]:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
