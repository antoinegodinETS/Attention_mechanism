"""Derive data_splits/cifar10_hr_synsets.csv from CINIC-10's synset-to-CIFAR-10-class list.

Input: ``data_splits/cinic10_synsets-to-cifar-10-classes.txt``, a verbatim copy of
``synsets-to-cifar-10-classes.txt`` from https://github.com/BayesWatch/cinic-10 (see
``data_splits/README.md``). Each CIFAR-10 class name is followed by its WordNet subtree, one
synset per line, indented with dashes.

Output: one row per synset with ``synset, label, class, words, depth, cinic_images``.
``depth`` is the indentation level (0 = the subtree root). ``cinic_images`` is how many
images CINIC-10 actually took from that synset, read from CINIC's
``imagenet-contributors.csv`` when ``--contributors`` is given (a path or URL), else blank.

One synset is listed under two classes: n03770679 "minivan" (automobile and truck).
CINIC-10's contributors file uses it for automobile only, so it is kept under automobile.

Usage:
  python scripts/make_hr_synsets.py --contributors \
      https://raw.githubusercontent.com/BayesWatch/cinic-10/master/imagenet-contributors.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from attnvit.data.common import CIFAR10_CLASSES, REPO_ROOT

DEFAULT_LIST = REPO_ROOT / "data_splits" / "cinic10_synsets-to-cifar-10-classes.txt"
DEFAULT_OUT = REPO_ROOT / "data_splits" / "cifar10_hr_synsets.csv"
# Synsets listed under more than one class: the class CINIC-10 used them for.
CROSS_CLASS_RESOLUTION = {"n03770679": "automobile"}
LINE = re.compile(r"^(-*)(n\d{8}): (.*)$")


def parse_cinic_list(text: str) -> list[dict]:
    rows: list[dict] = []
    seen: dict[str, str] = {}
    cls = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line in CIFAR10_CLASSES:
            cls = line
            continue
        m = LINE.match(line)
        if not m or cls is None:
            raise ValueError(f"unexpected line in CINIC-10 list: {line!r}")
        dashes, wnid, words = m.groups()
        if wnid in seen:
            prev = seen[wnid]
            if prev == cls:
                continue  # repeated inside the same class subtree
            winner = CROSS_CLASS_RESOLUTION.get(wnid)
            if winner not in (prev, cls):
                raise ValueError(f"{wnid} is listed under {prev} and {cls} with no resolution rule")
            if winner == prev:
                continue
            rows = [r for r in rows if r["synset"] != wnid]
        seen[wnid] = cls
        rows.append(
            {"synset": wnid, "label": CIFAR10_CLASSES.index(cls), "class": cls, "words": words, "depth": len(dashes)}
        )
    # Depth relative to each class root (the shallowest synset of the class).
    root_dashes = {c: min(r["depth"] for r in rows if r["class"] == c) for c in {r["class"] for r in rows}}
    for r in rows:
        r["depth"] = (r["depth"] - root_dashes[r["class"]]) // 4
    rows.sort(key=lambda r: (r["label"], r["synset"]))
    return rows


def read_text(src: str) -> str:
    if src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src) as resp:
            return resp.read().decode()
    return Path(src).read_text(encoding="utf-8")


def contributor_counts(text: str) -> dict[tuple[str, str], int]:
    counts: Counter = Counter()
    for rec in csv.DictReader(io.StringIO(text), skipinitialspace=True):
        counts[(rec["synset"].strip(), rec["class"].strip())] += 1
    return dict(counts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cinic-list", default=str(DEFAULT_LIST))
    p.add_argument("--contributors", help="path or URL of CINIC-10's imagenet-contributors.csv")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args(argv)

    rows = parse_cinic_list(Path(args.cinic_list).read_text(encoding="utf-8"))
    counts = contributor_counts(read_text(args.contributors)) if args.contributors else None
    if counts is not None:
        used = {s for s, _ in counts}
        listed = {r["synset"] for r in rows}
        if used - listed:
            raise ValueError(f"contributors use synsets missing from the list: {sorted(used - listed)[:5]}")
        for (s, c), n in counts.items():
            row = next(r for r in rows if r["synset"] == s)
            if row["class"] != c:
                raise ValueError(f"{s} is mapped to {row['class']} but CINIC-10 used it for {c}")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["synset", "label", "class", "words", "depth", "cinic_images"])
        for r in rows:
            n = "" if counts is None else counts.get((r["synset"], r["class"]), 0)
            w.writerow([r["synset"], r["label"], r["class"], r["words"], r["depth"], n])
    per_class = Counter(r["class"] for r in rows)
    print(f"wrote {args.out}: {len(rows)} synsets; " + ", ".join(f"{c} {per_class[c]}" for c in CIFAR10_CLASSES))
    if counts is not None:
        used = Counter(c for (s, c) in counts)
        print("CINIC-10 contributing synsets: " + ", ".join(f"{c} {used[c]}" for c in CIFAR10_CLASSES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
