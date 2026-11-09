"""Build CIFAR-10-HR from ImageNet-21K: the ten CIFAR-10 classes at 160 px, plus a 32 px copy.

Steps, each resumable (a finished step leaves ``<work>/done/<step>`` and is skipped next time):

  mapping  list candidate images of every mapped synset (data_splits/cifar10_hr_synsets.csv)
           and report per-class candidate counts plus missing or empty synsets.
  filter   per class, walk the candidates in a seeded random order; drop corrupt files and
           images whose short side is < 160 px; resize + center-crop the rest to 160x160;
           stop at per-class + reserve accepted images (a uniform random sample of the
           usable images). Fails if a class has fewer than per-class usable images.
  sample   the first per-class (6,000) accepted images of each class; the rest is a reserve.
  split    per class, seeded: 1,000 test, then 10 % of the remaining 5,000 as val (500),
           4,500 train.
  dedupe   perceptual hash (imagehash.phash, 64 bit) of every accepted image; any pair in
           different splits with Hamming distance <= threshold is a leak: the val/test member
           (test when val vs test) is replaced by the next reserve image of its class, and the
           check repeats until no cross-split pair is left.
  write    images_160.npy and images_32.npy (uint8 memmaps, rows train, val, test), labels.npy,
           manifest.csv and build_info.json under --out; the manifest is also written gzipped to
           data_splits/cifar10_hr_manifest.csv.gz, which is the committed split record.
  qa       per-class counts, sample grid and histograms (results/data_qa, figures/data_qa).

Source layout (``--source``), one entry per synset:
  <source>/<wnid>/<wnid>_<n>.JPEG   flat files per synset folder, as on Nibi in
                                    /datashare/imagenet/winter21_whole (the primary route)
  <source>/<wnid>.tar               alternative: an uncompressed per-synset tar as downloaded
                                    from image-net.org, read in place without extracting

Usage (see README "CIFAR-10-HR"):
  python scripts/build_cifar10_hr.py --source /datashare/imagenet/winter21_whole --release winter21
  python scripts/build_cifar10_hr.py ... --steps mapping      # only list and report candidates
  python scripts/build_cifar10_hr.py ... --redo dedupe        # rerun dedupe and every later step
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from attnvit.data import cifar10_hr as hr
from attnvit.data.common import (
    CIFAR10_CLASSES,
    REPO_ROOT,
    channel_stats,
    data_root,
    read_json,
    sha256_bytes,
    sha256_file,
    splits_dir,
    write_json,
)
from attnvit.data.imageops import downscale, load_candidate, near_pairs, phash64

STEPS = ("mapping", "filter", "sample", "split", "dedupe", "write", "qa")
SPLIT_ORDER = ("train", "val", "test")
IMAGE_EXT = (".jpeg", ".jpg", ".png")
# Work files each step produces; --redo deletes them for the step and every later one, so a
# partly resumable step (filter keeps finished classes, dedupe caches hashes) starts clean.
STEP_OUTPUTS = {
    "mapping": ("candidates.csv", "synsets_report.csv", "mapping.json"),
    "filter": ("filter", "pool"),
    "sample": ("sample.csv",),
    "split": ("split.csv",),
    "dedupe": ("phash.npz", "dedupe.csv", "dedupe_log.csv", "dedupe.json"),
}
# Parameters that change the output; a work dir refuses to resume under different values.
KEY_PARAMS = (
    "source",
    "release",
    "synsets_sha256",
    "per_class",
    "test_per_class",
    "val_fraction",
    "reserve",
    "min_side",
    "size",
    "small",
    "seed",
    "hash_threshold",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="ImageNet-21K root (one directory or tar per synset)")
    p.add_argument("--release", required=True, help="ImageNet-21K release name recorded in the build (e.g. winter21)")
    p.add_argument("--root", help="data root (default: $SYS819_DATA or ~/data/sys819)")
    p.add_argument("--out", help="output directory (default: <root>/cifar10_hr)")
    p.add_argument("--work", help="work directory (default: <out>/work)")
    p.add_argument("--synsets", default=str(REPO_ROOT / "data_splits" / hr.SYNSET_FILE))
    p.add_argument("--splits", help="where the gzipped manifest is written (default: <repo>/data_splits)")
    p.add_argument("--results-dir", default=str(REPO_ROOT / "results" / "data_qa"))
    p.add_argument("--figures-dir", default=str(REPO_ROOT / "figures" / "data_qa"))
    p.add_argument("--per-class", type=int, default=6000)
    p.add_argument("--test-per-class", type=int, default=1000)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--reserve", type=int, default=600, help="extra accepted images per class for dedupe")
    p.add_argument("--min-side", type=int, default=160)
    p.add_argument("--size", type=int, default=160)
    p.add_argument("--small", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hash-threshold", type=int, default=4, help="max phash Hamming distance of a duplicate")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--steps", default="all", help=f"comma list of {','.join(STEPS)} or 'all'")
    p.add_argument("--redo", choices=STEPS, help="forget this step and every later one, then run")
    p.add_argument("--drop-pool", action="store_true", help="delete <work>/pool (~5 GB) after a successful write")
    return p.parse_args(argv)


class Build:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out) if args.out else data_root(args.root) / hr.NAME
        self.work = Path(args.work) if args.work else self.out / "work"
        self.splits = splits_dir(args.splits)
        self.per_class = args.per_class
        self.n_test = args.test_per_class
        self.n_val = int(round(args.val_fraction * (args.per_class - args.test_per_class)))
        self.target = args.per_class + args.reserve
        self.params = {
            "source": str(Path(args.source).resolve()),
            "release": args.release,
            "synsets_sha256": sha256_file(args.synsets),
            "per_class": args.per_class,
            "test_per_class": args.test_per_class,
            "val_fraction": args.val_fraction,
            "reserve": args.reserve,
            "min_side": args.min_side,
            "size": args.size,
            "small": args.small,
            "seed": args.seed,
            "hash_threshold": args.hash_threshold,
        }

    def worker_pool(self):
        """A process pool, or None for --workers 1. Spawned, not forked: the parent already runs
        torch threads, and forking a multi-threaded process can deadlock the children."""
        if self.args.workers <= 1:
            return None
        return mp.get_context("spawn").Pool(self.args.workers)

    # -- bookkeeping -----------------------------------------------------------------
    def done(self, step: str) -> bool:
        return (self.work / "done" / step).exists()

    def mark(self, step: str) -> None:
        (self.work / "done").mkdir(parents=True, exist_ok=True)
        (self.work / "done" / step).write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z\n"))

    def prepare(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        state_path = self.work / "state.json"
        if state_path.exists():
            old = read_json(state_path)["params"]
            diff = {k: (old.get(k), self.params[k]) for k in KEY_PARAMS if old.get(k) != self.params[k]}
            if diff and self.args.redo != "mapping":
                raise SystemExit(
                    f"{self.work} was started with other parameters {diff}; use a new --work or --redo mapping"
                )
        if self.args.redo:
            for s in STEPS[STEPS.index(self.args.redo) :]:
                (self.work / "done" / s).unlink(missing_ok=True)
                for name in STEP_OUTPUTS.get(s, ()):
                    target = self.work / name
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink(missing_ok=True)
        write_json(state_path, {"params": self.params})

    def csv_write(self, path: Path, header: list[str], rows) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)
        tmp.replace(path)

    @staticmethod
    def csv_read(path: Path) -> list[dict]:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    # -- 1. mapping ------------------------------------------------------------------
    def step_mapping(self) -> None:
        src = Path(self.args.source)
        if not src.is_dir():
            raise SystemExit(f"--source {src} is not a directory")
        synsets = self.csv_read(Path(self.args.synsets))
        cand_rows, report = [], []
        for s in synsets:
            wnid, label = s["synset"], int(s["label"])
            files, kind = [], "missing"
            d = src / wnid
            tar_path = src / f"{wnid}.tar"
            if d.is_dir():
                kind = "dir"
                with os.scandir(d) as it:
                    names = sorted(e.name for e in it if e.is_file() and e.name.lower().endswith(IMAGE_EXT))
                files = [(n, str(d / n), -1, 0) for n in names]
            elif tar_path.is_file():
                kind = "tar"
                try:
                    with tarfile.open(tar_path, "r:") as tf:
                        members = [
                            m for m in tf if m.isfile() and m.name.lower().endswith(IMAGE_EXT)
                        ]
                    files = sorted(
                        (os.path.basename(m.name), str(tar_path), m.offset_data, m.size) for m in members
                    )
                except tarfile.TarError as e:
                    kind = f"bad-tar ({e})"
            if kind in ("dir", "tar") and not files:
                kind = "empty"
            report.append([wnid, label, s["class"], kind, len(files)])
            cand_rows += [[label, wnid, name, loc, off, size] for name, loc, off, size in files]
        self.csv_write(self.work / "synsets_report.csv", ["synset", "label", "class", "found", "n_files"], report)
        self.csv_write(self.work / "candidates.csv", ["label", "synset", "file", "loc", "offset", "nbytes"], cand_rows)

        per_class = np.bincount([r[0] for r in cand_rows], minlength=10)
        kinds: dict[str, int] = {}
        for r in report:
            kinds[r[3].split(" ")[0]] = kinds.get(r[3].split(" ")[0], 0) + 1
        log(f"mapping: {len(synsets)} mapped synsets in {src}: {kinds}")
        log("mapping: candidate files per class (before filtering):")
        for c, name in enumerate(CIFAR10_CLASSES):
            n_syn = sum(1 for r in report if r[1] == c and r[4] > 0)
            n_all = sum(1 for r in report if r[1] == c)
            log(f"  {c} {name:<11} {per_class[c]:>9} files from {n_syn}/{n_all} synsets")
        short = [CIFAR10_CLASSES[c] for c in range(10) if per_class[c] < self.per_class]
        write_json(
            self.work / "mapping.json",
            {
                "source": self.params["source"],
                "release": self.args.release,
                "synsets_found": kinds,
                "missing_or_empty": [r[0] for r in report if r[4] == 0],
                "candidates_per_class": per_class.tolist(),
            },
        )
        log(f"mapping: details in {self.work / 'synsets_report.csv'}")
        if short:
            raise SystemExit(f"mapping: fewer than {self.per_class} candidate files for {short}; cannot build")

    # -- 2. filter -------------------------------------------------------------------
    def step_filter(self) -> None:
        cands = self.csv_read(self.work / "candidates.csv")
        by_class: dict[int, list[dict]] = {c: [] for c in range(10)}
        for i, r in enumerate(cands):
            r["cand_index"] = i
            by_class[int(r["label"])].append(r)
        (self.work / "pool").mkdir(exist_ok=True)
        (self.work / "filter").mkdir(exist_ok=True)
        pool = self.worker_pool()
        try:
            for c in range(10):
                summary_path = self.work / "filter" / f"{c}.json"
                if summary_path.exists():
                    log(f"filter: class {c} {CIFAR10_CLASSES[c]} already done")
                    continue
                self._filter_class(c, by_class[c], pool)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        summaries = [read_json(self.work / "filter" / f"{c}.json") for c in range(10)]
        log("filter: usable images per class (accepted / examined; rejected small, corrupt):")
        for c, s in enumerate(summaries):
            log(
                f"  {c} {CIFAR10_CLASSES[c]:<11} {s['accepted']:>5} / {s['examined']:>6}"
                f"  small {s['small']:>5}  corrupt {s['corrupt']:>3}  (candidates {s['candidates']})"
            )
        short = [CIFAR10_CLASSES[c] for c, s in enumerate(summaries) if s["accepted"] < self.per_class]
        if short:
            for c in range(10):
                if summaries[c]["accepted"] < self.per_class:
                    (self.work / "filter" / f"{c}.json").unlink()
            raise SystemExit(
                f"filter: fewer than {self.per_class} usable images (short side >= {self.args.min_side}) "
                f"for {short}"
            )

    def _filter_class(self, c: int, cands: list[dict], pool) -> None:
        t0 = time.time()
        cands = sorted(cands, key=lambda r: (r["synset"], r["file"]))
        order = np.random.default_rng([self.args.seed, c]).permutation(len(cands))
        arr = open_memmap(
            self.work / "pool" / f"{c}.npy", mode="w+", dtype=np.uint8, shape=(self.target, self.args.size, self.args.size, 3)
        )
        rows, accepted = [], 0
        counts = {"ok": 0, "small": 0, "corrupt": 0}
        batch = max(64, 64 * self.args.workers)
        pos = 0
        while accepted < self.target and pos < len(order):
            ranks = range(pos, min(pos + batch, len(order)))
            tasks = [
                (
                    cands[order[k]]["loc"],
                    int(cands[order[k]]["offset"]),
                    int(cands[order[k]]["nbytes"]),
                    self.args.min_side,
                    self.args.size,
                )
                for k in ranks
            ]
            results = pool.imap(load_candidate, tasks, chunksize=8) if pool else map(load_candidate, tasks)
            for k, (status, img, (w, h)) in zip(ranks, results):
                if accepted >= self.target:
                    continue  # drain the iterator; results past the target are not recorded
                cand = cands[order[k]]
                slot = -1
                if status == "ok":
                    slot = accepted
                    arr[slot] = img
                    accepted += 1
                counts[status] += 1
                rows.append([k, cand["cand_index"], cand["synset"], cand["file"], status, w, h, slot])
            pos += batch
        arr.flush()
        del arr
        self.csv_write(
            self.work / "filter" / f"{c}.csv",
            ["rank", "cand_index", "synset", "file", "status", "width", "height", "pool_slot"],
            rows,
        )
        write_json(
            self.work / "filter" / f"{c}.json",
            {"candidates": len(cands), "examined": len(rows), "accepted": accepted, **{k: counts[k] for k in ("small", "corrupt")}},
        )
        log(f"filter: class {c} {CIFAR10_CLASSES[c]}: {accepted} accepted of {len(rows)} examined in {time.time() - t0:.0f} s")

    # -- 3. sample / 4. split ----------------------------------------------------------
    def step_sample(self) -> None:
        rows = []
        for c in range(10):
            for r in self.csv_read(self.work / "filter" / f"{c}.csv"):
                slot = int(r["pool_slot"])
                if slot >= 0:
                    role = "sample" if slot < self.per_class else "reserve"
                    rows.append([c, slot, role, r["rank"], r["synset"], r["file"], r["width"], r["height"]])
        self.csv_write(
            self.work / "sample.csv", ["label", "pool_slot", "role", "rank", "synset", "file", "width", "height"], rows
        )
        n_res = [sum(1 for r in rows if r[0] == c and r[2] == "reserve") for c in range(10)]
        log(f"sample: {self.per_class} per class; reserve per class {n_res}")

    def step_split(self) -> None:
        rows = []
        for c in range(10):
            perm = np.random.default_rng([self.args.seed, 100 + c]).permutation(self.per_class)
            split = np.empty(self.per_class, dtype=object)
            split[perm[: self.n_test]] = "test"
            split[perm[self.n_test : self.n_test + self.n_val]] = "val"
            split[perm[self.n_test + self.n_val :]] = "train"
            rows += [[c, slot, split[slot]] for slot in range(self.per_class)]
        self.csv_write(self.work / "split.csv", ["label", "slot", "split"], rows)
        log(
            f"split: per class {self.per_class - self.n_test - self.n_val} train, {self.n_val} val, "
            f"{self.n_test} test"
        )

    # -- 5. dedupe -------------------------------------------------------------------
    def _hashes(self) -> list[np.ndarray]:
        path = self.work / "phash.npz"
        if path.exists():
            z = np.load(path)
            return [z[str(c)] for c in range(10)]
        out = []
        pool = self.worker_pool()
        try:
            for c in range(10):
                n = read_json(self.work / "filter" / f"{c}.json")["accepted"]
                arr = np.load(self.work / "pool" / f"{c}.npy", mmap_mode="r")
                imgs = (np.asarray(arr[i]) for i in range(min(n, self.target)))
                it = pool.imap(phash64, imgs, chunksize=32) if pool else map(phash64, imgs)
                out.append(np.fromiter(it, dtype=np.uint64))
                log(f"dedupe: hashed class {c} ({len(out[-1])} images)")
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        np.savez(path, **{str(c): h for c, h in enumerate(out)})
        return out

    def step_dedupe(self) -> None:
        hashes = self._hashes()
        thr = self.args.hash_threshold
        split_rows = self.csv_read(self.work / "split.csv")
        # Slots: one entry per (class, slot); each slot holds one pool image at a time.
        slot_label = np.array([int(r["label"]) for r in split_rows])
        slot_split = np.array([SPLIT_ORDER.index(r["split"]) for r in split_rows])
        occupant = np.array([int(r["slot"]) for r in split_rows])  # pool slot; starts as slot == pool slot
        next_reserve = {c: self.per_class for c in range(10)}
        available = {c: len(hashes[c]) for c in range(10)}
        log_rows = []

        def slot_hash(idx):
            return np.array([hashes[slot_label[i]][occupant[i]] for i in idx], dtype=np.uint64)

        all_idx = np.arange(len(occupant))
        active = all_idx
        within_split_pairs = None
        for it in range(1000):
            h_all = slot_hash(all_idx)
            pairs = near_pairs(h_all[active], h_all, thr)
            i = active[pairs[:, 0]]
            j = pairs[:, 1]
            if it == 0:
                within_split_pairs = int(np.sum((i < j) & (slot_split[i] == slot_split[j])))
            cross = slot_split[i] != slot_split[j]
            i, j = i[cross], j[cross]
            if len(i) == 0:
                break
            victims = {}
            for a, b in zip(i.tolist(), j.tolist()):
                v, keep = (a, b) if slot_split[a] > slot_split[b] else (b, a)
                victims.setdefault(v, keep)
            replaced = []
            for v, keep in sorted(victims.items()):
                c = int(slot_label[v])
                if next_reserve[c] >= available[c]:
                    raise SystemExit(f"dedupe: reserve of class {CIFAR10_CLASSES[c]} exhausted; rerun with a larger --reserve")
                dist = int(bin(int(hashes[c][occupant[v]]) ^ int(hashes[slot_label[keep]][occupant[keep]])).count("1"))
                log_rows.append(
                    [it, c, v, SPLIT_ORDER[slot_split[v]], int(occupant[v]), next_reserve[c],
                     int(slot_label[keep]), SPLIT_ORDER[slot_split[keep]], int(occupant[keep]), dist]
                )
                occupant[v] = next_reserve[c]
                next_reserve[c] += 1
                replaced.append(v)
            active = np.array(sorted(replaced), dtype=np.int64)
            log(f"dedupe: round {it}: replaced {len(replaced)} images")
        else:
            raise SystemExit("dedupe: did not converge")
        self.csv_write(
            self.work / "dedupe_log.csv",
            ["round", "label", "slot_id", "split", "removed_pool_slot", "new_pool_slot",
             "dup_label", "dup_split", "dup_pool_slot", "distance"],
            log_rows,
        )
        self.csv_write(
            self.work / "dedupe.csv",
            ["label", "slot", "split", "pool_slot", "phash"],
            [
                [int(slot_label[k]), split_rows[k]["slot"], SPLIT_ORDER[slot_split[k]], int(occupant[k]),
                 f"{int(hashes[slot_label[k]][occupant[k]]):016x}"]
                for k in all_idx
            ],
        )
        summary = {
            "hash": "imagehash.phash (64 bit)",
            "threshold": thr,
            "replaced": len(log_rows),
            "replaced_per_split": {s: sum(1 for r in log_rows if r[3] == s) for s in ("val", "test")},
            "reserve_used_per_class": [next_reserve[c] - self.per_class for c in range(10)],
            "within_split_near_pairs_initial": within_split_pairs,
        }
        write_json(self.work / "dedupe.json", summary)
        log(f"dedupe: {summary}")

    # -- 6. write --------------------------------------------------------------------
    def step_write(self) -> None:
        t0 = time.time()
        a = self.args
        final = self.csv_read(self.work / "dedupe.csv")
        sample = {(int(r["label"]), int(r["pool_slot"])): r for r in self.csv_read(self.work / "sample.csv")}
        final.sort(key=lambda r: (SPLIT_ORDER.index(r["split"]), int(r["label"]), int(r["slot"])))
        n = len(final)
        self.out.mkdir(parents=True, exist_ok=True)
        p160, p32 = self.out / f"images_{a.size}.npy", self.out / f"images_{a.small}.npy"
        t160, t32 = p160.with_suffix(".tmp.npy"), p32.with_suffix(".tmp.npy")
        x160 = open_memmap(t160, mode="w+", dtype=np.uint8, shape=(n, a.size, a.size, 3))
        labels = np.empty(n, dtype=np.int64)
        pools = {c: np.load(self.work / "pool" / f"{c}.npy", mmap_mode="r") for c in range(10)}
        rows = []
        for i, r in enumerate(final):
            c, ps = int(r["label"]), int(r["pool_slot"])
            x160[i] = pools[c][ps]
            labels[i] = c
            s = sample[(c, ps)]
            rows.append([i, r["split"], c, CIFAR10_CLASSES[c], s["synset"], f"{s['synset']}/{s['file']}",
                         s["rank"], s["width"], s["height"], r["phash"]])
        x160.flush()
        x32 = open_memmap(t32, mode="w+", dtype=np.uint8, shape=(n, a.small, a.small, 3))
        for s in range(0, n, 1024):
            x32[s : s + 1024] = downscale(x160[s : s + 1024], a.small)
        x32.flush()
        n_train = sum(1 for r in final if r["split"] == "train")
        stats = {}
        for res, arr in ((a.size, x160), (a.small, x32)):
            mean, std = channel_stats(arr[:n_train])
            stats[str(res)] = {"mean": mean, "std": std}
        del x160, x32, pools

        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["index", "split", "label", "class", "synset", "file", "rank", "width", "height", "phash"])
        w.writerows(rows)
        manifest = buf.getvalue().encode()
        np.save(self.out / "labels.npy", labels)
        (self.out / "manifest.csv").write_bytes(manifest)
        self.splits.mkdir(parents=True, exist_ok=True)
        gz = io.BytesIO()
        with gzip.GzipFile(fileobj=gz, mode="wb", mtime=0) as f:
            f.write(manifest)
        (self.splits / hr.MANIFEST_FILE).write_bytes(gz.getvalue())
        t160.replace(p160)
        t32.replace(p32)

        filt = [read_json(self.work / "filter" / f"{c}.json") for c in range(10)]
        info = {
            "dataset": hr.NAME,
            "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "release": a.release,
            "source": self.params["source"],
            "params": self.params,
            "n_images": n,
            "counts": {s: sum(1 for r in final if r["split"] == s) for s in SPLIT_ORDER},
            "mapping": read_json(self.work / "mapping.json"),
            "filter": filt,
            "dedupe": read_json(self.work / "dedupe.json"),
            "crop": f"short side >= {a.min_side}; Lanczos resize of the short side to {a.size}, center crop",
            "downscale": f"{a.size} -> {a.small} px with a box filter",
            "stats": stats,
            "stats_note": "per-channel mean/std in [0, 1] units over the train split",
            "manifest_sha256": sha256_bytes(manifest),
            "manifest_committed_as": f"data_splits/{hr.MANIFEST_FILE}",
        }
        write_json(self.out / "build_info.json", info)
        log(f"write: {n} images -> {p160}, {p32}; manifest -> {self.splits / hr.MANIFEST_FILE} ({time.time() - t0:.0f} s)")
        if a.drop_pool:
            shutil.rmtree(self.work / "pool")
            log("write: removed the work pool")

    # -- 7. qa -----------------------------------------------------------------------
    def step_qa(self) -> None:
        from attnvit.data.qa import write_qa

        info = read_json(self.out / "build_info.json")
        x = np.load(self.out / f"images_{self.args.size}.npy", mmap_mode="r")
        labels = np.load(self.out / "labels.npy")
        import pandas as pd

        manifest = pd.read_csv(self.out / "manifest.csv")
        qa = write_qa(
            hr.NAME,
            x,
            labels,
            manifest["split"].to_numpy().astype(str),
            CIFAR10_CLASSES,
            Path(self.args.results_dir),
            Path(self.args.figures_dir),
            seed=self.args.seed,
            extra={
                "release": info["release"],
                "manifest": f"data_splits/{hr.MANIFEST_FILE}",
                "manifest_sha256": info["manifest_sha256"],
                "candidates_per_class": info["mapping"]["candidates_per_class"],
                "missing_or_empty_synsets": len(info["mapping"]["missing_or_empty"]),
                "filter": info["filter"],
                "dedupe": info["dedupe"],
                "stats": info["stats"],
            },
        )
        log(f"qa: counts {qa['counts']} balanced={qa['balanced']}")
        if not qa["balanced"]:
            raise SystemExit("qa: classes are not balanced")

    # -- driver ----------------------------------------------------------------------
    def run(self, steps) -> None:
        self.prepare()
        for step in STEPS:
            if step not in steps:
                continue
            if self.done(step):
                log(f"{step}: done earlier, skipped")
                continue
            log(f"{step}: start")
            getattr(self, f"step_{step}")()
            self.mark(step)


def main(argv=None) -> int:
    args = parse_args(argv)
    steps = STEPS if args.steps == "all" else tuple(s.strip() for s in args.steps.split(","))
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise SystemExit(f"unknown steps {sorted(unknown)}; expected {STEPS}")
    build = Build(args)
    log(f"cifar10_hr build: release {args.release}, source {args.source}, out {build.out}, work {build.work}")
    build.run(steps)
    log("finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
