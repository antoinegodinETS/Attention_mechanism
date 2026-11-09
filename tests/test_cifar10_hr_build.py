"""End-to-end test of scripts/build_cifar10_hr.py on a small synthetic ImageNet-21K tree.

The fake source uses real synset ids from data_splits/cifar10_hr_synsets.csv (two per class,
one of them stored as an uncompressed tar) with random smooth images of varied sizes, some
under 160 px, one corrupt file and planted duplicates, then checks filter, sample, split,
dedupe, output shapes, resumability and the get_dataset loader.
"""

import csv
import gzip
import io
import tarfile

import numpy as np
import pandas as pd
import pytest
from conftest import REPO_ROOT, load_script
from PIL import Image

pytest.importorskip("imagehash")

from attnvit.data import get_dataset
from attnvit.data.common import read_json
from attnvit.data.imageops import near_pairs, phash64, popcount64

PER_CLASS, TEST, VAL_FRAC, RESERVE = 20, 5, 0.2, 8  # -> 12 train, 3 val, 5 test per class
IMAGES_PER_SYNSET = 16  # 2 under 160 px -> 28 usable per class = per-class + reserve
SYNSETS_PER_CLASS = 2
DUP_CLASS, DUP_COPIES = 3, 8  # cat: one picture copied into many files


def smooth_image(rng, w, h):
    """Random low-frequency colour image, so perceptual hashes differ between images."""
    small = rng.integers(0, 256, size=(6, 6, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((w, h), Image.Resampling.BICUBIC)


def pick_synsets():
    with open(REPO_ROOT / "data_splits" / "cifar10_hr_synsets.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        out.setdefault(int(r["label"]), [])
        if len(out[int(r["label"])]) < SYNSETS_PER_CLASS:
            out[int(r["label"])].append(r["synset"])
    return out


@pytest.fixture(scope="module")
def fake_source(tmp_path_factory):
    rng = np.random.default_rng(123)
    src = tmp_path_factory.mktemp("winter21_fake")
    synsets = pick_synsets()
    expect = {"small": 0, "corrupt": 0}
    for label, wnids in synsets.items():
        for s_i, wnid in enumerate(wnids):
            files = {}
            for k in range(IMAGES_PER_SYNSET):
                if k % 6 == 5:  # under the 160 px short-side threshold
                    w, h = int(rng.integers(60, 160)), int(rng.integers(170, 300))
                    expect["small"] += 1
                else:
                    w, h = int(rng.integers(160, 320)), int(rng.integers(160, 320))
                buf = io.BytesIO()
                smooth_image(rng, w, h).save(buf, format="JPEG", quality=92)
                files[f"{wnid}_{k}.JPEG"] = buf.getvalue()
            if label == 0 and s_i == 0:
                files[f"{wnid}_corrupt.JPEG"] = b"\xff\xd8\xff\xe0 this is not a jpeg"
                expect["corrupt"] += 1
            if label == DUP_CLASS and s_i == 0:
                buf = io.BytesIO()
                smooth_image(rng, 240, 200).save(buf, format="JPEG", quality=92)
                for d in range(DUP_COPIES):
                    files[f"{wnid}_dup{d}.JPEG"] = buf.getvalue()
            if label == 9 and s_i == 1:  # one synset as a per-synset tar, read in place
                with tarfile.open(src / f"{wnid}.tar", "w") as tf:
                    for name, data in files.items():
                        info = tarfile.TarInfo(f"{wnid}/{name}")
                        info.size = len(data)
                        tf.addfile(info, io.BytesIO(data))
            else:
                (src / wnid).mkdir()
                for name, data in files.items():
                    (src / wnid / name).write_bytes(data)
    (src / "n99999999").mkdir()  # an unmapped synset that must be ignored
    return src, expect


def run_build(src, base, *extra):
    build = load_script("build_cifar10_hr")
    argv = [
        "--source", str(src), "--release", "fake21",
        "--root", str(base / "data"), "--splits", str(base / "splits"),
        "--results-dir", str(base / "results"), "--figures-dir", str(base / "figures"),
        "--per-class", str(PER_CLASS), "--test-per-class", str(TEST),
        "--val-fraction", str(VAL_FRAC), "--reserve", str(RESERVE),
        "--workers", "2", *extra,
    ]  # fmt: skip
    return build.main(argv)


@pytest.fixture(scope="module")
def built(fake_source, tmp_path_factory):
    src, expect = fake_source
    base = tmp_path_factory.mktemp("hr_build")
    assert run_build(src, base) == 0
    return src, base, expect


def test_outputs_have_expected_shapes(built):
    _, base, _ = built
    out = base / "data" / "cifar10_hr"
    n = 10 * PER_CLASS
    x160 = np.load(out / "images_160.npy", mmap_mode="r")
    x32 = np.load(out / "images_32.npy", mmap_mode="r")
    labels = np.load(out / "labels.npy")
    assert x160.shape == (n, 160, 160, 3) and x160.dtype == np.uint8
    assert x32.shape == (n, 32, 32, 3) and x32.dtype == np.uint8
    assert np.bincount(labels).tolist() == [PER_CLASS] * 10
    # The 32 px copy is the 5x5 box average of the 160 px image.
    ref = x160[7].reshape(32, 5, 32, 5, 3).mean(axis=(1, 3))
    assert np.abs(ref - x32[7]).max() <= 1


def test_filter_drops_small_and_corrupt(built):
    _, base, expect = built
    info = read_json(base / "data" / "cifar10_hr" / "build_info.json")
    filt = info["filter"]
    assert filt[0]["corrupt"] == 1
    assert sum(f["corrupt"] for f in filt) == expect["corrupt"]
    assert sum(f["small"] for f in filt) > 0
    man = pd.read_csv(base / "data" / "cifar10_hr" / "manifest.csv")
    assert not man["file"].str.contains("corrupt").any()
    assert (np.minimum(man["width"], man["height"]) >= 160).all()
    assert man["file"].is_unique


def test_split_counts_and_order(built):
    _, base, _ = built
    man = pd.read_csv(base / "data" / "cifar10_hr" / "manifest.csv")
    counts = man.groupby(["split", "label"]).size().unstack()
    n_val = round(VAL_FRAC * (PER_CLASS - TEST))
    assert (counts.loc["train"] == PER_CLASS - TEST - n_val).all()
    assert (counts.loc["val"] == n_val).all()
    assert (counts.loc["test"] == TEST).all()
    order = man["split"].map({"train": 0, "val": 1, "test": 2}).to_numpy()
    assert (np.diff(order) >= 0).all(), "rows must be ordered train, val, test"
    assert (man["index"] == np.arange(len(man))).all()


def test_no_cross_split_duplicates(built):
    _, base, _ = built
    out = base / "data" / "cifar10_hr"
    man = pd.read_csv(out / "manifest.csv")
    x160 = np.load(out / "images_160.npy", mmap_mode="r")
    hashes = np.array([phash64(x160[i]) for i in range(len(x160))], dtype=np.uint64)
    assert [f"{h:016x}" for h in hashes.tolist()] == man["phash"].tolist()
    splits = man["split"].to_numpy()
    pairs = near_pairs(hashes, hashes, 4)
    cross = pairs[splits[pairs[:, 0]] != splits[pairs[:, 1]]]
    assert len(cross) == 0
    dedupe = read_json(out / "build_info.json")["dedupe"]
    assert dedupe["replaced"] >= 1, "the planted duplicates should have forced a replacement"


def test_rerun_is_a_noop_and_manifest_is_committed(built, fake_source, capsys):
    src, base, _ = built
    out = base / "data" / "cifar10_hr"
    before = (out / "manifest.csv").read_bytes()
    assert run_build(src, base) == 0
    log = capsys.readouterr().out
    assert log.count("done earlier, skipped") == 7
    assert (out / "manifest.csv").read_bytes() == before
    assert gzip.decompress((base / "splits" / "cifar10_hr_manifest.csv.gz").read_bytes()) == before
    # Redoing the split onwards reproduces the same manifest (determinism).
    assert run_build(src, base, "--redo", "split") == 0
    assert (out / "manifest.csv").read_bytes() == before


def test_get_dataset_on_built_hr(built):
    _, base, _ = built
    kw = dict(root=base / "data", splits=base / "splits")
    n_val = round(VAL_FRAC * (PER_CLASS - TEST))
    n_train = PER_CLASS - TEST - n_val
    for res in (32, 160):
        lens = {s: len(get_dataset("cifar10_hr", res, s, **kw)) for s in ("train", "val", "trainval", "test")}
        assert lens == {
            "train": 10 * n_train,
            "val": 10 * n_val,
            "trainval": 10 * (n_train + n_val),
            "test": 10 * TEST,
        }
        ds = get_dataset("cifar10_hr", res, "val", **kw)
        x, y = ds[0]
        assert x.shape == (3, res, res) and str(x.dtype) == "torch.uint8"
        assert 0 <= y < 10
        assert ds.class_counts().tolist() == [n_val] * 10
    # A manifest that does not match the arrays is refused.
    other = base / "other_splits"
    other.mkdir()
    (other / "cifar10_hr_manifest.csv.gz").write_bytes(gzip.compress(b"index,split\n"))
    with pytest.raises(ValueError, match="checksum"):
        get_dataset("cifar10_hr", 32, "train", root=base / "data", splits=other)


def test_qa_outputs(built):
    _, base, _ = built
    for f in ("cifar10_hr_counts.csv", "cifar10_hr_qa.json"):
        assert (base / "results" / f).stat().st_size > 0
    for f in ("cifar10_hr_grid.png", "cifar10_hr_hist.png"):
        assert (base / "figures" / f).stat().st_size > 0
    qa = read_json(base / "results" / "cifar10_hr_qa.json")
    assert qa["balanced"]


def test_mapping_reports_and_fails_when_a_class_is_short(fake_source, tmp_path, capsys):
    src, _ = fake_source
    with pytest.raises(SystemExit, match="candidate files"):
        run_build(src, tmp_path, "--steps", "mapping", "--per-class", "500")
    log = capsys.readouterr().out
    assert "candidate files per class" in log
    report = pd.read_csv(tmp_path / "data" / "cifar10_hr" / "work" / "synsets_report.csv")
    found = report.set_index("synset")["found"]
    assert (found == "tar").sum() == 1
    assert (found == "dir").sum() == 10 * SYNSETS_PER_CLASS - 1
    assert (found == "missing").sum() == len(report) - 10 * SYNSETS_PER_CLASS


def test_changed_parameters_refuse_to_resume(built, fake_source):
    src, base, _ = built
    with pytest.raises(SystemExit, match="other parameters"):
        run_build(src, base, "--seed", "1")


def test_near_pairs_and_popcount():
    a = np.array([0, 0xFF, 2**64 - 1], dtype=np.uint64)
    assert popcount64(a).tolist() == [0, 8, 64]
    b = np.array([1, 0xF0F0], dtype=np.uint64)
    pairs = near_pairs(a, b, 1, max_elems=2)
    assert pairs.tolist() == [[0, 0]]


def test_redo_discards_stale_step_outputs(fake_source, tmp_path):
    src, _ = fake_source
    work = tmp_path / "data" / "cifar10_hr" / "work"
    assert run_build(src, tmp_path, "--steps", "mapping,filter") == 0
    first = (work / "filter" / "0.csv").read_text()
    # A new seed needs --redo mapping; filter must then redo every class, not reuse class files.
    assert run_build(src, tmp_path, "--steps", "mapping,filter", "--redo", "mapping", "--seed", "1") == 0
    assert (work / "filter" / "0.csv").read_text() != first
    assert read_json(work / "state.json")["params"]["seed"] == 1
