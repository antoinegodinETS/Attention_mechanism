"""get_dataset shapes, dtypes and lengths on the real data, skipped when the data root lacks it."""

import pickle

import numpy as np
import pytest
import torch

from attnvit.data import data_root, get_dataset

CIFAR_LENGTHS = {"train": 45000, "val": 5000, "trainval": 50000, "test": 10000}
IMAGENETTE_LENGTHS = {"train": 8522, "val": 947, "trainval": 9469, "test": 3925}


def _have(name: str) -> bool:
    root = data_root()
    if name == "cifar10":
        return (root / "cifar10" / "cifar-10-batches-py").is_dir()
    return (root / name / "build_info.json").exists()


needs_cifar = pytest.mark.skipif(not _have("cifar10"), reason="CIFAR-10 not under $SYS819_DATA")
needs_imagenette = pytest.mark.skipif(not _have("imagenette"), reason="Imagenette not built under $SYS819_DATA")


def test_unknown_names_are_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        get_dataset("mnist", 32, "train")
    with pytest.raises(ValueError, match="unknown split"):
        get_dataset("cifar10", 32, "dev")
    with pytest.raises(ValueError, match="32"):
        get_dataset("cifar10", 160, "train")


def test_missing_data_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="cifar10"):
        get_dataset("cifar10", 32, "train", root=tmp_path)
    with pytest.raises(FileNotFoundError, match="build_imagenette"):
        get_dataset("imagenette", 32, "train", root=tmp_path)


@needs_cifar
@pytest.mark.parametrize("split", list(CIFAR_LENGTHS))
def test_cifar10(split):
    ds = get_dataset("cifar10", 32, split)
    assert len(ds) == CIFAR_LENGTHS[split]
    x, y = ds[0]
    assert x.shape == (3, 32, 32) and x.dtype == torch.uint8
    assert isinstance(y, int) and 0 <= y < 10
    assert ds.images.dtype == np.uint8 and ds.images.shape == (len(ds), 32, 32, 3)
    counts = ds.class_counts()
    assert counts.min() == counts.max() == CIFAR_LENGTHS[split] // 10


@needs_cifar
def test_cifar10_train_val_disjoint_and_normalizer():
    train = get_dataset("cifar10", 32, "train")
    val = get_dataset("cifar10", 32, "val")
    assert not set(train.indices.tolist()) & set(val.indices.tolist())
    xs, ys = val.tensors()
    assert xs.shape == (5000, 3, 32, 32) and xs.dtype == torch.uint8 and ys.dtype == torch.int64
    z = val.normalizer()(xs[:256])
    assert z.dtype == torch.float32 and abs(float(z.mean())) < 0.5


@needs_imagenette
@pytest.mark.parametrize("res", [32, 160])
@pytest.mark.parametrize("split", list(IMAGENETTE_LENGTHS))
def test_imagenette(res, split):
    ds = get_dataset("imagenette", res, split)
    assert len(ds) == IMAGENETTE_LENGTHS[split]
    x, y = ds[len(ds) - 1]
    assert x.shape == (3, res, res) and x.dtype == torch.uint8
    assert 0 <= y < 10
    assert ds.class_counts().sum() == len(ds) and (ds.class_counts() > 0).all()


@needs_imagenette
def test_imagenette_memmap_pickles_without_the_array():
    ds = get_dataset("imagenette", 160, "train")
    assert "memmap" in repr(ds)
    blob = pickle.dumps(ds)
    assert len(blob) < 5_000_000, "a memmap-backed dataset must not pickle its pixels"
    clone = pickle.loads(blob)
    a, b = ds[5], clone[5]
    assert torch.equal(a[0], b[0]) and a[1] == b[1]
    # The 32 px row is the box-downscaled 160 px row.
    small = get_dataset("imagenette", 32, "train")
    ref = a[0].numpy().astype(np.float32).reshape(3, 32, 5, 32, 5).mean(axis=(2, 4))
    assert np.abs(ref - small[5][0].numpy()).max() <= 1
