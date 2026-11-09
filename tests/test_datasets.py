"""get_dataset shapes, dtypes and lengths on real CIFAR-10, skipped when the data root lacks it.

CIFAR-10-HR is covered on a synthetic build in test_cifar10_hr_build.py.
"""

import numpy as np
import pytest
import torch

from attnvit.data import data_root, get_dataset

CIFAR_LENGTHS = {"train": 45000, "val": 5000, "trainval": 50000, "test": 10000}


needs_cifar = pytest.mark.skipif(
    not (data_root() / "cifar10" / "cifar-10-batches-py").is_dir(), reason="CIFAR-10 not under $SYS819_DATA"
)


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
    with pytest.raises(FileNotFoundError, match="build_cifar10_hr"):
        get_dataset("cifar10_hr", 32, "train", root=tmp_path)


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
