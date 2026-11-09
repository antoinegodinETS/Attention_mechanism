"""Committed split files: stratification, determinism and disjointness (CPU only, no data needed
except where marked)."""

import numpy as np
import pytest

from attnvit.data import cifar10
from attnvit.data.common import read_json, splits_dir, stratified_split

SPLITS = splits_dir()


def test_stratified_split_is_deterministic_and_stratified():
    labels = np.repeat(np.arange(10), [100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
    a = stratified_split(labels, 0.1, seed=0)
    b = stratified_split(labels, 0.1, seed=0)
    c = stratified_split(labels, 0.1, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.bincount(labels[a], minlength=10).tolist() == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert len(np.unique(a)) == len(a)
    assert np.all(np.diff(a) > 0)


def test_stratified_split_per_class_is_independent_of_other_classes():
    labels = np.repeat(np.arange(3), 50)
    full = stratified_split(labels, 0.2, seed=7)
    labels2 = labels.copy()
    labels2[labels == 2] = 5  # rename one class; the others must keep their picks
    other = stratified_split(labels2, 0.2, seed=7)
    assert np.array_equal(full[labels[full] < 2], other[labels2[other] < 2])


def test_cifar10_split_file_structure():
    spec = read_json(SPLITS / cifar10.SPLIT_FILE)
    val = np.array(spec["val_idx"])
    assert len(val) == spec["n_val"] == 5000
    assert spec["val_per_class"] == [500] * 10
    assert spec["n_source"] == 50000
    assert len(np.unique(val)) == 5000 and val.min() >= 0 and val.max() < 50000
    assert np.all(np.diff(val) > 0)


def test_cifar10_split_matches_data_and_is_reproducible():
    try:
        _, labels = cifar10.load_arrays(True)
    except FileNotFoundError:
        pytest.skip("CIFAR-10 not downloaded under the data root")
    spec = read_json(SPLITS / cifar10.SPLIT_FILE)
    assert spec == cifar10.make_split(labels)
    val = np.array(spec["val_idx"])
    train = cifar10.split_indices("train", labels)
    assert len(train) == 45000
    assert not set(train.tolist()) & set(val.tolist())
    assert np.bincount(labels[train], minlength=10).tolist() == [4500] * 10
