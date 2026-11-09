"""Shared test helpers."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    """Import scripts/<name>.py as a module (scripts are not part of the package)."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while it executes
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return load_script


# A tiny model on 256 synthetic 32 px images: a CPU training epoch takes a fraction of a second.
TINY = [
    "data.name=synthetic",
    "data.num_workers=0",
    "device=cpu",
    "model.patch_size=8",
    "model.dim=64",
    "model.depth=2",
    "model.num_heads=2",
    "optim.batch_size=32",
    "optim.warmup_epochs=1",
]
NO_AUGMENT = [
    "augment.crop_padding=0",
    "augment.hflip=false",
    "augment.randaugment_ops=0",
    "augment.mix_prob=0",
    "augment.label_smoothing=0",
]


@pytest.fixture
def tiny_cfg():
    """``tiny_cfg(*overrides)``: configs/base.yaml with the TINY overrides, then ``overrides``."""
    from attnvit.config import load_config

    def make(*overrides):
        return load_config(REPO_ROOT / "configs" / "base.yaml", TINY + list(overrides))

    return make
