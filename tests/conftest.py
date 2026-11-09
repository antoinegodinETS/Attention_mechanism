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
