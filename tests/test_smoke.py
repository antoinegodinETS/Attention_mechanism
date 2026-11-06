"""CPU-only smoke tests for the phase 0 scaffolding."""

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_check_env():
    path = REPO_ROOT / "scripts" / "check_env.py"
    spec = importlib.util.spec_from_file_location("check_env", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while it executes
    spec.loader.exec_module(module)
    return module


def test_package_imports():
    import attnvit
    import attnvit.data
    import attnvit.models

    assert attnvit.__version__


def test_check_env_main_on_cpu(capsys, monkeypatch):
    # Force the CPU path so the test behaves the same on the GPU PC and Rorqual.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    check_env = _load_check_env()
    assert check_env.main([]) == 0
    out = capsys.readouterr().out
    assert "torch" in out
    assert "CUDA          unavailable" in out
