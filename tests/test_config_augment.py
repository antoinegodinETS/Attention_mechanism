"""Config loading, run ids, LR schedule and the augmentation recipe (CPU)."""

import math

import pytest
import torch
import yaml

from attnvit.config import (
    ConfigError,
    ExperimentConfig,
    load_config,
    run_id,
    runs_root,
    to_dict,
)
from attnvit.data.augment import MixupCutmix, Normalize, build_train_transform, smooth_one_hot
from attnvit.train import lr_at

from conftest import REPO_ROOT

BASE = REPO_ROOT / "configs" / "base.yaml"


def test_base_yaml_matches_dataclass_defaults():
    with open(BASE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert load_config(BASE) == ExperimentConfig()
    assert set(data) == set(to_dict(ExperimentConfig()))  # every key is written out


def test_overrides_inheritance_and_run_id(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text(f"base: {BASE}\ntag: hr\ndata:\n  name: cifar10_hr\n  res: 160\n")
    cfg = load_config(child, ["seed=2", "model.patch_size=8", "optim.lr=3e-4"])
    assert cfg.data.res == 160 and cfg.data.num_workers == 4  # merged with the parent
    assert cfg.optim.lr == pytest.approx(3e-4)  # "3e-4" is a YAML 1.1 string; coerced
    assert run_id(cfg) == "cifar10_hr-160-p8-full-hr-s2"


@pytest.mark.parametrize(
    "overrides, message",
    [
        (["optim.lrr=1"], "unknown config key"),
        (["model.mechanism=swin"], "not registered"),
        (["optim.accum_steps=3"], "multiple of"),
        (["data.res=30"], "not divisible"),
        (["tag=a-b"], "tag"),
        (["optim.epochs=ten"], "expected an integer"),
    ],
)
def test_config_errors(overrides, message):
    with pytest.raises(ConfigError, match=message):
        load_config(BASE, overrides)


def test_runs_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SYS819_RUNS", str(tmp_path))
    assert runs_root() == tmp_path
    assert runs_root(tmp_path / "x") == tmp_path / "x"


def test_lr_schedule_warmup_then_cosine():
    total, warm, base, low = 100, 10, 1e-3, 1e-5
    lrs = [lr_at(s, total, warm, base, low) for s in range(total)]
    assert lrs[0] == pytest.approx(base / warm)
    assert lrs[warm - 1] == pytest.approx(base)
    assert lrs[warm] == pytest.approx(base)
    assert all(a >= b for a, b in zip(lrs[warm:], lrs[warm + 1:]))
    assert lrs[-1] == pytest.approx(low + 0.5 * (base - low) * (1 + math.cos(math.pi * 89 / 90)))
    assert lr_at(total, total, warm, base, low) == pytest.approx(low)


def test_train_transform_keeps_uint8_shape():
    cfg = ExperimentConfig()
    tf = build_train_transform(32, cfg.augment)
    x = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    out = tf(x)
    assert out.dtype == torch.uint8 and out.shape == (3, 32, 32)
    assert build_train_transform(160, cfg.augment).transforms[0].padding == [20, 20, 20, 20]
    cfg.augment.crop_padding, cfg.augment.hflip, cfg.augment.randaugment_ops = 0, False, 0
    assert build_train_transform(32, cfg.augment) is None


def test_normalize():
    norm = Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    x = torch.full((1, 3, 2, 2), 255, dtype=torch.uint8)
    assert torch.allclose(norm(x), torch.full((1, 3, 2, 2), 2.0))


def test_label_smoothing_targets():
    t = smooth_one_hot(torch.tensor([0, 3]), 10, 0.1)
    assert torch.allclose(t.sum(1), torch.ones(2))
    assert t[0, 0] == pytest.approx(0.91) and t[0, 1] == pytest.approx(0.01)


@pytest.mark.parametrize("mixup, cutmix", [(0.8, 0.0), (0.0, 1.0), (0.8, 1.0)])
def test_mixup_cutmix_targets(mixup, cutmix):
    mixer = MixupCutmix(10, mixup_alpha=mixup, cutmix_alpha=cutmix, prob=1.0, label_smoothing=0.1, seed=0)
    x = torch.randn(8, 3, 32, 32)
    x0 = x.clone()
    y = torch.arange(8)
    for _ in range(5):
        xm, t = mixer(x, y)
        assert xm.shape == x.shape
        assert torch.allclose(t.sum(1), torch.ones(8), atol=1e-6)
        # each row mixes its own label with the mirrored one
        lam = (t[0, 0] - 0.01) / 0.9
        assert torch.allclose(t[0, 7], 0.01 + 0.9 * (1 - lam), atol=1e-6)
    assert torch.equal(x, x0)  # the input batch is not modified in place


def test_mixer_off_and_reseed():
    off = MixupCutmix(10, prob=0.0, label_smoothing=0.0)
    x, y = torch.randn(4, 3, 8, 8), torch.tensor([1, 2, 3, 4])
    xm, t = off(x, y)
    assert xm is x and torch.equal(t.argmax(1), y)
    a = MixupCutmix(10, seed=5)
    b = MixupCutmix(10, seed=6)
    b.reseed(5)
    assert torch.equal(a(x, y)[0], b(x, y)[0])
