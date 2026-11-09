"""Experiment configuration: YAML files read into dataclasses (no Hydra).

A config file may name a parent with ``base: <path>`` (relative to the file); keys are merged
recursively, child over parent. ``key.sub=value`` overrides (values parsed as YAML) are applied
last. Unknown keys are errors, so a typo fails at load time, not at 3 a.m.

The dataclass defaults are the reference recipe and match ``configs/base.yaml``.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ENV = "SYS819_RUNS"
DEFAULT_RUNS_ROOT = "~/runs/sys819"
TAG_RE = re.compile(r"^[A-Za-z0-9_.]+$")

# Fields that do not change what a run computes; they may differ when a run is resumed.
NON_SEMANTIC_KEYS = {"device", "eval_batch_size", "data.num_workers"}


@dataclass
class DataConfig:
    name: str = "cifar10"  # any get_dataset() name, or "synthetic" (tests, smoke runs)
    res: int = 32
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    num_workers: int = 4
    synthetic_size: int = 256  # images per split when name == "synthetic"
    synthetic_classes: int = 10


@dataclass
class ModelConfig:
    mechanism: str = "full"
    head: str = "auto"  # "auto" takes the registry's head for the mechanism; or "cls" / "gap"
    patch_size: int = 4
    dim: int = 192
    depth: int = 9
    num_heads: int = 3
    mlp_ratio: float = 2.0
    drop_path: float = 0.1
    attn_kernel: str = "sdpa"  # "sdpa" or "naive" (memory benchmark only)
    sdpa_backend: str = "efficient"  # forced on CUDA so every GPU runs the same kernel
    attn: dict = field(default_factory=dict)  # extra keyword arguments for the mechanism


@dataclass
class AugmentConfig:
    crop_padding: float = 0.125  # random crop after padding by this fraction of the side
    hflip: bool = True
    randaugment_ops: int = 2  # 0 disables RandAugment
    randaugment_magnitude: int = 9
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mix_prob: float = 1.0  # probability that a batch is mixed at all (0 disables)
    switch_prob: float = 0.5  # probability of cutmix rather than mixup when both are on
    label_smoothing: float = 0.1


@dataclass
class OptimConfig:
    epochs: int = 100
    batch_size: int = 128  # effective batch per optimizer step
    accum_steps: int = 1  # micro-batches per step: micro-batch = batch_size / accum_steps
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 0.05
    warmup_epochs: float = 10.0
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float | None = None


@dataclass
class ExperimentConfig:
    tag: str = "base"
    seed: int = 0
    device: str = "auto"  # "auto" (CUDA when available), "cuda", "cpu"
    amp: bool = True  # fp16 autocast + GradScaler on CUDA; ignored on CPU
    eval_test: bool = False  # evaluate the test split once, after the last epoch
    eval_batch_size: int = 256
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    @property
    def micro_batch_size(self) -> int:
        return self.optim.batch_size // self.optim.accum_steps


class ConfigError(ValueError):
    pass


# -- dict <-> dataclass ---------------------------------------------------------------------


def to_dict(cfg: ExperimentConfig) -> dict:
    return dataclasses.asdict(cfg)


def _coerce(value: Any, hint: Any, key: str) -> Any:
    origin = getattr(hint, "__origin__", None)
    args = getattr(hint, "__args__", ())
    if hint is Any or hint is dict:
        if hint is dict and not isinstance(value, dict):
            raise ConfigError(f"{key}: expected a mapping, got {value!r}")
        return copy.deepcopy(value)
    if args and type(None) in args:  # Optional[X]
        if value is None:
            return None
        inner = [a for a in args if a is not type(None)]
        return _coerce(value, inner[0], key)
    if origin is not None:
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{key}: expected true/false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{key}: expected an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, str):  # YAML 1.1 reads "1e-3" (no dot) as a string
            try:
                return float(value)
            except ValueError:
                pass
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{key}: expected a number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{key}: expected a string, got {value!r}")
        return value
    return value


def _from_dict(cls, data: dict, prefix: str = ""):
    if not isinstance(data, dict):
        raise ConfigError(f"{prefix or 'config'}: expected a mapping, got {data!r}")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        where = prefix.rstrip(".") or "top level"
        raise ConfigError(f"unknown config key(s) at {where}: {', '.join(unknown)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        key = prefix + name
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _from_dict(hint, value, key + ".")
        else:
            kwargs[name] = _coerce(value, hint, key)
    return cls(**kwargs)


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "attn":
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml_chain(path: Path, seen: tuple = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"config inheritance loop through {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent = data.pop("base", None)
    if parent is None:
        return data
    return _merge(_read_yaml_chain(path.parent / parent, seen + (path,)), data)


def parse_override(text: str) -> tuple[list[str], Any]:
    if "=" not in text:
        raise ConfigError(f"override {text!r} is not of the form key=value")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"override {text!r} has an empty key")
    return key.split("."), yaml.safe_load(raw) if raw.strip() else ""


def apply_overrides(data: dict, overrides) -> dict:
    data = copy.deepcopy(data)
    for text in overrides:
        keys, value = parse_override(text)
        node = data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override {text!r}: {k} is not a section")
        node[keys[-1]] = value
    return data


def resolve_config_path(path: str | os.PathLike) -> Path:
    """A config path as given, else relative to the repository root."""
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p
    alt = REPO_ROOT / p
    return alt if alt.exists() else p


def load_config(path: str | os.PathLike | None = None, overrides=()) -> ExperimentConfig:
    data = to_dict(ExperimentConfig())
    if path is not None:
        p = resolve_config_path(path)
        if not p.exists():
            raise ConfigError(f"config file {path} not found")
        data = _merge(data, _read_yaml_chain(p))
    data = apply_overrides(data, overrides)
    cfg = _from_dict(ExperimentConfig, data)
    validate(cfg)
    return cfg


def config_from_dict(data: dict) -> ExperimentConfig:
    cfg = _from_dict(ExperimentConfig, _merge(to_dict(ExperimentConfig()), data))
    validate(cfg)
    return cfg


def validate(cfg: ExperimentConfig) -> None:
    from .models.attention import HEADS, KERNELS, MECHANISMS, SDPA_BACKENDS

    m, o, d, a = cfg.model, cfg.optim, cfg.data, cfg.augment
    errors = []
    if not TAG_RE.match(cfg.tag):
        errors.append(f"tag {cfg.tag!r} may only contain letters, digits, '_' and '.'")
    if m.mechanism not in MECHANISMS:
        errors.append(f"model.mechanism {m.mechanism!r} is not registered ({sorted(MECHANISMS)})")
    if m.head not in ("auto",) + HEADS:
        errors.append(f"model.head must be auto, cls or gap, got {m.head!r}")
    if m.attn_kernel not in KERNELS:
        errors.append(f"model.attn_kernel must be one of {KERNELS}")
    if m.sdpa_backend not in SDPA_BACKENDS:
        errors.append(f"model.sdpa_backend must be one of {SDPA_BACKENDS}")
    if d.res % m.patch_size:
        errors.append(f"data.res {d.res} is not divisible by model.patch_size {m.patch_size}")
    if m.dim % m.num_heads:
        errors.append(f"model.dim {m.dim} is not divisible by model.num_heads {m.num_heads}")
    if o.accum_steps < 1 or o.batch_size % o.accum_steps:
        errors.append(f"optim.batch_size {o.batch_size} must be a multiple of optim.accum_steps {o.accum_steps}")
    if o.epochs < 1:
        errors.append("optim.epochs must be >= 1")
    if not 0 <= a.mix_prob <= 1 or not 0 <= a.switch_prob <= 1:
        errors.append("augment.mix_prob and augment.switch_prob must be in [0, 1]")
    if not 0 <= a.label_smoothing < 1:
        errors.append("augment.label_smoothing must be in [0, 1)")
    if cfg.device not in ("auto", "cpu", "cuda") and not cfg.device.startswith("cuda:"):
        errors.append(f"device must be auto, cpu or cuda[:N], got {cfg.device!r}")
    if errors:
        raise ConfigError("; ".join(errors))


# -- run identity and locations -------------------------------------------------------------


def run_id(cfg: ExperimentConfig) -> str:
    """``<dataset>-<res>-p<P>-<mech>-<tag>-s<seed>``."""
    return f"{cfg.data.name}-{cfg.data.res}-p{cfg.model.patch_size}-{cfg.model.mechanism}-{cfg.tag}-s{cfg.seed}"


def runs_root(override: str | os.PathLike | None = None) -> Path:
    """``override``, else ``$SYS819_RUNS``, else ``~/runs/sys819`` (outside the repository)."""
    root = override or os.environ.get(RUNS_ENV) or DEFAULT_RUNS_ROOT
    return Path(root).expanduser()


def semantic_dict(data: dict) -> dict:
    """A config dict without the keys listed in :data:`NON_SEMANTIC_KEYS`."""
    data = copy.deepcopy(data)
    for key in NON_SEMANTIC_KEYS:
        *parents, last = key.split(".")
        node = data
        for p in parents:
            node = node.get(p, {})
        node.pop(last, None)
    return data


def dump_yaml(cfg: ExperimentConfig, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
    tmp.replace(path)
