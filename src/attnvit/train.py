"""Training loop: AdamW, warmup + cosine, label smoothing, fp16 + GradScaler, accumulation.

One run lives in ``<runs root>/<run_id>/``:

- ``config.yaml``    the resolved configuration (a resumed run must match it);
- ``metrics.jsonl``  one line per completed epoch (losses, accuracies, times, peak memory);
- ``last.pt``        checkpoint written after every epoch; a new call resumes from it;
- ``summary.json``   written once the run is complete; its presence means "done".

Determinism: the model is initialised from ``seed``; every epoch reseeds the global RNGs, the
sampler, the DataLoader workers and the mixup generator from ``(seed, epoch)``. A run resumed
from ``last.pt`` therefore continues exactly as an uninterrupted one would (bitwise on CPU).

Timing: ``epoch_time_s`` is the training part of an epoch, data loading included. It splits into
``data_wait_s``, the host time spent blocked waiting for the next batch from the DataLoader, and
``compute_time_s = epoch_time_s - data_wait_s``. Mechanisms are compared on the compute time;
the loader-inclusive time is reported beside it (a starved loader is a property of the node).

Peak memory: ``peak_mem_mib`` is the allocator's allocated peak (``max_memory_allocated``), the
memory tensors actually used; ``peak_mem_reserved_mib`` is the reserved peak
(``max_memory_reserved``), which also counts the allocator's cache and fragmentation.

Interruption: SIGUSR1 (Slurm's warning before the time limit, ``--signal=B:USR1@300``) stops
the run cleanly. If the current epoch is expected to end within ``stop_grace_s`` it is finished
and checkpointed first; otherwise the partial epoch is dropped and ``last.pt`` (the previous
epoch) stays the resume point. SIGTERM stops at the next step. The CLI then exits with
:data:`EXIT_INTERRUPTED` so a job script can resubmit itself.

Usage: ``python -m attnvit.train configs/base.yaml [key=value ...] [--runs-root DIR]``
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import (
    ConfigError,
    ExperimentConfig,
    check_test_allowed,
    dump_yaml,
    load_config,
    run_id,
    runs_root,
    semantic_dict,
    to_dict,
)
from .data.augment import MixupCutmix, Normalize, build_train_transform
from .data.synthetic import SyntheticImages
from .models.vit import VisionTransformer
from .utils import (
    Timer,
    append_jsonl,
    atomic_torch_save,
    derive_seed,
    host_info,
    peak_memory_mib,
    peak_reserved_memory_mib,
    read_json,
    read_jsonl,
    reset_peak_memory,
    resolve_device,
    seed_everything,
    write_json,
)

EXIT_INTERRUPTED = 3
SYNTHETIC_DATA_SEED = 1234

# Seed streams derived from (seed, epoch, stream).
_STREAM_GLOBAL, _STREAM_MIX, _STREAM_LOADER, _STREAM_SAMPLER = 0, 1, 2, 3


# -- data -----------------------------------------------------------------------------------


@dataclass
class DataBundle:
    train: Dataset
    val: Dataset
    num_classes: int
    mean: list[float]
    std: list[float]
    load_test: Callable[[], Dataset]


def load_datasets(cfg: ExperimentConfig, *, augment: bool = True) -> DataBundle:
    """Train split with the augmentation recipe, val split without, test split on demand."""
    d = cfg.data
    tf = build_train_transform(d.res, cfg.augment) if augment else None
    if d.name == "synthetic":
        def make(split, transform=None):
            return SyntheticImages(d.synthetic_size, d.res, d.synthetic_classes, seed=SYNTHETIC_DATA_SEED,
                                   split=split, transform=transform)
    else:
        from .data import get_dataset

        def make(split, transform=None):
            return get_dataset(d.name, d.res, split, transform=transform)

    def load_test():
        check_test_allowed(cfg)  # search runs never see the test split
        return make(d.test_split)

    train = make(d.train_split, tf)
    val = make(d.val_split)
    return DataBundle(train, val, len(train.classes), list(train.mean), list(train.std), load_test)


class EpochSampler(Sampler[int]):
    """The first ``count`` indices of a random permutation of ``range(n)`` fixed by ``seed``."""

    def __init__(self, n: int, count: int, seed: int):
        self.n, self.count, self.seed = n, count, seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        return iter(torch.randperm(self.n, generator=g)[: self.count].tolist())

    def __len__(self) -> int:
        return self.count


def _worker_init(worker_id: int) -> None:
    import random

    import numpy as np

    seed = torch.initial_seed() % 2**32  # set by the DataLoader from its seeded generator
    random.seed(seed)
    np.random.seed(seed)


class TimedIterator:
    """Iterates ``iterable`` and adds the time spent blocked in ``iter``/``next`` to ``wait_s``."""

    def __init__(self, iterable):
        t0 = time.perf_counter()
        self._it = iter(iterable)  # starts the DataLoader workers
        self.wait_s = time.perf_counter() - t0

    def __iter__(self):
        return self

    def __next__(self):
        t0 = time.perf_counter()
        try:
            return next(self._it)
        finally:
            self.wait_s += time.perf_counter() - t0


def train_loader(ds: Dataset, cfg: ExperimentConfig, epoch: int, steps: int, device: torch.device) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.micro_batch_size,
        sampler=EpochSampler(len(ds), steps * cfg.optim.batch_size, derive_seed(cfg.seed, epoch, _STREAM_SAMPLER)),
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_init,
        generator=torch.Generator().manual_seed(derive_seed(cfg.seed, epoch, _STREAM_LOADER)),
    )


def _steps_per_epoch(data: DataBundle, cfg: ExperimentConfig) -> int:
    """Optimizer steps per epoch; the last incomplete effective batch is dropped."""
    steps = len(data.train) // cfg.optim.batch_size
    if steps < 1:
        raise ConfigError(f"train split has {len(data.train)} images, fewer than one batch of {cfg.optim.batch_size}")
    return steps


def eval_loader(ds: Dataset, cfg: ExperimentConfig, device: torch.device) -> DataLoader:
    return DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.data.num_workers,
                      pin_memory=device.type == "cuda", persistent_workers=cfg.data.num_workers > 0)


# -- model and optimisation -----------------------------------------------------------------


def build_model(cfg: ExperimentConfig, num_classes: int) -> VisionTransformer:
    m = cfg.model
    return VisionTransformer(
        img_size=cfg.data.res,
        patch_size=m.patch_size,
        num_classes=num_classes,
        dim=m.dim,
        depth=m.depth,
        num_heads=m.num_heads,
        mlp_ratio=m.mlp_ratio,
        drop_path=m.drop_path,
        mechanism=m.mechanism,
        head=m.head,
        attn_kernel=m.attn_kernel,
        sdpa_backend=m.sdpa_backend,
        attn_kwargs=m.attn,
    )


def build_optimizer(model: nn.Module, cfg: ExperimentConfig, device: torch.device) -> torch.optim.AdamW:
    """AdamW; no weight decay on biases, norm weights, position embeddings and the CLS token."""
    skip = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name in skip else decay).append(p)
    o = cfg.optim
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": o.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=o.lr, betas=(o.beta1, o.beta2), fused=device.type == "cuda",
    )


def lr_at(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    """Linear warmup over ``warmup_steps`` optimizer steps, then cosine decay to ``min_lr``."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum(-target * F.log_softmax(logits.float(), dim=-1), dim=-1).mean()


def train_step(model, optimizer, scaler, micro_batches, *, normalize: Normalize, mixer: MixupCutmix,
               use_amp: bool, grad_clip: float | None = None):
    """One optimizer step over ``micro_batches`` (a list of uint8 images and labels on the device).

    Mixup/cutmix is applied once, to the whole effective batch, which is then split back into
    micro-batches of the original sizes: the same images are paired and the same random draws
    are used whatever ``accum_steps`` is, so accumulation does not change the training recipe.

    Returns device tensors ``(sum of per-sample losses, number correct)`` and the sample count;
    the caller decides when to synchronise.
    """
    optimizer.zero_grad(set_to_none=True)
    accum = len(micro_batches)
    sizes = [len(y) for _, y in micro_batches]
    labels = torch.cat([y for _, y in micro_batches])
    images, targets = mixer(normalize(torch.cat([x for x, _ in micro_batches])), labels)
    loss_sum = torch.zeros((), device=labels.device)
    correct = torch.zeros((), device=loss_sum.device, dtype=torch.long)
    n = 0
    for x, target, y in zip(images.split(sizes), targets.split(sizes), labels.split(sizes)):
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = soft_cross_entropy(logits, target)
        scaler.scale(loss / accum).backward()
        loss_sum += loss.detach() * len(y)
        correct += (logits.detach().argmax(-1) == y).sum()
        n += len(y)
    if grad_clip:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return loss_sum, correct, n


@torch.no_grad()
def evaluate(model, loader, normalize: Normalize, device: torch.device, use_amp: bool) -> tuple[float, float]:
    """(mean cross-entropy, top-1 accuracy) over ``loader``, without augmentation."""
    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    n = 0
    for x, y in loader:
        x = normalize(x.to(device, non_blocking=True))
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum")
        correct += (logits.argmax(-1) == y).sum()
        n += len(y)
    return loss_sum.item() / max(n, 1), correct.item() / max(n, 1)


# -- interruption ---------------------------------------------------------------------------


class StopSignals:
    """Context manager recording SIGUSR1 / SIGTERM instead of dying (main thread only)."""

    def __init__(self, log=print):
        self.received: int | None = None
        self.log = log
        self._previous: dict[int, object] = {}

    def _handle(self, signum, frame):
        self.received = signum
        self.log(f"received {signal.Signals(signum).name}: stopping at the next safe point")

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGUSR1", "SIGTERM"):
                sig = getattr(signal, name, None)
                if sig is not None:
                    self._previous[sig] = signal.signal(sig, self._handle)
        return self

    def __exit__(self, *exc):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)
        self._previous.clear()
        return False


# -- the run --------------------------------------------------------------------------------


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k != "attn":
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[prefix + k] = v
    return out


def _check_same_config(saved: dict, cfg: ExperimentConfig, path: Path) -> None:
    a = _flatten(semantic_dict(saved))
    b = _flatten(semantic_dict(to_dict(cfg)))
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    if diff:
        detail = ", ".join(f"{k}: {a.get(k)!r} -> {b.get(k)!r}" for k in diff)
        raise ConfigError(f"{path} exists with a different configuration ({detail}); "
                          "use another tag or delete the run directory")


class Trainer:
    """Trains one configuration into ``<runs root>/<run_id>/``, resuming from ``last.pt``."""

    def __init__(self, cfg: ExperimentConfig, *, runs_root_dir=None, log: Callable[[str], None] = print,
                 step_hook: Callable | None = None, stop_grace_s: float = 120.0,
                 stop: "StopSignals | None" = None):
        self.cfg = cfg
        self.run_id = run_id(cfg)
        self.run_dir = runs_root(runs_root_dir) / self.run_id
        self.device = resolve_device(cfg.device)
        self.use_amp = cfg.amp and self.device.type == "cuda"
        self.log = lambda msg: log(f"[{self.run_id}] {msg}")
        self.step_hook = step_hook
        self.stop_grace_s = stop_grace_s
        self.stop = stop  # a caller-owned handler (the queue); else run() installs its own
        self.model: VisionTransformer | None = None

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def ckpt_path(self) -> Path:
        return self.run_dir / "last.pt"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    def _prepare_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.run_dir / "config.yaml"
        if cfg_path.exists():
            import yaml

            with open(cfg_path, encoding="utf-8") as f:
                _check_same_config(yaml.safe_load(f), self.cfg, cfg_path)
        else:
            dump_yaml(self.cfg, cfg_path)

    def run(self, stop_after_epochs: int | None = None) -> str:
        """Train to the end (``"done"``), or stop early: ``"interrupted"`` by a signal,
        ``"paused"`` after ``stop_after_epochs`` epochs of this call."""
        if self.summary_path.exists():
            self.log("summary.json exists: already done")
            return "done"
        self._prepare_dir()
        if self.stop is not None:
            return self._run(self.stop, stop_after_epochs)
        with StopSignals(self.log) as stop:
            return self._run(stop, stop_after_epochs)

    def _run(self, stop: StopSignals, stop_after_epochs: int | None) -> str:
        cfg, dev = self.cfg, self.device
        o = cfg.optim
        data = load_datasets(cfg)
        steps_per_epoch = _steps_per_epoch(data, cfg)
        total_steps = o.epochs * steps_per_epoch
        warmup_steps = int(round(o.warmup_epochs * steps_per_epoch))

        seed_everything(cfg.seed)
        model = build_model(cfg, data.num_classes).to(dev)
        self.model = model
        optimizer = build_optimizer(model, cfg, dev)
        scaler = torch.amp.GradScaler(dev.type, enabled=self.use_amp)
        mixer = MixupCutmix.from_config(cfg.augment, data.num_classes)
        normalize = Normalize(data.mean, data.std, device=dev)
        val_loader = eval_loader(data.val, cfg, dev)
        n_params = sum(p.numel() for p in model.parameters())

        state = {"epoch": 0, "global_step": 0, "train_time_s": 0.0, "data_wait_s": 0.0, "eval_time_s": 0.0, "wall_time_s": 0.0,
                 "peak_mem_mib": None, "peak_mem_reserved_mib": None, "best_val_acc": None, "best_epoch": None, "chunks": 0,
                 "last_val": None}
        if self.ckpt_path.exists():
            ckpt = torch.load(self.ckpt_path, map_location=dev, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            state.update(ckpt["state"])
            self._truncate_metrics(state["epoch"])
            self.log(f"resuming from {self.ckpt_path} after epoch {state['epoch']}/{o.epochs}")
        else:
            self.log(f"start: {n_params / 1e6:.2f} M params, {steps_per_epoch} steps/epoch, device {dev}, "
                     f"micro-batch {cfg.micro_batch_size} x {o.accum_steps}")
        state["chunks"] += 1

        chunk_start = time.perf_counter()
        wall_before = state["wall_time_s"]
        epochs_this_call = 0
        last_epoch_time = last_eval_time = None
        while state["epoch"] < o.epochs:
            epoch = state["epoch"]
            seed_everything(derive_seed(cfg.seed, epoch, _STREAM_GLOBAL))
            mixer.reseed(derive_seed(cfg.seed, epoch, _STREAM_MIX))
            loader = train_loader(data.train, cfg, epoch, steps_per_epoch, dev)

            model.train()
            reset_peak_memory(dev)
            timer = Timer(dev)
            loss_sum = torch.zeros((), device=dev)
            correct = torch.zeros((), device=dev, dtype=torch.long)
            seen = 0
            buf = []
            step_in_epoch = 0
            finishing = False
            dropped = False
            lr = o.lr
            batches = TimedIterator(loader)
            for x, y in batches:
                buf.append((x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)))
                if len(buf) < o.accum_steps:
                    continue
                lr = lr_at(state["global_step"], total_steps, warmup_steps, o.lr, o.min_lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                l_sum, c, n = train_step(model, optimizer, scaler, buf, normalize=normalize, mixer=mixer,
                                         use_amp=self.use_amp, grad_clip=o.grad_clip)
                buf = []
                loss_sum += l_sum
                correct += c
                seen += n
                state["global_step"] += 1
                step_in_epoch += 1
                if self.step_hook is not None:
                    self.step_hook(self, epoch, step_in_epoch)
                if stop.received is not None:
                    usr1 = stop.received == getattr(signal, "SIGUSR1", None)
                    if usr1 and not finishing:
                        frac = step_in_epoch / steps_per_epoch
                        est = last_epoch_time if last_epoch_time else timer.elapsed() / frac
                        remaining = est * (1 - frac) + (last_eval_time or 0.0)
                        if remaining <= self.stop_grace_s:
                            finishing = True
                            self.log(f"finishing epoch {epoch + 1} first (~{remaining:.0f} s left)")
                    if not (usr1 and finishing):
                        dropped = True
                        break
            if dropped:
                self.log(f"stopped during epoch {epoch + 1}; resume point is epoch {state['epoch']} (last.pt)")
                return "interrupted"

            epoch_time = timer.elapsed()
            data_wait = min(batches.wait_s, epoch_time)
            compute_time = epoch_time - data_wait
            peak = peak_memory_mib(dev)
            peak_reserved = peak_reserved_memory_mib(dev)
            eval_timer = Timer(dev)
            val_loss, val_acc = evaluate(model, val_loader, normalize, dev, self.use_amp)
            eval_time = eval_timer.elapsed()
            last_epoch_time, last_eval_time = epoch_time, eval_time

            state["epoch"] = epoch + 1
            state["train_time_s"] += epoch_time
            state["data_wait_s"] += data_wait
            state["eval_time_s"] += eval_time
            if peak is not None:
                state["peak_mem_mib"] = max(state["peak_mem_mib"] or 0.0, peak)
                state["peak_mem_reserved_mib"] = max(state.get("peak_mem_reserved_mib") or 0.0, peak_reserved)
            if state["best_val_acc"] is None or val_acc > state["best_val_acc"]:
                state["best_val_acc"], state["best_epoch"] = val_acc, epoch + 1
            state["last_val"] = {"loss": val_loss, "acc": val_acc}
            state["wall_time_s"] = wall_before + (time.perf_counter() - chunk_start)
            record = {
                "epoch": epoch + 1,
                "step": state["global_step"],
                "lr": lr,
                "train_loss": loss_sum.item() / max(seen, 1),
                "train_acc": correct.item() / max(seen, 1),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch_time_s": epoch_time,
                "data_wait_s": data_wait,
                "compute_time_s": compute_time,
                "eval_time_s": eval_time,
                "images_per_s": seen / epoch_time if epoch_time > 0 else None,
                "peak_mem_mib": peak,
                "peak_mem_reserved_mib": peak_reserved,
                "time": time.time(),
            }
            append_jsonl(self.metrics_path, record)
            atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "scaler": scaler.state_dict(), "state": state, "config": to_dict(cfg),
                               "run_id": self.run_id}, self.ckpt_path)
            peak_txt = f" | peak {peak:.0f} MiB" if peak is not None else ""
            self.log(f"epoch {epoch + 1}/{o.epochs} loss {record['train_loss']:.4f} acc {record['train_acc']:.4f} "
                     f"| val loss {val_loss:.4f} acc {val_acc:.4f} | {epoch_time:.1f} s ({data_wait:.1f} s data wait, +{eval_time:.1f} s eval)"
                     f"{peak_txt} | lr {lr:.2e}")
            epochs_this_call += 1
            if stop.received is not None:
                return "interrupted"
            if stop_after_epochs is not None and epochs_this_call >= stop_after_epochs and state["epoch"] < o.epochs:
                return "paused"

        test_loss = test_acc = None
        if cfg.eval_test:
            check_test_allowed(cfg)
            test_loss, test_acc = evaluate(model, eval_loader(data.load_test(), cfg, dev), normalize, dev,
                                           self.use_amp)
            self.log(f"test loss {test_loss:.4f} acc {test_acc:.4f}")
        self._write_summary(state, n_params, test_loss, test_acc, steps_per_epoch)
        return "done"

    def _truncate_metrics(self, epoch: int) -> None:
        """Drop metric lines past the checkpoint (an epoch logged but not checkpointed)."""
        rows = read_jsonl(self.metrics_path)
        keep = [r for r in rows if r["epoch"] <= epoch]
        if len(keep) != len(rows):
            tmp = self.metrics_path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for r in keep:
                    f.write(json.dumps(r) + "\n")
            tmp.replace(self.metrics_path)

    def _write_summary(self, state: dict, n_params: int, test_loss, test_acc, steps_per_epoch: int) -> None:
        cfg, dev = self.cfg, self.device
        epochs = state["epoch"]
        summary = {
            "run_id": self.run_id,
            "dataset": cfg.data.name,
            "res": cfg.data.res,
            "patch_size": cfg.model.patch_size,
            "mechanism": cfg.model.mechanism,
            "head": self.model.head_type,
            "tag": cfg.tag,
            "seed": cfg.seed,
            "params": n_params,
            "epochs": epochs,
            "steps": state["global_step"],
            "steps_per_epoch": steps_per_epoch,
            "batch_size": cfg.optim.batch_size,
            "micro_batch_size": cfg.micro_batch_size,
            "accum_steps": cfg.optim.accum_steps,
            "final_val_acc": state["last_val"]["acc"],
            "final_val_loss": state["last_val"]["loss"],
            "best_val_acc": state["best_val_acc"],
            "best_epoch": state["best_epoch"],
            "test_acc": test_acc,
            "test_loss": test_loss,
            "train_time_s": state["train_time_s"],
            "time_per_epoch_s": state["train_time_s"] / max(epochs, 1),
            "data_wait_s": state["data_wait_s"],
            "compute_time_s": state["train_time_s"] - state["data_wait_s"],
            "compute_time_per_epoch_s": (state["train_time_s"] - state["data_wait_s"]) / max(epochs, 1),
            "eval_time_s": state["eval_time_s"],
            "wall_time_s": state["wall_time_s"],
            "peak_mem_mib": state["peak_mem_mib"],
            "peak_mem_reserved_mib": state.get("peak_mem_reserved_mib"),
            "chunks": state["chunks"],
            "device": str(dev),
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
            "num_workers": cfg.data.num_workers,
            **host_info(),
            "amp_fp16": self.use_amp,
            "sdpa_backend": cfg.model.sdpa_backend,
            "torch": torch.__version__,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        write_json(self.summary_path, summary)
        self.log(f"done: val acc {summary['final_val_acc']:.4f}, {summary['train_time_s'] / 3600:.2f} h training")


def load_summary(run_dir: Path) -> dict | None:
    p = run_dir / "summary.json"
    return read_json(p) if p.exists() else None


# -- smoke check ----------------------------------------------------------------------------


def smoke_check(cfg: ExperimentConfig) -> float:
    """One optimizer step on one real training batch and one eval batch; writes nothing.

    Returns the training loss. Raises on any error, including a non-finite loss.
    """
    dev = resolve_device(cfg.device)
    use_amp = cfg.amp and dev.type == "cuda"
    data = load_datasets(cfg)
    _steps_per_epoch(data, cfg)
    seed_everything(cfg.seed)
    model = build_model(cfg, data.num_classes).to(dev)
    optimizer = build_optimizer(model, cfg, dev)
    scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)
    mixer = MixupCutmix.from_config(cfg.augment, data.num_classes)
    normalize = Normalize(data.mean, data.std, device=dev)
    loader = DataLoader(data.train, batch_size=cfg.micro_batch_size, shuffle=True, num_workers=0)
    it = iter(loader)
    batches = [tuple(t.to(dev) for t in next(it)) for _ in range(cfg.optim.accum_steps)]
    model.train()
    loss_sum, _, n = train_step(model, optimizer, scaler, batches, normalize=normalize, mixer=mixer,
                                use_amp=use_amp, grad_clip=cfg.optim.grad_clip)
    loss = loss_sum.item() / n
    if not math.isfinite(loss):
        raise RuntimeError(f"non-finite training loss {loss}")
    x, y = next(iter(DataLoader(data.val, batch_size=min(cfg.eval_batch_size, 64))))
    model.eval()
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
        out = model(normalize(x.to(dev)))
    if out.shape != (len(y), data.num_classes):
        raise RuntimeError(f"unexpected logits shape {tuple(out.shape)}")
    return loss


# -- CLI ------------------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("config", help="YAML config, e.g. configs/base.yaml")
    p.add_argument("overrides", nargs="*", help="key=value overrides, e.g. seed=1 optim.epochs=2")
    p.add_argument("--runs-root", default=None, help="runs root (default $SYS819_RUNS or ~/runs/sys819)")
    p.add_argument("--stop-after-epochs", type=int, default=None, help="pause after this many epochs")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    trainer = Trainer(cfg, runs_root_dir=args.runs_root, log=lambda m: print(m, flush=True))
    status = trainer.run(stop_after_epochs=args.stop_after_epochs)
    return EXIT_INTERRUPTED if status == "interrupted" else 0


if __name__ == "__main__":
    sys.exit(main())
