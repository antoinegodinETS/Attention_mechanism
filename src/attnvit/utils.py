"""Seeding, devices, timing and small file helpers."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import socket
import time
from pathlib import Path

import numpy as np
import torch


def derive_seed(*parts: int) -> int:
    """A 31-bit seed that depends only on ``parts`` (e.g. run seed, epoch, stream)."""
    digest = hashlib.sha256(",".join(str(int(p)) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {name!r} requested but CUDA is not available")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device: torch.device) -> float:
    """Seconds taken by ``fn()``, with the device synchronised before and after."""
    synchronize(device)
    t0 = time.perf_counter()
    fn()
    synchronize(device)
    return time.perf_counter() - t0


def free_device_memory(device: torch.device) -> None:
    """Collect garbage and return cached CUDA blocks, e.g. between benchmark cases or after an OOM."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


class Timer:
    """Wall-clock timer that synchronises the device before reading the clock."""

    def __init__(self, device: torch.device):
        self.device = device
        self.start = self._now()

    def _now(self) -> float:
        synchronize(self.device)
        return time.perf_counter()

    def elapsed(self) -> float:
        return self._now() - self.start


def usable_cpu_count() -> int:
    """CPUs this process may run on (the Slurm allocation, not the whole node)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def host_info() -> dict:
    """The CPU side of a run: usable CPUs, Slurm allocation and host name."""
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    return {
        "usable_cpus": usable_cpu_count(),
        "slurm_cpus_per_task": int(cpus_per_task) if cpus_per_task and cpus_per_task.isdigit() else cpus_per_task,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
    }


def peak_memory_mib(device: torch.device) -> float | None:
    """Peak memory handed out by the CUDA caching allocator to tensors (``max_memory_allocated``)."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**20


def peak_reserved_memory_mib(device: torch.device) -> float | None:
    """Peak memory the CUDA caching allocator reserved from the driver (``max_memory_reserved``).

    It is at least the allocated peak; the gap is cached, fragmented blocks.
    """
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_reserved(device) / 2**20


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    tmp.replace(path)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, obj) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
