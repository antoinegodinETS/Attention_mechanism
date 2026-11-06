"""Report the PyTorch environment for the SYS819 attention project.

Prints Python and torch versions, CUDA availability and, when a GPU is visible,
its name, compute capability and VRAM, whether one fp16 autocast + GradScaler
training step runs, and which scaled-dot-product-attention (SDPA) backends run a
small forward and backward pass when forced with ``torch.nn.attention.sdpa_kernel``.

Exit status: 0 on a CPU-only machine (GPU checks are skipped) and when every
required GPU check passes; 1 if CUDA is visible but the fp16 step or the
memory-efficient SDPA backend fails. Flash, cuDNN and math are reported only.

Usage: python scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Backends tried in this order. Names follow torch.nn.attention.SDPBackend members;
# any member missing from the installed torch is reported as unavailable.
SDPA_BACKENDS = ("MATH", "EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION")
REQUIRED_SDPA_BACKEND = "EFFICIENT_ATTENTION"


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _short_error(exc: BaseException) -> str:
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {first_line}"[:160]


def check_fp16_step(device: torch.device) -> CheckResult:
    """Run one training step of a tiny MLP under fp16 autocast with a GradScaler."""
    try:
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 10)
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(device.type)
        x = torch.randn(32, 64, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize(device)
        if logits.dtype != torch.float16:
            return CheckResult("fp16 autocast + GradScaler", False, f"logits dtype {logits.dtype}")
        if not torch.isfinite(loss):
            return CheckResult("fp16 autocast + GradScaler", False, f"non-finite loss {loss.item()}")
        return CheckResult("fp16 autocast + GradScaler", True, f"loss {loss.item():.4f}")
    except Exception as exc:  # report, do not crash
        return CheckResult("fp16 autocast + GradScaler", False, _short_error(exc))


def check_sdpa_backend(backend_name: str, device: torch.device) -> CheckResult:
    """Force one SDPA backend and run a small fp16 forward + backward pass."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError as exc:
        return CheckResult(backend_name, False, f"torch.nn.attention unavailable ({exc})")
    backend = getattr(SDPBackend, backend_name, None)
    if backend is None:
        return CheckResult(backend_name, False, "not exposed by this torch")
    try:
        torch.manual_seed(0)
        # (batch, heads, tokens, head_dim): 3 heads of 64, like the d=192 reference model.
        shape = (2, 3, 256, 64)
        q, k, v = (
            torch.randn(shape, device=device, dtype=torch.float16, requires_grad=True)
            for _ in range(3)
        )
        with sdpa_kernel(backend):
            out = F.scaled_dot_product_attention(q, k, v)
            out.float().sum().backward()
        torch.cuda.synchronize(device)
        if not (torch.isfinite(out).all() and torch.isfinite(q.grad).all()):
            return CheckResult(backend_name, False, "non-finite output or gradient")
        return CheckResult(backend_name, True, "forward+backward ok")
    except Exception as exc:
        return CheckResult(backend_name, False, _short_error(exc))


def main(argv: list[str] | None = None) -> int:
    del argv  # no options yet
    print(f"Python        {platform.python_version()} ({sys.executable})")
    print(f"torch         {torch.__version__}")
    print(f"torch CUDA    {torch.version.cuda or 'none (CPU build)'}")
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    print(f"cuDNN         {cudnn or 'none'}")
    print(f"CPU threads   {torch.get_num_threads()}")

    if not torch.cuda.is_available():
        print("CUDA          unavailable: GPU checks skipped")
        return 0

    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    print("CUDA          available")
    print(f"GPU count     {torch.cuda.device_count()}")
    print(f"Device        {props.name}")
    print(f"Capability    {major}.{minor}")
    print(f"VRAM total    {props.total_memory / 2**30:.2f} GiB")
    print(f"bf16 support  {torch.cuda.is_bf16_supported()} (the project uses fp16 everywhere)")

    failures: list[str] = []

    fp16 = check_fp16_step(device)
    print(f"[{'ok' if fp16.ok else 'FAIL'}] {fp16.name}: {fp16.detail}")
    if not fp16.ok:
        failures.append(fp16.name)

    print("SDPA backends (fp16, forced with sdpa_kernel):")
    for name in SDPA_BACKENDS:
        result = check_sdpa_backend(name, device)
        required = name == REQUIRED_SDPA_BACKEND
        tag = "ok" if result.ok else ("FAIL" if required else "no")
        suffix = " (required)" if required else ""
        print(f"  [{tag}] {name}{suffix}: {result.detail}")
        if required and not result.ok:
            failures.append(name)

    peak = torch.cuda.max_memory_allocated(device) / 2**20
    print(f"Peak memory   {peak:.1f} MiB during checks")

    if failures:
        print(f"RESULT        FAIL: {', '.join(failures)}")
        return 1
    print("RESULT        ok: CUDA, fp16 and memory-efficient SDPA work")
    return 0


if __name__ == "__main__":
    sys.exit(main())
