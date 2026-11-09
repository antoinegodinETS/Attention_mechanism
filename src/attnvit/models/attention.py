"""Attention mechanisms behind one pluggable slot, and the registry that names them.

Every mechanism is an ``nn.Module`` built as ``cls(dim, num_heads, layer_idx=..., depth=...,
kernel=..., sdpa_backend=..., **attn_kwargs)`` whose ``forward(x, H, W)`` maps ``x`` of shape
``(B, N, dim)`` to the same shape. ``N = num_prefix + H * W``: the patch tokens are the last
``H * W`` rows in row-major order, preceded by ``num_prefix`` extra tokens (the CLS token, so
0 or 1). Mechanisms that need the 2-D grid (shifted windows, axial) reshape the patch tokens
and run :class:`MultiHeadAttention` on token groups, so all of them share one attention call.

A mechanism registers itself with :func:`register_mechanism`, which also records the
classification head its original paper uses (``"cls"`` or ``"gap"``) and whether it accepts
prefix tokens. Phase 3 only adds two classes decorated with ``@register_mechanism(...)``.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

KERNELS = ("sdpa", "naive")
SDPA_BACKENDS = ("auto", "efficient", "flash", "math", "cudnn")
HEADS = ("cls", "gap")


@dataclass(frozen=True)
class MechanismSpec:
    """A registered attention mechanism."""

    name: str
    cls: type[nn.Module]
    head: str  # default classification head: "cls" (CLS token) or "gap" (average pooling)
    supports_prefix_tokens: bool


MECHANISMS: dict[str, MechanismSpec] = {}


def register_mechanism(name: str, *, head: str, supports_prefix_tokens: bool = False):
    """Class decorator: register an attention mechanism under ``name``."""
    if head not in HEADS:
        raise ValueError(f"head must be one of {HEADS}, got {head!r}")

    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if name in MECHANISMS and MECHANISMS[name].cls is not cls:
            raise ValueError(f"attention mechanism {name!r} is already registered")
        MECHANISMS[name] = MechanismSpec(name, cls, head, supports_prefix_tokens)
        return cls

    return decorator


def get_mechanism(name: str) -> MechanismSpec:
    try:
        return MECHANISMS[name]
    except KeyError:
        raise ValueError(f"unknown attention mechanism {name!r}; registered: {sorted(MECHANISMS)}") from None


# -- the shared attention call --------------------------------------------------------------


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor | None = None):
    """softmax(q k^T / sqrt(d)) v with the full (n, n) weight matrix materialised.

    Same semantics as ``F.scaled_dot_product_attention``: a boolean mask is True where
    attention is allowed, a float mask is added to the logits. Used by the memory benchmark
    to show the O(N^2) memory that fused kernels avoid.
    """
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    return scores.softmax(dim=-1) @ v


def _sdpa_context(backend: str, is_cuda: bool):
    """Force one SDPA backend on CUDA tensors; a no-op for ``auto`` and on CPU."""
    if backend == "auto" or not is_cuda:
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    member = {
        "efficient": "EFFICIENT_ATTENTION",
        "flash": "FLASH_ATTENTION",
        "math": "MATH",
        "cudnn": "CUDNN_ATTENTION",
    }[backend]
    return sdpa_kernel(getattr(SDPBackend, member))


def attend(q, k, v, attn_mask=None, *, kernel: str = "sdpa", sdpa_backend: str = "auto"):
    """Scaled dot-product attention over ``(..., n, head_dim)`` tensors with the chosen kernel."""
    if kernel == "naive":
        return naive_attention(q, k, v, attn_mask)
    if kernel != "sdpa":
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
    with _sdpa_context(sdpa_backend, q.is_cuda):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


class MultiHeadAttention(nn.Module):
    """QKV projection, :func:`attend`, output projection, over token groups ``(G, n, dim)``.

    Every mechanism is "group the tokens, call this, ungroup", so only the grouping differs.
    """

    def __init__(self, dim: int, num_heads: int, *, qkv_bias: bool = True, kernel: str = "sdpa",
                 sdpa_backend: str = "auto"):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
        if sdpa_backend not in SDPA_BACKENDS:
            raise ValueError(f"sdpa_backend must be one of {SDPA_BACKENDS}, got {sdpa_backend!r}")
        self.num_heads = num_heads
        self.kernel = kernel
        self.sdpa_backend = sdpa_backend
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        G, n, d = x.shape
        q, k, v = self.qkv(x).view(G, n, 3, self.num_heads, d // self.num_heads).permute(2, 0, 3, 1, 4)
        out = attend(q, k, v, attn_mask, kernel=self.kernel, sdpa_backend=self.sdpa_backend)
        return self.proj(out.transpose(1, 2).reshape(G, n, d))


def set_attention_kernel(model: nn.Module, kernel: str) -> None:
    """Switch every attention call in ``model`` to ``kernel`` (``"sdpa"`` or ``"naive"``)."""
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
    for m in model.modules():
        if isinstance(m, MultiHeadAttention):
            m.kernel = kernel


# -- mechanisms -----------------------------------------------------------------------------


@register_mechanism("full", head="cls", supports_prefix_tokens=True)
class FullAttention(nn.Module):
    """Global multi-head self-attention over all tokens (ViT), O(N^2) in the token count."""

    def __init__(self, dim: int, num_heads: int, *, layer_idx: int = 0, depth: int = 1,
                 kernel: str = "sdpa", sdpa_backend: str = "auto", qkv_bias: bool = True):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, qkv_bias=qkv_bias, kernel=kernel,
                                       sdpa_backend=sdpa_backend)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return self.attn(x)
