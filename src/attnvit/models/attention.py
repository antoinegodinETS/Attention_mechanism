"""Attention mechanisms behind one pluggable slot, and the registry that names them.

Every mechanism is an ``nn.Module`` built as ``cls(dim, num_heads, layer_idx=..., depth=...,
kernel=..., sdpa_backend=..., **attn_kwargs)`` whose ``forward(x, H, W)`` maps ``x`` of shape
``(B, N, dim)`` to the same shape. ``N = num_prefix + H * W``: the patch tokens are the last
``H * W`` rows in row-major order, preceded by ``num_prefix`` extra tokens (the CLS token, so
0 or 1). Mechanisms that need the 2-D grid (shifted windows, axial) reshape the patch tokens
and run :class:`MultiHeadAttention` on token groups, so all of them share one attention call.
A mechanism may also define ``check_grid(H, W)``, which the model calls once at construction
to reject a patch grid it cannot handle.

A mechanism registers itself with :func:`register_mechanism`, which also records the
classification head its original paper uses (``"cls"`` or ``"gap"``) and whether it accepts
prefix tokens. Registered: ``full`` (ViT, CLS head), ``windows`` (flat Swin, average pooling),
``axial`` (average pooling) and ``axial_shared`` (the parameter-matched axial ablation).
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


def _check_grid(x: torch.Tensor, H: int, W: int, name: str) -> None:
    if x.shape[1] != H * W:
        raise ValueError(f"{name} takes only the H * W = {H * W} patch tokens (no CLS token), got {x.shape[1]}")


def window_partition(x: torch.Tensor, M: int) -> torch.Tensor:
    """``(B, H, W, d)`` -> ``(B * (H/M) * (W/M), M * M, d)``: non-overlapping M x M windows, row-major."""
    B, H, W, d = x.shape
    return x.view(B, H // M, M, W // M, M, d).permute(0, 1, 3, 2, 4, 5).reshape(-1, M * M, d)


def window_reverse(windows: torch.Tensor, M: int, H: int, W: int) -> torch.Tensor:
    """Inverse of :func:`window_partition`: ``(B * nW, M * M, d)`` -> ``(B, H, W, d)``."""
    d = windows.shape[-1]
    B = windows.shape[0] // ((H // M) * (W // M))
    return windows.view(B, H // M, W // M, M, M, d).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, d)


def shifted_window_mask(H: int, W: int, M: int, shift: int, device=None) -> torch.Tensor:
    """Boolean mask ``(nW, M * M, M * M)`` for shifted windows, True where attention is allowed.

    After the cyclic shift by ``-shift`` the last row and column of windows hold tokens from
    opposite edges of the image. Label the shifted grid with the 3 x 3 regions of Swin (the
    bulk, the ``M - shift`` band and the wrapped ``shift`` band in each direction); two tokens
    of one window may attend to each other only when they carry the same label, i.e. when they
    were neighbours before the shift.
    """
    labels = torch.zeros(H, W, dtype=torch.long, device=device)
    bands = (slice(0, -M), slice(-M, -shift), slice(-shift, None))
    for i, hs in enumerate(bands):
        for j, ws in enumerate(bands):
            labels[hs, ws] = 3 * i + j
    win = window_partition(labels.view(1, H, W, 1), M).squeeze(-1)  # (nW, M*M)
    return win[:, :, None] == win[:, None, :]


@register_mechanism("windows", head="gap")
class ShiftedWindowAttention(nn.Module):
    """Swin's (shifted) window attention in a flat model: W-MSA on even blocks, SW-MSA on odd ones.

    The ``H x W`` patch grid is cut into non-overlapping ``M x M`` windows and every window runs
    :class:`MultiHeadAttention` on its ``M^2`` tokens. On odd blocks (``layer_idx % 2 == 1``) the
    grid is first rolled by ``-floor(M/2)`` in both directions, so windows straddle the previous
    block's window borders, and :func:`shifted_window_mask` stops tokens wrapped across the image
    edge from attending to each other; the result is rolled back. No patch merging and no relative
    position bias (the model has learned absolute position embeddings). When ``M`` equals the grid
    there is one window and, as in Swin, no shift: the layer is full attention.
    Cost per layer: ``4 N d^2 + 2 M^2 N d`` multiply-adds (Report 1).
    """

    def __init__(self, dim: int, num_heads: int, *, layer_idx: int = 0, depth: int = 1, kernel: str = "sdpa",
                 sdpa_backend: str = "auto", qkv_bias: bool = True, window_size: int = 4, shift: bool = True):
        super().__init__()
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift and layer_idx % 2 == 1 else 0
        self.attn = MultiHeadAttention(dim, num_heads, qkv_bias=qkv_bias, kernel=kernel, sdpa_backend=sdpa_backend)
        self._masks: dict = {}

    def check_grid(self, H: int, W: int) -> None:
        M = self.window_size
        if H % M or W % M:
            raise ValueError(f"window size {M} does not divide the {H}x{W} patch grid; pick a window size "
                             f"that divides it (M = 4 fits 8x8 at 32 px / P4, 40x40 at 160 px / P4, "
                             f"20x20 at 160 px / P8)")

    def _shift(self, H: int, W: int) -> int:
        return 0 if (H, W) == (self.window_size, self.window_size) else self.shift_size

    def _mask(self, H: int, W: int, device) -> torch.Tensor:
        key = (H, W, str(device))
        if key not in self._masks:
            self._masks[key] = shifted_window_mask(H, W, self.window_size, self.shift_size, device)
        return self._masks[key]

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        _check_grid(x, H, W, "windows")
        self.check_grid(H, W)
        B, N, d = x.shape
        M, s = self.window_size, self._shift(H, W)
        g = x.view(B, H, W, d)
        mask = None
        if s:
            g = torch.roll(g, shifts=(-s, -s), dims=(1, 2))
            m = self._mask(H, W, x.device)  # (nW, M*M, M*M)
            mask = m.unsqueeze(0).expand(B, -1, -1, -1).reshape(-1, 1, M * M, M * M)
        y = window_reverse(self.attn(window_partition(g, M), mask), M, H, W)
        if s:
            y = torch.roll(y, shifts=(s, s), dims=(1, 2))
        return y.reshape(B, N, d)


@register_mechanism("axial", head="gap")
class AxialAttention(nn.Module):
    """Axial attention: a column pass then a row pass, each with its own QKV and output projections.

    The column pass attends within each column of the ``H x W`` patch grid (``W`` groups of ``H``
    tokens), the row pass within each row of the column pass's output (``H`` groups of ``W``
    tokens). Each pass alone reaches exactly one column or row; their composition reaches every
    token in one layer. The two projection sets give 1.5x the attention parameters of full
    attention (Report 1); :class:`SharedAxialAttention` is the parameter-matched ablation.
    Cost per layer: ``2 (4 N d^2 + 2 N sqrt(N) d)`` multiply-adds on a square grid (Report 1).
    """

    def __init__(self, dim: int, num_heads: int, *, layer_idx: int = 0, depth: int = 1, kernel: str = "sdpa",
                 sdpa_backend: str = "auto", qkv_bias: bool = True):
        super().__init__()
        self.col = MultiHeadAttention(dim, num_heads, qkv_bias=qkv_bias, kernel=kernel, sdpa_backend=sdpa_backend)
        self.row = MultiHeadAttention(dim, num_heads, qkv_bias=qkv_bias, kernel=kernel, sdpa_backend=sdpa_backend)

    def column_pass(self, g: torch.Tensor) -> torch.Tensor:
        """``(B, H, W, d)`` -> ``(B, H, W, d)``, attention along H within each column."""
        B, H, W, d = g.shape
        cols = g.permute(0, 2, 1, 3).reshape(B * W, H, d)
        return self.col(cols).view(B, W, H, d).permute(0, 2, 1, 3)

    def row_pass(self, g: torch.Tensor) -> torch.Tensor:
        """``(B, H, W, d)`` -> ``(B, H, W, d)``, attention along W within each row."""
        B, H, W, d = g.shape
        return self.row(g.reshape(B * H, W, d)).view(B, H, W, d)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        _check_grid(x, H, W, "axial")
        B, N, d = x.shape
        return self.row_pass(self.column_pass(x.view(B, H, W, d))).reshape(B, N, d)


@register_mechanism("axial_shared", head="gap")
class SharedAxialAttention(AxialAttention):
    """Parameter-matched axial ablation: both passes share one set of projections.

    Same compute as :class:`AxialAttention`, same attention parameter count as full attention
    (``4 d^2 + 4 d``). Registered for the optional ablation only; no default run uses it.
    """

    def __init__(self, dim: int, num_heads: int, *, layer_idx: int = 0, depth: int = 1, kernel: str = "sdpa",
                 sdpa_backend: str = "auto", qkv_bias: bool = True):
        nn.Module.__init__(self)
        self.col = MultiHeadAttention(dim, num_heads, qkv_bias=qkv_bias, kernel=kernel, sdpa_backend=sdpa_backend)
        self.row = self.col
