"""Forward FLOPs of the reference ViT: the Report 1 formulas and ``FlopCounterMode`` counts.

Report 1 gives the cost of one attention layer in multiply-adds, for ``N`` tokens of width ``d``:

* full attention (MSA):   ``4 N d^2 + 2 N^2 d``
* windows (W-MSA, M x M): ``4 N d^2 + 2 M^2 N d``
* axial (two passes):     ``2 (4 N d^2 + 2 N sqrt(N) d)`` on a square grid

:func:`analytic_flops` turns these into FLOPs (2 per multiply-add, the convention of
``torch.utils.flop_counter``) for the whole model, adding the layers every mechanism shares:
the 2-layer MLP (``2 r N d^2`` multiply-adds per block), the patch embedding and the linear
head. LayerNorm, softmax, GELU and the residual additions are not counted, by either side.
The full-attention model carries a CLS token, so its blocks see ``N + 1`` tokens.

:func:`count_flops` runs the model under ``FlopCounterMode`` and splits the count into the
attention sublayers (``blocks.<i>.attn``) and the rest. A shifted-window mask does not change
the count: the masked logits are computed anyway.
"""

from __future__ import annotations

import re
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

_ATTN_MODULE = re.compile(r"\.blocks\.\d+\.attn$")


def attention_layer_macs(mechanism: str, grid_h: int, grid_w: int, dim: int, *, num_prefix: int = 0,
                         window_size: int = 4) -> int:
    """Multiply-adds of one attention layer (projections plus mixing) per Report 1.

    ``axial_shared`` costs the same as ``axial``: sharing the projections saves parameters, not
    compute. Windows whose size equals the grid are one window, i.e. full attention.
    """
    n_patch = grid_h * grid_w
    n = n_patch + num_prefix
    if mechanism == "full":
        return 4 * n * dim**2 + 2 * n * n * dim
    if mechanism == "windows":
        return 4 * n_patch * dim**2 + 2 * window_size**2 * n_patch * dim
    if mechanism in ("axial", "axial_shared"):
        # Column pass: groups of grid_h tokens; row pass: groups of grid_w tokens. On a square
        # grid this is 2 (4 N d^2 + 2 N sqrt(N) d).
        return 8 * n_patch * dim**2 + 2 * n_patch * (grid_h + grid_w) * dim
    raise ValueError(f"no analytic cost for mechanism {mechanism!r}")


def analytic_flops(mechanism: str, *, img_size: int, patch_size: int, dim: int, depth: int, mlp_ratio: float,
                   num_classes: int = 10, num_prefix: int = 0, window_size: int = 4,
                   in_chans: int = 3) -> dict[str, int]:
    """Forward FLOPs per image: ``attention`` (Report 1, all blocks), ``shared`` (MLP, patch
    embedding, head) and their sum ``total``."""
    grid = img_size // patch_size
    n_patch = grid * grid
    n = n_patch + num_prefix
    attn = depth * attention_layer_macs(mechanism, grid, grid, dim, num_prefix=num_prefix, window_size=window_size)
    mlp = depth * 2 * n * dim * int(dim * mlp_ratio)
    embed = n_patch * in_chans * patch_size**2 * dim
    head = dim * num_classes
    shared = mlp + embed + head
    return {"attention": 2 * attn, "shared": 2 * shared, "total": 2 * (attn + shared)}


def model_analytic_flops(model: nn.Module, *, num_classes: int | None = None) -> dict[str, int]:
    """:func:`analytic_flops` with the dimensions read from a :class:`~attnvit.models.VisionTransformer`."""
    block = model.blocks[0]
    dim = block.norm1.normalized_shape[0]
    proj = model.patch_embed.proj
    return analytic_flops(
        model.mechanism,
        img_size=model.grid * proj.kernel_size[0],
        patch_size=proj.kernel_size[0],
        dim=dim,
        depth=len(model.blocks),
        mlp_ratio=block.mlp.fc1.out_features / dim,
        num_classes=num_classes if num_classes is not None else model.head.out_features,
        num_prefix=model.num_prefix,
        window_size=getattr(block.attn, "window_size", 4),
        in_chans=proj.in_channels,
    )


def count_flops(model: nn.Module, x: torch.Tensor, *, backward: bool = False) -> dict[str, int]:
    """FLOPs counted by ``FlopCounterMode`` for one call ``model(x)``, summed over the batch.

    Returns ``total`` and ``attention`` (the ``blocks.<i>.attn`` modules) for the forward pass;
    with ``backward=True`` also ``train`` (forward plus the backward of ``model(x).sum()``).
    The model's train/eval mode is left as it is; gradients are cleared afterwards.
    """
    out: dict[str, int] = {}
    ctx = nullcontext() if backward else torch.no_grad()
    with ctx:
        with FlopCounterMode(display=False) as fwd:
            y = model(x)
        out["total"] = fwd.get_total_flops()
        out["attention"] = sum(sum(ops.values()) for name, ops in fwd.get_flop_counts().items()
                               if _ATTN_MODULE.search(name))
        if backward:
            with FlopCounterMode(display=False) as bwd:
                y.float().sum().backward()
            out["train"] = out["total"] + bwd.get_total_flops()
            model.zero_grad(set_to_none=True)
    return out
