"""Flat (isotropic) Vision Transformer with a pluggable attention slot.

Patch embedding (a P x P stride-P convolution), learned absolute position embeddings,
``depth`` pre-LN blocks ``x + DropPath(Attn(LN(x), H, W))`` then ``x + DropPath(MLP(LN(x)))``
with a 2-layer GELU MLP, a final LayerNorm and a linear classifier. Only the attention
module changes between models. The head follows each mechanism's original paper through the
registry in :mod:`attnvit.models.attention`: a CLS token for full attention (ViT), global
average pooling over the patch tokens for shifted windows (Swin) and axial attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import HEADS, get_mechanism


class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch per sample with probability ``p``."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"p={self.p}"


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, attn: nn.Module, mlp_ratio: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, dim: int):
        super().__init__()
        if img_size % patch_size:
            raise ValueError(f"image size {img_size} is not divisible by patch size {patch_size}")
        self.grid = img_size // patch_size
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, H*W, dim), row-major


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        dim: int = 192,
        depth: int = 9,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
        mechanism: str = "full",
        head: str = "auto",
        attn_kernel: str = "sdpa",
        sdpa_backend: str = "auto",
        attn_kwargs: dict | None = None,
    ):
        super().__init__()
        spec = get_mechanism(mechanism)
        head = spec.head if head == "auto" else head
        if head not in HEADS:
            raise ValueError(f"head must be 'auto' or one of {HEADS}, got {head!r}")
        if head == "cls" and not spec.supports_prefix_tokens:
            raise ValueError(f"mechanism {mechanism!r} does not accept a CLS token; use head 'gap'")
        self.mechanism = mechanism
        self.head_type = head
        self.num_prefix = 1 if head == "cls" else 0

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, dim)
        self.grid = self.patch_embed.grid
        num_tokens = self.num_prefix + self.grid * self.grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if self.num_prefix else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim))

        rates = [drop_path * i / max(depth - 1, 1) for i in range(depth)]  # linear, 0 at block 0
        attn_kwargs = dict(attn_kwargs or {})
        self.blocks = nn.ModuleList(
            Block(
                dim,
                spec.cls(dim, num_heads, layer_idx=i, depth=depth, kernel=attn_kernel,
                         sdpa_backend=sdpa_backend, **attn_kwargs),
                mlp_ratio,
                rates[i],
            )
            for i in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def no_weight_decay(self) -> set[str]:
        """Parameter names excluded from weight decay besides biases and norm weights."""
        return {"pos_embed", "cls_token"}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x, self.grid, self.grid)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        pooled = x[:, 0] if self.head_type == "cls" else x.mean(dim=1)
        return self.head(pooled)
