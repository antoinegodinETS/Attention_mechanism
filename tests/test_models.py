"""Backbone, heads, attention registry and the naive vs SDPA kernels (CPU)."""

import pytest
import torch
import torch.nn.functional as F

from attnvit.models import (
    MECHANISMS,
    AxialAttention,
    FullAttention,
    MultiHeadAttention,
    SharedAxialAttention,
    ShiftedWindowAttention,
    VisionTransformer,
    get_mechanism,
    naive_attention,
    register_mechanism,
    set_attention_kernel,
)


def small_vit(**kw):
    args = dict(img_size=32, patch_size=4, dim=48, depth=3, num_heads=3, mlp_ratio=2.0)
    args.update(kw)
    return VisionTransformer(**args)


def test_full_is_registered_with_cls_head():
    # Each technique's own head: CLS for full attention (ViT), average pooling for windows and axial.
    spec = get_mechanism("full")
    assert spec.cls is FullAttention
    assert spec.head == "cls" and spec.supports_prefix_tokens
    for name, cls in [("windows", ShiftedWindowAttention), ("axial", AxialAttention),
                      ("axial_shared", SharedAxialAttention)]:
        spec = get_mechanism(name)
        assert spec.cls is cls
        assert spec.head == "gap" and not spec.supports_prefix_tokens
        model = small_vit(mechanism=name)
        assert model.head_type == "gap" and model.num_prefix == 0 and model.cls_token is None
        with pytest.raises(ValueError, match="CLS token"):
            small_vit(mechanism=name, head="cls")
    assert small_vit(mechanism="full").head_type == "cls"
    with pytest.raises(ValueError, match="unknown attention mechanism"):
        get_mechanism("nope")


@pytest.mark.parametrize("head, prefix", [("auto", 1), ("cls", 1), ("gap", 0)])
@pytest.mark.parametrize("img, patch", [(32, 4), (32, 8), (64, 8)])
def test_forward_shapes_both_heads(head, prefix, img, patch):
    model = small_vit(img_size=img, patch_size=patch, head=head, num_classes=7)
    grid = img // patch
    assert model.num_prefix == prefix
    assert model.pos_embed.shape == (1, prefix + grid * grid, 48)
    assert (model.cls_token is not None) == (prefix == 1)
    x = torch.randn(2, 3, img, img)
    assert model.forward_features(x).shape == (2, prefix + grid * grid, 48)
    assert model(x).shape == (2, 7)


def test_reference_model_parameter_counts():
    # Plan, "Compute budget": 2.70 M at 32 px / P4, 2.99 M at 160 px / P4, 2.79 M at 160 px / P8.
    for img, patch, millions in [(32, 4, 2.70), (160, 4, 2.99), (160, 8, 2.79)]:
        model = VisionTransformer(img_size=img, patch_size=patch, drop_path=0.1)
        n = sum(p.numel() for p in model.parameters()) / 1e6
        assert abs(n - millions) < 0.01, (img, patch, n)


def test_head_pools_cls_token_or_mean():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)
    cls_model, gap_model = small_vit(head="cls").eval(), small_vit(head="gap").eval()
    feats = cls_model.forward_features(x)
    assert torch.allclose(cls_model(x), cls_model.head(feats[:, 0]))
    feats = gap_model.forward_features(x)
    assert torch.allclose(gap_model(x), gap_model.head(feats.mean(1)))


def test_registry_head_selection_and_prefix_check():
    @register_mechanism("_test_gap", head="gap")
    class GridOnly(torch.nn.Module):
        def __init__(self, dim, num_heads, **kw):
            super().__init__()
            self.attn = MultiHeadAttention(dim, num_heads)

        def forward(self, x, H, W):
            assert x.shape[1] == H * W
            return self.attn(x)

    try:
        model = small_vit(mechanism="_test_gap")
        assert model.head_type == "gap" and model.num_prefix == 0
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
        with pytest.raises(ValueError, match="CLS token"):
            small_vit(mechanism="_test_gap", head="cls")
    finally:
        MECHANISMS.pop("_test_gap", None)


def test_mechanism_receives_layer_index():
    seen = []

    @register_mechanism("_test_idx", head="cls", supports_prefix_tokens=True)
    class Recorder(FullAttention):
        def __init__(self, dim, num_heads, *, layer_idx, depth, **kw):
            super().__init__(dim, num_heads, **kw)
            seen.append((layer_idx, depth))

    try:
        small_vit(mechanism="_test_idx", depth=4)
        assert seen == [(0, 4), (1, 4), (2, 4), (3, 4)]
    finally:
        MECHANISMS.pop("_test_idx", None)


@pytest.mark.parametrize("mask_kind", [None, "bool", "float"])
def test_naive_matches_sdpa(mask_kind):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 17, 16) for _ in range(3))
    mask = None
    if mask_kind == "bool":
        mask = torch.rand(17, 17) > 0.3
        mask |= torch.eye(17, dtype=torch.bool)  # every query keeps at least one key
    elif mask_kind == "float":
        mask = torch.randn(17, 17)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(naive_attention(q, k, v, mask), ref, atol=1e-5)


@pytest.mark.parametrize("mechanism", ["full", "windows", "axial"])
def test_naive_and_sdpa_models_agree_forward_and_backward(mechanism):
    torch.manual_seed(0)
    model = small_vit(drop_path=0.0, mechanism=mechanism)
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    out = {}
    for kernel in ("sdpa", "naive"):
        set_attention_kernel(model, kernel)
        assert all(m.kernel == kernel for m in model.modules() if isinstance(m, MultiHeadAttention))
        model.zero_grad()
        logits = model(x)
        F.cross_entropy(logits, y).backward()
        out[kernel] = (logits.detach(), [p.grad.clone() for p in model.parameters()])
    assert torch.allclose(out["sdpa"][0], out["naive"][0], atol=1e-5)
    for a, b in zip(out["sdpa"][1], out["naive"][1]):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("mechanism", ["full", "windows", "axial"])
def test_naive_kernel_from_constructor(mechanism):
    torch.manual_seed(0)
    a = small_vit(attn_kernel="sdpa", mechanism=mechanism).eval()
    b = small_vit(attn_kernel="naive", mechanism=mechanism).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(a(x), b(x), atol=1e-5)


def test_drop_path_only_in_training():
    torch.manual_seed(0)
    model = small_vit(drop_path=0.5)
    x = torch.randn(8, 3, 32, 32)
    model.eval()
    assert torch.equal(model(x), model(x))
    model.train()
    assert not torch.equal(model(x), model(x))
    rates = [b.drop_path.p for b in model.blocks]
    assert rates[0] == 0.0 and rates[-1] == pytest.approx(0.5)
