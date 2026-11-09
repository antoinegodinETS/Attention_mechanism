"""Shifted-window and axial attention (CPU): equivalences, the shift mask, a timm oracle,
receptive fields, gradients and FLOP counts against Report 1's cost formulas."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from attnvit.models import (
    AxialAttention,
    FullAttention,
    MultiHeadAttention,
    SharedAxialAttention,
    ShiftedWindowAttention,
    VisionTransformer,
    shifted_window_mask,
    window_partition,
    window_reverse,
)

DIM, HEADS = 48, 3


def grid_tokens(B, H, W, seed=0):
    return torch.randn(B, H * W, DIM, generator=torch.Generator().manual_seed(seed))


def changed_tokens(module, H, W, r, c):
    """Grid positions whose output moves when the input token at (r, c) is perturbed."""
    torch.manual_seed(0)
    x = grid_tokens(1, H, W)
    y = x.clone()
    y[0, r * W + c] += 1.0
    with torch.no_grad():
        diff = (module(y, H, W) - module(x, H, W)).abs().amax(-1)[0]
    return {(i // W, i % W) for i in torch.nonzero(diff > 1e-6).flatten().tolist()}


# -- shifted windows ------------------------------------------------------------------------


@pytest.mark.parametrize("H, W, M", [(8, 8, 4), (40, 40, 4), (20, 20, 4), (8, 16, 2)])
def test_window_partition_and_reverse(H, W, M):
    g = torch.arange(H * W, dtype=torch.float).view(1, H, W, 1).expand(2, -1, -1, 3).contiguous()
    win = window_partition(g, M)
    assert win.shape == (2 * (H // M) * (W // M), M * M, 3)
    assert win[0, :, 0].tolist() == [r * W + c for r in range(M) for c in range(M)]  # top-left window
    assert win[1, :, 0].tolist() == [r * W + c for r in range(M) for c in range(M, 2 * M)]  # its right neighbour
    assert torch.equal(window_reverse(win, M, H, W), g)


def test_only_odd_blocks_shift():
    shifts = [ShiftedWindowAttention(DIM, HEADS, layer_idx=i, window_size=4).shift_size for i in range(4)]
    assert shifts == [0, 2, 0, 2]
    assert ShiftedWindowAttention(DIM, HEADS, layer_idx=1, window_size=5).shift_size == 2  # floor(M/2)
    assert ShiftedWindowAttention(DIM, HEADS, layer_idx=1, shift=False).shift_size == 0


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_window_equal_to_grid_is_full_attention(layer_idx):
    full = FullAttention(DIM, HEADS)
    win = ShiftedWindowAttention(DIM, HEADS, layer_idx=layer_idx, window_size=8)
    win.load_state_dict(full.state_dict())
    x = grid_tokens(2, 8, 8)
    assert torch.allclose(win(x, 8, 8), full(x, 8, 8), atol=1e-5)


def test_window_model_equal_to_grid_is_full_attention_model():
    torch.manual_seed(0)
    kw = dict(img_size=32, patch_size=4, dim=DIM, depth=3, num_heads=HEADS, head="gap")
    full = VisionTransformer(mechanism="full", **kw).eval()
    win = VisionTransformer(mechanism="windows", attn_kwargs={"window_size": 8}, **kw).eval()
    win.load_state_dict(full.state_dict())
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(win(x), full(x), atol=1e-5)


@pytest.mark.parametrize("H, M", [(8, 4), (40, 4), (20, 4), (12, 6)])
def test_shift_mask_allows_exactly_the_original_neighbours(H, M):
    # Independent oracle: undo the cyclic shift and allow a pair only when both tokens fit in one
    # M x M patch of the original grid without wrapping around the image edge.
    s = M // 2
    mask = shifted_window_mask(H, H, M, s)
    assert mask.dtype == torch.bool and mask.shape == ((H // M) ** 2, M * M, M * M)
    pos = torch.arange(H * H).view(1, H, H, 1)
    win = window_partition(pos, M).squeeze(-1)  # shifted-grid index of each window slot
    r, c = (win // H + s) % H, (win % H + s) % H  # original coordinates
    near = ((r[:, :, None] - r[:, None, :]).abs() < M) & ((c[:, :, None] - c[:, None, :]).abs() < M)
    assert torch.equal(mask, near)
    n = H // M
    edge = torch.tensor([i // n == n - 1 or i % n == n - 1 for i in range(n * n)])
    assert mask[~edge].all()  # windows away from the last window row and column are unmasked
    assert not mask[edge].all((1, 2)).any()  # every wrapped window has a masked pair


@pytest.mark.parametrize("kernel", ["sdpa", "naive"])
def test_shifted_block_corner_token_reaches_only_its_corner(kernel):
    # (0, 0) lands in the bottom-right window after the shift, next to tokens from the other three
    # image corners. The mask must zero those weights: only the s x s top-left corner may change.
    H, M = 8, 4
    shifted = ShiftedWindowAttention(DIM, HEADS, layer_idx=1, window_size=M, kernel=kernel)
    assert changed_tokens(shifted, H, H, 0, 0) == {(r, c) for r in range(2) for c in range(2)}
    assert changed_tokens(shifted, H, H, 7, 7) == {(r, c) for r in range(6, 8) for c in range(6, 8)}
    # Interior token (3, 3) sits in the shifted window covering rows/cols 2..5 of the original grid.
    assert changed_tokens(shifted, H, H, 3, 3) == {(r, c) for r in range(2, 6) for c in range(2, 6)}
    plain = ShiftedWindowAttention(DIM, HEADS, layer_idx=0, window_size=M, kernel=kernel)
    plain.load_state_dict(shifted.state_dict())
    assert changed_tokens(plain, H, H, 0, 0) == {(r, c) for r in range(4) for c in range(4)}


@pytest.mark.parametrize("grid", [8, 40])
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_windows_match_timm_swin_block(grid, layer_idx):
    # timm's Swin block (relative position bias zeroed) as a test-only reference for the
    # shift, partition, mask and reverse. Its mask is -100 instead of -inf, which is equal after softmax.
    swin = pytest.importorskip("timm.models.swin_transformer")
    M = 4
    ours = ShiftedWindowAttention(DIM, HEADS, layer_idx=layer_idx, window_size=M).eval()
    ref = swin.SwinTransformerBlock(DIM, (grid, grid), num_heads=HEADS, window_size=M,
                                    shift_size=ours.shift_size).eval()
    assert tuple(ref.shift_size) == (ours.shift_size,) * 2
    with torch.no_grad():
        ref.attn.relative_position_bias_table.zero_()
        for name in ("qkv", "proj"):
            getattr(ref.attn, name).load_state_dict(getattr(ours.attn, name).state_dict())
        x = grid_tokens(2, grid, grid)
        expected = ref._attn(x.view(2, grid, grid, DIM)).reshape(2, grid * grid, DIM)
        assert torch.allclose(ours(x, grid, grid), expected, atol=1e-5)


def test_window_size_must_divide_the_grid():
    with pytest.raises(ValueError, match="does not divide the 8x8 patch grid"):
        VisionTransformer(img_size=32, patch_size=4, dim=DIM, depth=2, num_heads=HEADS, mechanism="windows",
                          attn_kwargs={"window_size": 3})
    with pytest.raises(ValueError, match="does not divide the 6x6 patch grid"):
        VisionTransformer(img_size=24, patch_size=4, dim=DIM, depth=2, num_heads=HEADS, mechanism="windows")
    with pytest.raises(ValueError, match="does not divide"):
        ShiftedWindowAttention(DIM, HEADS)(grid_tokens(1, 6, 6), 6, 6)
    for img, patch in [(32, 4), (160, 4), (160, 8)]:  # the study's grids: 8x8, 40x40, 20x20
        VisionTransformer(img_size=img, patch_size=patch, dim=DIM, depth=2, num_heads=HEADS, mechanism="windows")


# -- axial ----------------------------------------------------------------------------------


def test_axial_passes_have_separate_projections():
    ax = AxialAttention(DIM, HEADS)
    full = FullAttention(DIM, HEADS)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert ax.col is not ax.row and isinstance(ax.col, MultiHeadAttention)
    assert count(ax) == 2 * count(full) == 2 * (4 * DIM * DIM + 4 * DIM)
    shared = SharedAxialAttention(DIM, HEADS)
    assert shared.col is shared.row and count(shared) == count(full)


def test_axial_passes_match_full_attention_per_column_and_row():
    ax = AxialAttention(DIM, HEADS)
    H, W = 4, 6
    x = grid_tokens(2, H, W)
    g = x.view(2, H, W, DIM)
    with torch.no_grad():
        cols = ax.column_pass(g)
        rows = ax.row_pass(g)
        for j in range(W):
            assert torch.allclose(cols[:, :, j], ax.col(g[:, :, j]), atol=1e-6)
        for i in range(H):
            assert torch.allclose(rows[:, i], ax.row(g[:, i]), atol=1e-6)
        assert torch.allclose(ax(x, H, W), ax.row_pass(ax.column_pass(g)).reshape(2, H * W, DIM))


class _Pass(torch.nn.Module):
    def __init__(self, ax, which):
        super().__init__()
        self.ax, self.which = ax, which

    def forward(self, x, H, W):
        g = x.view(x.shape[0], H, W, -1)
        out = self.ax.column_pass(g) if self.which == "col" else self.ax.row_pass(g)
        return out.reshape(x.shape)


@pytest.mark.parametrize("r, c", [(0, 0), (2, 5), (4, 7)])
def test_axial_perturbation_reaches_its_row_and_column(r, c):
    H, W = 5, 8
    ax = AxialAttention(DIM, HEADS)
    assert changed_tokens(_Pass(ax, "col"), H, W, r, c) == {(i, c) for i in range(H)}
    assert changed_tokens(_Pass(ax, "row"), H, W, r, c) == {(r, j) for j in range(W)}
    # Column then row: the token's column reaches every row, so one layer is global.
    assert changed_tokens(ax, H, W, r, c) == {(i, j) for i in range(H) for j in range(W)}


# -- shared ---------------------------------------------------------------------------------


@pytest.mark.parametrize("mechanism", ["windows", "axial", "axial_shared"])
def test_rejects_prefix_tokens(mechanism):
    from attnvit.models import get_mechanism

    attn = get_mechanism(mechanism).cls(DIM, HEADS)
    with pytest.raises(ValueError, match="no CLS token"):
        attn(torch.randn(1, 65, DIM), 8, 8)


@pytest.mark.parametrize("mechanism", ["full", "windows", "axial", "axial_shared"])
def test_gradients_reach_every_parameter(mechanism):
    torch.manual_seed(0)
    model = VisionTransformer(img_size=32, patch_size=4, dim=DIM, depth=2, num_heads=HEADS, mechanism=mechanism)
    x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))
    F.cross_entropy(model(x), y).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    xg = torch.randn(2, 3, 32, 32, requires_grad=True)
    model(xg)[:, 0].sum().backward()
    assert (xg.grad.abs().sum((0, 1)) > 0).all()  # every pixel reaches the output


# -- FLOPs ----------------------------------------------------------------------------------


def report1_flops(mechanism, img, patch, *, d=192, depth=9, mlp_ratio=2, M=4):
    """Report 1's per-layer cost in multiply-adds plus the shared MLP, times depth, as FLOPs (x2)."""
    n_grid = (img // patch) ** 2
    if mechanism == "full":
        N = n_grid + 1  # CLS token
        attn = 4 * N * d * d + 2 * N * N * d
    elif mechanism == "windows":
        N = n_grid
        attn = 4 * N * d * d + 2 * M * M * N * d
    else:  # axial: two passes, each with its own (or shared) projections
        N = n_grid
        attn = 2 * (4 * N * d * d + 2 * N * math.isqrt(N) * d)
    mlp = 2 * N * d * (mlp_ratio * d)
    return 2 * depth * (attn + mlp)


@pytest.mark.parametrize("mechanism", ["full", "windows", "axial", "axial_shared"])
@pytest.mark.parametrize("img, patch", [(32, 4), (160, 4), (160, 8)])
def test_flop_counter_matches_report1(mechanism, img, patch):
    # Shapes only: on the meta device nothing is computed, so 160 px full attention is instant.
    with torch.device("meta"):
        model = VisionTransformer(img_size=img, patch_size=patch, mechanism=mechanism)
        x = torch.randn(1, 3, img, img)
        with FlopCounterMode(display=False) as counter:
            model(x)
    counted, expected = counter.get_total_flops(), report1_flops(mechanism, img, patch)
    # The rest (patch embedding, head) is under 1.5 % of the total at these sizes.
    assert counted == pytest.approx(expected, rel=0.015), (counted, expected)
    assert counted > expected
