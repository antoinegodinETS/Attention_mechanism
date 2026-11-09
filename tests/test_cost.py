"""Phase 4: analytic vs counted FLOPs, scripts/benchmark_cost.py on CPU and the cost figure."""

import csv

import pytest
import torch

from attnvit.cost import analytic_flops, attention_layer_macs, count_flops, model_analytic_flops
from attnvit.models import VisionTransformer
from conftest import TINY

MODEL_SETS = [x for x in TINY if x.startswith("model.") and "patch_size" not in x]


@pytest.mark.parametrize("mechanism", ["full", "windows", "axial", "axial_shared"])
@pytest.mark.parametrize("img,patch", [(32, 4), (64, 8), (48, 4)])
def test_analytic_matches_counted_flops(mechanism, img, patch):
    """The reference model (d 192, depth 9, 3 heads, MLP ratio 2) at the study's patch sizes."""
    if mechanism == "windows" and (img // patch) % 4:
        pytest.skip("window size 4 does not divide the grid")
    model = VisionTransformer(img_size=img, patch_size=patch, mechanism=mechanism).eval()
    counted = count_flops(model, torch.randn(2, 3, img, img))
    analytic = model_analytic_flops(model)
    assert counted["total"] == pytest.approx(2 * analytic["total"], rel=0.02)  # batch of 2
    assert counted["attention"] == pytest.approx(2 * analytic["attention"], rel=0.02)


def test_report1_formulas():
    N, d, M = 1600, 192, 4
    assert attention_layer_macs("full", 40, 40, d) == 4 * N * d**2 + 2 * N**2 * d
    assert attention_layer_macs("windows", 40, 40, d, window_size=M) == 4 * N * d**2 + 2 * M**2 * N * d
    assert attention_layer_macs("axial", 40, 40, d) == 2 * (4 * N * d**2 + 2 * N * 40 * d)
    assert attention_layer_macs("full", 40, 40, d, num_prefix=1) == 4 * 1601 * d**2 + 2 * 1601**2 * d
    # One window covering the whole grid is full attention.
    assert attention_layer_macs("windows", 4, 4, d, window_size=4) == attention_layer_macs("full", 4, 4, d)
    # The plan's table: 160 px / P4, about 26.2 / 8.7 / 13.6 GFLOPs per image.
    kw = dict(img_size=160, patch_size=4, dim=192, depth=9, mlp_ratio=2.0)
    assert analytic_flops("full", num_prefix=1, **kw)["total"] / 1e9 == pytest.approx(26.2, abs=0.1)
    assert analytic_flops("windows", **kw)["total"] / 1e9 == pytest.approx(8.7, abs=0.1)
    assert analytic_flops("axial", **kw)["total"] / 1e9 == pytest.approx(13.6, abs=0.1)
    with pytest.raises(ValueError):
        attention_layer_macs("nope", 4, 4, d)


def test_count_flops_backward_is_about_three_times_forward():
    model = VisionTransformer(img_size=32, patch_size=8, dim=64, depth=2, num_heads=2)
    c = count_flops(model, torch.randn(1, 3, 32, 32), backward=True)
    assert 2.5 < c["train"] / c["total"] < 3.5
    assert all(p.grad is None for p in model.parameters())


def _run_benchmark(script, tmp_path, *extra):
    bench = script("benchmark_cost")
    out = tmp_path / "cost.csv"
    argv = ["--device", "cpu", "--settings", "32:8,48:8,64:8", "--batch-size", "4", "--steps", "3",
            "--warmup", "1", "--eval-iters", "2", "--out", str(out), *extra]
    for s in MODEL_SETS:
        argv += ["--set", s]
    assert bench.main(argv) == 0
    return list(csv.DictReader(open(out)))


def test_benchmark_runs_on_cpu(script, tmp_path, capsys):
    rows = _run_benchmark(script, tmp_path)
    assert [(r["mechanism"], r["res"]) for r in rows] == [
        (m, res) for res in ("32", "48", "64") for m in ("full", "windows", "axial")]
    win48 = rows[4]
    assert win48["status"] == "skipped" and "does not divide" in win48["reason"] and win48["tokens"] == "36"
    ok = [r for r in rows if r is not win48]
    assert all(r["status"] == "ok" and r["naive_status"] == "ok" for r in ok)
    for r in ok:
        assert float(r["train_ms_per_image"]) > 0 and float(r["eval_ms_per_image"]) > 0
        assert float(r["naive_train_ms_per_image"]) > 0
        assert float(r["flops_ratio"]) == pytest.approx(1.0, abs=0.02)
        assert float(r["gflops_train_counted"]) > 2 * float(r["gflops_fwd_counted"])
        assert r["peak_alloc_mib_sdpa"] == "" and r["batch_size"] == "4"  # no CUDA memory on CPU
        assert r["device"] == "cpu" and r["torch"] == torch.__version__ and r["timestamp"]
    assert rows[0]["num_prefix"] == "1" and rows[2]["num_prefix"] == "0"  # CLS token for full only
    assert rows[0]["tokens"] == "16" and rows[6]["tokens"] == "64"
    assert int(rows[2]["params"]) > int(rows[1]["params"])  # axial: two projection sets per block
    text = capsys.readouterr().out
    assert "not on a GPU" in text and "vs full" in text


def test_benchmark_records_oom_and_carries_on(script, tmp_path, monkeypatch):
    """A failed allocation in the naive kernel is a result; the SDPA columns and later cases still run."""
    import attnvit.models.attention as attention

    def oom(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 80.00 GiB (mocked)")

    monkeypatch.setattr(attention, "naive_attention", oom)
    rows = _run_benchmark(script, tmp_path)
    ok = [r for r in rows if r["status"] != "skipped"]
    assert ok and all(r["status"] == "ok" and r["naive_status"] == "oom" for r in ok)
    assert all(r["naive_train_ms_per_image"] == "" and float(r["train_ms_per_image"]) > 0 for r in ok)


def test_benchmark_sdpa_oom(script, tmp_path, monkeypatch):
    import attnvit.models.attention as attention

    real = attention.MultiHeadAttention.forward

    def forward(self, x, attn_mask=None):  # out of memory in every training step, not in eval mode
        if self.training:
            raise torch.OutOfMemoryError("mocked")
        return real(self, x, attn_mask)

    monkeypatch.setattr(attention.MultiHeadAttention, "forward", forward)
    rows = _run_benchmark(script, tmp_path, "--no-naive")
    ok = [r for r in rows if r["status"] != "skipped"]
    # FLOP counting (eval mode) still works; training steps fail.
    assert all(r["status"] == "oom" and r["train_ms_per_image"] == "" and r["gflops_fwd_counted"] for r in ok)
    assert all(r["naive_status"] == "not run" for r in ok)


FIXTURE = [
    # mechanism, res, patch, tokens, gflops, ms/img, sdpa MiB, naive status, naive MiB
    ("full", 32, 4, 64, 0.375, 0.08, 520, "ok", 600),
    ("full", 160, 4, 1600, 26.2, 1.06, 10184, "oom", ""),
    ("full", 160, 8, 400, 3.27, 0.21, 2734, "ok", 5000),
    ("windows", 32, 4, 64, 0.348, 0.08, 500, "ok", 520),
    ("windows", 160, 4, 1600, 8.70, 0.50, 6000, "ok", 7000),
    ("axial", 32, 4, 64, 0.518, 0.11, 700, "ok", 720),
    ("axial", 160, 4, 1600, 13.6, 0.70, 9000, "ok", 11000),
]


def test_cost_figure_from_fixture_csv(script, tmp_path):
    bench, fig = script("benchmark_cost"), script("make_cost_figure")
    rows = [dict(timestamp="t", device="cuda", gpu="H100", torch="2", mechanism=m, res=res, patch=p, tokens=n,
                 batch_size=128, status="ok", gflops_fwd_counted=g, gflops_fwd_analytic=g * 0.999,
                 train_ms_per_image=t, peak_alloc_mib_sdpa=mem, naive_status=ns, peak_alloc_mib_naive=nm)
            for m, res, p, n, g, t, mem, ns, nm in FIXTURE]
    rows.append(dict(mechanism="windows", res=48, patch=4, tokens=144, status="skipped", reason="grid"))
    csv_path = tmp_path / "cost.csv"
    bench.write_csv(csv_path, rows)
    out = tmp_path / "fig" / "cost_vs_tokens.png"
    assert fig.main(["--csv", str(csv_path), "--out", str(out)]) == 0
    assert out.stat().st_size > 10_000 and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(fig.load_rows(csv_path)) == len(FIXTURE)  # the skipped case is not plotted


def test_cost_figure_rejects_empty_csv(script, tmp_path):
    bench, fig = script("benchmark_cost"), script("make_cost_figure")
    csv_path = tmp_path / "empty.csv"
    bench.write_csv(csv_path, [])
    with pytest.raises(SystemExit):
        fig.main(["--csv", str(csv_path), "--out", str(tmp_path / "x.png")])
