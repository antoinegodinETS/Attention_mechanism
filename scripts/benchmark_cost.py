"""Cost benchmark (phase 4, no training): FLOPs, time and peak memory per mechanism and resolution.

For each attention mechanism (default: the study's full, windows and axial) and each setting
(default 32, 64, 96, 128 and 160 px at P4, plus 160 px at P8) this builds the reference model from
``configs/base.yaml`` and records, in one CSV row per case:

* tokens ``N`` (patch tokens; full attention adds a CLS token) and parameters;
* forward FLOPs per image counted by ``FlopCounterMode`` (whole model and attention sublayers),
  the analytic count (Report 1 attention formulas plus the MLP, patch embedding and head, see
  :mod:`attnvit.cost`) and their ratio, and the counted forward plus backward FLOPs;
* forward plus backward time per image in the training setting (fp16 autocast, the SDPA backend
  of the config, train mode), on a fixed synthetic batch already on the device: ``--warmup``
  untimed steps, then the median (and p10/p90) of ``--steps`` synchronised steps. One step is
  ``zero_grad``, forward, cross-entropy and backward, without the optimizer update or mixup;
* eval-mode forward time per image (no grad, same batch);
* peak allocated and reserved CUDA memory over the timed steps, under SDPA and again with the
  naive kernel, which materialises the ``(n, n)`` attention matrices (O(N^2) memory).

The batch is the same for every mechanism and setting (``--batch-size``, default the training
batch of 128) and is in the CSV. An out-of-memory error is a result (status ``oom``), not a crash;
memory is freed between cases. A setting a mechanism cannot run (a window size that does not
divide the patch grid) is recorded as ``skipped`` with the reason.

Usage: python scripts/benchmark_cost.py [--out results/cost_benchmark.csv] [--no-naive]
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from attnvit.config import REPO_ROOT, load_config
from attnvit.cost import count_flops, model_analytic_flops
from attnvit.models.attention import MECHANISMS, set_attention_kernel
from attnvit.train import build_model
from attnvit.utils import (free_device_memory, peak_memory_mib, peak_reserved_memory_mib, reset_peak_memory,
                           resolve_device, timed)

DEFAULT_SETTINGS = "32:4,64:4,96:4,128:4,160:4,160:8"
STUDY = ("full", "windows", "axial")
NUM_CLASSES = 10
FIELDS = [
    "timestamp", "device", "gpu", "torch", "mechanism", "res", "patch", "grid", "tokens", "num_prefix",
    "window_size", "params", "batch_size", "amp_fp16", "sdpa_backend", "status", "reason",
    "gflops_fwd_counted", "gflops_fwd_analytic", "flops_ratio", "gflops_attn_counted", "gflops_attn_analytic",
    "gflops_train_counted",
    "train_ms_median", "train_ms_p10", "train_ms_p90", "train_ms_per_image", "eval_ms_per_image",
    "peak_alloc_mib_sdpa", "peak_reserved_mib_sdpa",
    "naive_status", "naive_train_ms_per_image", "peak_alloc_mib_naive", "peak_reserved_mib_naive",
]


def parse_settings(text: str) -> list[tuple[int, int]]:
    out = []
    for item in text.split(","):
        res, patch = item.split(":")
        out.append((int(res), int(patch)))
    return out


def _gflops(x: int) -> float:
    return round(x / 1e9, 6)


def _mib(x: float | None):
    return "" if x is None else round(x, 1)


def time_train(model, x, y, device, *, use_amp: bool, steps: int, warmup: int) -> dict:
    """Forward plus backward steps on one batch; ``status`` is ``ok`` or ``oom``.

    Peak memory is over the timed steps (after a reset) and includes the weights, the gradients
    and the batch. Everything the steps allocated is released before returning.
    """
    model.train()

    def step():
        model.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        F.cross_entropy(logits.float(), y).backward()

    try:
        for _ in range(warmup):
            step()
        reset_peak_memory(device)
        times = [timed(step, device) for _ in range(steps)]
        out = {"status": "ok", "times": times, "peak_alloc": peak_memory_mib(device),
               "peak_reserved": peak_reserved_memory_mib(device)}
    except torch.OutOfMemoryError:
        out = {"status": "oom"}
    model.zero_grad(set_to_none=True)
    free_device_memory(device)
    return out


def time_eval(model, x, device, *, use_amp: bool, iters: int) -> dict:
    """Eval-mode forward passes (no grad) on one batch; ``status`` is ``ok`` or ``oom``."""
    model.eval()

    @torch.no_grad()
    def forward():
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            model(x)

    try:
        forward()
        out = {"status": "ok", "times": [timed(forward, device) for _ in range(iters)]}
    except torch.OutOfMemoryError:
        out = {"status": "oom"}
    free_device_memory(device)
    return out


def measure(cfg, device, *, steps: int, warmup: int, eval_iters: int, naive: bool) -> dict:
    """One case (mechanism x resolution x patch); returns a CSV row without the shared columns."""
    res, patch, B = cfg.data.res, cfg.model.patch_size, cfg.optim.batch_size
    use_amp = cfg.amp and device.type == "cuda"
    grid = res // patch
    row = {"mechanism": cfg.model.mechanism, "res": res, "patch": patch, "grid": grid, "tokens": grid * grid,
           "batch_size": B,
           "amp_fp16": use_amp, "sdpa_backend": cfg.model.sdpa_backend,
           "window_size": cfg.model.attn.get("window_size", 4) if cfg.model.mechanism == "windows" else "",
           "naive_status": "pending" if naive else "not run"}
    try:
        torch.manual_seed(0)
        model = build_model(cfg, NUM_CLASSES)
    except ValueError as e:  # e.g. a window size that does not divide the grid
        row.update(status="skipped", reason=str(e), naive_status="skipped")
        return row
    row.update(num_prefix=model.num_prefix, params=sum(p.numel() for p in model.parameters()))
    model = model.to(device)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, 3, res, res, generator=g).to(device)
    y = torch.randint(0, NUM_CLASSES, (B,), generator=g).to(device)
    try:
        # FLOPs of one image in fp32; counts depend on shapes only, not on the kernel or dtype.
        model.eval()
        counted = count_flops(model, x[:1], backward=True)
        analytic = model_analytic_flops(model, num_classes=NUM_CLASSES)
        row.update(gflops_fwd_counted=_gflops(counted["total"]), gflops_fwd_analytic=_gflops(analytic["total"]),
                   flops_ratio=round(counted["total"] / analytic["total"], 4),
                   gflops_attn_counted=_gflops(counted["attention"]),
                   gflops_attn_analytic=_gflops(analytic["attention"]),
                   gflops_train_counted=_gflops(counted["train"]))
        free_device_memory(device)

        set_attention_kernel(model, "sdpa")
        t = time_train(model, x, y, device, use_amp=use_amp, steps=steps, warmup=warmup)
        row["status"] = t["status"]
        if t["status"] == "ok":
            q = statistics.quantiles(t["times"], n=10) if len(t["times"]) >= 2 else [t["times"][0]] * 9
            med = statistics.median(t["times"])
            row.update(train_ms_median=round(med * 1e3, 3), train_ms_p10=round(q[0] * 1e3, 3),
                       train_ms_p90=round(q[-1] * 1e3, 3), train_ms_per_image=round(med * 1e3 / B, 5),
                       peak_alloc_mib_sdpa=_mib(t["peak_alloc"]), peak_reserved_mib_sdpa=_mib(t["peak_reserved"]))
            e = time_eval(model, x, device, use_amp=use_amp, iters=eval_iters)
            if e["status"] == "ok":
                row["eval_ms_per_image"] = round(statistics.median(e["times"]) * 1e3 / B, 5)
            else:
                row["reason"] = "eval forward out of memory"

        if naive:
            set_attention_kernel(model, "naive")
            t = time_train(model, x, y, device, use_amp=use_amp, steps=steps, warmup=warmup)
            row["naive_status"] = t["status"]
            if t["status"] == "ok":
                row.update(naive_train_ms_per_image=round(statistics.median(t["times"]) * 1e3 / B, 5),
                           peak_alloc_mib_naive=_mib(t["peak_alloc"]),
                           peak_reserved_mib_naive=_mib(t["peak_reserved"]))
    finally:
        del model, x, y
        free_device_memory(device)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    tmp.replace(path)


def _fmt(row: dict) -> str:
    if row["status"] == "skipped":
        return f"skipped: {row['reason']}"
    s = (f"N={row['tokens']:5d}  {row['gflops_fwd_counted']:7.2f} GFLOP (x{row['flops_ratio']:.3f} analytic)  ")
    if row["status"] != "ok":
        return s + "sdpa OUT OF MEMORY"
    s += f"train {row['train_ms_per_image']:.4f} ms/img  eval {row.get('eval_ms_per_image', float('nan')):.4f} ms/img"
    if row.get("peak_alloc_mib_sdpa") not in (None, ""):
        s += f"  peak {row['peak_alloc_mib_sdpa']:.0f} MiB"
    if row["naive_status"] == "ok" and row.get("peak_alloc_mib_naive") not in (None, ""):
        s += f"  naive {row['peak_alloc_mib_naive']:.0f} MiB"
    elif row["naive_status"] == "oom":
        s += "  naive OOM"
    return s


def hypothesis_summary(rows: list[dict]) -> list[str]:
    """Windows and axial relative to full attention at each setting (FLOPs, time, SDPA memory)."""
    lines = []
    for res, patch in dict.fromkeys((r["res"], r["patch"]) for r in rows):
        sel = {r["mechanism"]: r for r in rows if (r["res"], r["patch"]) == (res, patch) and r["status"] == "ok"}
        full = sel.get("full")
        if not full:
            continue
        parts = []
        for mech, r in sel.items():
            if mech == "full":
                continue
            p = f"{mech} FLOPs x{r['gflops_fwd_counted'] / full['gflops_fwd_counted']:.2f}"
            p += f", time x{r['train_ms_per_image'] / full['train_ms_per_image']:.2f}"
            if full.get("peak_alloc_mib_sdpa") not in (None, "") and r.get("peak_alloc_mib_sdpa") not in (None, ""):
                p += f", memory x{r['peak_alloc_mib_sdpa'] / full['peak_alloc_mib_sdpa']:.2f}"
            parts.append(p)
        if parts:
            lines.append(f"  {res:3d} px P{patch} vs full: " + "; ".join(parts))
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--mechanisms", default="study",
                   help="comma list, 'study' (%s) or 'all' registered" % ",".join(STUDY))
    p.add_argument("--settings", default=DEFAULT_SETTINGS, help="res:patch list (default %(default)s)")
    p.add_argument("--batch-size", type=int, default=128, help="images per step, every case (default: training)")
    p.add_argument("--steps", type=int, default=50, help="timed steps; the median is reported")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--no-naive", dest="naive", action="store_false", help="skip the naive-kernel memory run")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "cost_benchmark.csv"))
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    if args.mechanisms == "study":
        mechs = list(STUDY)
    elif args.mechanisms == "all":
        mechs = sorted(MECHANISMS)
    else:
        mechs = args.mechanisms.split(",")
    unknown = [m for m in mechs if m not in MECHANISMS]
    if unknown:
        p.error(f"unknown mechanism(s) {', '.join(unknown)}; registered: {', '.join(sorted(MECHANISMS))}")
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else ""
    print(f"device {device} {gpu} | torch {torch.__version__} | batch {args.batch_size} | "
          f"mechanisms {', '.join(mechs)}")
    if device.type != "cuda":
        print("WARNING: not on a GPU; these times are not the plan's H100 measurements (no memory columns)")

    shared = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": str(device), "gpu": gpu,
              "torch": torch.__version__}
    out = Path(args.out)
    rows = []
    for res, patch in parse_settings(args.settings):
        for mech in mechs:
            cfg = load_config(args.config, [f"model.mechanism={mech}", f"data.res={res}", f"model.patch_size={patch}",
                                            f"optim.batch_size={args.batch_size}", "optim.accum_steps=1",
                                            "model.attn_kernel=sdpa", f"device={args.device}"] + args.overrides)
            row = {**shared, **measure(cfg, device, steps=args.steps, warmup=args.warmup,
                                       eval_iters=args.eval_iters, naive=args.naive)}
            rows.append(row)
            write_csv(out, rows)  # after every case, so a killed job keeps what it measured
            print(f"{mech:12s} {res:3d} px P{patch:<2d} {_fmt(row)}", flush=True)
    print(f"wrote {out}")
    summary = hypothesis_summary(rows)
    if summary:
        print("\nrelative cost (counted forward FLOPs, train time per image, SDPA peak memory):")
        print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
