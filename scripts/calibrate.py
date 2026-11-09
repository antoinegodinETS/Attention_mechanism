"""Time real training steps per mechanism and project the 160 px GPU budget.

For each of the study's attention mechanisms (full, windows, axial; ``--mechanisms`` picks
others, such as the ``axial_shared`` ablation) and every setting (default 32 px / P4, 160 px / P4
and 160 px / P8) this builds the reference model from ``configs/base.yaml`` and times
``--steps`` full optimizer steps (fp16 autocast + GradScaler, mixup/cutmix, AdamW: the same
``train_step`` the training loop uses) on synthetic uint8 batches already on the GPU, after
``--warmup`` untimed steps. Each step is synchronised; the median is reported. Peak memory is
``torch.cuda.max_memory_allocated`` over the timed steps after a reset. Data loading is not
included (see ``--loader-check``). An out-of-memory error is recorded as a result.

It writes ``results/calibration.csv`` and prints the projected GPU-hours of the 9-run 160 px
block (3 mechanisms x 3 seeds) at 100 and 50 epochs against the plan's 40-hour decision rule.
A study mechanism left out of the measurement is projected with the slowest measured one.

Usage: python scripts/calibrate.py [--out results/calibration.csv] [--loader-check]
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

import torch

from attnvit.config import REPO_ROOT, load_config
from attnvit.data.augment import MixupCutmix, Normalize
from attnvit.models.attention import MECHANISMS
from attnvit.train import build_model, build_optimizer, train_step
from attnvit.utils import resolve_device, timed

DEFAULT_SETTINGS = "32:4,160:4,160:8"
STUDY = ("full", "windows", "axial")  # the 3 mechanisms of the study; axial_shared is an ablation
STUDY_MECHANISMS = len(STUDY)
FIELDS = [
    "timestamp", "device", "gpu", "torch", "mechanism", "res", "patch", "tokens", "params",
    "batch_size", "micro_batch", "accum_steps", "amp_fp16", "attn_kernel", "sdpa_backend", "status",
    "step_ms_median", "step_ms_p10", "step_ms_p90", "eval_ms_median", "images_per_s", "peak_mem_mib",
    "peak_mem_reserved_mib", "steps_per_epoch", "est_epoch_s", "est_run_h_100ep", "est_run_h_50ep",
]


def parse_settings(text: str) -> list[tuple[int, int]]:
    out = []
    for item in text.split(","):
        res, patch = item.split(":")
        out.append((int(res), int(patch)))
    return out


def measure(cfg, device, *, steps: int, warmup: int, eval_iters: int, train_images: int,
            val_images: int) -> dict:
    """Time one configuration; returns a CSV row (without the shared columns)."""
    use_amp = cfg.amp and device.type == "cuda"
    res, patch = cfg.data.res, cfg.model.patch_size
    row = {"mechanism": cfg.model.mechanism, "res": res, "patch": patch, "tokens": (res // patch) ** 2,
           "batch_size": cfg.optim.batch_size, "micro_batch": cfg.micro_batch_size,
           "accum_steps": cfg.optim.accum_steps, "amp_fp16": use_amp, "attn_kernel": cfg.model.attn_kernel,
           "sdpa_backend": cfg.model.sdpa_backend}
    model = optimizer = batches = None
    try:
        torch.manual_seed(0)
        model = build_model(cfg, 10).to(device)
        row["params"] = sum(p.numel() for p in model.parameters())
        optimizer = build_optimizer(model, cfg, device)
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        mixer = MixupCutmix.from_config(cfg.augment, 10)
        normalize = Normalize([0.5] * 3, [0.25] * 3, device=device)
        g = torch.Generator().manual_seed(0)
        batches = [
            (torch.randint(0, 256, (cfg.micro_batch_size, 3, res, res), dtype=torch.uint8, generator=g).to(device),
             torch.randint(0, 10, (cfg.micro_batch_size,), generator=g).to(device))
            for _ in range(cfg.optim.accum_steps)
        ]
        model.train()

        def step():
            train_step(model, optimizer, scaler, batches, normalize=normalize, mixer=mixer, use_amp=use_amp,
                       grad_clip=cfg.optim.grad_clip)

        for _ in range(warmup):
            step()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = [timed(step, device) for _ in range(steps)]
        if device.type == "cuda":
            row["peak_mem_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
            row["peak_mem_reserved_mib"] = round(torch.cuda.max_memory_reserved(device) / 2**20, 1)

        model.eval()
        xe = batches[0][0][: cfg.eval_batch_size]

        @torch.no_grad()
        def forward():
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                model(normalize(xe))

        forward()
        eval_times = [timed(forward, device) for _ in range(eval_iters)]
    except torch.OutOfMemoryError:
        row["status"] = "oom"
        return row
    finally:
        del model, optimizer, batches
        if device.type == "cuda":
            torch.cuda.empty_cache()

    q = statistics.quantiles(times, n=10) if len(times) >= 2 else [times[0]] * 9
    step_s = statistics.median(times)
    eval_s = statistics.median(eval_times) * cfg.eval_batch_size / len(xe)  # per eval batch
    steps_per_epoch = train_images // cfg.optim.batch_size
    epoch_s = step_s * steps_per_epoch + eval_s * math.ceil(val_images / cfg.eval_batch_size)
    row.update({
        "status": "ok",
        "step_ms_median": round(step_s * 1e3, 3),
        "step_ms_p10": round(q[0] * 1e3, 3),
        "step_ms_p90": round(q[-1] * 1e3, 3),
        "eval_ms_median": round(eval_s * 1e3, 3),
        "images_per_s": round(cfg.optim.batch_size / step_s, 1),
        "steps_per_epoch": steps_per_epoch,
        "est_epoch_s": round(epoch_s, 2),
        "est_run_h_100ep": round(epoch_s * 100 / 3600, 3),
        "est_run_h_50ep": round(epoch_s * 50 / 3600, 3),
    })
    return row


def block_projection(rows: list[dict], *, res: int = 160, seeds: int = 3,
                     mechanisms: int = STUDY_MECHANISMS) -> dict[tuple[int, int], dict]:
    """GPU-hours of the ``mechanisms x seeds`` block at ``res`` per (patch, epochs).

    Unmeasured mechanisms (fewer than ``mechanisms`` registered) use the slowest measured run;
    an out-of-memory mechanism makes the projection infinite.
    """
    out = {}
    for patch in sorted({r["patch"] for r in rows if r["res"] == res}):
        sel = [r for r in rows if r["res"] == res and r["patch"] == patch]
        for epochs in (100, 50):
            runs = [r[f"est_run_h_{epochs}ep"] if r["status"] == "ok" else math.inf for r in sel]
            stand_in = max(runs)
            missing = max(0, mechanisms - len(runs))
            hours = seeds * (sum(runs) + missing * stand_in)
            out[(patch, epochs)] = {"hours": hours, "measured": [r["mechanism"] for r in sel], "stand_ins": missing}
    return out


def decide(proj: dict[tuple[int, int], dict], budget_h: float = 40.0) -> str:
    """The plan's 160 px decision rule, applied to :func:`block_projection`."""
    p4_100 = proj.get((4, 100), {}).get("hours", math.inf)
    p4_50 = proj.get((4, 50), {}).get("hours", math.inf)
    if p4_100 <= budget_h:
        return f"option 1: P4, 100 epochs, as documented ({p4_100:.1f} GPU-h <= {budget_h:g})"
    if p4_50 <= budget_h:
        return f"option 2: P4, 50 epochs; state the shorter schedule ({p4_50:.1f} GPU-h <= {budget_h:g})"
    return "option 3: P8, 100 epochs; state the weaker axial contrast"


def loader_check(cfg, batches: int) -> None:
    """Throughput of the real CIFAR-10 training loader with the augmentation recipe."""
    from attnvit.train import load_datasets

    try:
        data = load_datasets(cfg)
    except Exception as e:  # noqa: BLE001 - report and carry on, timing is still valid
        print(f"loader check skipped: {type(e).__name__}: {e}")
        return
    loader = torch.utils.data.DataLoader(data.train, batch_size=cfg.micro_batch_size, shuffle=True,
                                         num_workers=cfg.data.num_workers, drop_last=True)
    it = iter(loader)
    next(it)  # start the workers
    t0 = time.perf_counter()
    n = 0
    for _ in range(batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        n += len(x)
    dt = time.perf_counter() - t0
    print(f"loader check: {cfg.data.name} {cfg.data.res} px, {cfg.data.num_workers} workers: "
          f"{n / dt:.0f} images/s -> {len(data.train) / (n / dt):.1f} s per epoch if loader-bound")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--mechanisms", default="study",
                   help="comma list, 'study' (%s) or 'all' registered" % ",".join(STUDY))
    p.add_argument("--settings", default=DEFAULT_SETTINGS, help="res:patch list (default %(default)s)")
    p.add_argument("--batch-size", type=int, default=128, help="effective batch (plan: 128)")
    p.add_argument("--accum", type=int, default=1, help="micro-batches per step (1 on an H100)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--eval-iters", type=int, default=10)
    p.add_argument("--train-images", type=int, default=45000)
    p.add_argument("--val-images", type=int, default=5000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--budget-hours", type=float, default=40.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "calibration.csv"))
    p.add_argument("--loader-check", action="store_true", help="also time the real 32 px training loader")
    p.add_argument("--loader-batches", type=int, default=100)
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
    print(f"device {device} {gpu} | torch {torch.__version__} | mechanisms {', '.join(mechs)}")
    if device.type != "cuda":
        print("WARNING: not on a GPU; these times are not the plan's H100 measurements")

    shared = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "device": str(device), "gpu": gpu,
              "torch": torch.__version__}
    rows = []
    for res, patch in parse_settings(args.settings):
        for mech in mechs:
            cfg = load_config(args.config, [f"model.mechanism={mech}", f"data.res={res}", f"model.patch_size={patch}",
                                            f"optim.batch_size={args.batch_size}", f"optim.accum_steps={args.accum}",
                                            f"device={args.device}"] + args.overrides)
            row = {**shared, **measure(cfg, device, steps=args.steps, warmup=args.warmup, eval_iters=args.eval_iters,
                                       train_images=args.train_images, val_images=args.val_images)}
            rows.append(row)
            if row["status"] == "ok":
                peak = f"{row['peak_mem_mib']:.0f} MiB" if row.get("peak_mem_mib") is not None else "-"
                print(f"{mech:12s} {res:3d} px P{patch} N={row['tokens']:5d}  step {row['step_ms_median']:8.2f} ms "
                      f"(p10 {row['step_ms_p10']:.2f}, p90 {row['step_ms_p90']:.2f})  peak {peak}  "
                      f"epoch ~{row['est_epoch_s']:.1f} s  run ~{row['est_run_h_100ep']:.2f} h @100ep")
            else:
                print(f"{mech:12s} {res:3d} px P{patch}  OUT OF MEMORY")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"wrote {out}")

    proj = block_projection(rows, seeds=args.seeds)
    if proj:
        any_p = next(iter(proj.values()))
        n_runs = STUDY_MECHANISMS * args.seeds
        stand = (f"; {any_p['stand_ins']} unmeasured mechanism(s) projected with the slowest measured one"
                 if any_p["stand_ins"] else "")
        print(f"\n160 px block: {STUDY_MECHANISMS} mechanisms x {args.seeds} seeds = {n_runs} runs, "
              f"measured {', '.join(any_p['measured'])}{stand}")
        for (patch, epochs), v in sorted(proj.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
            print(f"  P{patch} {epochs:3d} epochs: {v['hours']:7.1f} GPU-h")
        print(f"decision rule ({args.budget_hours:g} GPU-h cap): {decide(proj, args.budget_hours)}")

    if args.loader_check:
        loader_check(load_config(args.config, ["data.res=32", f"device={args.device}"] + args.overrides),
                     args.loader_batches)
    return 0


if __name__ == "__main__":
    sys.exit(main())
