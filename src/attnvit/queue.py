"""Resumable sequential run queue: ``python -m attnvit.queue <queue file>``.

A queue file holds one run per line: a config path followed by ``key=value`` overrides.
Blank lines and ``#`` comments are ignored. Config paths are relative to the current
directory or to the repository root. Example::

    # phase 2: full-attention sanity run on CIFAR-10
    configs/base.yaml tag=sanity seed=0

Before training, every pending run gets a 1-step smoke check (data, model, one optimizer
step), so a typo fails in seconds. Runs with a ``summary.json`` are skipped; a run with a
``last.pt`` resumes from it. A failed run is reported and the queue moves on. At the end
the queue prints one line per run. Exit status: 0 when every run is done, 3 when a signal
interrupted the queue (resubmit to continue), 1 if any run failed.
"""

from __future__ import annotations

import argparse
import gc
import shlex
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import ExperimentConfig, load_config, run_id, runs_root
from .train import EXIT_INTERRUPTED, StopSignals, Trainer, load_summary, smoke_check


@dataclass
class QueueEntry:
    line_no: int
    text: str
    cfg: ExperimentConfig
    run_id: str


def parse_queue(path: str | Path, extra_overrides=()) -> list[QueueEntry]:
    """Read and validate every line; raises ValueError naming the first bad line."""
    entries, seen = [], {}
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for i, raw in enumerate(lines, start=1):
        text = raw.split("#", 1)[0].strip()
        if not text:
            continue
        parts = shlex.split(text)
        try:
            cfg = load_config(parts[0], list(parts[1:]) + list(extra_overrides))
        except Exception as e:
            raise ValueError(f"{path}:{i}: {e}") from e
        rid = run_id(cfg)
        if rid in seen:
            raise ValueError(f"{path}:{i}: run {rid} duplicates line {seen[rid]}")
        seen[rid] = i
        entries.append(QueueEntry(i, text, cfg, rid))
    return entries


def _fmt_summary(s: dict) -> str:
    test = f"{s['test_acc']:.4f}" if s.get("test_acc") is not None else "-"
    peak = f"{s['peak_mem_mib']:.0f}MiB" if s.get("peak_mem_mib") is not None else "-"
    return (f"val={s['final_val_acc']:.4f} test={test} train={s['train_time_s'] / 3600:.2f}h "
            f"epoch={s['time_per_epoch_s']:.1f}s peak={peak}")


def run_queue(path, *, root=None, smoke: bool = True, extra_overrides=(), log=print) -> int:
    with StopSignals(log) as stop:
        return _run_queue(path, root, smoke, extra_overrides, log, stop)


def _run_queue(path, root, smoke, extra_overrides, log, stop: StopSignals) -> int:
    entries = parse_queue(path, extra_overrides)
    rroot = runs_root(root)
    log(f"queue {path}: {len(entries)} run(s), runs root {rroot}")
    pending = [e for e in entries if not (rroot / e.run_id / "summary.json").exists()]

    if smoke:
        for e in pending:
            try:
                loss = smoke_check(e.cfg)
            except Exception as exc:
                log(f"smoke check FAILED for {e.run_id} (line {e.line_no}): {type(exc).__name__}: {exc}")
                return 1
            log(f"smoke ok  {e.run_id}  loss {loss:.3f}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if stop.received is not None:
                log("stopped by a signal during the smoke checks")
                return EXIT_INTERRUPTED

    results: dict[str, str] = {}
    interrupted = False
    for e in entries:
        if stop.received is not None:
            interrupted = True
            break
        run_dir = rroot / e.run_id
        if (run_dir / "summary.json").exists():
            results[e.run_id] = "skip"
            log(f"skip      {e.run_id} (summary.json exists)")
            continue
        try:
            status = Trainer(e.cfg, runs_root_dir=rroot, log=log, stop=stop).run()
        except Exception:
            log(f"FAILED    {e.run_id}\n{traceback.format_exc()}")
            status = "failed"
        results[e.run_id] = status
        # A Trainer is a reference cycle (its log closure): collect it now so its model and
        # gradients do not count towards the next run's peak memory.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if status == "interrupted":
            interrupted = True
            break

    log("-- queue summary --")
    for e in entries:
        status = results.get(e.run_id, "pending")
        s = load_summary(rroot / e.run_id)
        detail = _fmt_summary(s) if s else ""
        log(f"{status.upper():11s} {e.run_id}  {detail}".rstrip())
    if interrupted:
        return EXIT_INTERRUPTED
    return 1 if "failed" in results.values() else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run every configuration in a queue file, resumably.")
    p.add_argument("queue", help="queue file, e.g. configs/queue/phase2_sanity.txt")
    p.add_argument("--runs-root", default=None, help="runs root (default $SYS819_RUNS or ~/runs/sys819)")
    p.add_argument("--no-smoke", action="store_true", help="skip the 1-step smoke check")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override applied to every line (e.g. data.num_workers=6); repeatable")
    args = p.parse_args(argv)
    return run_queue(args.queue, root=args.runs_root, smoke=not args.no_smoke,
                     extra_overrides=args.overrides, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    sys.exit(main())
