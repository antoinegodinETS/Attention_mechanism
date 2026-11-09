"""Show done, running and pending runs for a queue file.

A run is ``done`` when its ``summary.json`` exists, ``running?`` when its last checkpoint or
metrics line was written in the last ``--active-minutes`` minutes, ``partial`` when it has a
checkpoint but no recent activity (resume it by re-running the queue), else ``pending``.

Usage: python scripts/status.py configs/queue/phase2_sanity.txt [--runs-root DIR]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from attnvit.config import runs_root
from attnvit.queue import parse_queue
from attnvit.utils import read_json, read_jsonl


def run_status(run_dir: Path, epochs: int, active_minutes: float, now: float) -> tuple[str, str]:
    summary = run_dir / "summary.json"
    if summary.exists():
        s = read_json(summary)
        test = f" test {s['test_acc']:.4f}" if s.get("test_acc") is not None else ""
        return "done", f"val {s['final_val_acc']:.4f}{test}, {s['train_time_s'] / 3600:.2f} h"
    rows = read_jsonl(run_dir / "metrics.jsonl")
    files = [p for p in (run_dir / "last.pt", run_dir / "metrics.jsonl") if p.exists()]
    if not files:
        return "pending", ""
    age_min = (now - max(p.stat().st_mtime for p in files)) / 60
    state = "running?" if age_min <= active_minutes else "partial"
    done = rows[-1]["epoch"] if rows else 0
    val = f", val {rows[-1]['val_acc']:.4f}" if rows else ""
    return state, f"epoch {done}/{epochs}{val}, updated {age_min:.0f} min ago"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("queue")
    p.add_argument("--runs-root", default=None)
    p.add_argument("--active-minutes", type=float, default=15.0)
    args = p.parse_args(argv)
    root = runs_root(args.runs_root)
    entries = parse_queue(args.queue)
    now = time.time()
    counts: dict[str, int] = {}
    print(f"runs root {root}")
    for e in entries:
        state, detail = run_status(root / e.run_id, e.cfg.optim.epochs, args.active_minutes, now)
        counts[state] = counts.get(state, 0) + 1
        print(f"{state:9s} {e.run_id}  {detail}".rstrip())
    print(", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "empty queue")
    return 0


if __name__ == "__main__":
    sys.exit(main())
