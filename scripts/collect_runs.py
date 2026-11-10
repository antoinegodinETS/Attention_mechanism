"""Collect finished run folders into two small committed CSVs.

Reads every ``<runs>/<run_id>/`` that holds a ``summary.json`` (plus its ``config.yaml`` and
``metrics.jsonl``) and writes:

- ``results/final_runs.csv``: one row per run, from ``summary.json`` and the run's config
  (condition, mechanism, seed, hyperparameters, test accuracy, times, peak memory, hardware);
- ``results/final_curves.csv``: one row per run and epoch, from ``metrics.jsonl``.

A folder without ``summary.json`` is a run still in progress: it is reported and skipped, so
re-running the collector once it finishes adds it with no code change. Only runs whose tag is
``--tag`` (default ``final``) are kept. Raw folders and checkpoints never enter the repository.

Usage: python scripts/collect_runs.py [RUNS_DIR] [--tag final] [--out-dir results]
RUNS_DIR defaults to ``$SYS819_RUNS``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# (column, source): "s" = summary.json key, "c" = dotted config.yaml key.
RUN_COLUMNS = [
    ("run_id", "s:run_id"),
    ("condition", None),
    ("dataset", "s:dataset"),
    ("res", "s:res"),
    ("patch_size", "s:patch_size"),
    ("mechanism", "s:mechanism"),
    ("seed", "s:seed"),
    ("tag", "s:tag"),
    ("head", "s:head"),
    ("params", "s:params"),
    ("epochs", "s:epochs"),
    ("batch_size", "s:batch_size"),
    ("accum_steps", "s:accum_steps"),
    ("lr", "c:optim.lr"),
    ("weight_decay", "c:optim.weight_decay"),
    ("drop_path", "c:model.drop_path"),
    ("warmup_epochs", "c:optim.warmup_epochs"),
    ("dim", "c:model.dim"),
    ("depth", "c:model.depth"),
    ("num_heads", "c:model.num_heads"),
    ("test_acc", "s:test_acc"),
    ("test_loss", "s:test_loss"),
    ("final_val_acc", "s:final_val_acc"),
    ("best_val_acc", "s:best_val_acc"),
    ("best_epoch", "s:best_epoch"),
    ("train_time_s", "s:train_time_s"),
    ("time_per_epoch_s", "s:time_per_epoch_s"),
    ("compute_time_s", "s:compute_time_s"),
    ("data_wait_s", "s:data_wait_s"),
    ("eval_time_s", "s:eval_time_s"),
    ("wall_time_s", "s:wall_time_s"),
    ("peak_mem_mib", "s:peak_mem_mib"),
    ("peak_mem_reserved_mib", "s:peak_mem_reserved_mib"),
    ("gpu", "s:gpu"),
    ("amp_fp16", "s:amp_fp16"),
    ("sdpa_backend", "s:sdpa_backend"),
    ("torch", "s:torch"),
    ("slurm_job_id", "s:slurm_job_id"),
    ("finished_at", "s:finished_at"),
]
CURVE_COLUMNS = ["run_id", "condition", "mechanism", "seed", "epoch", "lr", "train_loss", "train_acc", "val_loss",
                 "val_acc", "epoch_time_s", "images_per_s", "peak_mem_mib"]


def condition_name(dataset: str, res) -> str:
    """``cifar10``/32 -> ``cifar10-32``; the key the aggregate groups runs by."""
    return f"{dataset}-{int(res)}"


def _dotted(cfg: dict, key: str):
    for part in key.split("."):
        if not isinstance(cfg, dict) or part not in cfg:
            return None
        cfg = cfg[part]
    return cfg


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def read_run(run_dir: Path) -> tuple[dict, list[dict]]:
    """One run folder -> (run row, curve rows). Raises if ``summary.json`` is missing."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    cfg_path = run_dir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    row = {}
    for col, src in RUN_COLUMNS:
        if src is None:
            continue
        kind, key = src.split(":", 1)
        row[col] = summary.get(key) if kind == "s" else _dotted(cfg, key)
    row["condition"] = condition_name(row["dataset"], row["res"])
    curves = []
    metrics = run_dir / "metrics.jsonl"
    if metrics.exists():
        for line in metrics.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            c = {k: m.get(k) for k in CURVE_COLUMNS}
            c.update(run_id=row["run_id"], condition=row["condition"], mechanism=row["mechanism"], seed=row["seed"])
            curves.append(c)
    return row, curves


def collect(runs_dir: Path, tag: str | None = "final") -> tuple[list[dict], list[dict], list[str]]:
    """Scan ``runs_dir`` -> (run rows, curve rows, names of folders skipped as unfinished)."""
    rows, curves, pending = [], [], []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if not (d / "summary.json").exists():
            if (d / "config.yaml").exists() or (d / "metrics.jsonl").exists():
                pending.append(d.name)
            continue
        row, c = read_run(d)
        if tag and row.get("tag") != tag:
            continue
        rows.append(row)
        curves.extend(c)
    key = lambda r: (r["condition"], r["mechanism"], int(r["seed"]))  # noqa: E731
    rows.sort(key=key)
    curves.sort(key=lambda r: (*key(r), int(r["epoch"])))
    return rows, curves, pending


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in columns})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs_dir", nargs="?", default=os.environ.get("SYS819_RUNS"),
                   help="folder holding one sub-folder per run (default: $SYS819_RUNS)")
    p.add_argument("--tag", default="final", help="keep only runs with this tag ('' keeps all)")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    args = p.parse_args(argv)
    if not args.runs_dir:
        p.error("give RUNS_DIR or set SYS819_RUNS")
    runs_dir = Path(args.runs_dir).expanduser()
    if not runs_dir.is_dir():
        p.error(f"not a directory: {runs_dir}")
    rows, curves, pending = collect(runs_dir, args.tag or None)
    out = Path(args.out_dir)
    write_csv(out / "final_runs.csv", [c for c, _ in RUN_COLUMNS], rows)
    write_csv(out / "final_curves.csv", CURVE_COLUMNS, curves)
    print(f"{len(rows)} finished runs, {len(curves)} epoch rows -> {out / 'final_runs.csv'}, {out / 'final_curves.csv'}")
    for name in pending:
        print(f"  skipped (no summary.json yet): {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
