"""Build every phase 7 table and figure from the committed CSVs, and nothing else.

Inputs (all under ``results/``): ``final_runs.csv`` and ``final_curves.csv`` (written by
``scripts/collect_runs.py``), ``cost_benchmark.csv`` (phase 4) and ``hpsearch_cifar10.csv``
(phase 5, for the selected hyperparameters' search trial).

Outputs:

- ``results/main_table.csv`` / ``.md``: test accuracy mean ± std over seeds per mechanism and
  condition, with training time, peak memory, parameters and forward GFLOPs;
- ``results/hypotheses.csv`` / ``.md``: the H1-H4 verdicts (Report 1 p.10), each with the
  numbers it rests on; H3 and H4 per condition;
- ``figures/cost_vs_tokens.png`` (drawn by ``make_cost_figure.py``), ``accuracy_by_mechanism.png``,
  ``accuracy_vs_cost.png``, ``val_curves.png`` and ``attention_keys.png``.

A condition/mechanism cell with no finished run is reported as pending (and one with fewer
than ``SEEDS`` runs as partial) instead of being dropped; H3/H4 are marked provisional while any
cell is incomplete. Re-running the collector and this script after more runs land gives the
complete outputs with no code change.

Usage: python scripts/aggregate_results.py [--results results] [--figures figures]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# The experiment matrix (project plan): every condition x mechanism should end with SEEDS runs.
CONDITIONS = [
    ("cifar10-32", "CIFAR-10 32 px"),
    ("cifar10_hr-32", "CIFAR-10-HR 32 px"),
    ("cifar10_hr-160", "CIFAR-10-HR 160 px"),
]
MECHANISMS = ["full", "windows", "axial"]
SEEDS = 3
PATCH = 4

# Verdict thresholds, stated in every verdict that uses them.
H1_SAME_TOL = 0.10    # H1 "about the same": windows within 10 % of full forward FLOPs at 32 px
H2_CUT_MIN = 0.25     # H2 "cut FLOPs a lot": at least 25 % fewer forward FLOPs than full at 160 px
H4_CLOSE_PTS = 1.0    # H4 "close": axial mean test accuracy within 1.0 point of full
# H3 "beat": windows' mean exceeds full's by more than the combined seed spread sqrt(sd_w^2 + sd_f^2).


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_COST_FIG = _load_script("make_cost_figure")
COLORS, FALLBACK = _COST_FIG.COLORS, _COST_FIG.FALLBACK
INK, MUTED, GRID = _COST_FIG.INK, _COST_FIG.MUTED, _COST_FIG.GRID
COND_LABEL = dict(CONDITIONS)
COND_MARKER = {"cifar10-32": "o", "cifar10_hr-32": "s", "cifar10_hr-160": "D"}


def _res(cond: str) -> int:
    return int(cond.rsplit("-", 1)[1])


# --------------------------------------------------------------------------- tables


def load_inputs(results: Path) -> dict[str, pd.DataFrame]:
    def read(name, required=True):
        path = results / name
        if not path.exists():
            if required:
                raise FileNotFoundError(f"{path} is missing (run scripts/collect_runs.py first)")
            return pd.DataFrame()
        return pd.read_csv(path)

    return {
        "runs": read("final_runs.csv"),
        "curves": read("final_curves.csv"),
        "cost": read("cost_benchmark.csv"),
        "hps": read("hpsearch_cifar10.csv", required=False),
    }


def cost_lookup(cost: pd.DataFrame, mech: str, res: int, patch: int = PATCH) -> pd.Series | None:
    sel = cost[(cost["mechanism"] == mech) & (cost["res"] == res) & (cost["patch"] == patch)
               & (cost["status"] == "ok")]
    return None if sel.empty else sel.iloc[0]


def main_table(runs: pd.DataFrame, cost: pd.DataFrame) -> pd.DataFrame:
    """One row per condition x mechanism of the full matrix, finished or not."""
    out = []
    for cond, label in CONDITIONS:
        for mech in MECHANISMS:
            sel = runs[(runs["condition"] == cond) & (runs["mechanism"] == mech)] if len(runs) else runs
            n = len(sel)
            c = cost_lookup(cost, mech, _res(cond))
            row = {
                "condition": cond, "condition_label": label, "res": _res(cond), "mechanism": mech,
                "tokens": int(c["tokens"]) if c is not None else None,
                "n_seeds": n,
                "status": "complete" if n >= SEEDS else ("pending" if n == 0 else "partial"),
                "seeds": " ".join(str(int(s)) for s in sorted(sel["seed"])) if n else "",
                "gflops_fwd": float(c["gflops_fwd_counted"]) if c is not None else None,
                "bench_train_ms_per_image": float(c["train_ms_per_image"]) if c is not None else None,
            }
            if n:
                acc = sel["test_acc"] * 100
                row.update(
                    test_acc_mean=acc.mean(), test_acc_std=acc.std(ddof=1) if n > 1 else float("nan"),
                    best_val_acc_mean=sel["best_val_acc"].mean() * 100,
                    train_time_min_mean=sel["train_time_s"].mean() / 60,
                    train_time_min_std=(sel["train_time_s"] / 60).std(ddof=1) if n > 1 else float("nan"),
                    time_per_epoch_s_mean=sel["time_per_epoch_s"].mean(),
                    peak_mem_gib_max=sel["peak_mem_mib"].max() / 1024,
                    params_m=sel["params"].iloc[0] / 1e6,
                    gpu=sel["gpu"].iloc[0],
                )
            out.append(row)
    cols = ["condition", "condition_label", "res", "tokens", "mechanism", "status", "n_seeds", "seeds",
            "test_acc_mean", "test_acc_std", "best_val_acc_mean", "train_time_min_mean", "train_time_min_std",
            "time_per_epoch_s_mean", "peak_mem_gib_max", "params_m", "gflops_fwd", "bench_train_ms_per_image", "gpu"]
    return pd.DataFrame(out).reindex(columns=cols)


def _pm(mean, std, digits=2) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "pending"
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _get(table: pd.DataFrame, cond: str, mech: str) -> dict:
    sel = table[(table["condition"] == cond) & (table["mechanism"] == mech)]
    return sel.iloc[0].to_dict() if len(sel) else {}


def _finished(cell: dict) -> bool:
    return cell.get("n_seeds", 0) > 0


def _acc_txt(mech: str, cell: dict) -> str:
    if not _finished(cell):
        return f"{mech} pending"
    n = int(cell["n_seeds"])
    seeds = "" if n >= SEEDS else f", {n}/{SEEDS} seeds"
    return f"{mech} {_pm(cell['test_acc_mean'], cell['test_acc_std'])} %{seeds}"


def hypotheses(table: pd.DataFrame, cost: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_complete = bool((table["status"] == "complete").all())

    # H1: cost at 32 px (phase 4 benchmark, plus the measured final-run training time on CIFAR-10).
    c = {m: cost_lookup(cost, m, 32) for m in MECHANISMS}
    if all(v is not None for v in c.values()):
        g = {m: float(c[m]["gflops_fwd_counted"]) for m in MECHANISMS}
        t = {m: float(c[m]["train_ms_per_image"]) for m in MECHANISMS}
        rw, ra = g["windows"] / g["full"], g["axial"] / g["full"]
        same = abs(rw - 1) <= H1_SAME_TOL
        axial_most = g["axial"] == max(g.values())
        tt = {m: _get(table, "cifar10-32", m).get("train_time_min_mean") for m in MECHANISMS}
        ev = (f"forward GFLOPs per image full / windows / axial = {g['full']:.3f} / {g['windows']:.3f} / "
              f"{g['axial']:.3f} (windows {rw:.2f}x, axial {ra:.2f}x full); benchmark training time "
              f"{t['full']:.3f} / {t['windows']:.3f} / {t['axial']:.3f} ms per image "
              f"({t['windows'] / t['full']:.2f}x, {t['axial'] / t['full']:.2f}x)")
        if all(v is not None and not math.isnan(v) for v in tt.values()):
            ev += (f"; CIFAR-10 final runs {tt['full']:.1f} / {tt['windows']:.1f} / {tt['axial']:.1f} min per "
                   "100 epochs")
        if same and axial_most:
            verdict = "supported"
            note = (f"full and windows within {H1_SAME_TOL:.0%} in FLOPs and axial the most expensive "
                    f"({ra:.2f}x); in measured time windows is {t['windows'] / t['full']:.2f}x full despite fewer FLOPs")
        else:
            verdict = "not supported"
            note = (f"windows/full FLOPs ratio {rw:.2f} (tolerance {H1_SAME_TOL:.0%}); "
                    f"axial {'is' if axial_most else 'is not'} the most expensive")
        rows.append(dict(hypothesis="H1", condition="32 px (cost benchmark)", verdict=verdict, status="final",
                         claim="At 32 px all three cost about the same, axial the most.", evidence=ev, note=note))
    else:
        rows.append(dict(hypothesis="H1", condition="32 px (cost benchmark)", verdict="pending", status="pending",
                         claim="At 32 px all three cost about the same, axial the most.",
                         evidence="no 32 px / P4 rows in results/cost_benchmark.csv", note=""))

    # H2: cost at 160 px / P4.
    c = {m: cost_lookup(cost, m, 160) for m in MECHANISMS}
    claim2 = "At 160 px windows and axial cut FLOPs a lot."
    if all(v is not None for v in c.values()):
        g = {m: float(c[m]["gflops_fwd_counted"]) for m in MECHANISMS}
        t = {m: float(c[m]["train_ms_per_image"]) for m in MECHANISMS}
        mem = {m: float(c[m]["peak_alloc_mib_sdpa"]) / 1024 for m in MECHANISMS}
        cut = {m: 1 - g[m] / g["full"] for m in ("windows", "axial")}
        ok = all(v >= H2_CUT_MIN for v in cut.values())
        ev = (f"forward GFLOPs per image at 1,600 tokens full / windows / axial = {g['full']:.1f} / "
              f"{g['windows']:.1f} / {g['axial']:.1f} (windows -{cut['windows']:.0%}, axial -{cut['axial']:.0%}); "
              f"benchmark training time {t['full']:.2f} / {t['windows']:.2f} / {t['axial']:.2f} ms per image; "
              f"peak memory (SDPA) {mem['full']:.1f} / {mem['windows']:.1f} / {mem['axial']:.1f} GiB")
        tt = {m: _get(table, "cifar10_hr-160", m) for m in MECHANISMS}
        parts = [f"{m} {tt[m]['train_time_min_mean']:.0f} min" if _finished(tt[m]) else f"{m} pending"
                 for m in MECHANISMS]
        ev += "; HR-160 final runs per 100 epochs: " + ", ".join(parts)
        note = (f"both cut forward FLOPs by at least {H2_CUT_MIN:.0%}; the time saving is smaller "
                f"(windows {t['windows'] / t['full']:.2f}x, axial {t['axial'] / t['full']:.2f}x full) and peak memory "
                "is not reduced with the memory-efficient kernel") if ok else \
            f"cut below {H2_CUT_MIN:.0%}: windows {cut['windows']:.0%}, axial {cut['axial']:.0%}"
        rows.append(dict(hypothesis="H2", condition="160 px / P4 (cost benchmark)",
                         verdict="supported" if ok else "not supported", status="final", claim=claim2,
                         evidence=ev, note=note))
    else:
        rows.append(dict(hypothesis="H2", condition="160 px / P4 (cost benchmark)", verdict="pending",
                         status="pending", claim=claim2, evidence="no 160 px / P4 rows in cost_benchmark.csv", note=""))

    # H3 / H4: accuracy, per condition.
    status_all = "final" if all_complete else "provisional"
    for hyp, other, claim in (("H3", "windows", "Windows attention beats full attention on accuracy."),
                              ("H4", "axial", "Axial attention's accuracy is close to full attention's.")):
        for cond, label in CONDITIONS:
            f, o = _get(table, cond, "full"), _get(table, cond, other)
            ev = f"test accuracy {_acc_txt('full', f)}, {_acc_txt(other, o)}"
            if not (_finished(f) and _finished(o)):
                rows.append(dict(hypothesis=hyp, condition=label, verdict="pending", status="pending", claim=claim,
                                 evidence=ev, note="waiting for runs"))
                continue
            diff = o["test_acc_mean"] - f["test_acc_mean"]
            sds = [x for x in (o["test_acc_std"], f["test_acc_std"]) if not math.isnan(x)]
            spread = math.sqrt(sum(s * s for s in sds)) if sds else 0.0
            ev += f"; difference {other} - full = {diff:+.2f} points (combined seed spread {spread:.2f})"
            if hyp == "H3":
                if diff > spread:
                    verdict, note = "supported", f"windows ahead by more than the seed spread"
                elif diff < -spread:
                    verdict, note = "not supported", f"full ahead by {-diff:.2f} points, beyond the seed spread"
                else:
                    verdict, note = "inconclusive", "difference within the seed spread"
            else:
                if abs(diff) <= H4_CLOSE_PTS:
                    verdict, note = "supported", f"within {H4_CLOSE_PTS:.1f} point of full"
                    if abs(diff) + spread > H4_CLOSE_PTS:
                        note += f" (borderline: the seed spread {spread:.2f} reaches past the tolerance)"
                else:
                    verdict = "not supported"
                    note = f"{abs(diff):.2f} points {'below' if diff < 0 else 'above'} full (tolerance {H4_CLOSE_PTS:.1f})"
            complete = f["status"] == "complete" and o["status"] == "complete"
            rows.append(dict(hypothesis=hyp, condition=label, verdict=verdict,
                             status=status_all if complete else "provisional", claim=claim, evidence=ev, note=note))
    return pd.DataFrame(rows, columns=["hypothesis", "condition", "verdict", "status", "claim", "evidence", "note"])


def main_table_markdown(table: pd.DataFrame) -> str:
    lines = [
        "# Main results",
        "",
        f"Test accuracy (%) is the mean ± sample standard deviation over {SEEDS} seeds, evaluated once on the "
        "test split after 100 epochs. Training time is per 100-epoch run on one H100; peak memory is the "
        "largest allocated over the seeds. Forward GFLOPs per image come from `results/cost_benchmark.csv` "
        f"(P{PATCH}). Generated by `scripts/aggregate_results.py`; do not edit by hand.",
        "",
        "| Condition | Mechanism | Seeds | Test acc. (%) | Train time (min) | Peak mem. (GiB) | Params (M) | GFLOPs fwd |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        done = r["n_seeds"] > 0
        seeds = f"{int(r['n_seeds'])}/{SEEDS}" + ("" if r["status"] == "complete" else f" ({r['status']})")
        lines.append("| " + " | ".join([
            r["condition_label"], r["mechanism"], seeds,
            _pm(r["test_acc_mean"], r["test_acc_std"]) if done else "pending",
            _pm(r["train_time_min_mean"], r["train_time_min_std"], 1) if done else "pending",
            f"{r['peak_mem_gib_max']:.2f}" if done else "pending",
            f"{r['params_m']:.2f}" if done else "",
            f"{r['gflops_fwd']:.2f}" if pd.notna(r["gflops_fwd"]) else "",
        ]) + " |")
    return "\n".join(lines) + "\n"


def selected_hyperparameters(runs: pd.DataFrame, hps: pd.DataFrame) -> pd.DataFrame:
    """Per mechanism: the hyperparameters the final runs used, and the phase 5 trial they came from."""
    rows = []
    for mech in MECHANISMS:
        sel = runs[runs["mechanism"] == mech] if len(runs) else runs
        if sel.empty:
            continue
        r = sel.iloc[0]
        row = {"mechanism": mech, "lr": r["lr"], "weight_decay": r["weight_decay"], "drop_path": r["drop_path"],
               "warmup_epochs": r["warmup_epochs"], "search_trial": None, "search_val_acc_epoch30": None,
               "search_trials": None}
        if len(hps):
            h = hps[(hps["mechanism"] == mech) & (hps["status"] == "done")]
            row["search_trials"] = len(h)
            match = h[((h["lr"] - r["lr"]).abs() <= 1e-9) & ((h["weight_decay"] - r["weight_decay"]).abs() <= 1e-9)
                      & ((h["drop_path"] - r["drop_path"]).abs() <= 1e-9)]
            if len(match):
                row["search_trial"] = int(match.iloc[0]["trial"])
                row["search_val_acc_epoch30"] = float(match.iloc[0]["val_acc_epoch30"]) * 100
        rows.append(row)
    return pd.DataFrame(rows)


def hyperparameters_markdown(sel: pd.DataFrame) -> str:
    if sel.empty:
        return ""
    lines = [
        "",
        "## Selected hyperparameters",
        "",
        "Chosen per mechanism by the phase 5 search on CIFAR-10 32 px (`results/hpsearch_cifar10.csv`, highest "
        "validation accuracy at epoch 30) and reused unchanged for every condition.",
        "",
        "| Mechanism | lr | Weight decay | Drop-path | Warmup (epochs) | Search trial | Search val. acc. @30 (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in sel.iterrows():
        trial = "" if pd.isna(r["search_trial"]) else f"{int(r['search_trial'])} of {int(r['search_trials'])}"
        acc = "" if pd.isna(r["search_val_acc_epoch30"]) else f"{r['search_val_acc_epoch30']:.2f}"
        lines.append(f"| {r['mechanism']} | {r['lr']:g} | {r['weight_decay']:g} | {r['drop_path']:g} | "
                     f"{r['warmup_epochs']:g} | {trial} | {acc} |")
    return "\n".join(lines) + "\n"


def hypotheses_markdown(hyp: pd.DataFrame) -> str:
    lines = [
        "# Hypotheses H1-H4 (Report 1 p.10)",
        "",
        "Thresholds: H1 \"about the same\" = windows within "
        f"{H1_SAME_TOL:.0%} of full forward FLOPs; H2 \"a lot\" = at least {H2_CUT_MIN:.0%} fewer forward "
        "FLOPs than full; H3 \"beat\" = windows' mean ahead of full's by more than the combined seed spread "
        f"sqrt(sd_w² + sd_f²); H4 \"close\" = axial's mean within {H4_CLOSE_PTS:.1f} point of full's. "
        "A verdict marked provisional may change once every condition has all its seeds. "
        "Generated by `scripts/aggregate_results.py`; do not edit by hand.",
        "",
        "| Hypothesis | Condition | Verdict | Status | Measured | Note |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in hyp.iterrows():
        lines.append(f"| {r['hypothesis']}: {r['claim']} | {r['condition']} | {r['verdict']} | {r['status']} | "
                     f"{r['evidence']} | {r['note']} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- figures


def _style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, which="both")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _mech_handles(extra=()):
    from matplotlib.lines import Line2D

    return [Line2D([], [], color=COLORS.get(m, FALLBACK), lw=2, marker="o", label=m) for m in MECHANISMS] + list(extra)


def fig_accuracy_by_mechanism(runs: pd.DataFrame, table: pd.DataFrame, out: Path) -> Path:
    plt = _plt()
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(13, 4.4), constrained_layout=True)
    for ax, (cond, label) in zip(axes, CONDITIONS):
        vals = []
        for i, mech in enumerate(MECHANISMS):
            color = COLORS.get(mech, FALLBACK)
            cell = _get(table, cond, mech)
            if not _finished(cell):
                ax.text(i, 0.5, "pending", transform=ax.get_xaxis_transform(), ha="center", va="center",
                        color=MUTED, fontsize=10, rotation=90)
                continue
            seeds = runs[(runs["condition"] == cond) & (runs["mechanism"] == mech)]["test_acc"] * 100
            vals += list(seeds)
            ax.scatter([i] * len(seeds), seeds, color=color, s=22, alpha=0.55, zorder=3, edgecolors="none")
            sd = cell["test_acc_std"] if not math.isnan(cell["test_acc_std"]) else 0
            ax.errorbar(i + 0.18, cell["test_acc_mean"], yerr=sd, fmt="o", color=color, ms=7, capsize=4, lw=1.8,
                        zorder=4)
            ax.annotate(f"{cell['test_acc_mean']:.2f}", (i + 0.18, cell["test_acc_mean"]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set_xticks(range(len(MECHANISMS)), MECHANISMS)
        ax.set_xlim(-0.5, len(MECHANISMS) - 0.3)
        if vals:
            lo, hi = min(vals), max(vals)
            pad = max(0.6, (hi - lo) * 0.25)
            ax.set_ylim(lo - pad, hi + pad)
        _style(ax, label, ylabel="test accuracy (%)" if ax is axes[0] else None)
    handles = [Line2D([], [], color=MUTED, ls="none", marker="o", alpha=0.55, ms=5, label="one seed"),
               Line2D([], [], color=MUTED, marker="o", ms=7, label=f"mean ± std over seeds")]
    fig.suptitle("Test accuracy by mechanism (y-axes differ per condition)", color=INK, fontsize=12, x=0.01,
                 ha="left")
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_accuracy_vs_cost(table: pd.DataFrame, out: Path) -> Path:
    plt = _plt()
    from matplotlib.lines import Line2D

    from matplotlib.ticker import NullFormatter, ScalarFormatter

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    panels = [("gflops_fwd", None, "forward GFLOPs per image (cost benchmark)", "Accuracy vs forward FLOPs"),
              ("train_time_min_mean", "train_time_min_std", "training time per 100-epoch run (min, one H100)",
               "Accuracy vs training time")]
    done = table[table["n_seeds"] > 0]
    for ax, (xcol, xerr, xlabel, title) in zip(axes, panels):
        for cond, _ in CONDITIONS:
            sel = done[done["condition"] == cond]
            if sel.empty:
                continue
            sel = sel.sort_values(xcol)
            ax.plot(sel[xcol], sel["test_acc_mean"], color=GRID, lw=1.2, zorder=1)
            for _, r in sel.iterrows():
                color = COLORS.get(r["mechanism"], FALLBACK)
                ax.errorbar(r[xcol], r["test_acc_mean"], yerr=0 if math.isnan(r["test_acc_std"]) else r["test_acc_std"],
                            xerr=None if xerr is None or math.isnan(r[xerr]) else r[xerr],
                            fmt=COND_MARKER[cond], color=color, ms=7, capsize=3, lw=1.4, zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        _style(ax, title, xlabel, "test accuracy (%)" if ax is axes[0] else None)
    pending = table[table["n_seeds"] == 0]
    if len(pending):
        txt = "pending: " + ", ".join(f"{COND_LABEL[r.condition]} {r.mechanism}" for r in pending.itertuples())
        fig.text(0.99, 0.995, txt, ha="right", va="top", color=MUTED, fontsize=9)
    handles = _mech_handles([Line2D([], [], color=MUTED, ls="none", marker=COND_MARKER[c], label=l)
                             for c, l in CONDITIONS])
    fig.suptitle("Accuracy against cost (mean ± std over seeds)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_val_curves(curves: pd.DataFrame, table: pd.DataFrame, out: Path) -> Path:
    plt = _plt()
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(14, 4.4), constrained_layout=True, sharey=False)
    for ax, (cond, label) in zip(axes, CONDITIONS):
        missing = []
        for mech in MECHANISMS:
            sel = curves[(curves["condition"] == cond) & (curves["mechanism"] == mech)] if len(curves) else curves
            if sel.empty:
                missing.append(mech)
                continue
            g = sel.groupby("epoch")["val_acc"]
            mean, lo, hi = g.mean() * 100, g.min() * 100, g.max() * 100
            color = COLORS.get(mech, FALLBACK)
            ax.fill_between(mean.index, lo, hi, color=color, alpha=0.18, lw=0)
            n = sel["run_id"].nunique()
            ax.plot(mean.index, mean, color=color, lw=1.8,
                    label=f"{mech} ({mean.iloc[-1]:.1f} %" + ("" if n >= SEEDS else f", {n}/{SEEDS} seeds") + ")")
        if missing:
            ax.text(0.98, 0.04, "pending: " + ", ".join(missing), transform=ax.transAxes, ha="right", va="bottom",
                    color=MUTED, fontsize=9)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right" if not missing else "center right", frameon=False, fontsize=9)
        _style(ax, label, "epoch", "validation accuracy (%)" if ax is axes[0] else None)
    handles = [Patch(color=MUTED, alpha=0.25, label="min-max over seeds"),
               plt.Line2D([], [], color=MUTED, lw=1.8, label="mean over seeds (final-epoch value in the legend)")]
    fig.suptitle("Validation accuracy per epoch", color=INK, fontsize=12, x=0.01, ha="left")
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def attention_key_maps(grid: int = 8, window: int = 4, query: tuple[int, int] = (5, 6)) -> dict:
    """``{panel title: (grid, grid) bool array}`` of the keys one query token attends to.

    Built from the model code itself: the shifted-window mask is ``attention.shifted_window_mask``
    applied after the same cyclic roll the windows layer uses; axial is the column pass then the
    row pass of ``AxialAttention``.
    """
    import numpy as np
    import torch

    from attnvit.models.attention import shifted_window_mask, window_partition

    qi, qj = query
    H = W = grid
    maps = {"Full (and the CLS token)": np.ones((H, W), bool)}

    win = np.zeros((H, W), bool)
    r0, c0 = (qi // window) * window, (qj // window) * window
    win[r0:r0 + window, c0:c0 + window] = True
    maps[f"Windows, even block ({window}x{window})"] = win

    shift = window // 2
    idx = torch.arange(H * W).view(1, H, W, 1)
    rolled = torch.roll(idx, shifts=(-shift, -shift), dims=(1, 2))
    wins = window_partition(rolled, window).squeeze(-1)                  # (nW, M*M) original token ids
    mask = shifted_window_mask(H, W, window, shift)                      # (nW, M*M, M*M)
    q = qi * W + qj
    w_idx, pos = map(int, (wins == q).nonzero()[0])
    keys = wins[w_idx][mask[w_idx, pos]]
    sw = np.zeros(H * W, bool)
    sw[keys.numpy()] = True
    maps[f"Windows, odd block (shift {shift})"] = sw.reshape(H, W)

    col = np.zeros((H, W), bool)
    col[:, qj] = True
    maps["Axial, column pass"] = col
    row = np.zeros((H, W), bool)
    row[qi, :] = True
    maps["Axial, row pass"] = row
    return maps


def fig_attention_keys(out: Path, grid: int = 8, window: int = 4, query: tuple[int, int] = (5, 6)) -> Path:
    plt = _plt()
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    maps = attention_key_maps(grid, window, query)
    colors = [COLORS["full"], COLORS["windows"], COLORS["windows"], COLORS["axial"], COLORS["axial"]]
    fig, axes = plt.subplots(1, len(maps), figsize=(3.3 * len(maps), 3.9), constrained_layout=True)
    for ax, (title, m), color in zip(axes, maps.items(), colors):
        ax.imshow(m.astype(int), cmap=ListedColormap(["#f4f3ef", color]), vmin=0, vmax=1)
        ax.add_patch(Rectangle((query[1] - 0.5, query[0] - 0.5), 1, 1, fill=True, color=INK, zorder=3))
        ax.set_xticks(np.arange(-0.5, grid, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid, 1), minor=True)
        ax.grid(True, which="minor", color="white", lw=1.2)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        for s in ax.spines.values():
            s.set_visible(False)
        if "block" in title:
            off = 0 if "even" in title else -(window // 2)
            for b in range(off, grid + window, window):
                ax.axhline(b - 0.5, color=MUTED, lw=1.0, ls="--")
                ax.axvline(b - 0.5, color=MUTED, lw=1.0, ls="--")
            ax.set_xlim(-0.5, grid - 0.5)
            ax.set_ylim(grid - 0.5, -0.5)
        ax.set_title(f"{title}\n{int(m.sum())} of {grid * grid} patch keys", color=INK, fontsize=10, loc="left")
    fig.suptitle(f"Keys one query token (black) attends to in one layer, {grid}x{grid} patch grid (32 px / P4).\n"
                 "Odd window blocks roll the grid and mask tokens wrapped across the edge; axial runs a column "
                 "pass then a row pass, so one layer reaches the whole grid.", color=INK, fontsize=10, x=0.01,
                 ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- driver


def run(results: Path, figures: Path) -> list[Path]:
    data = load_inputs(results)
    runs, curves, cost = data["runs"], data["curves"], data["cost"]
    table = main_table(runs, cost)
    hyp = hypotheses(table, cost)
    written = []
    table.to_csv(results / "main_table.csv", index=False, float_format="%.4f", lineterminator="\n")
    hp_md = hyperparameters_markdown(selected_hyperparameters(runs, data["hps"]))
    (results / "main_table.md").write_text(main_table_markdown(table) + hp_md, encoding="utf-8")
    hyp.to_csv(results / "hypotheses.csv", index=False, lineterminator="\n")
    (results / "hypotheses.md").write_text(hypotheses_markdown(hyp), encoding="utf-8")
    written += [results / n for n in ("main_table.csv", "main_table.md", "hypotheses.csv", "hypotheses.md")]
    cost_rows = _COST_FIG.load_rows(results / "cost_benchmark.csv")
    if cost_rows:
        written.append(_COST_FIG.make_figure(cost_rows, figures / "cost_vs_tokens.png"))
    written.append(fig_accuracy_by_mechanism(runs, table, figures / "accuracy_by_mechanism.png"))
    written.append(fig_accuracy_vs_cost(table, figures / "accuracy_vs_cost.png"))
    written.append(fig_val_curves(curves, table, figures / "val_curves.png"))
    written.append(fig_attention_keys(figures / "attention_keys.png"))
    return written


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", default=str(REPO_ROOT / "results"))
    p.add_argument("--figures", default=str(REPO_ROOT / "figures"))
    args = p.parse_args(argv)
    for path in run(Path(args.results), Path(args.figures)):
        print(f"wrote {path}")
    table = pd.read_csv(Path(args.results) / "main_table.csv")
    todo = table[table["status"] != "complete"]
    if len(todo):
        print("incomplete: " + ", ".join(f"{r.condition} {r.mechanism} ({r.n_seeds}/{SEEDS})" for r in todo.itertuples()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
