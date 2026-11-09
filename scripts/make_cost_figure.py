"""Draw the cost-vs-tokens figure from ``results/cost_benchmark.csv`` (and nothing else).

Three log-log panels against the token count N: forward GFLOPs per image (counted, solid, with
the analytic Report 1 count dashed on top), forward plus backward time per image, and peak
allocated memory (SDPA solid with filled markers, naive kernel dotted with hollow markers; a
naive out-of-memory case is an ``x`` on the top edge). One colour per mechanism. Lines join the
P4 resolutions; the 160 px / P8 cases are unjoined diamonds.

Usage: python scripts/make_cost_figure.py [--csv results/cost_benchmark.csv] [--out figures/cost_vs_tokens.png]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Categorical slots 1-3 of the reference palette, in fixed order: full, windows, axial.
COLORS = {"full": "#2a78d6", "windows": "#eb6834", "axial": "#1baf7a", "axial_shared": "#6250d6"}
FALLBACK = "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MAIN_PATCH = 4


def _num(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("res", "patch", "tokens"):
            r[k] = int(float(r[k])) if r.get(k) else None
    return [r for r in rows if r["status"] != "skipped" and r["tokens"]]


def _series(rows, mech, patch_main: bool):
    sel = [r for r in rows if r["mechanism"] == mech and (r["patch"] == MAIN_PATCH) == patch_main]
    return sorted(sel, key=lambda r: r["tokens"])


def _plot_metric(ax, rows, mechs, key, *, style="-", marker="o", filled=True, label_suffix="", zorder=3,
                 lw=2.0):
    for mech in mechs:
        color = COLORS.get(mech, FALLBACK)
        mfc = color if filled else "white"
        for main in (True, False):
            pts = [(r["tokens"], _num(r.get(key))) for r in _series(rows, mech, main)]
            pts = [(n, v) for n, v in pts if v is not None and v > 0]
            if not pts:
                continue
            xs, ys = zip(*pts)
            if main:
                ax.plot(xs, ys, linestyle=style, color=color, lw=lw, marker=marker or "none", ms=6, mfc=mfc, mec=color, mew=1.5,
                        label=f"{mech}{label_suffix}", zorder=zorder)
            else:
                ax.plot(xs, ys, linestyle="none", color=color, marker="D", ms=7, mfc=mfc, mec=color, mew=1.5,
                        zorder=zorder + 1)


def _style(ax, title, ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("tokens N (patch tokens)", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.grid(False, which="minor")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, which="both")


def make_figure(rows: list[dict], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mechs = list(dict.fromkeys(r["mechanism"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    # Counted FLOPs as markers on the dashed analytic curve, so a mismatch shows as an off-curve marker.
    _plot_metric(ax, rows, mechs, "gflops_fwd_analytic", style="--", marker="", label_suffix=" (analytic)",
                 lw=1.5)
    _plot_metric(ax, rows, mechs, "gflops_fwd_counted", style="none", zorder=5)
    _style(ax, "Forward FLOPs per image (markers: counted)", "GFLOPs")
    for mech in mechs:  # direct labels at the right end of each P4 curve
        last = [r for r in _series(rows, mech, True) if _num(r.get("gflops_fwd_counted"))]
        if last:
            ax.annotate(mech, (last[-1]["tokens"], _num(last[-1]["gflops_fwd_counted"])), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=9)

    ax = axes[1]
    _plot_metric(ax, rows, mechs, "train_ms_per_image")
    _style(ax, "Forward + backward time per image", "ms per image")

    ax = axes[2]
    _plot_metric(ax, rows, mechs, "peak_alloc_mib_sdpa")
    _plot_metric(ax, rows, mechs, "peak_alloc_mib_naive", style=":", filled=False)
    _style(ax, "Peak memory per training step", "peak allocated MiB")
    ooms = [r for r in rows if r.get("naive_status") == "oom" or r.get("status") == "oom"]
    if ooms:
        top = ax.get_ylim()[1]
        for r in ooms:
            ax.plot(r["tokens"], top, marker="x", ms=9, mew=2, color=COLORS.get(r["mechanism"], FALLBACK),
                    clip_on=False, zorder=6, linestyle="none")
        ax.set_ylim(top=top)
        ax.annotate("x = out of memory", xy=(0.02, 0.97), xycoords="axes fraction", va="top", color=MUTED,
                    fontsize=9)

    batch = sorted({r.get("batch_size") for r in rows if r.get("batch_size")})
    gpu = sorted({r.get("gpu") or r.get("device") for r in rows})
    fig.suptitle(f"Cost vs tokens, P4 at {', '.join(str(r) for r in sorted({r['res'] for r in rows if r['patch'] == MAIN_PATCH}))} px"
                 f" (diamonds: 160 px / P8) | batch {', '.join(batch)} | {', '.join(gpu)}",
                 color=INK, fontsize=12, x=0.01, ha="left")

    handles = [Line2D([], [], color=COLORS.get(m, FALLBACK), lw=2, marker="o", label=m) for m in mechs]
    handles += [
        Line2D([], [], color=MUTED, lw=1.5, ls="--", label="analytic FLOPs (Report 1 + shared layers)"),
        Line2D([], [], color=MUTED, lw=2, marker="o", label="SDPA (training kernel)"),
        Line2D([], [], color=MUTED, lw=2, ls=":", marker="o", mfc="white", label="naive attention"),
        Line2D([], [], color=MUTED, ls="none", marker="D", label="160 px / P8"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", default=str(REPO_ROOT / "results" / "cost_benchmark.csv"))
    p.add_argument("--out", default=str(REPO_ROOT / "figures" / "cost_vs_tokens.png"))
    args = p.parse_args(argv)
    rows = load_rows(Path(args.csv))
    if not rows:
        p.error(f"no measured rows in {args.csv}")
    print(f"wrote {make_figure(rows, Path(args.out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
