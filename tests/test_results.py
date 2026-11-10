"""CPU tests for the phase 7 collector (scripts/collect_runs.py) and aggregate (scripts/aggregate_results.py)."""

import json
import shutil

import pandas as pd
import pytest
import yaml

from conftest import REPO_ROOT

ACC = {  # (condition, mechanism) -> test accuracy per seed
    ("cifar10-32", "full"): [0.88, 0.89, 0.90],
    ("cifar10-32", "windows"): [0.86, 0.86, 0.86],
    ("cifar10-32", "axial"): [0.875, 0.885, 0.88],
    ("cifar10_hr-32", "full"): [0.71, 0.72, 0.71],
    ("cifar10_hr-32", "windows"): [0.70, 0.70, 0.70],
    ("cifar10_hr-32", "axial"): [0.70, 0.71, 0.70],
    ("cifar10_hr-160", "windows"): [0.81, 0.80, 0.82],
}
LATE = {("cifar10_hr-160", "full"): [0.79, 0.78, 0.80], ("cifar10_hr-160", "axial"): [0.785, 0.79, 0.78]}


def make_run(root, cond, mech, seed, acc, *, tag="final", epochs=3, finished=True):
    dataset, res = cond.rsplit("-", 1)
    run_id = f"{dataset}-{res}-p4-{mech}-{tag}-s{seed}"
    d = root / run_id
    d.mkdir(parents=True)
    cfg = {"tag": tag, "seed": seed, "data": {"name": dataset, "res": int(res)},
           "model": {"mechanism": mech, "patch_size": 4, "dim": 192, "depth": 9, "num_heads": 3, "drop_path": 0.1},
           "optim": {"lr": 0.000901, "weight_decay": 0.0953, "warmup_epochs": 10.0}}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    lines = [json.dumps({"epoch": e, "lr": 1e-3, "train_loss": 2.0 / e, "train_acc": 0.1 * e, "val_loss": 1.0 / e,
                         "val_acc": acc * e / epochs, "epoch_time_s": 6.0, "images_per_s": 7000.0,
                         "peak_mem_mib": 500.0}) for e in range(1, epochs + 1)]
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n")
    if finished:
        summary = {"run_id": run_id, "dataset": dataset, "res": int(res), "patch_size": 4, "mechanism": mech,
                   "head": "gap", "tag": tag, "seed": seed, "params": 2_700_000, "epochs": epochs, "batch_size": 128,
                   "accum_steps": 1, "final_val_acc": acc, "best_val_acc": acc, "best_epoch": epochs,
                   "test_acc": acc, "test_loss": 0.5, "train_time_s": 600.0 + seed, "time_per_epoch_s": 6.0,
                   "compute_time_s": 580.0, "data_wait_s": 20.0, "eval_time_s": 10.0, "wall_time_s": 700.0,
                   "peak_mem_mib": 512.0, "peak_mem_reserved_mib": 600.0, "gpu": "NVIDIA H100 80GB HBM3",
                   "amp_fp16": True, "sdpa_backend": "efficient", "torch": "2.14.1", "slurm_job_id": "1",
                   "finished_at": "2026-11-09T19:41:54-0500"}
        (d / "summary.json").write_text(json.dumps(summary))
    return d


def populate(root, accs):
    for (cond, mech), seeds in accs.items():
        for seed, acc in enumerate(seeds):
            make_run(root, cond, mech, seed, acc)


@pytest.fixture
def workspace(tmp_path):
    """Fixture runs with HR-160 full and axial missing, plus results/ holding the committed cost CSVs."""
    runs = tmp_path / "runs"
    populate(runs, ACC)
    make_run(runs, "cifar10_hr-160", "full", 0, 0.5, finished=False)          # still training
    make_run(runs, "cifar10-32", "full", 0, 0.7, tag="search.t0")             # not a final run
    results = tmp_path / "results"
    results.mkdir()
    for name in ("cost_benchmark.csv", "hpsearch_cifar10.csv"):
        shutil.copy(REPO_ROOT / "results" / name, results / name)
    return runs, results, tmp_path / "figures"


def test_collector_reads_finished_final_runs_only(script, workspace, capsys):
    collect = script("collect_runs")
    runs, results, _ = workspace
    assert collect.main([str(runs), "--out-dir", str(results)]) == 0
    assert "skipped (no summary.json yet): cifar10_hr-160-p4-full-final-s0" in capsys.readouterr().out
    df = pd.read_csv(results / "final_runs.csv")
    assert len(df) == 21 and set(df["tag"]) == {"final"}
    assert set(df["condition"]) == {"cifar10-32", "cifar10_hr-32", "cifar10_hr-160"}
    row = df[df["run_id"] == "cifar10-32-p4-windows-final-s1"].iloc[0]
    assert row["lr"] == pytest.approx(0.000901) and row["drop_path"] == pytest.approx(0.1)
    curves = pd.read_csv(results / "final_curves.csv")
    assert len(curves) == 21 * 3 and curves.groupby("run_id")["epoch"].max().eq(3).all()


def test_collector_reads_sys819_runs(script, workspace, monkeypatch):
    collect = script("collect_runs")
    runs, results, _ = workspace
    monkeypatch.setenv("SYS819_RUNS", str(runs))
    assert collect.main(["--out-dir", str(results)]) == 0
    assert len(pd.read_csv(results / "final_runs.csv")) == 21


def test_aggregate_marks_missing_conditions_pending_then_completes(script, workspace):
    collect, agg = script("collect_runs"), script("aggregate_results")
    runs, results, figures = workspace
    collect.main([str(runs), "--out-dir", str(results)])
    agg.main(["--results", str(results), "--figures", str(figures)])

    table = pd.read_csv(results / "main_table.csv")
    assert len(table) == 9
    hr160 = table[table["condition"] == "cifar10_hr-160"].set_index("mechanism")
    assert hr160.loc["full", "status"] == "pending" and hr160.loc["axial", "status"] == "pending"
    assert hr160.loc["windows", "status"] == "complete"
    full32 = table[(table["condition"] == "cifar10-32") & (table["mechanism"] == "full")].iloc[0]
    assert full32["test_acc_mean"] == pytest.approx(89.0) and full32["test_acc_std"] == pytest.approx(1.0)
    assert "| CIFAR-10-HR 160 px | full | 0/3 (pending) | pending |" in (results / "main_table.md").read_text()

    hyp = pd.read_csv(results / "hypotheses.csv")
    assert list(hyp["hypothesis"]) == ["H1", "H2"] + ["H3"] * 3 + ["H4"] * 3
    assert set(hyp.loc[hyp["hypothesis"].isin(["H1", "H2"]), "verdict"]) == {"supported"}
    acc = hyp[hyp["hypothesis"].isin(["H3", "H4"])]
    assert set(acc.loc[acc["condition"] == "CIFAR-10-HR 160 px", "verdict"]) == {"pending"}
    assert set(acc.loc[acc["condition"] != "CIFAR-10-HR 160 px", "status"]) == {"provisional"}
    h3 = hyp[(hyp["hypothesis"] == "H3") & (hyp["condition"] == "CIFAR-10 32 px")].iloc[0]
    assert h3["verdict"] == "not supported" and "-3.00 points" in h3["evidence"]
    h4 = hyp[(hyp["hypothesis"] == "H4") & (hyp["condition"] == "CIFAR-10 32 px")].iloc[0]
    assert h4["verdict"] == "supported" and "88.00 ± 0.50" in h4["evidence"]
    for name in ("cost_vs_tokens", "accuracy_by_mechanism", "accuracy_vs_cost", "val_curves", "attention_keys"):
        assert (figures / f"{name}.png").stat().st_size > 0

    # The late runs arrive in the same folder: the same two commands give complete outputs.
    shutil.rmtree(runs / "cifar10_hr-160-p4-full-final-s0")
    populate(runs, LATE)
    collect.main([str(runs), "--out-dir", str(results)])
    agg.main(["--results", str(results), "--figures", str(figures)])
    assert set(pd.read_csv(results / "main_table.csv")["status"]) == {"complete"}
    hyp = pd.read_csv(results / "hypotheses.csv")
    assert "pending" not in set(hyp["verdict"]) and set(hyp["status"]) == {"final"}
    assert "pending" not in (results / "main_table.md").read_text()


def test_aggregate_reports_partial_seeds(script, workspace):
    collect, agg = script("collect_runs"), script("aggregate_results")
    runs, results, figures = workspace
    shutil.rmtree(runs / "cifar10_hr-160-p4-full-final-s0")
    make_run(runs, "cifar10_hr-160", "full", 0, 0.79)
    collect.main([str(runs), "--out-dir", str(results)])
    agg.main(["--results", str(results), "--figures", str(figures)])
    table = pd.read_csv(results / "main_table.csv")
    cell = table[(table["condition"] == "cifar10_hr-160") & (table["mechanism"] == "full")].iloc[0]
    assert cell["status"] == "partial" and cell["n_seeds"] == 1
    h3 = pd.read_csv(results / "hypotheses.csv")
    h3 = h3[(h3["hypothesis"] == "H3") & (h3["condition"] == "CIFAR-10-HR 160 px")].iloc[0]
    assert h3["status"] == "provisional" and "1/3 seeds" in h3["evidence"]


def test_aggregate_needs_collected_runs(script, tmp_path):
    agg = script("aggregate_results")
    with pytest.raises(FileNotFoundError, match="collect_runs"):
        agg.run(tmp_path, tmp_path / "figures")


def test_attention_key_maps_match_the_mechanisms(script):
    agg = script("aggregate_results")
    maps = list(agg.attention_key_maps(grid=8, window=4, query=(5, 6)).values())
    assert [int(m.sum()) for m in maps] == [64, 16, 8, 8, 8]
    shifted = maps[2]
    # Shifted window around (5, 6): rows 2-5, columns 6-7; the wrapped columns 0-1 are masked.
    assert shifted[2:6, 6:8].all() and not shifted[:, :6].any()
    assert maps[3][:, 6].all() and maps[4][5, :].all()
