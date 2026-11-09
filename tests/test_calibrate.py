"""scripts/calibrate.py on CPU with a tiny model, and the 160 px decision rule."""

import csv
import math

import pytest

from conftest import TINY


def test_calibrate_writes_csv_on_cpu(script, tmp_path, capsys):
    cal = script("calibrate")
    out = tmp_path / "calibration.csv"
    sets = [x for x in TINY if x.startswith("model.") and "patch_size" not in x]
    argv = ["--device", "cpu", "--settings", "32:8,64:8", "--batch-size", "8", "--accum", "2", "--steps", "3",
            "--warmup", "1", "--eval-iters", "2", "--out", str(out)]
    for s in sets:
        argv += ["--set", s]
    assert cal.main(argv) == 0
    rows = list(csv.DictReader(open(out)))
    assert [(r["mechanism"], r["res"], r["patch"]) for r in rows] == [("full", "32", "8"), ("full", "64", "8")]
    assert all(r["status"] == "ok" and float(r["step_ms_median"]) > 0 for r in rows)
    assert rows[0]["micro_batch"] == "4" and rows[0]["tokens"] == "16" and rows[1]["tokens"] == "64"
    assert rows[0]["steps_per_epoch"] == str(45000 // 8)
    assert "not on a GPU" in capsys.readouterr().out


def test_block_projection_and_decision(script):
    cal = script("calibrate")
    rows = [
        {"mechanism": "full", "res": 160, "patch": 4, "status": "ok", "est_run_h_100ep": 2.0, "est_run_h_50ep": 1.0},
        {"mechanism": "full", "res": 160, "patch": 8, "status": "ok", "est_run_h_100ep": 0.3, "est_run_h_50ep": 0.15},
        {"mechanism": "full", "res": 32, "patch": 4, "status": "ok", "est_run_h_100ep": 0.2, "est_run_h_50ep": 0.1},
    ]
    proj = cal.block_projection(rows)
    # Only full is registered: 3 seeds x 3 mechanisms, the two missing ones at full's cost.
    assert proj[(4, 100)]["hours"] == 18.0 and proj[(4, 100)]["stand_ins"] == 2
    assert proj[(8, 100)]["hours"] == pytest.approx(2.7)
    assert cal.decide(proj).startswith("option 1")
    assert cal.decide(proj, budget_h=10).startswith("option 2")
    assert cal.decide(proj, budget_h=5).startswith("option 3")
    rows[0]["status"] = "oom"
    assert math.isinf(cal.block_projection(rows)[(4, 50)]["hours"])
