"""scripts/hpsearch.py: trial generation, collection and selection on fixture run folders (CPU)."""

import csv
import math

import pytest
import yaml

from attnvit.config import dump_yaml, is_search_run, load_config, run_id, to_dict
from attnvit.queue import parse_queue
from attnvit.utils import append_jsonl, write_json


@pytest.fixture
def hps(script):
    return script("hpsearch")


def test_generation_is_deterministic_and_matches_the_committed_files(hps, tmp_path):
    assert hps.sample_trials() == hps.sample_trials()
    assert hps.sample_trials(seed=1) != hps.sample_trials()
    a = hps.generated_files(tmp_path / "a.csv", tmp_path / "qa")
    b = hps.generated_files(tmp_path / "b.csv", tmp_path / "qb")
    assert list(a.values()) == list(b.values())
    assert hps.main(["generate", "--check"]) == 0  # the committed trial list and queues are current
    assert hps.read_trials() == hps.sample_trials()


def test_trial_i_is_identical_for_every_mechanism(hps):
    per_mech = {m: parse_queue(hps.queue_path(m)) for m in hps.MECHANISMS}
    assert all(len(entries) == hps.N_TRIALS for entries in per_mech.values())
    for i in range(hps.N_TRIALS):
        dicts = []
        for m, entries in per_mech.items():
            cfg = entries[i].cfg
            assert cfg.model.mechanism == m and cfg.tag == f"search.t{i}" and cfg.seed == i
            assert is_search_run(cfg) and not cfg.eval_test
            assert cfg.optim.epochs == 30 and cfg.data.train_split == "train" and cfg.data.val_split == "val"
            d = to_dict(cfg)
            d["model"].pop("mechanism")
            dicts.append(d)
        assert dicts[0] == dicts[1] == dicts[2]
    # Different trials differ, and every queue line differs from the base recipe only where expected.
    full = [e.cfg for e in per_mech["full"]]
    assert len({(c.optim.lr, c.optim.weight_decay) for c in full}) == hps.N_TRIALS
    base = to_dict(load_config("configs/base.yaml"))
    for c in full:
        d = to_dict(c)
        assert d["augment"] == base["augment"] and d["data"] == base["data"]
        assert d["optim"]["batch_size"] == 128 and d["optim"]["beta1"] == base["optim"]["beta1"]
        assert {k: v for k, v in d["model"].items() if k not in ("drop_path",)} == \
               {k: v for k, v in base["model"].items() if k not in ("drop_path",)}


def test_values_fall_in_the_space(hps):
    trials = hps.sample_trials(n=4000, seed=3)
    lr = [t["lr"] for t in trials]
    wd = [t["weight_decay"] for t in trials]
    assert 3e-4 <= min(lr) and max(lr) <= 3e-3
    assert 0.01 <= min(wd) and max(wd) <= 0.1
    assert {t["drop_path"] for t in trials} == {0.0, 0.05, 0.1}
    assert {t["warmup_epochs"] for t in trials} == {5, 10}
    # Log-uniform: log values are uniform, so their mean is the log-midpoint and each decade-half
    # of the range holds about half the draws.
    for xs, (lo, hi) in ((lr, (3e-4, 3e-3)), (wd, (0.01, 0.1))):
        u = [(math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo)) for x in xs]
        assert abs(sum(u) / len(u) - 0.5) < 0.02
        assert abs(sum(v < 0.5 for v in u) / len(u) - 0.5) < 0.03
    # Uniform choices: each option within a few points of its share.
    for key, options in (("drop_path", (0.0, 0.05, 0.1)), ("warmup_epochs", (5, 10))):
        for o in options:
            assert abs(sum(t[key] == o for t in trials) / len(trials) - 1 / len(options)) < 0.03
    # The committed eight trials too.
    for t in hps.read_trials():
        assert 3e-4 <= t["lr"] <= 3e-3 and 0.01 <= t["weight_decay"] <= 0.1
        assert t["drop_path"] in (0.0, 0.05, 0.1) and t["warmup_epochs"] in (5, 10)


def test_trial_schedule_is_defined_over_30_epochs(hps):
    """Warmup and cosine run over the trial's 30 epochs: the lr reaches min_lr at the last step."""
    from attnvit.train import lr_at

    cfg = hps.trial_config(hps.read_trials()[0], "full")
    steps = 45000 // cfg.optim.batch_size
    total, warm = cfg.optim.epochs * steps, int(round(cfg.optim.warmup_epochs * steps))
    assert total == 30 * steps
    assert lr_at(warm - 1, total, warm, cfg.optim.lr, cfg.optim.min_lr) == pytest.approx(cfg.optim.lr)
    assert lr_at(total, total, warm, cfg.optim.lr, cfg.optim.min_lr) == pytest.approx(cfg.optim.min_lr)


# -- fixture run folders --------------------------------------------------------------------


def fake_run(hps, root, trial, mech, final_acc, best_acc=None, *, epochs=30, test_acc=None, done=True):
    cfg = hps.trial_config(trial, mech)
    d = root / run_id(cfg)
    d.mkdir(parents=True)
    dump_yaml(cfg, d / "config.yaml")
    best_acc = final_acc if best_acc is None else best_acc
    for e in range(1, epochs + 1):
        acc = final_acc if e == epochs else best_acc if e == epochs - 1 else 0.1
        append_jsonl(d / "metrics.jsonl", {"epoch": e, "val_acc": acc})
    if done:
        write_json(d / "summary.json", {
            "epochs": epochs, "best_val_acc": max(best_acc, final_acc), "best_epoch": epochs - 1,
            "final_val_acc": final_acc, "test_acc": test_acc, "test_loss": None, "train_time_s": 100.0,
            "compute_time_s": 80.0, "wall_time_s": 120.0, "gpu": "fake"})
    return d


# Val accuracy at epoch 30 per (mechanism, trial). full: trial 5 wins outright. windows: trials 2
# and 6 tie at epoch 30, trial 6 has the higher best val accuracy. axial: trials 1 and 4 tie on
# both, so the lower index wins.
ACC = {
    "full": {i: 0.70 + 0.01 * i * (i != 7) for i in range(8)},
    "windows": {i: 0.6 for i in range(8)} | {2: 0.8, 6: 0.8},
    "axial": {i: 0.5 for i in range(8)} | {1: 0.75, 4: 0.75},
}
BEST = {("windows", 6): 0.82, ("windows", 2): 0.81}


def populate(hps, root):
    for m in hps.MECHANISMS:
        for t in hps.read_trials():
            fake_run(hps, root, t, m, ACC[m][t["trial"]], BEST.get((m, t["trial"])))


def test_collect_and_select_on_fixture_runs(hps, tmp_path, capsys):
    root, out, final = tmp_path / "runs", tmp_path / "hp.csv", tmp_path / "final"
    populate(hps, root)
    assert hps.main(["collect", "--runs-root", str(root), "--out", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 24 and all(r["status"] == "done" for r in rows)
    assert list(rows[0]) == hps.RESULT_COLUMNS
    trials = {t["trial"]: t for t in hps.read_trials()}
    for r in rows:
        t = trials[int(r["trial"])]
        assert float(r["lr"]) == t["lr"] and float(r["weight_decay"]) == t["weight_decay"]
        assert float(r["val_acc_epoch30"]) == pytest.approx(ACC[r["mechanism"]][int(r["trial"])])
    assert hps.main(["select", "--results", str(out), "--out-dir", str(final)]) == 0
    picks = {m: yaml.safe_load(open(final / f"{m}.yaml")) for m in hps.MECHANISMS}
    for m, trial in (("full", 6), ("windows", 6), ("axial", 1)):
        d, t = picks[m], trials[trial]
        assert d["model"]["mechanism"] == m and d["optim"]["epochs"] == 100
        assert d["optim"]["lr"] == t["lr"] and d["optim"]["weight_decay"] == t["weight_decay"]
        assert d["model"]["drop_path"] == t["drop_path"] and d["optim"]["warmup_epochs"] == t["warmup_epochs"]
        assert d["eval_test"] is True and d["tag"] == "final"
        text = (final / f"{m}.yaml").read_text()
        assert f"# Trial {trial} of 8" in text
        cfg = load_config(final / f"{m}.yaml")  # loadable as is, and not a search run
        assert not is_search_run(cfg) and cfg.optim.epochs == 100
    assert "full     trial 6" in capsys.readouterr().out


def test_collect_reports_missing_runs_and_select_refuses(hps, tmp_path):
    root, out = tmp_path / "runs", tmp_path / "hp.csv"
    trials = hps.read_trials()
    fake_run(hps, root, trials[0], "full", 0.7)
    fake_run(hps, root, trials[1], "full", 0.7, epochs=12, done=False)
    assert hps.main(["collect", "--runs-root", str(root), "--out", str(out)]) == 1
    status = {(r["mechanism"], r["trial"]): r["status"] for r in csv.DictReader(open(out))}
    assert status[("full", "0")] == "done" and status[("full", "1")] == "partial"
    assert status[("axial", "7")] == "missing"
    assert hps.main(["select", "--results", str(out), "--out-dir", str(tmp_path / "f")]) == 2
    assert not (tmp_path / "f").exists()


def test_collect_rejects_runs_that_saw_the_test_split_or_another_config(hps, tmp_path):
    trials = hps.read_trials()
    root = tmp_path / "a"
    fake_run(hps, root, trials[0], "full", 0.7, test_acc=0.69)
    with pytest.raises(ValueError, match="evaluated the test split"):
        hps.collect(trials, root)
    root = tmp_path / "b"
    d = fake_run(hps, root, trials[0], "full", 0.7)
    saved = yaml.safe_load(open(d / "config.yaml"))
    saved["optim"]["lr"] = 0.5
    yaml.safe_dump(saved, open(d / "config.yaml", "w"))
    with pytest.raises(ValueError, match="optim.lr"):
        hps.collect(trials, root)
