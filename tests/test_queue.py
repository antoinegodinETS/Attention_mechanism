"""Run queue: smoke check, skip finished runs, resume partial ones; scripts/status.py (CPU)."""

import pytest

from attnvit.queue import main as queue_main
from attnvit.queue import parse_queue, run_queue
from attnvit.train import Trainer
from attnvit.utils import read_jsonl

from conftest import TINY


def write_queue(path, lines):
    path.write_text("# test queue\n\n" + "\n".join(lines) + "\n")
    return path


def tiny_line(*extra):
    return " ".join(["configs/base.yaml", *TINY, "optim.epochs=2", *extra])


def test_queue_runs_skips_and_resumes(tmp_path):
    root = tmp_path / "runs"
    q = write_queue(tmp_path / "q.txt", [tiny_line("tag=a"), tiny_line("tag=b  # trailing comment")])
    entries = parse_queue(q)
    assert [e.run_id for e in entries] == ["synthetic-32-p8-full-a-s0", "synthetic-32-p8-full-b-s0"]

    # Run b is half done before the queue starts: the queue must resume it, not restart it.
    Trainer(entries[1].cfg, runs_root_dir=root, log=lambda m: None).run(stop_after_epochs=1)
    log = []
    assert run_queue(q, root=root, log=log.append) == 0
    text = "\n".join(log)
    assert "smoke ok  synthetic-32-p8-full-a-s0" in text
    assert "resuming from" in text and "after epoch 1/2" in text
    assert [r["epoch"] for r in read_jsonl(root / entries[1].run_id / "metrics.jsonl")] == [1, 2]
    for e in entries:
        assert (root / e.run_id / "summary.json").exists()
    assert text.count("DONE ") == 2

    # Second pass: everything is skipped, no smoke check, no training.
    log.clear()
    assert run_queue(q, root=root, log=log.append) == 0
    text = "\n".join(log)
    assert text.count("skip      ") == 2 and "smoke ok" not in text and "epoch 1/2" not in text
    assert text.count("SKIP ") == 2 and "val=" in text


def test_bad_line_fails_before_any_training(tmp_path):
    q = write_queue(tmp_path / "q.txt", [tiny_line("tag=a"), tiny_line("tag=b", "optim.lrr=1")])
    with pytest.raises(ValueError, match=r"q.txt:4: unknown config key"):
        parse_queue(q)
    dup = write_queue(tmp_path / "dup.txt", [tiny_line("tag=a"), tiny_line("tag=a")])
    with pytest.raises(ValueError, match="duplicates line 3"):
        parse_queue(dup)


def test_smoke_failure_stops_the_queue(tmp_path, capsys):
    root = tmp_path / "runs"
    # The second run's batch (512) is larger than the synthetic split: its smoke check fails.
    q = write_queue(tmp_path / "q.txt", [tiny_line("tag=a"), tiny_line("tag=b", "optim.batch_size=512")])
    assert queue_main([str(q), "--runs-root", str(root)]) == 1
    assert "smoke check FAILED for synthetic-32-p8-full-b-s0" in capsys.readouterr().out
    assert not root.exists() or not any(root.iterdir())


def test_status_script(tmp_path, script, capsys):
    status = script("status")
    root = tmp_path / "runs"
    q = write_queue(tmp_path / "q.txt", [tiny_line("tag=a"), tiny_line("tag=b"), tiny_line("tag=c")])
    entries = parse_queue(q)
    Trainer(entries[0].cfg, runs_root_dir=root, log=lambda m: None).run()
    Trainer(entries[1].cfg, runs_root_dir=root, log=lambda m: None).run(stop_after_epochs=1)
    assert status.main([str(q), "--runs-root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "done      synthetic-32-p8-full-a-s0  val " in out
    assert "running?  synthetic-32-p8-full-b-s0  epoch 1/2" in out
    assert "pending   synthetic-32-p8-full-c-s0" in out
    assert status.main([str(q), "--runs-root", str(root), "--active-minutes", "-1"]) == 0
    assert "partial   synthetic-32-p8-full-b-s0" in capsys.readouterr().out


def test_signal_between_runs_stops_the_queue(tmp_path, monkeypatch):
    import os
    import signal

    import attnvit.queue as queue_mod

    root = tmp_path / "runs"
    q = write_queue(tmp_path / "q.txt", [tiny_line("tag=a"), tiny_line("tag=b")])
    real_smoke = queue_mod.smoke_check

    def smoke_then_signal(cfg):  # USR1 lands outside any training loop
        loss = real_smoke(cfg)
        os.kill(os.getpid(), signal.SIGUSR1)
        return loss

    monkeypatch.setattr(queue_mod, "smoke_check", smoke_then_signal)
    assert run_queue(q, root=root, log=lambda m: None) == 3
    assert not root.exists() or not any(root.iterdir())
    assert signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL


def test_phase3_smoke_queue_covers_the_three_mechanisms():
    from conftest import REPO_ROOT

    entries = parse_queue(REPO_ROOT / "configs" / "queue" / "phase3_smoke.txt")
    assert [e.run_id for e in entries] == [
        f"{data}-{res}-p4-{mech}-smoke-s0"
        for data, res in (("cifar10", 32), ("synthetic", 160))
        for mech in ("full", "windows", "axial")
    ]
    assert all(e.cfg.optim.epochs == 2 for e in entries)


def test_finished_run_is_freed_before_the_next_one(tmp_path, monkeypatch):
    # A Trainer refers to itself through its log closure, so only the cyclic garbage collector
    # frees its model. On a GPU a model still alive when the next run starts inflates that run's
    # peak memory (each phase 5 trial peaked about one model + gradients above the one before).
    import gc
    import weakref

    import attnvit.queue as queue_mod

    models, alive_at_start = [], []

    class RecordingTrainer(Trainer):
        def run(self, *args, **kwargs):
            alive_at_start.append(sum(r() is not None for r in models))
            status = super().run(*args, **kwargs)
            models.append(weakref.ref(self.model))
            return status

    monkeypatch.setattr(queue_mod, "Trainer", RecordingTrainer)
    q = write_queue(tmp_path / "q.txt", [tiny_line("tag=a"), tiny_line("tag=b")])
    gc.disable()  # the collector must not free the cycle by chance
    try:
        assert run_queue(q, root=tmp_path / "runs", smoke=False, log=lambda m: None) == 0
    finally:
        gc.enable()
    assert alive_at_start == [0, 0]
