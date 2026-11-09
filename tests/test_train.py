"""Training loop on CPU: overfit, determinism, resume, accumulation, interruption."""

import os
import signal
import socket
import time

import pytest
import torch

from attnvit.config import ConfigError
from attnvit.data.augment import MixupCutmix, Normalize
from attnvit.data import data_root
from attnvit.data.synthetic import SyntheticImages
from attnvit.train import Trainer, TimedIterator, build_model, load_datasets, smoke_check, train_step
from attnvit.utils import peak_memory_mib, peak_reserved_memory_mib, read_json, read_jsonl

from conftest import NO_AUGMENT


def quiet(*_):
    pass


def train(cfg, root, **kw):
    stop_after = kw.pop("stop_after_epochs", None)
    trainer = Trainer(cfg, runs_root_dir=root, log=quiet, **kw)
    return trainer, trainer.run(stop_after_epochs=stop_after)


def final_state(trainer):
    return torch.load(trainer.ckpt_path, weights_only=False)["model"]


def assert_same_weights(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_overfits_256_images(tiny_cfg, tmp_path):
    # Memorise 256 random images with random labels: >= 95 % accuracy (eval mode, no augmentation).
    cfg = tiny_cfg(*NO_AUGMENT, "optim.epochs=20", "optim.lr=2e-3", "optim.weight_decay=0",
                   "model.drop_path=0", "tag=overfit")
    trainer, status = train(cfg, tmp_path)
    assert status == "done"
    s = read_json(trainer.summary_path)
    assert s["final_val_acc"] >= 0.95, s  # synthetic val = the 256 training images
    assert s["steps"] == 20 * 8 and s["epochs"] == 20
    assert s["head"] == "cls" and s["peak_mem_mib"] is None and s["test_acc"] is None
    assert s["peak_mem_reserved_mib"] is None  # no CUDA
    assert s["train_time_s"] > 0 and s["time_per_epoch_s"] == pytest.approx(s["train_time_s"] / 20)
    assert len(read_jsonl(trainer.metrics_path)) == 20
    assert (trainer.run_dir / "config.yaml").exists()


def test_same_seed_same_losses(tiny_cfg, tmp_path):
    cfg = tiny_cfg("optim.epochs=2")  # full augmentation recipe on
    a, _ = train(cfg, tmp_path / "a")
    b, _ = train(cfg, tmp_path / "b")
    la = [r["train_loss"] for r in read_jsonl(a.metrics_path)]
    lb = [r["train_loss"] for r in read_jsonl(b.metrics_path)]
    assert la == lb
    assert_same_weights(final_state(a), final_state(b))
    c, _ = train(tiny_cfg("optim.epochs=2", "seed=1"), tmp_path / "c")
    assert [r["train_loss"] for r in read_jsonl(c.metrics_path)] != la


def test_resume_matches_uninterrupted_run(tiny_cfg, tmp_path):
    cfg = tiny_cfg("optim.epochs=3", "optim.accum_steps=2")
    straight, _ = train(cfg, tmp_path / "straight")
    part, status = train(cfg, tmp_path / "chunks", stop_after_epochs=1)
    assert status == "paused" and not part.summary_path.exists()
    assert len(read_jsonl(part.metrics_path)) == 1
    resumed, status = train(cfg, tmp_path / "chunks")
    assert status == "done"
    assert_same_weights(final_state(straight), final_state(resumed))
    rows_a, rows_b = read_jsonl(straight.metrics_path), read_jsonl(resumed.metrics_path)
    assert [r["train_loss"] for r in rows_a] == [r["train_loss"] for r in rows_b]
    assert [r["epoch"] for r in rows_b] == [1, 2, 3]
    assert read_json(resumed.summary_path)["chunks"] == 2
    # A finished run is not trained again.
    assert train(cfg, tmp_path / "chunks")[1] == "done"


def test_resume_drops_metrics_past_the_checkpoint(tiny_cfg, tmp_path):
    cfg = tiny_cfg("optim.epochs=2")
    t, _ = train(cfg, tmp_path, stop_after_epochs=1)
    with open(t.metrics_path, "a") as f:  # an epoch logged but never checkpointed
        f.write('{"epoch": 2, "train_loss": 0}\n')
    t, _ = train(cfg, tmp_path)
    assert [r["epoch"] for r in read_jsonl(t.metrics_path)] == [1, 2]


def test_resume_refuses_a_different_config(tiny_cfg, tmp_path):
    train(tiny_cfg("optim.epochs=2"), tmp_path, stop_after_epochs=1)
    with pytest.raises(ConfigError, match="optim.lr"):
        train(tiny_cfg("optim.epochs=2", "optim.lr=0.1"), tmp_path)
    # Non-semantic keys may change between chunks.
    assert train(tiny_cfg("optim.epochs=2", "eval_batch_size=64"), tmp_path)[1] == "done"


def _send_usr1_at(epoch, step):
    def hook(trainer, e, s):
        if (e, s) == (epoch, step):
            os.kill(os.getpid(), signal.SIGUSR1)

    return hook


def test_sigusr1_drops_partial_epoch_then_resumes(tiny_cfg, tmp_path):
    cfg = tiny_cfg("optim.epochs=3")
    straight, _ = train(cfg, tmp_path / "straight")
    t, status = train(cfg, tmp_path / "x", step_hook=_send_usr1_at(1, 3), stop_grace_s=0)
    assert status == "interrupted"
    assert torch.load(t.ckpt_path, weights_only=False)["state"]["epoch"] == 1
    assert len(read_jsonl(t.metrics_path)) == 1
    assert signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL  # handler restored
    resumed, status = train(cfg, tmp_path / "x")
    assert status == "done"
    assert_same_weights(final_state(straight), final_state(resumed))


def test_sigusr1_finishes_a_short_epoch_first(tiny_cfg, tmp_path):
    cfg = tiny_cfg("optim.epochs=3")
    t, status = train(cfg, tmp_path, step_hook=_send_usr1_at(1, 3), stop_grace_s=3600)
    assert status == "interrupted"
    assert torch.load(t.ckpt_path, weights_only=False)["state"]["epoch"] == 2
    assert train(cfg, tmp_path)[1] == "done"


@pytest.mark.parametrize("mix", [
    pytest.param({"prob": 0.0}, id="no-mix"),
    pytest.param({"prob": 1.0, "cutmix_alpha": 0.0}, id="mixup"),
    pytest.param({"prob": 1.0, "mixup_alpha": 0.0}, id="cutmix"),
    pytest.param({"prob": 1.0}, id="mixup-or-cutmix"),
])
def test_gradient_accumulation_matches_full_batch(tiny_cfg, mix):
    # Mixing is applied once per optimizer step, so accumulating micro-batches must give the
    # gradient of the full batch even with mixup/cutmix on (same mixer seed for every split).
    cfg = tiny_cfg(*NO_AUGMENT, "model.drop_path=0")
    torch.manual_seed(0)
    x = torch.randint(0, 256, (32, 3, 32, 32), dtype=torch.uint8)
    y = torch.randint(0, 10, (32,))
    norm = Normalize([0.5] * 3, [0.25] * 3)
    grads = []
    for chunks in (1, 2, 4):
        mixer = MixupCutmix(10, label_smoothing=0.1, seed=7, **mix)
        torch.manual_seed(0)
        model = build_model(cfg, 10)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)  # keep the weights, inspect the grads
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        batches = list(zip(x.chunk(chunks), y.chunk(chunks)))
        loss_sum, correct, n = train_step(model, opt, scaler, batches, normalize=norm, mixer=mixer, use_amp=False)
        assert n == 32
        grads.append([p.grad.clone() for p in model.parameters()])
    for other in grads[1:]:
        for a, b in zip(grads[0], other):
            assert torch.allclose(a, b, atol=1e-6)


def test_smoke_check(tiny_cfg):
    loss = smoke_check(tiny_cfg())
    assert 0 < loss < 10


@pytest.mark.skipif(not (data_root() / "cifar10" / "cifar-10-batches-py").is_dir(),
                    reason="CIFAR-10 not under $SYS819_DATA")
def test_real_cifar10_wiring(tiny_cfg):
    # The training loop reads CIFAR-10 through get_dataset(name, res, split) with the recipe applied.
    cfg = tiny_cfg("data.name=cifar10")
    data = load_datasets(cfg)
    assert (len(data.train), len(data.val), data.num_classes) == (45000, 5000, 10)
    x, y = data.train[0]
    assert x.dtype == torch.uint8 and x.shape == (3, 32, 32)
    assert data.train.transform is not None and data.val.transform is None
    assert 0 < smoke_check(cfg) < 10


def test_timed_iterator_counts_only_the_wait():
    def slow():
        for i in range(3):
            time.sleep(0.02)
            yield i

    it = TimedIterator(slow())
    for _ in it:
        time.sleep(0.03)  # "compute" between batches is not waiting
    assert 0.06 <= it.wait_s < 0.09


def test_data_wait_and_compute_time_are_logged(tiny_cfg, tmp_path, monkeypatch):
    # A slow dataset: every training image costs 2 ms to load, so most of the epoch is data wait.
    original = SyntheticImages.__getitem__

    def slow_getitem(self, i):
        if self.transform is not None:  # the training split only
            time.sleep(0.002)
        return original(self, i)

    monkeypatch.setattr(SyntheticImages, "__getitem__", slow_getitem)
    trainer, status = train(tiny_cfg("optim.epochs=2"), tmp_path)
    assert status == "done"
    rows = read_jsonl(trainer.metrics_path)
    for r in rows:
        assert r["data_wait_s"] + r["compute_time_s"] == pytest.approx(r["epoch_time_s"])
        assert r["data_wait_s"] >= 256 * 0.002 and r["compute_time_s"] > 0
    s = read_json(trainer.summary_path)
    assert s["data_wait_s"] == pytest.approx(sum(r["data_wait_s"] for r in rows))
    assert s["compute_time_s"] == pytest.approx(sum(r["compute_time_s"] for r in rows))
    assert s["data_wait_s"] + s["compute_time_s"] == pytest.approx(s["train_time_s"])
    assert s["compute_time_per_epoch_s"] == pytest.approx(s["compute_time_s"] / 2)


def test_summary_records_the_cpu_side(tiny_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    trainer, _ = train(tiny_cfg("optim.epochs=1"), tmp_path)
    s = read_json(trainer.summary_path)
    assert s["num_workers"] == 0
    assert s["usable_cpus"] == len(os.sched_getaffinity(0))
    assert (s["slurm_cpus_per_task"], s["slurm_job_id"]) == (8, "123456")
    assert s["hostname"] == socket.gethostname()
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    monkeypatch.delenv("SLURM_JOB_ID")
    trainer, _ = train(tiny_cfg("optim.epochs=1", "tag=noslurm"), tmp_path)
    s = read_json(trainer.summary_path)
    assert s["slurm_cpus_per_task"] is None and s["slurm_job_id"] is None


def test_peak_memory_reports_allocated_and_reserved(monkeypatch):
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 3 * 2**20)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 5 * 2**20)
    cuda = torch.device("cuda", 0)  # never touched: the torch.cuda queries are faked
    assert (peak_memory_mib(cuda), peak_reserved_memory_mib(cuda)) == (3.0, 5.0)
    cpu = torch.device("cpu")
    assert peak_memory_mib(cpu) is None and peak_reserved_memory_mib(cpu) is None
