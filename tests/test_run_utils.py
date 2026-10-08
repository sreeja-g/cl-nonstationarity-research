"""Run-directory utilities: fingerprint, lock, metrics.jsonl truncation, W&B mode, eval mask."""

import json
import sys

import pytest
from omegaconf import OmegaConf

from clns.utils.fingerprint import (
    FINGERPRINT_FILE,
    check_or_write_fingerprint,
    config_fingerprint,
)


def _cfg(**train):
    base = {
        "seed": 0,
        "device": "cpu",
        "deterministic": "strict",
        "run_dir": "/x",
        "paths": {"data_dir": "/d", "output_dir": "/o"},
        "logger": {"name": "none"},
        "benchmark": {"name": "synthetic", "image_size": 32},
        "train": {"lr": 0.01, "num_workers": 0, "eval_mb_size": 16,
                  "stop_after_experiences": None},
    }
    base["train"].update(train)
    return OmegaConf.create(base)


def test_fingerprint_ignores_volatile_keys():
    a, _ = config_fingerprint(_cfg())
    b, _ = config_fingerprint(_cfg(num_workers=8, eval_mb_size=512, stop_after_experiences=1))
    c = _cfg()
    c.paths.output_dir = "/elsewhere"
    c.device = "cuda"
    c.logger = {"name": "wandb", "mode": "offline"}
    c.deterministic = "warn"  # documented strict -> warn recovery must resume (exit 0, not 4)
    assert a == b == config_fingerprint(c)[0]
    assert a != config_fingerprint(_cfg(lr=0.1))[0]


def test_fingerprint_file_roundtrip(tmp_path):
    ok, diff = check_or_write_fingerprint(tmp_path, _cfg())
    assert ok and diff == [] and (tmp_path / FINGERPRINT_FILE).exists()
    assert check_or_write_fingerprint(tmp_path, _cfg(num_workers=3)) == (True, [])
    bad = _cfg(lr=0.5)
    bad.benchmark.image_size = 64
    ok, diff = check_or_write_fingerprint(tmp_path, bad)
    assert not ok and diff == ["benchmark.image_size", "train.lr"]
    stored = json.loads((tmp_path / FINGERPRINT_FILE).read_text())
    assert stored["config"]["train"]["lr"] == 0.01  # never overwritten


def test_reapply_eval_mb_size_after_resume():
    """A resume's train.eval_mb_size overrides the value unpickled from the checkpoint."""
    from types import SimpleNamespace

    from clns.utils.checkpoint import reapply_eval_mb_size

    restored = SimpleNamespace(eval_mb_size=256, train_mb_size=64)  # as unpickled
    assert reapply_eval_mb_size(restored, "32") == 256
    assert restored.eval_mb_size == 32 and restored.train_mb_size == 64


@pytest.mark.skipif(sys.platform.startswith("win"), reason="fcntl lock is a no-op on Windows")
def test_run_dir_lock(tmp_path):
    from clns.utils.lock import RunDirLock, RunLockError

    with RunDirLock(tmp_path):
        with pytest.raises(RunLockError):
            RunDirLock(tmp_path).acquire()
    lk = RunDirLock(tmp_path)  # released -> can be taken again
    lk.acquire()
    lk.release()


def test_truncate_jsonl(tmp_path):
    pytest.importorskip("avalanche")
    from clns.evaluation.plugins import truncate_jsonl

    p = tmp_path / "metrics.jsonl"
    recs = [
        {"name": "a", "phase": "eval", "n_trained": 0},
        {"name": "b", "phase": "train", "n_trained": 0},
        {"name": "c", "phase": "eval", "n_trained": 1},
        {"name": "d", "phase": "train", "n_trained": 1},
        {"name": "e", "phase": "eval", "n_trained": 2},
    ]
    text = "".join(json.dumps(r) + "\n" for r in recs) + '{"partial'
    p.write_text(text)
    truncate_jsonl(p, 1)
    assert [json.loads(x)["name"] for x in p.read_text().splitlines()] == ["a", "b", "c"]
    p.write_text(text)
    truncate_jsonl(p, 0)  # exp-0 crash: everything is redone -> drop all
    assert p.read_text() == ""


def test_resolve_wandb_mode(monkeypatch):
    pytest.importorskip("avalanche")
    from clns.evaluation.evaluator import resolve_wandb_mode

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert resolve_wandb_mode("auto")[0] == "offline"
    monkeypatch.setenv("WANDB_API_KEY", "dummy-test-value")
    assert resolve_wandb_mode("auto")[0] == "online"
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert resolve_wandb_mode("auto")[0] == "disabled"
    assert resolve_wandb_mode("offline")[0] == "offline"
    with pytest.raises(ValueError):
        resolve_wandb_mode("sometimes")


def test_wandb_init_failure_falls_back_offline(tmp_path, monkeypatch):
    pytest.importorskip("avalanche")
    import clns.evaluation.evaluator as ev

    calls = []

    def fake_make(cfg, run_dir, mode):
        calls.append(mode)
        if mode == "online":
            raise TimeoutError("simulated outage")
        return "WB-LOGGER"

    monkeypatch.setattr(ev, "_make_wandb_logger", fake_make)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_API_KEY", "dummy-test-value")
    cfg = OmegaConf.create({"logger": {"name": "wandb", "project": "p", "mode": "auto"}})
    status: dict = {}
    loggers = ev.build_loggers(cfg, tmp_path, status=status)
    try:
        assert calls == ["online", "offline"]
        assert loggers[-1] == "WB-LOGGER"
        assert status["wandb_fallback_offline"] is True and status["wandb_mode"] == "offline"
        assert "simulated outage" in status["wandb_error"]
    finally:
        ev_loggers = [lg for lg in loggers if lg != "WB-LOGGER"]
        for lg in ev_loggers:
            f = getattr(lg, "file", None)
            if f is not None and f not in (sys.stdout, sys.stderr):
                f.close()


class _Clock:
    def __init__(self, n):
        self.train_exp_counter = n


class _Strategy:
    def __init__(self, n, out):
        self.clock = _Clock(n)
        self.mb_output = out


def test_eval_class_mask_plugin():
    torch = pytest.importorskip("torch")
    pytest.importorskip("avalanche")
    from clns.evaluation.plugins import EvalClassMaskPlugin

    plug = EvalClassMaskPlugin([[3, 1], [0, 2]], num_classes=5)
    logits = torch.zeros(2, 5)
    logits[:, 4] = 10.0  # an unseen class wins without the mask

    s = _Strategy(0, logits.clone())  # random-init row: no mask
    plug.before_eval(s)
    plug.after_eval_forward(s)
    assert torch.equal(s.mb_output, logits)

    s = _Strategy(1, logits.clone())  # after exp 0: classes {1, 3} visible
    plug.before_eval(s)
    plug.after_eval_forward(s)
    visible = torch.isfinite(s.mb_output[0]).nonzero().flatten().tolist()
    assert visible == [1, 3]
    assert s.mb_output.argmax(dim=1).tolist() == [1, 1]

    s = _Strategy(1, logits.clone())  # nested eval during training of exp 1: {0,1,2,3}
    plug.before_training(s)
    plug.before_eval(s)
    plug.after_eval_forward(s)
    plug.after_training(s)
    assert torch.isfinite(s.mb_output[0]).nonzero().flatten().tolist() == [0, 1, 2, 3]
