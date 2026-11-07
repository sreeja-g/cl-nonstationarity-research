"""clns.tune: search space, trial command, resumable seeded study, logs, --write-best."""

import csv
import json
import subprocess
from pathlib import Path

import pytest
import yaml

pytest.importorskip("optuna")
pytest.importorskip("torch")
pytest.importorskip("avalanche")
pytest.importorskip("hydra")

import optuna  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from optuna.trial import TrialState  # noqa: E402

from clns import tune  # noqa: E402
from clns.utils.fingerprint import comparable_config  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
LR_DIST = optuna.distributions.FloatDistribution(0.001, 0.1, log=True)


def _extra(tmp_path: Path) -> list[str]:
    return ["benchmark.train_per_class=8", "benchmark.test_per_class=4", "model.cifar_stem=true",
            "train.mb_size=8", "train.eval_mb_size=16", "train.num_workers=0", "device=cpu",
            "deterministic=warn", f"paths.data_dir={(tmp_path / 'data').as_posix()}"]


def _base(out: Path, method: str = "naive", arm: str = "r18_scratch", epochs: int = 1) -> Path:
    return tune.log_base(tune.tune_root(out), "synthetic", arm, method, epochs)


def _jsonl(base: Path) -> list[dict]:
    path = base.parent / f"{base.name}.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()]


def _fake_trial(values=None, behaviours=None):
    """A run_trial replacement: ``behaviours`` = list of exceptions / None per call."""
    calls = []

    def fake(benchmark, arm, method, epochs, params, root, extra, attrs=None):
        attrs = attrs if attrs is not None else {}
        calls.append(dict(params))
        attrs.update(run_dir=f"rd{len(calls)}", exit_code=0, wall_s=1.0)
        b = behaviours[len(calls) - 1] if behaviours and len(calls) <= len(behaviours) else None
        if b is not None:
            if isinstance(b, tune.TrialFailed):
                attrs["hp_failure"] = True
            raise b
        return 0.5 if values is None else values[(len(calls) - 1) % len(values)]

    fake.calls = calls
    return fake


def test_search_space_starts():
    assert tune.search_space("naive", "r18_scratch") == {"train.lr": 0.01}
    assert tune.search_space("er", "r18_places") == {"train.lr": 0.001}
    assert tune.search_space("ewc", "r50_scratch") == {
        "train.lr": 0.01, "method.ewc_lambda": 1000.0}
    assert tune.search_space("lwf", "r18_places") == {"train.lr": 0.001, "method.alpha": 1.0}
    assert tune.search_space("derpp", "r18_scratch") == {
        "train.lr": 0.01, "method.alpha": 0.1, "method.beta": 0.5}
    assert tune.bounds(0.1) == (pytest.approx(0.01), pytest.approx(1.0))
    assert tune.default_trials("clear10") == tune.default_trials("domainnet") == 10
    assert tune.default_trials("imagenet100") == tune.default_trials("core50_nic") == 20
    with pytest.raises(ValueError):
        tune.search_space("nope", "r18_scratch")


def test_search_space_matches_method_configs():
    """Tuned keys exist in configs/method; fixed keys are never in the space."""
    for m in tune.METHODS:
        cfg = yaml.safe_load((CONFIG_DIR / "method" / f"{m}.yaml").read_text())
        for key in tune.search_space(m, "r18_scratch"):
            if key.startswith("method."):
                assert key.split(".", 1)[1] in cfg
    fixed = {"method.mem_size", "method.batch_size_mem", "method.temperature", "method.storage"}
    for m in tune.METHODS:
        assert not fixed & set(tune.search_space(m, "r18_scratch"))


def test_check_extra_allowlist():
    tune.check_extra(["train.num_workers=0", "train.eval_mb_size=16", "device=cpu",
                      "deterministic=warn", "paths.data_dir=/x", "++paths.weights_dir=/w"])
    tune.check_extra(["train.mb_size=8", "benchmark.train_per_class=8", "model.cifar_stem=true"],
                     "synthetic")
    for bad in ("train.lr=0.1", "method.mem_size=10", "method.ewc_lambda=3", "seed=1",
                "train.epochs=3", "+method.alpha=1", "run_dir=x", "oops", "train.momentum=0.5",
                "benchmark.max_experiences=5", "benchmark.val_fraction=0.5", "logger=wandb"):
        with pytest.raises(ValueError):
            tune.check_extra([bad], "synthetic")
    for syn_only in ("train.mb_size=8", "benchmark.train_per_class=8"):
        with pytest.raises(ValueError, match="synthetic"):
            tune.check_extra([syn_only], "clear10")


def test_output_path_validation_and_quoting():
    from hydra.core.override_parser.overrides_parser import OverridesParser

    for bad in ("/a/b'c", "/a/$HOME", "/a\\b", "/a\nb", '/a/"b'):
        with pytest.raises(ValueError, match="output path"):
            tune.check_output_path(bad)
    for good in ("/work space/a,b=c/x", "Z:/Research CL-x/out", "/a/b[1]/c"):
        tune.check_output_path(good)
        (o,) = OverridesParser.create().parse_overrides([tune._hydra_assign("run_dir", good)])
        assert o.value() == good  # one string, whatever commas / '=' / spaces it holds
    with pytest.raises(ValueError):
        tune._hydra_assign("run_dir", "/a/b'c")
    with pytest.raises(ValueError, match="output path"):
        tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=1, epochs=1,
                         output_dir="/tmp/it's")


def _compose(cmd: list[str], hydra_cfg: bool = False):
    args = cmd[cmd.index("clns.train") + 1:]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="train", overrides=args, return_hydra_config=hydra_cfg)


def test_trial_command_run_dir_redirect_and_volatility(tmp_path):
    rd = tmp_path / "x" / "p0123" / "seed0"
    cmd = tune.build_trial_command("clear10", "r18_scratch", "ewc", 2,
                                   {"train.lr": 0.01, "method.ewc_lambda": 1000.0}, rd, [])
    assert cmd[1:4] == ["-m", "clns.train", "experiment=tune"]
    for tok in ("benchmark=clear10", "model=r18_scratch", "method=ewc", "seed=0",
                "train.epochs=2", "logger=none", "train.lr=0.01", "method.ewc_lambda=1000.0",
                f"run_dir='{rd.as_posix()}'"):
        assert tok in cmd
    cfg = _compose(cmd)
    assert Path(str(cfg.run_dir)) == rd
    assert Path(str(_compose(cmd, hydra_cfg=True).hydra.run.dir)) == rd
    assert cfg.hparams.mode == "off" and cfg.train.lr == 0.01
    assert cfg.method.ewc_lambda == 1000.0 and cfg.benchmark.max_experiences == 3
    # run_dir is volatile in the fingerprint, so a moved tuning root does not exit 4
    from omegaconf import OmegaConf

    assert "run_dir" not in comparable_config(OmegaConf.to_container(cfg, resolve=True))


def test_run_dir_with_commas_and_equals_composes(tmp_path):
    rd = tmp_path / "a,b=c d" / "seed0"
    cmd = tune.build_trial_command("clear10", "r18_scratch", "naive", 1, {"train.lr": 0.01}, rd, [])
    assert Path(str(_compose(cmd).run_dir)) == rd


def test_core50_trial_command_uses_whole_session_validation(tmp_path):
    cmd = tune.build_trial_command("core50_nic", "r18_places", "derpp", 1,
                                   {"train.lr": 0.001, "method.alpha": 0.1, "method.beta": 0.5},
                                   tmp_path / "seed0", [])
    assert "benchmark.val_fraction=0.0" in cmd and "benchmark.val_sessions=[1]" in cmd
    cfg = _compose(cmd)
    assert cfg.benchmark.name == "core50_nic" and cfg.benchmark.val_fraction == 0.0
    assert list(cfg.benchmark.val_sessions) == [1] and cfg.benchmark.max_experiences == 3
    assert cfg.model.name == "r18_places" and cfg.method.name == "derpp"
    # without the amendment tune.yaml's val_fraction=0.2 would be raised by CORe50
    other = tune.build_trial_command("clear10", "r18_scratch", "naive", 1, {"train.lr": 0.01},
                                     tmp_path / "s", [])
    assert not any(t.startswith("benchmark.val_") for t in other)
    assert _compose(other).benchmark.val_fraction == 0.2


def test_study_attrs_record_core50_overrides(tmp_path):
    study = tune.open_study("core50_nic", "r18_scratch", "naive", 1, 0, [],
                            tune.tune_root(tmp_path))
    assert study.user_attrs["extra_nonvolatile"] == list(tune.CORE50_OVERRIDES)
    assert study.user_attrs["extra"] == ["benchmark.val_fraction=0.0", "benchmark.val_sessions=[1]"]


def test_score_valid_matrix(tmp_path):
    (tmp_path / "valid_matrix.json").write_text(json.dumps(
        {"rows": [[0.5, None, None], [0.4, 0.6, None], [0.3, 0.6, 0.9]]}))
    assert tune.score_valid_matrix(tmp_path, 3) == pytest.approx(0.6)
    assert tune.score_valid_matrix(tmp_path, 2) == pytest.approx(0.5)
    with pytest.raises(tune.StudyAborted):
        tune.score_valid_matrix(tmp_path, 4)
    with pytest.raises(tune.StudyAborted):
        tune.score_valid_matrix(tmp_path / "missing", 3)


def test_two_trial_study_resume_log_and_write_best(tmp_path):
    out, extra = tmp_path / "out", _extra(tmp_path)
    kw = {"epochs": 1, "sampler_seed": 0, "extra": extra, "output_dir": out}
    study = tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=2, **kw)
    trials = study.trials
    assert [t.state for t in trials] == [TrialState.COMPLETE] * 2
    assert trials[0].params["train.lr"] == pytest.approx(0.01)  # the paper value is trial 0
    for t in trials:
        assert 0.001 * 0.999 <= t.params["train.lr"] <= 0.1 * 1.001
        assert 0.0 <= t.value <= 1.0 and Path(t.user_attrs["run_dir"], "DONE").exists()
        assert t.user_attrs["exit_code"] == 0
    assert study.user_attrs["n_startup_trials"] == 3  # max(3, 2 // 4)

    # resume 2 -> 3 trials runs exactly one more; asking for 2 again runs nothing
    assert len(tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=2, **kw).trials) == 2
    study = tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=3, **kw)
    assert len(study.trials) == 3 and trials[0].params == study.trials[0].params

    base = _base(out)
    assert tune.db_path(tune.tune_root(out), "synthetic", "r18_scratch", "naive").exists()
    lines = _jsonl(base)
    assert [r["number"] for r in lines] == [0, 1, 2]
    assert all(r["state"] == "COMPLETE" and r["study"] == study.study_name for r in lines)
    rows = list(csv.DictReader((base.parent / f"{base.name}.csv").open(encoding="utf-8")))
    assert len(rows) == 3 and float(rows[0]["train.lr"]) == pytest.approx(0.01)
    assert {"study", "number", "state", "value", "exit_code", "wall_s", "run_dir"} <= set(rows[0])

    # a different seed against the same db is an error; volatile extras may change
    with pytest.raises(ValueError, match="other settings"):
        tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=3,
                         **{**kw, "sampler_seed": 1})
    changed = [("device=cuda" if e == "device=cpu" else e) for e in extra]
    assert len(tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=3,
                                **{**kw, "extra": changed}).trials) == 3
    with pytest.raises(ValueError, match="other settings"):
        tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=3,
                         **{**kw, "extra": [*extra, "benchmark.train_per_class=9"]})

    hp = tmp_path / "hp"
    wb = {"n_trials": 3, "epochs": 1, "sampler_seed": 0, "extra": extra, "hparams_dir": hp,
          "root": tune.tune_root(out)}
    with pytest.raises(ValueError, match="only 3/4"):
        tune.write_best(study, "synthetic", "r18_scratch", "naive", **{**wb, "n_trials": 4})
    path = tune.write_best(study, "synthetic", "r18_scratch", "naive", **wb)
    data = yaml.safe_load(path.read_text())
    assert path == hp / "synthetic" / "r18_scratch.yaml" and data["tuned_transitions"] == [0, 1]
    assert data["tuning_seed"] == 0 and "task_order_sha256" in data
    best = max(study.trials, key=lambda t: t.value)
    ent = data["methods"]["naive"]
    assert ent["values"] == {"train.lr": best.params["train.lr"]}
    prov = ent["provenance"]
    assert prov["n_trials"] == 3 and prov["n_trials_total"] == 3 and prov["n_complete"] == 3
    assert prov["best_value"] == best.value and prov["n_startup_trials"] == 3
    assert prov["sampler_seed"] == 0 and prov["tuning_epochs"] == 1 and prov["tuning_seed"] == 0
    assert prov["start_values"] == {"train.lr": 0.01} and prov["tuned_experiences"] == [0, 1, 2]
    assert prov["fixed_config_sha256"] == study.user_attrs["fixed_config_sha256"]
    assert not list(path.parent.glob("*.tmp"))

    # merging another method keeps the first one; an incomplete study needs force
    study_ewc = tune.open_study("synthetic", "r18_scratch", "ewc", 1, 0, extra,
                                tune.tune_root(out))
    lam = optuna.distributions.FloatDistribution(100.0, 10000.0, log=True)
    study_ewc.add_trial(optuna.trial.create_trial(
        params={"train.lr": 0.02, "method.ewc_lambda": 500.0}, value=0.3,
        distributions={"train.lr": LR_DIST, "method.ewc_lambda": lam}))
    with pytest.raises(ValueError, match="--force"):
        tune.write_best(study_ewc, "synthetic", "r18_scratch", "ewc", **wb)
    tune.write_best(study_ewc, "synthetic", "r18_scratch", "ewc", force=True, **wb)
    merged = yaml.safe_load(path.read_text())
    assert set(merged["methods"]) == {"naive", "ewc"}
    assert merged["methods"]["ewc"]["values"] == {"train.lr": 0.02, "method.ewc_lambda": 500.0}


def test_fixed_config_hash_is_compared(tmp_path, monkeypatch):
    root = tune.tune_root(tmp_path)
    s1 = tune.open_study("synthetic", "r18_scratch", "naive", 1, 0, [], root)
    assert len(s1.user_attrs["fixed_config_sha256"]) == 64
    assert tune.fixed_config_sha256("naive") != tune.fixed_config_sha256("ewc")
    monkeypatch.setattr(tune, "fixed_config_sha256", lambda m: "0" * 64)
    with pytest.raises(ValueError, match="fixed_config_sha256"):
        tune.open_study("synthetic", "r18_scratch", "naive", 1, 0, [], root)


def _proc(rc, stderr=""):
    return subprocess.CompletedProcess([], rc, "", stderr)


def _run_one(tmp_path, monkeypatch, rc, failure=None, matrix=None):
    root = tune.tune_root(tmp_path)
    rd = tune.trial_run_dir(root, "synthetic", "r18_scratch", "naive", 1, {"train.lr": 0.01})
    rd.mkdir(parents=True)
    if failure is not None:
        (rd / "failure.json").write_text(json.dumps(failure))
    if matrix is not None:
        (rd / "valid_matrix.json").write_text(json.dumps({"rows": matrix}))
    monkeypatch.setattr(tune.subprocess, "run", lambda *a, **k: _proc(rc, "boom"))
    attrs: dict = {}
    try:
        return tune.run_trial("synthetic", "r18_scratch", "naive", 1, {"train.lr": 0.01}, root,
                              [], attrs)
    finally:
        _run_one.attrs = attrs


def test_run_trial_failure_classification(tmp_path, monkeypatch):
    with pytest.raises(tune.TrialFailed):  # training crashed (counted)
        _run_one(tmp_path / "a", monkeypatch, 1, {"type": "RuntimeError", "message": "loss nan"})
    assert _run_one.attrs["hp_failure"] and _run_one.attrs["exit_code"] == 1
    for name, rc, fail in (("b", 1, {"type": "FileNotFoundError", "message": "x"}),
                           ("c", 1, {"type": "RuntimeError", "message": "CLEAR10 not found"}),
                           ("d", 1, None), ("e", 2, None), ("f", 3, {"type": "OOM", "message": ""}),
                           ("g", 4, None), ("h", 5, None)):
        with pytest.raises(tune.StudyAborted):
            _run_one(tmp_path / name, monkeypatch, rc, fail)
        assert not _run_one.attrs.get("hp_failure")
    with pytest.raises(tune.TrialFailed, match="non-finite"):
        _run_one(tmp_path / "n", monkeypatch, 0, matrix=[[float("nan")]] * 3)
    with pytest.raises(tune.StudyAborted):  # exit 0 but no valid matrix
        _run_one(tmp_path / "m", monkeypatch, 0)
    assert _run_one(tmp_path / "ok", monkeypatch, 0, matrix=[[0.2], [0.3], [0.4]]) == 0.4


def test_hp_failure_counts_and_is_logged(tmp_path, monkeypatch):
    fake = _fake_trial(behaviours=[tune.TrialFailed("exit code 1: nan")])
    monkeypatch.setattr(tune, "run_trial", fake)
    study = tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=2, epochs=1,
                             output_dir=tmp_path)
    states = [t.state for t in study.trials]
    assert states == [TrialState.FAIL, TrialState.COMPLETE] and len(fake.calls) == 2
    recs = _jsonl(_base(tmp_path))
    assert [r["state"] for r in recs] == ["FAIL", "COMPLETE"] and "nan" in recs[0]["note"]


def test_abort_is_not_counted_and_requeued(tmp_path, monkeypatch):
    fake = _fake_trial(behaviours=[tune.StudyAborted("exit 2: bad config")])
    monkeypatch.setattr(tune, "run_trial", fake)
    with pytest.raises(tune.StudyAborted, match="bad config"):
        tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=1, epochs=1,
                         output_dir=tmp_path)
    assert _jsonl(_base(tmp_path))[0]["state"] == "FAIL"
    # the cause is fixed: trial 0 (paper value) is evaluated again, then the budget is met
    fake2 = _fake_trial()
    monkeypatch.setattr(tune, "run_trial", fake2)
    study = tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=1, epochs=1,
                             output_dir=tmp_path)
    assert fake2.calls == [{"train.lr": 0.01}] and len(study.trials) == 2
    assert [t.state for t in study.trials] == [TrialState.FAIL, TrialState.COMPLETE]
    assert tune.n_finished(study) == 1


def test_interrupt_is_logged_and_trial_zero_still_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, "run_trial", _fake_trial(behaviours=[KeyboardInterrupt()]))
    with pytest.raises(KeyboardInterrupt):
        tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=2, epochs=1,
                         output_dir=tmp_path)
    (rec,) = _jsonl(_base(tmp_path))
    assert rec["state"] == "FAIL" and rec["note"] == "interrupted"
    fake = _fake_trial()
    monkeypatch.setattr(tune, "run_trial", fake)
    study = tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=2, epochs=1,
                             output_dir=tmp_path)
    assert fake.calls[0] == {"train.lr": 0.01}
    assert tune.n_finished(study) == 2


def test_stale_running_trial_is_failed_and_reenqueued(tmp_path, monkeypatch):
    out = tmp_path / "out"
    root = tune.tune_root(out)
    study = tune.open_study("synthetic", "r18_scratch", "naive", 1, 0, [], root)
    study.enqueue_trial({"train.lr": 0.01})
    study.ask({"train.lr": LR_DIST})
    assert study.trials[0].state == TrialState.RUNNING
    fake = _fake_trial()
    monkeypatch.setattr(tune, "run_trial", fake)
    study = tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=2, epochs=1,
                             output_dir=out)
    states = [t.state for t in study.trials]
    assert states[0] == TrialState.FAIL and states.count(TrialState.COMPLETE) == 2
    assert fake.calls[0] == {"train.lr": 0.01}  # re-enqueued stale params run first
    recs = _jsonl(_base(out))
    assert recs[0]["state"] == "FAIL" and "stale" in recs[0]["note"] and len(recs) == 3


def test_study_lock_refuses_second_process(tmp_path, monkeypatch):
    from clns.utils import lock as lock_mod

    monkeypatch.setattr(tune, "run_trial", _fake_trial())
    if lock_mod.fcntl is None:  # Windows: flock unavailable, simulate a held lock
        def held(self):
            raise lock_mod.RunLockError("held")

        monkeypatch.setattr(lock_mod.RunDirLock, "acquire", held)
        with pytest.raises(tune.StudyLocked, match="refusing to start"):
            tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=1, epochs=1,
                             output_dir=tmp_path)
    else:
        db = tune.db_path(tune.tune_root(tmp_path), "synthetic", "r18_scratch", "naive")
        with tune.FileLock(db.parent / "naive.db.lock"):
            with pytest.raises(tune.StudyLocked, match="refusing to start"):
                tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=1, epochs=1,
                                 output_dir=tmp_path)
        assert len(tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=1, epochs=1,
                                    output_dir=tmp_path).trials) == 1


def test_max_hours_stops_cleanly(tmp_path, monkeypatch):
    fake = _fake_trial()
    monkeypatch.setattr(tune, "run_trial", fake)
    monkeypatch.setattr(tune, "hours_used", lambda root: 5.0)
    study = tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=3, epochs=1,
                             output_dir=tmp_path, max_hours=1.0)
    assert fake.calls == [] and study.user_attrs["stopped_by_max_hours"] is True
    used = iter([0.1, 0.2, 0.9, 1.5, 1.6])
    monkeypatch.setattr(tune, "hours_used", lambda root: next(used))
    study = tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=5, epochs=1,
                             output_dir=tmp_path / "b", max_hours=1.0)
    assert len(fake.calls) == 2 and tune.n_finished(study) == 2  # stopped once 1.5 h >= 1 h


def test_main_default_max_hours_from_env_and_method_all(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, "run_trial", _fake_trial())
    seen = []
    monkeypatch.setattr(tune, "hours_used", lambda root: seen.append(root) or 0.0)
    monkeypatch.setenv("GPU_HOURS_CAP", "7")
    hp = tmp_path / "hp"
    code = tune.main(["--benchmark", "synthetic", "--arm", "r18_places", "--n-trials", "2",
                      "--output-dir", str(tmp_path / "out"), "--write-best",
                      "--hparams-dir", str(hp)])
    assert code == 0 and seen and seen[0] == tmp_path / "out"
    data = yaml.safe_load((hp / "synthetic" / "r18_places.yaml").read_text())
    assert list(data["methods"]) == list(tune.METHODS)
    assert data["methods"]["derpp"]["values"].keys() == {"train.lr", "method.alpha", "method.beta"}
    # constant score: the best trial is trial 0 = the start value
    assert data["methods"]["lwf"]["values"] == {"train.lr": 0.001, "method.alpha": 1.0}


def test_write_best_task_order_hash_recorded_and_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, "run_trial", _fake_trial())
    monkeypatch.setattr(tune, "_task_order_sha256", lambda rd: "abc")
    hp = tmp_path / "hp"
    kw = {"n_trials": 1, "epochs": 1, "sampler_seed": 0, "extra": [], "hparams_dir": hp,
          "root": tune.tune_root(tmp_path)}
    study = tune.tune_method("synthetic", "r18_scratch", "naive", n_trials=1, epochs=1,
                             output_dir=tmp_path)
    path = tune.write_best(study, "synthetic", "r18_scratch", "naive", **kw)
    data = yaml.safe_load(path.read_text())
    assert data["task_order_sha256"] == "abc" and data["tuning_seed"] == 0
    assert data["methods"]["naive"]["provenance"]["task_order_sha256"] == "abc"
    monkeypatch.setattr(tune, "_task_order_sha256", lambda rd: "other")
    with pytest.raises(ValueError, match="task_order_sha256"):
        tune.write_best(study, "synthetic", "r18_scratch", "naive", **kw)
