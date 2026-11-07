"""Optuna hyper-parameter tuning: ``python -m clns.tune``.

Per benchmark x model arm x method one resumable, seeded TPE study; every trial is one
``python -m clns.train experiment=tune ...`` subprocess (tuning stream = the benchmark's first
``max_experiences`` experiences, scored on the validation stream carved from TRAIN, never test).
Example::

    python -m clns.tune --benchmark clear10 --arm r18_scratch --method all --epochs 2 --write-best

Design (decided in advance):

* TUNED: ``train.lr`` (all methods), ``method.ewc_lambda`` (ewc), ``method.alpha`` (lwf, derpp),
  ``method.beta`` (derpp). Each is searched on a LOG scale over [start/10, start*10]; the start
  value (the paper/default value) is the FIRST trial. Starts: lr 0.01 (scratch arms) / 0.001
  (pretrained arms), ewc_lambda 1000, lwf alpha 1.0, derpp alpha 0.1 / beta 0.5.
* FIXED (never tuned, identical across methods): ``method.mem_size``, ``method.batch_size_mem``,
  ``method.temperature``, ``method.storage``, ewc ``mode`` / ``decay_factor`` (configs/method).
* EQUAL budget per method (``--n-trials``; default 20, 10 for clear10 and domainnet), sized for
  DER++ (3 tuned parameters); 1 seed (``seed=0``), shortened schedule (``--epochs``).
* Sampler: ``TPESampler(seed=--sampler-seed)``. A resume re-creates the sampler with the same
  seed, so the continued proposals are not identical to an uninterrupted study (the history is).
* TPE: ``n_startup_trials = max(3, n_trials // 4)`` (random trials before the model is used),
  fixed when the study is created and stored in the study attrs / provenance (a resume reuses it).
* Budget: only COMPLETE trials and FAIL trials caused by the hyper-parameters (training ran and
  crashed / produced NaN: exit 1 with a training error) count towards ``--n-trials``. Interrupted
  / stale trials are re-enqueued on resume and do not count (trial 0, the paper value, is always
  evaluated eventually); a trial that fails for another reason (exit 2/3/4/5, missing data, bad
  config or run dir) aborts the study with an error (and is re-enqueued on resume).
  ``--max-hours`` (default $GPU_HOURS_CAP, else none) stops the study cleanly before a trial when
  scripts/gpu_hours.py reports that many pod hours under the output dir.
* One process per study: ``<method>.db.lock`` (fcntl flock; a no-op on Windows). Keep
  ``$CLNS_OUTPUT_DIR/tune/optuna`` on the pod's local disk, not on a network share (sqlite).
* Records: sqlite ``<output>/tune/optuna/<benchmark>/<arm>/<method>.db`` + ``<study>.jsonl`` +
  ``<study>.csv`` (every trial, incl. failed / interrupted / stale ones; ``study`` column).
  Trial run dirs:
  ``<output>/tune/runs/<benchmark>/<arm>/<method>/optuna_e<epochs>/p<hash of params>/seed0``.
* ``--write-best`` merges the best trial into ``configs/hparams/<benchmark>/<arm>.yaml``.
* CORe50-NIC: no frame-level validation split exists, so trials get
  ``benchmark.val_fraction=0.0 benchmark.val_sessions=[1]`` (whole-session hold-out).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from clns.models import ARM_SPECS
from clns.utils.hparams import default_hparams_dir, load_hparams
from clns.utils.lock import RunDirLock, RunLockError

log = logging.getLogger("clns.tune")

METHODS = ("naive", "ewc", "lwf", "er", "derpp")
DEFAULT_TRIALS = 20
TRIALS_BY_BENCHMARK = {"clear10": 10, "domainnet": 10}
TUNED_EXPERIENCES = 3                     # tune.yaml benchmark.max_experiences
LOG_RANGE = 10.0                          # [start / 10, start * 10]
SCRATCH_LR, PRETRAINED_LR = 0.01, 0.001
TUNING_SEED = 0
CORE50_OVERRIDES = ("benchmark.val_fraction=0.0", "benchmark.val_sessions=[1]")
REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs"
# --extra allowlist: volatile / harmless keys only (never part of the config fingerprint's
# results-relevant content); everything else, incl. every tuned / fixed hyper-parameter, raises.
VOLATILE_EXTRA_KEYS = frozenset({"train.num_workers", "train.eval_mb_size", "device",
                                 "deterministic"})
VOLATILE_EXTRA_PREFIXES = ("paths.",)
# Test-sized caps, synthetic benchmark only.
SYNTHETIC_EXTRA_KEYS = frozenset({
    "benchmark.subsample_per_class", "benchmark.train_per_class", "benchmark.test_per_class",
    "benchmark.image_size", "benchmark.n_experiences", "model.cifar_stem", "train.mb_size"})
# failure.json types / messages that mean "the environment is broken", not "these hyper-parameters
# failed": the study aborts instead of recording a (counted) FAIL.
INFRA_ERROR_TYPES = frozenset({"FileNotFoundError", "NotADirectoryError", "PermissionError",
                               "OSError", "ImportError", "ModuleNotFoundError"})
INFRA_MESSAGE_RE = re.compile(r"not found|no such file|does not exist|\.done|download", re.I)
UNSAFE_PATH_CHARS = re.compile(r"""['"$\\\n\r\t]""")


class TrialFailed(RuntimeError):
    """The hyper-parameters failed (training ran, then crashed / NaN): a FAIL that counts."""


class StudyAborted(RuntimeError):
    """A trial failed for a reason unrelated to its hyper-parameters (exit 2/3/4/5, missing
    data, bad config / run dir, ...): the study stops; the trial is re-enqueued on resume."""


class StudyLocked(RuntimeError):
    """Another process holds the lock of this study / hparams file."""


# --------------------------------------------------------------------------- #
# Search space / extras
# --------------------------------------------------------------------------- #
def search_space(method: str, arm: str) -> dict[str, float]:
    """Tuned parameters of ``method`` and their start (paper/default) values."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {list(METHODS)}")
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown model arm {arm!r}; expected one of {list(ARM_SPECS)}")
    space = {"train.lr": SCRATCH_LR if ARM_SPECS[arm][1] == "scratch" else PRETRAINED_LR}
    if method == "ewc":
        space["method.ewc_lambda"] = 1000.0
    elif method == "lwf":
        space["method.alpha"] = 1.0
    elif method == "derpp":
        space["method.alpha"] = 0.1
        space["method.beta"] = 0.5
    return space


def bounds(start: float) -> tuple[float, float]:
    return start / LOG_RANGE, start * LOG_RANGE


def default_trials(benchmark: str) -> int:
    return TRIALS_BY_BENCHMARK.get(benchmark, DEFAULT_TRIALS)


def n_startup_trials(n_trials: int) -> int:
    return max(3, int(n_trials) // 4)


def _extra_key(item: str) -> str:
    return item.lstrip("+~").split("=", 1)[0]


def is_volatile_extra(item: str) -> bool:
    key = _extra_key(item)
    return key in VOLATILE_EXTRA_KEYS or key.startswith(VOLATILE_EXTRA_PREFIXES)


def check_extra(extra: list[str], benchmark: str | None = None) -> None:
    """Only volatile keys (and, for ``benchmark == "synthetic"``, test-size caps) are allowed."""
    for item in extra:
        if "=" not in item:
            raise ValueError(f"--extra {item!r}: expected key=value")
        key = _extra_key(item)
        if is_volatile_extra(item):
            continue
        if key in SYNTHETIC_EXTRA_KEYS:
            if benchmark == "synthetic":
                continue
            raise ValueError(f"--extra {item!r}: {key} is allowed on the synthetic benchmark only")
        raise ValueError(
            f"--extra {item!r}: not allowed. Only volatile keys ({sorted(VOLATILE_EXTRA_KEYS)}, "
            "paths.*) are accepted; tuned / fixed hyper-parameters and everything else change "
            "the tuning protocol (fixed values live in configs/method/*.yaml)")


def auto_overrides(benchmark: str) -> list[str]:
    """Benchmark-specific overrides added to every trial (recorded in the study attrs)."""
    return list(CORE50_OVERRIDES) if benchmark == "core50_nic" else []


def n_expected_rows(overrides: list[str]) -> int:
    for item in overrides:
        if item.startswith("benchmark.max_experiences="):
            return int(item.split("=", 1)[1])
    return TUNED_EXPERIENCES


def check_output_path(path: str | Path) -> None:
    """Reject paths that cannot be passed safely as a single-quoted Hydra override value."""
    text = Path(path).as_posix() if not isinstance(path, str) else path
    if UNSAFE_PATH_CHARS.search(text):
        raise ValueError(
            f"output path {text!r} contains a quote, '$', backslash or control character; "
            "use a plain path (letters, digits, space, . _ - / , = : are fine)")


# --------------------------------------------------------------------------- #
# Paths / names
# --------------------------------------------------------------------------- #
def tune_root(output_dir: str | Path | None = None) -> Path:
    base = output_dir or os.environ.get("CLNS_OUTPUT_DIR", "/workspace/outputs")
    return Path(base) / "tune"


def db_path(root: Path, benchmark: str, arm: str, method: str) -> Path:
    return root / "optuna" / benchmark / arm / f"{method}.db"


def study_name(benchmark: str, arm: str, method: str, epochs: int) -> str:
    return f"{benchmark}__{arm}__{method}__e{epochs}"


def log_base(root: Path, benchmark: str, arm: str, method: str, epochs: int) -> Path:
    """``<db dir>/<study name>``; ``.jsonl`` / ``.csv`` are appended."""
    return db_path(root, benchmark, arm, method).parent / study_name(benchmark, arm, method, epochs)


def params_hash(params: dict[str, float]) -> str:
    canon = json.dumps({k: repr(float(v)) for k, v in sorted(params.items())})
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:10]


def trial_run_dir(root: Path, benchmark: str, arm: str, method: str, epochs: int,
                  params: dict[str, float]) -> Path:
    return (root / "runs" / benchmark / arm / method / f"optuna_e{epochs}"
            / f"p{params_hash(params)}" / "seed0")


def _hydra_assign(key: str, value: str) -> str:
    """``key='value'``: always single-quoted (commas, '=', spaces are then plain characters).

    ``value`` must pass :func:`check_output_path` (no quote / ``$`` / backslash), so no escaping
    is needed.
    """
    check_output_path(value)
    return f"{key}='{value}'"


def build_trial_command(benchmark: str, arm: str, method: str, epochs: int,
                        params: dict[str, float], run_dir: Path,
                        extra: list[str] | None = None) -> list[str]:
    """The ``clns.train`` command of one trial (benchmark overrides, then ``extra``)."""
    return [
        sys.executable, "-m", "clns.train", "experiment=tune", f"benchmark={benchmark}",
        f"model={arm}", f"method={method}", f"seed={TUNING_SEED}", f"train.epochs={epochs}",
        "logger=none",
        *(f"{k}={float(v)!r}" for k, v in params.items()),
        *auto_overrides(benchmark), *(extra or []),
        _hydra_assign("run_dir", Path(run_dir).as_posix()),
    ]


# --------------------------------------------------------------------------- #
# Locks
# --------------------------------------------------------------------------- #
class FileLock(RunDirLock):
    """Exclusive flock on ``path`` (RunDirLock with a custom file; no-op without fcntl)."""

    def __init__(self, path: str | Path):
        super().__init__(Path(path).parent)
        self.path = Path(path)

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            super().acquire()
        except RunLockError as exc:
            raise StudyLocked(f"{self.path} is locked by another running process "
                              "(clns.tune already runs on this study / file); refusing to start"
                              ) from exc


# --------------------------------------------------------------------------- #
# Trial
# --------------------------------------------------------------------------- #
def score_valid_matrix(run_dir: Path, n_rows: int) -> float:
    """Mean validation accuracy of row ``n_rows - 1`` of ``valid_matrix.json`` (maximised)."""
    path = Path(run_dir) / "valid_matrix.json"
    if not path.is_file():
        raise StudyAborted(f"no valid_matrix.json in {run_dir} (no validation stream?)")
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
    if len(rows) < n_rows:
        raise StudyAborted(f"valid_matrix.json has {len(rows)} rows, expected {n_rows}")
    vals = [v for v in rows[n_rows - 1] if v is not None]
    if not vals:
        raise StudyAborted("last valid_matrix.json row has no values")
    return float(sum(vals) / len(vals))


def _read_failure(run_dir: Path) -> dict | None:
    try:
        blob = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
        return blob if isinstance(blob, dict) else None
    except (OSError, ValueError):
        return None


def run_trial(benchmark: str, arm: str, method: str, epochs: int, params: dict[str, float],
              root: Path, extra: list[str], attrs: dict | None = None) -> float:
    """Run one trial subprocess and return its score; ``attrs`` receives run_dir/exit_code/wall.

    Raises :class:`TrialFailed` (counted; ``attrs["hp_failure"]``) if training ran and failed or
    scored NaN, :class:`StudyAborted` for any other failure.
    """
    run_dir = trial_run_dir(root, benchmark, arm, method, epochs, params)
    cmd = build_trial_command(benchmark, arm, method, epochs, params, run_dir, extra)
    attrs = attrs if attrs is not None else {}
    attrs["run_dir"] = str(run_dir)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    attrs["wall_s"] = round(time.perf_counter() - t0, 2)
    attrs["exit_code"] = proc.returncode
    if proc.returncode != 0:
        fail = _read_failure(run_dir)
        msg = str((fail or {}).get("message") or proc.stderr or proc.stdout or "")[-500:].strip()
        if (proc.returncode == 1 and fail is not None
                and fail.get("type") not in INFRA_ERROR_TYPES
                and not INFRA_MESSAGE_RE.search(msg)):
            attrs["hp_failure"] = True
            raise TrialFailed(f"exit code 1: {msg}")
        raise StudyAborted(
            f"trial exited with code {proc.returncode}, which is not a hyper-parameter failure "
            f"(exit 2/3/4/5, missing data, bad config or run dir, ...): {msg}; run dir {run_dir}. "
            "Fix the cause and re-run (the trial is re-enqueued).")
    value = score_valid_matrix(run_dir, n_expected_rows([*auto_overrides(benchmark), *extra]))
    if not math.isfinite(value):
        attrs["hp_failure"] = True
        raise TrialFailed(f"non-finite validation score {value}")
    return value


# --------------------------------------------------------------------------- #
# Study
# --------------------------------------------------------------------------- #
class TrialLog:
    """Appends every trial outcome to ``<study>.jsonl`` and ``<study>.csv``."""

    FIXED = ["study", "number", "state", "value", "exit_code", "wall_s", "run_dir", "note"]

    def __init__(self, base: Path, study: str, param_names: list[str]):
        self.jsonl = base.parent / f"{base.name}.jsonl"
        self.csv = base.parent / f"{base.name}.csv"
        self.study = study
        self.param_names = sorted(param_names)

    def add(self, number: int, state: str, value: float | None, params: dict, attrs: dict,
            note: str | None = None) -> None:
        rec = {"study": self.study, "number": number, "state": state, "value": value,
               "params": params, "exit_code": attrs.get("exit_code"),
               "wall_s": attrs.get("wall_s"), "run_dir": attrs.get("run_dir"), "note": note,
               "logged_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        new = not self.csv.exists()
        with self.csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow([*self.FIXED, *self.param_names])
            w.writerow([rec[k] for k in self.FIXED] + [params.get(p) for p in self.param_names])


def fixed_config_sha256(method: str) -> str:
    """Hash of everything fixed for the study: the method config, the non-volatile ``train``
    block of train.yaml and the benchmark / train blocks of experiment/tune.yaml."""
    def load(rel: str) -> dict:
        return yaml.safe_load((CONFIG_DIR / rel).read_text(encoding="utf-8")) or {}

    volatile = {"num_workers", "eval_mb_size", "stop_after_experiences"}
    train = {k: v for k, v in (load("train.yaml").get("train") or {}).items()
             if k not in volatile}
    tune = load("experiment/tune.yaml")
    blob = {"method": load(f"method/{method}.yaml"), "train": train,
            "tune_benchmark": tune.get("benchmark"), "tune_train": tune.get("train")}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


def _git_commit() -> str | None:
    try:
        from clns.utils.provenance import git_info

        return git_info().get("commit")
    except Exception:  # noqa: BLE001 - provenance only
        return None


def open_study(benchmark: str, arm: str, method: str, epochs: int, sampler_seed: int,
               extra: list[str], root: Path, n_trials: int = DEFAULT_TRIALS):
    """Create / load the study (sqlite); a mismatch with the stored attrs raises ValueError.

    Compared: sampler seed, non-volatile extras (incl. CORe50 overrides), start values, epochs,
    benchmark / arm / method and the fixed-config hash. Only recorded: ``n_startup_trials``
    (reused on resume), git commit, the full extra list.
    """
    import optuna

    path = db_path(root, benchmark, arm, method)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = study_name(benchmark, arm, method, epochs)
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{path.resolve().as_posix()}", engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.create_study(study_name=name, storage=storage, direction="maximize",
                                load_if_exists=True)
    all_extra = [*auto_overrides(benchmark), *extra]
    wanted = {"sampler_seed": sampler_seed,
              "extra_nonvolatile": [e for e in all_extra if not is_volatile_extra(e)],
              "start_values": search_space(method, arm), "epochs": epochs,
              "benchmark": benchmark, "arm": arm, "method": method,
              "fixed_config_sha256": fixed_config_sha256(method)}
    stored = study.user_attrs
    if not stored:
        for k, v in wanted.items():
            study.set_user_attr(k, v)
        study.set_user_attr("n_startup_trials", n_startup_trials(n_trials))
        study.set_user_attr("extra", all_extra)
        study.set_user_attr("git_commit", _git_commit())
    else:
        bad = {k: (stored.get(k), v) for k, v in wanted.items() if stored.get(k) != v}
        if bad:
            raise ValueError(f"study {name} in {path} was created with other settings "
                             f"(stored, requested): {bad}; use another --output-dir")
    study.sampler = optuna.samplers.TPESampler(
        seed=sampler_seed, n_startup_trials=int(study.user_attrs["n_startup_trials"]))
    return study


def n_finished(study) -> int:
    """Trials that count against the budget: COMPLETE and hyper-parameter-caused FAIL."""
    from optuna.trial import TrialState

    return sum(1 for t in study.trials if t.state == TrialState.COMPLETE
               or (t.state == TrialState.FAIL and t.user_attrs.get("hp_failure")))


def _gpu_hours_module():
    spec = importlib.util.spec_from_file_location("gpu_hours", REPO / "scripts" / "gpu_hours.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def hours_used(output_root: Path) -> float:
    """Pod hours under ``output_root`` (scripts/gpu_hours.py, incl. tuning trial wall times)."""
    return float(_gpu_hours_module().hours_used(output_root))


def tune_method(benchmark: str, arm: str, method: str, *, n_trials: int, epochs: int,
                sampler_seed: int = 0, extra: list[str] | None = None,
                output_dir: str | Path | None = None, max_hours: float | None = None):
    """Run (or resume) the study of one method until ``n_trials`` trials have counted.

    Holds the study lock for the whole call (raises :class:`StudyLocked` otherwise). Stops
    cleanly (returns) when ``max_hours`` of pod time are used; raises :class:`StudyAborted`
    on a non-hyper-parameter failure.
    """
    from optuna.trial import TrialState

    extra = list(extra or [])
    check_extra(extra, benchmark)
    root = tune_root(output_dir)
    check_output_path(root.as_posix())
    space = search_space(method, arm)
    db = db_path(root, benchmark, arm, method)
    with FileLock(db.parent / f"{method}.db.lock"):
        study = open_study(benchmark, arm, method, epochs, sampler_seed, extra, root, n_trials)
        trial_log = TrialLog(log_base(root, benchmark, arm, method, epochs), study.study_name,
                             list(space))

        # Only now (lock held) are RUNNING trials stale: FAIL them, kept in the log.
        for t in study.get_trials(deepcopy=True, states=(TrialState.RUNNING,)):
            study.tell(t.number, state=TrialState.FAIL)
            trial_log.add(t.number, "FAIL", None, t.params, t.user_attrs,
                          "stale RUNNING trial marked FAIL (will be re-enqueued)")
        # FAIL trials that are not the hyper-parameters' fault (interrupted, aborted, stale) do
        # not count: evaluate their parameters again (trial 0 = the paper value, always).
        requeued = list(study.user_attrs.get("requeued", []))
        for t in study.trials:
            if (t.state == TrialState.FAIL and not t.user_attrs.get("hp_failure")
                    and t.number not in requeued and set(t.params) == set(space)):
                study.enqueue_trial(dict(t.params))
                requeued.append(t.number)
        study.set_user_attr("requeued", requeued)
        if not study.trials:
            study.enqueue_trial(dict(space))  # the paper value is trial 0

        def objective(trial) -> float:
            params = {k: trial.suggest_float(k, *bounds(s), log=True) for k, s in space.items()}
            attrs: dict[str, Any] = {}
            state, value, note = "FAIL", None, None
            try:
                value = run_trial(benchmark, arm, method, epochs, params, root, extra, attrs)
                state = "COMPLETE"
                return value
            except KeyboardInterrupt:
                note = "interrupted"
                raise
            except BaseException as exc:  # noqa: BLE001 - logged, then re-raised
                note = f"{type(exc).__name__}: {exc}"[:600]
                raise
            finally:
                for k, v in attrs.items():
                    trial.set_user_attr(k, v)
                trial_log.add(trial.number, state, value, params, attrs, note)

        stopped = False
        while n_trials - n_finished(study) > 0:
            used = _hours_or_none(root, max_hours)
            if max_hours is not None and used is not None and used >= max_hours:
                log.warning("%s: max-hours %.2f reached (%.2f h used); stopping with %d/%d "
                            "counted trials.", study.study_name, max_hours, used,
                            n_finished(study), n_trials)
                stopped = True
                break
            study.optimize(objective, n_trials=1, catch=(TrialFailed,))
            last = study.trials[-1]
            used = _hours_or_none(root, max_hours)
            log.info("trial %d %s value=%s params=%s; pod hours used: %s%s", last.number,
                     last.state.name, last.value, last.params,
                     "?" if used is None else f"{used:.3f}",
                     "" if max_hours is None else f" / {max_hours}")
        study.set_user_attr("stopped_by_max_hours", stopped)
    return study


def _hours_or_none(root: Path, max_hours: float | None) -> float | None:
    try:
        return hours_used(root.parent)
    except Exception as exc:  # noqa: BLE001
        if max_hours is not None:
            raise RuntimeError(f"GPU-hour accounting failed ({exc}); refusing to continue "
                               "under --max-hours") from exc
        log.warning("GPU-hour accounting failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# --write-best
# --------------------------------------------------------------------------- #
def _task_order_sha256(run_dir: str | None) -> str | None:
    if not run_dir:
        return None
    try:
        prov = json.loads((Path(run_dir) / "provenance.json").read_text(encoding="utf-8"))
        return (prov.get("benchmark_meta") or {}).get("task_order_sha256")
    except (OSError, ValueError):
        return None


def write_best(study, benchmark: str, arm: str, method: str, *, n_trials: int, epochs: int,
               sampler_seed: int, extra: list[str], hparams_dir: str | Path | None,
               root: Path, force: bool = False) -> Path:
    """Merge the best trial of ``study`` into ``<hparams_dir>/<benchmark>/<arm>.yaml``.

    Refuses (ValueError) unless ``n_trials`` trials are COMPLETE, or ``force``. The file is
    locked (``.<arm>.yaml.lock``), re-read under the lock and replaced atomically.
    """
    from optuna.trial import TrialState

    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete:
        raise ValueError(f"{study.study_name}: no COMPLETE trial; nothing to write")
    if len(complete) < n_trials:
        msg = f"{study.study_name}: only {len(complete)}/{n_trials} trials COMPLETE"
        if not force:
            raise ValueError(f"{msg}; finish the study or pass --force to write the best so far")
        log.warning("%s (--force): writing the best so far.", msg)
    best = study.best_trial
    sha = _task_order_sha256(best.user_attrs.get("run_dir"))
    if sha is None:
        log.warning("%s: no task_order_sha256 found in the best trial's provenance.",
                    study.study_name)
    n_exp = n_expected_rows([*auto_overrides(benchmark), *extra])
    path = Path(hparams_dir or default_hparams_dir()) / benchmark / f"{arm}.yaml"
    with FileLock(path.parent / f".{arm}.yaml.lock"):
        data = load_hparams(path) if path.is_file() else {}
        old = data.get("task_order_sha256")
        if old is not None and sha is not None and old != sha:
            raise ValueError(f"{path}: task_order_sha256 of the tuning runs changed "
                             f"({old} -> {sha}); methods of one file must share the tuning order")
        methods = data.setdefault("methods", {})
        methods[method] = {
            "values": {k: float(v) for k, v in best.params.items()},
            "provenance": {
                "n_trials": n_trials, "n_trials_total": len(study.trials),
                "n_complete": len(complete), "best_value": best.value,
                "best_trial": best.number,
                "objective": "mean validation accuracy, last tuned row of valid_matrix.json "
                             "(maximised)",
                "db": str(db_path(root, benchmark, arm, method)), "study": study.study_name,
                "date_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
                "tuning_epochs": epochs, "sampler_seed": sampler_seed,
                "n_startup_trials": study.user_attrs.get("n_startup_trials"),
                "start_values": search_space(method, arm),
                "tuned_experiences": list(range(n_exp)),
                "tuning_seed": TUNING_SEED, "task_order_sha256": sha,
                "fixed_config_sha256": study.user_attrs.get("fixed_config_sha256"),
                "git_commit": study.user_attrs.get("git_commit"),
                "extra": [*auto_overrides(benchmark), *extra],
            },
        }
        data["tuned_transitions"] = list(range(n_exp - 1))
        data["tuning_seed"] = TUNING_SEED
        data["task_order_sha256"] = sha if sha is not None else old
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Optuna tuning (one clns.train subprocess per trial).")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--arm", required=True, help="model arm, e.g. r18_scratch")
    ap.add_argument("--method", default="all", help=f"one of {list(METHODS)} or 'all'")
    ap.add_argument("--n-trials", type=int, default=None,
                    help="counted trials per method (default 20; 10 for clear10 and domainnet)")
    ap.add_argument("--epochs", type=int, default=1, help="tuning epochs (shortened schedule)")
    ap.add_argument("--sampler-seed", type=int, default=0)
    ap.add_argument("--extra", nargs="*", default=[], metavar="KEY=VALUE",
                    help="volatile clns.train overrides only (train.num_workers, "
                    "train.eval_mb_size, device, deterministic, paths.*)")
    ap.add_argument("--output-dir", default=None, help="default $CLNS_OUTPUT_DIR (-> <it>/tune)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop before a trial once this many pod hours are used under the "
                    "output dir (scripts/gpu_hours.py); default $GPU_HOURS_CAP, else none")
    ap.add_argument("--write-best", action="store_true",
                    help="merge the best trials into <hparams-dir>/<benchmark>/<arm>.yaml")
    ap.add_argument("--force", action="store_true",
                    help="--write-best even if fewer than --n-trials trials are COMPLETE")
    ap.add_argument("--hparams-dir", default=None, help="default <repo>/configs/hparams")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    methods = METHODS if args.method == "all" else (args.method,)
    n_trials = args.n_trials if args.n_trials is not None else default_trials(args.benchmark)
    max_hours = args.max_hours
    if max_hours is None and os.environ.get("GPU_HOURS_CAP"):
        max_hours = float(os.environ["GPU_HOURS_CAP"])
    check_extra(args.extra, args.benchmark)
    root = tune_root(args.output_dir)
    check_output_path(root.as_posix())
    rc = 0
    for m in methods:
        log.info("== %s / %s / %s: %d trials, %d epochs", args.benchmark, args.arm, m, n_trials,
                 args.epochs)
        study = tune_method(args.benchmark, args.arm, m, n_trials=n_trials, epochs=args.epochs,
                            sampler_seed=args.sampler_seed, extra=args.extra,
                            output_dir=args.output_dir, max_hours=max_hours)
        if args.write_best:
            try:
                out = write_best(study, args.benchmark, args.arm, m, n_trials=n_trials,
                                 epochs=args.epochs, sampler_seed=args.sampler_seed,
                                 extra=args.extra, hparams_dir=args.hparams_dir, root=root,
                                 force=args.force)
                log.info("wrote %s (method %s)", out, m)
            except ValueError as exc:
                log.error("not written (method %s): %s", m, exc)
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
