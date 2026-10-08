"""Training entry point: one benchmark x method x model x seed per invocation.

Usage (editable install, ``pip install -e .``; ``config_path`` is resolved relative to this
file, i.e. ``<repo>/configs``, so a non-editable install cannot find the configs)::

    python -m clns.train benchmark=imagenet100 method=er model=resnet18 seed=0
    python -m clns.train experiment=smoke                 # tiny synthetic run (-> <output>/smoke)
    clns-train experiment=smoke logger=wandb_offline      # console script, same thing

Run directory (contract): ``${paths.output_dir}/runs/<benchmark>/<model>/<method>/seed<seed>/``
holding config.yaml, provenance.json, fingerprint.json, RUNNING.lock, checkpoint.pt,
progress.json, accuracy_matrix.json, per_class_matrix.json, valid_matrix.json
(``train.eval_valid`` only), metrics.jsonl, summary.json, wandb_run_id.txt (W&B only), log.txt,
failure.json (failed attempts only; removed on success) and DONE (success only). No ``.hydra/``
(``hydra.output_subdir: null``): config.yaml + provenance.json are the config record.
``experiment=smoke*`` configs write under ``<output_dir>/smoke/runs/...`` so smoke results never
share a directory with real runs.

Flow: model-specific overrides (L2P: image_size 224 + ViT normalization) -> lock RUNNING.lock ->
config fingerprint check (BEFORE the DONE check) -> DONE? exit 0 -> device -> seed ->
provenance -> build data/loggers/evaluator/model/strategy -> load checkpoint -> truncate
outputs to the checkpoint -> (fresh run) eval FULL test stream = random-init row -> for each
remaining training experience: re-seed from (seed, exp), train, eval FULL test stream (+ valid
stream if ``train.eval_valid``), checkpoint, progress.json -> summary.json -> DONE.

Compute accounting (single source of truth, persisted in progress.json across resumes):
wall clock and CUDA peak memory (``reset_peak_memory_stats`` before each call) are measured
around every ``strategy.train()`` and test ``strategy.eval()`` call, so EWC's Fisher pass, the
replay-buffer update, LwF's teacher copy and evaluation are included.
``wall_clock_s_total`` = sum over attempts of (attempt start -> last committed checkpoint, or
the end of the run); time spent in an attempt after its last checkpoint (a crash) is excluded.

Exit codes: 0 success / already done / stopped by ``train.stop_after_experiences``;
1 any other error (failure.json written, exception re-raised);
3 CUDA out-of-memory (failure.json written);
4 config fingerprint mismatch: run_dir holds a run with a different (non-volatile) config;
5 run directory locked by another live process (RUNNING.lock);
143 SIGTERM (e.g. ``timeout``, ``run_all_baselines.sh --stop``, container stop): raised as
``SystemExit(143)`` through the failure path -> failure.json (``"interrupted": true``), W&B
run marked preempting and finished with exit code 143, RUNNING.lock released. The checkpoint of
the last completed experience is kept, so a re-run resumes.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path

# Sets CUBLAS_WORKSPACE_CONFIG before anything initialises CUDA.
from clns.utils.seed import seed_everything, seed_experience, seed_model_init  # isort: skip

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from clns.data import build_benchmark
from clns.utils.checkpoint import (
    atomic_write_json,
    load_checkpoint,
    reapply_eval_mb_size,
    save_checkpoint_atomic,
)
from clns.utils.fingerprint import FINGERPRINT_FILE, check_or_write_fingerprint
from clns.utils.lock import RunDirLock, RunLockError
from clns.utils.provenance import update_provenance, write_provenance

log = logging.getLogger("clns.train")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_OOM = 3
EXIT_CONFIG_MISMATCH = 4
EXIT_LOCKED = 5
EXIT_SIGTERM = 143  # 128 + SIGTERM


class Terminated(SystemExit):
    """Raised by the SIGTERM handler; a SystemExit so ``except Exception`` code never eats it."""

    def __init__(self) -> None:
        super().__init__(EXIT_SIGTERM)


def _install_sigterm_handler():
    """Turn SIGTERM into :class:`Terminated` in this (main) process. Returns the old handler.

    Forked children (DataLoader workers) inherit the handler; there it restores the default
    action and re-sends the signal, so workers die as before instead of unwinding Python code.
    Returns None (nothing installed) outside the main thread, where ``signal.signal`` fails.
    """
    owner = os.getpid()

    def _handler(signum, frame):  # noqa: ARG001
        if os.getpid() != owner:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)
            return
        log.error("SIGTERM received; stopping (exit %d).", EXIT_SIGTERM)
        raise Terminated()

    try:
        return signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError, AttributeError):  # not the main thread / unsupported
        return None

# Same values as clns.models.vit.VIT_IMAGE_SIZE / VIT_NORMALIZATION (kept literal here so the
# override does not import torch before the lock/fingerprint checks).
VIT_MODEL_NAME = "vit_b16_l2p"
VIT_IMAGE_SIZE = 224
VIT_NORMALIZATION = "vit"

# B2 keys (fixed, used by scripts/smoke_test.sh) + compute/macro additions.
B2_SUMMARY_KEYS = (
    "acc_final",
    "acc_avg_incremental",
    "bwt",
    "forgetting_max",
    "fwt",
    "fwt_degenerate",
    "peak_gpu_mem_mb",
    "peak_gpu_mem_mb_per_exp",
    "train_time_s_per_exp",
    "train_time_s_total",
)
SUMMARY_KEYS = B2_SUMMARY_KEYS + (
    "acc_final_macro",
    "peak_gpu_mem_mb_eval",
    "wall_clock_s_total",
)
PROGRESS_FILE = "progress.json"


def resolve_device(name: str):
    import torch

    name = str(name).lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return torch.device(name)


def apply_model_overrides(cfg: DictConfig) -> None:
    """Model-driven data settings, applied before the fingerprint so it records them.

    vit_b16_l2p: the timm ViT-B/16 checkpoint needs 224 x 224 inputs normalised with
    mean = std = 0.5 -> force ``benchmark.image_size=224`` and ``benchmark.normalization=vit``.
    """
    if str(cfg.model.name) != VIT_MODEL_NAME:
        return
    with open_dict(cfg):
        old_size = cfg.benchmark.get("image_size", None)
        if old_size is None or int(old_size) != VIT_IMAGE_SIZE:
            log.warning("model=%s: overriding benchmark.image_size %s -> %d (ViT input size).",
                        VIT_MODEL_NAME, old_size, VIT_IMAGE_SIZE)
            cfg.benchmark.image_size = VIT_IMAGE_SIZE
        old_norm = cfg.benchmark.get("normalization", None)
        if old_norm != VIT_NORMALIZATION:
            log.warning("model=%s: overriding benchmark.normalization %s -> %s (mean=std=0.5).",
                        VIT_MODEL_NAME, old_norm, VIT_NORMALIZATION)
            cfg.benchmark.normalization = VIT_NORMALIZATION


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _swap_loggers(strategy, fresh_loggers: list) -> None:
    """After a resume, replace the unpickled loggers by the freshly built ones.

    The unpickled W&B logger is not re-initialised (ClnsWandBLogger.__setstate__) and the
    unpickled ClnsTextLogger holds no file (ClnsTextLogger.__setstate__); the W&B step counters
    are carried over to the fresh logger, which already joined the same W&B run id.
    """
    from clns.evaluation.plugins import ClnsWandBLogger

    old = list(getattr(strategy.evaluator, "loggers", []) or [])
    old_wb = next((lg for lg in old if isinstance(lg, ClnsWandBLogger)), None)
    new_wb = next((lg for lg in fresh_loggers if isinstance(lg, ClnsWandBLogger)), None)
    if old_wb is not None and new_wb is not None:
        new_wb.step = getattr(old_wb, "step", 0)
        new_wb.exp_count = getattr(old_wb, "exp_count", 0)
    _close_loggers([lg for lg in old if lg not in fresh_loggers])
    strategy.evaluator.loggers = fresh_loggers


def _close_loggers(loggers: list) -> None:
    """Close file handles held by JSONL/Text loggers (InteractiveLogger writes to stdout)."""
    from avalanche.logging import InteractiveLogger, TextLogger

    from clns.evaluation.plugins import JSONLLogger

    for lg in loggers:
        try:
            if isinstance(lg, JSONLLogger):
                lg.close()
            elif isinstance(lg, TextLogger) and not isinstance(lg, InteractiveLogger):
                f = getattr(lg, "file", None)
                if f is not None and f not in (sys.stdout, sys.stderr) and hasattr(f, "close"):
                    f.close()
        except Exception:  # noqa: BLE001
            pass


def _log_extra(loggers: list, values: dict, n_trained: int, phase: str = "other") -> None:
    from clns.evaluation.plugins import ClnsWandBLogger, JSONLLogger

    for lg in loggers:
        if isinstance(lg, JSONLLogger):
            lg.log_dict(values, n_trained=n_trained, phase=phase)
        elif isinstance(lg, ClnsWandBLogger) and lg.wandb is not None and lg.wandb.run:
            lg.wandb.log(
                {k: v for k, v in values.items() if v is not None}, step=lg.step
            )


def _wandb_finish(
    loggers: list, summary: dict | None = None, exit_code: int = 0, preempting: bool = False
) -> None:
    """Finish the W&B run. ``preempting``: mark it preempted/interrupted (SIGTERM) first."""
    from clns.evaluation.plugins import ClnsWandBLogger

    for lg in loggers:
        if isinstance(lg, ClnsWandBLogger) and lg.wandb is not None and lg.wandb.run:
            try:
                if preempting and hasattr(lg.wandb, "mark_preempting"):
                    lg.wandb.mark_preempting()
                if summary:
                    lg.wandb.run.summary.update(
                        {f"final/{k}": v for k, v in summary.items()}
                    )
                lg.wandb.finish(exit_code=exit_code)
            except Exception as exc:  # noqa: BLE001 - never fail a finished run on W&B
                log.warning("wandb.finish failed: %s", exc)


def _matrix_plugin(strategy, stream_name: str):
    from clns.evaluation.plugins import AccuracyMatrixPlugin

    for p in strategy.plugins:
        if isinstance(p, AccuracyMatrixPlugin) and p.stream_name == stream_name:
            return p
    return None


# --------------------------------------------------------------------------- #
# Compute accounting (progress.json)
# --------------------------------------------------------------------------- #
def _fit(xs, n: int) -> list:
    xs = list(xs or [])[:n]
    return xs + [None] * (n - len(xs))


def load_progress(run_dir: Path, n_done: int) -> dict:
    """progress.json consistent with ``n_done`` completed (checkpointed) experiences.

    Lists: ``train_time_s_per_exp`` / ``peak_gpu_mem_mb_per_exp`` (one entry per training
    experience), ``eval_time_s`` / ``peak_gpu_mem_mb_eval`` (entry 0 = random-init eval, k =
    eval after training experience k-1). Entries beyond the checkpoint are dropped; missing
    ones (crash between checkpoint and progress write) are None. n_done == 0 -> fresh record.
    """
    n_done = int(n_done)
    prog: dict = {}
    path = Path(run_dir) / PROGRESS_FILE
    if n_done > 0 and path.exists():
        try:
            prog = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.warning("Unreadable %s; compute accounting restarts at this resume.", path)
            prog = {}
    n_eval = n_done + 1 if n_done > 0 else 0
    return {
        "n_done": n_done,
        "train_time_s_per_exp": _fit(prog.get("train_time_s_per_exp"), n_done),
        "peak_gpu_mem_mb_per_exp": _fit(prog.get("peak_gpu_mem_mb_per_exp"), n_done),
        "eval_time_s": _fit(prog.get("eval_time_s"), n_eval),
        "peak_gpu_mem_mb_eval": _fit(prog.get("peak_gpu_mem_mb_eval"), n_eval),
        "wall_clock_s_committed": float(prog.get("wall_clock_s_committed") or 0.0),
        "attempts": int(prog.get("attempts") or 0) + 1,
        # deterministic mode of every attempt (``deterministic`` is a volatile config key)
        "deterministic_modes": list(prog.get("deterministic_modes") or []),
        "wandb_history": list(prog.get("wandb_history") or []),
    }


def _set_at(xs: list, i: int, v) -> None:
    while len(xs) <= i:
        xs.append(None)
    xs[i] = v


def _measure(device, fn) -> tuple[float, float | None]:
    """Run ``fn``; return (wall seconds, CUDA peak MiB since reset or None on CPU)."""
    import torch

    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    fn()
    if cuda:
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None
    return dt, peak


def build_summary(
    matrix: dict,
    peak_per_exp: list,
    time_per_exp: list,
    clear: bool,
    *,
    peak_eval: list | None = None,
    wall_clock_s_total: float | None = None,
    per_class: dict | None = None,
) -> dict:
    from clns.evaluation.matrix_metrics import macro_matrix, matrix_metrics

    m = matrix_metrics(
        matrix["rows"],
        b=matrix["random_init"],
        scenario=matrix["scenario"],
        test_exp_to_train_exp=matrix["test_exp_to_train_exp"],
        clear=clear,
    )
    acc_final_macro = None
    pc_rows = (per_class or {}).get("rows")
    if pc_rows and len(pc_rows) == len(matrix["rows"]) and all(r is not None for r in pc_rows):
        macro = macro_matrix(pc_rows)
        if all(v is not None for row in macro for v in row):
            acc_final_macro = matrix_metrics(
                macro,
                scenario=matrix["scenario"],
                test_exp_to_train_exp=matrix["test_exp_to_train_exp"],
            )["acc_final"]
    peaks = [p for p in peak_per_exp if p is not None]
    times = [t for t in time_per_exp if t is not None]
    epeaks = [p for p in (peak_eval or []) if p is not None]
    summary = {k: m[k] for k in SUMMARY_KEYS if k in m}
    summary.update(
        {
            "peak_gpu_mem_mb": max(peaks) if peaks else None,
            "peak_gpu_mem_mb_per_exp": peak_per_exp if peaks else [],
            "train_time_s_per_exp": time_per_exp,
            "train_time_s_total": float(sum(times)) if times else None,
            "acc_final_macro": acc_final_macro,
            "peak_gpu_mem_mb_eval": max(epeaks) if epeaks else None,
            "wall_clock_s_total": wall_clock_s_total,
        }
    )
    summary = {k: summary.get(k) for k in SUMMARY_KEYS}
    if clear:
        summary.update({k: v for k, v in m.items() if k.startswith("clear_")})
    return summary


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def run(cfg: DictConfig) -> int:
    """Run (or resume) one training job. Returns the process exit code."""
    t_attempt = time.perf_counter()
    run_dir = Path(str(cfg.run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    apply_model_overrides(cfg)

    old_term = _install_sigterm_handler()
    try:
        return _run_with_lock(cfg, run_dir, t_attempt)
    finally:
        if old_term is not None:
            try:
                signal.signal(signal.SIGTERM, old_term)
            except (ValueError, OSError):
                pass


def _run_with_lock(cfg: DictConfig, run_dir: Path, t_attempt: float) -> int:
    lock = RunDirLock(run_dir)
    try:
        lock.acquire()
    except RunLockError as exc:
        log.error("%s (exit %d)", exc, EXIT_LOCKED)
        return EXIT_LOCKED
    try:
        ok, differing = check_or_write_fingerprint(run_dir, cfg)
        if not ok:
            log.error(
                "Config fingerprint mismatch in %s: this run directory belongs to a run with a "
                "different configuration; differing keys: %s. Refusing to resume/skip (exit "
                "%d). Use another paths.output_dir (or move the old run away) -- see %s.",
                run_dir, differing, EXIT_CONFIG_MISMATCH, run_dir / FINGERPRINT_FILE,
            )
            return EXIT_CONFIG_MISMATCH
        if (run_dir / "DONE").exists():
            log.info("DONE exists in %s; nothing to do.", run_dir)
            return EXIT_OK
        return _run_locked(cfg, run_dir, t_attempt)
    finally:
        lock.release()


def _run_locked(cfg: DictConfig, run_dir: Path, t_attempt: float) -> int:
    import torch

    device = resolve_device(cfg.device)
    seed = int(cfg.seed)
    det_mode = seed_everything(seed, cfg.deterministic)

    write_provenance(
        run_dir, cfg, extra={"device": str(device), "deterministic_mode": det_mode}
    )
    loggers: list = []
    n_done = 0
    try:
        from clns.evaluation.evaluator import build_evaluator, build_loggers
        from clns.evaluation.plugins import truncate_jsonl
        from clns.models import build_model
        from clns.strategies import build_strategy

        bundle = build_benchmark(cfg.benchmark, str(cfg.paths.data_dir), seed)
        update_provenance(run_dir, "benchmark_meta", _jsonable(bundle.meta))
        train_stream = bundle.benchmark.train_stream
        test_stream = bundle.benchmark.test_stream
        n_train, n_test = len(train_stream), len(test_stream)
        mapping = list(bundle.meta.get("test_exp_to_train_exp", range(n_test)))
        if len(mapping) != n_test:
            raise ValueError(
                f"test_exp_to_train_exp has {len(mapping)} entries, test stream has {n_test}"
            )
        valid_stream = getattr(bundle.benchmark, "valid_stream", None)
        eval_valid = bool(cfg.train.get("eval_valid", False))
        if eval_valid and valid_stream is None:
            log.warning("train.eval_valid=true but there is no valid stream "
                        "(benchmark.val_fraction=0); nothing to evaluate.")
        if not eval_valid:
            valid_stream = None  # B3: never evaluated unless explicitly requested
        n_valid = len(valid_stream) if valid_stream is not None else None

        wandb_status: dict = {}
        loggers = build_loggers(cfg, run_dir, status=wandb_status)
        update_provenance(run_dir, "wandb", wandb_status)
        evaluator, plugins = build_evaluator(
            cfg, loggers, run_dir, scenario=bundle.scenario,
            test_exp_to_train_exp=mapping, n_test=n_test,
            train_classes=bundle.meta.get("train_classes"), num_classes=bundle.num_classes,
            n_valid=n_valid,
        )
        # Model init (ResNet here, or the ViT that LearningToPrompt builds inside
        # build_strategy) uses a seed independent of anything consumed above.
        seed_model_init(seed)
        model = build_model(cfg.model, bundle.num_classes)
        strategy = build_strategy(
            cfg.method, cfg.model, cfg.train, model, bundle.num_classes,
            evaluator, plugins, device, scenario=bundle.scenario, n_train_experiences=n_train,
        )

        ckpt = run_dir / "checkpoint.pt"
        strategy, n_done = load_checkpoint(strategy, ckpt, device, unique_objects=bundle.benchmark)
        # The checkpoint restores the first attempt's eval_mb_size; the current (volatile)
        # train.eval_mb_size must win (e.g. lowered after an eval OOM).
        old_eval_mb = reapply_eval_mb_size(strategy, cfg.train.eval_mb_size)
        log.info("eval_mb_size = %d (restored from checkpoint: %s)",
                 strategy.eval_mb_size, old_eval_mb if n_done > 0 else "n/a")
        matrix_plugin = _matrix_plugin(strategy, "test")
        valid_plugin = _matrix_plugin(strategy, "valid") if valid_stream is not None else None
        for p in (matrix_plugin, valid_plugin):
            if p is not None:
                p.run_dir = str(run_dir)
        if n_done > 0:
            log.info("Resumed from %s: %d/%d training experiences done.", ckpt, n_done, n_train)
            _swap_loggers(strategy, loggers)
            # Drop anything written after the last checkpoint (crash between eval and save).
            for p in (matrix_plugin, valid_plugin):
                if p is not None:
                    p.truncate(n_done)
                    p.write()
        # Always: n_done == 0 drops every line (the random-init eval is redone too).
        truncate_jsonl(run_dir / "metrics.jsonl", n_done)
        progress = load_progress(run_dir, n_done)
        progress["wandb_history"].append(wandb_status)
        progress["deterministic_modes"].append(det_mode)
        committed_before = progress["wall_clock_s_committed"]

        def _write_progress() -> None:
            progress["n_done"] = n_done
            progress["wall_clock_s_committed"] = committed_before + (
                time.perf_counter() - t_attempt
            )
            progress["wandb_fallback_offline"] = any(
                bool(w.get("wandb_fallback_offline")) for w in progress["wandb_history"]
            )
            atomic_write_json(run_dir / PROGRESS_FILE, progress)

        nw = int(cfg.train.get("num_workers", 0))
        if n_done == 0:
            # random-init row b_j (no class mask before training)
            edt, epeak = _measure(device, lambda: strategy.eval(test_stream, num_workers=nw))
            _set_at(progress["eval_time_s"], 0, edt)
            _set_at(progress["peak_gpu_mem_mb_eval"], 0, epeak)
            _write_progress()

        stop_after = cfg.train.get("stop_after_experiences", None)
        for i in range(n_done, n_train):
            if stop_after is not None and i >= int(stop_after):
                log.info("stop_after_experiences=%s reached; exiting without DONE.", stop_after)
                _wandb_finish(loggers, exit_code=0)
                return EXIT_OK
            exp = train_stream[i]
            assert exp.current_experience == i, (exp.current_experience, i)
            seed_experience(seed, i)  # B4: identical data order with or without resume
            # eval_streams=[]: no Avalanche periodic eval on the training experience.
            tdt, tpeak = _measure(
                device, lambda: strategy.train(exp, eval_streams=[], num_workers=nw)
            )
            # Evaluate the FULL test stream after every training experience.
            edt, epeak = _measure(device, lambda: strategy.eval(test_stream, num_workers=nw))
            if valid_stream is not None:
                strategy.eval(valid_stream, num_workers=nw)  # -> valid_matrix.json only
            save_checkpoint_atomic(strategy, ckpt)
            n_done = i + 1
            _set_at(progress["train_time_s_per_exp"], i, tdt)
            _set_at(progress["peak_gpu_mem_mb_per_exp"], i, tpeak)
            _set_at(progress["eval_time_s"], i + 1, edt)
            _set_at(progress["peak_gpu_mem_mb_eval"], i + 1, epeak)
            _write_progress()
            _log_extra(
                loggers,
                {
                    "train/exp_index": i,
                    "train/exp_time_s": tdt,
                    "train/exp_peak_gpu_mem_mb": tpeak,
                    "eval/time_s": edt,
                    "eval/peak_gpu_mem_mb": epeak,
                },
                n_trained=n_done,
            )
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        matrix = matrix_plugin.to_dict()
        if len(matrix["rows"]) != n_train or matrix["random_init"] is None:
            raise RuntimeError(
                f"Incomplete accuracy matrix ({len(matrix['rows'])}/{n_train} rows, "
                f"random_init={'set' if matrix['random_init'] is not None else 'missing'})"
            )
        _write_progress()
        summary = build_summary(
            matrix,
            progress["peak_gpu_mem_mb_per_exp"],
            progress["train_time_s_per_exp"],
            clear=str(cfg.benchmark.name) == "clear10",
            peak_eval=progress["peak_gpu_mem_mb_eval"],
            wall_clock_s_total=progress["wall_clock_s_committed"],
            per_class=matrix_plugin.per_class_dict() if matrix_plugin.track_per_class else None,
        )
        atomic_write_json(run_dir / "summary.json", summary)
        _log_extra(loggers, {f"summary/{k}": v for k, v in summary.items()
                             if isinstance(v, (int, float))}, n_trained=n_done)
        (run_dir / "DONE").write_text("done\n", encoding="utf-8")
        try:
            (run_dir / "failure.json").unlink()
        except FileNotFoundError:
            pass
        log.info("Finished %s", run_dir)
        _wandb_finish(loggers, summary, exit_code=0)
        return EXIT_OK

    except BaseException as exc:  # noqa: BLE001 - always leave a failure record
        if isinstance(exc, KeyboardInterrupt):
            raise
        interrupted = isinstance(exc, Terminated)
        oom = False if interrupted else _is_oom(exc)
        atomic_write_json(
            run_dir / "failure.json",
            {
                "type": type(exc).__name__,
                "message": "SIGTERM" if interrupted else str(exc)[:4000],
                "oom": oom,
                "interrupted": interrupted,
                "completed_experiences": n_done,
                "traceback": traceback.format_exc()[-20000:],
            },
        )
        if interrupted:
            _wandb_finish(loggers, exit_code=EXIT_SIGTERM, preempting=True)
            raise
        _wandb_finish(loggers, exit_code=EXIT_OOM if oom else EXIT_ERROR)
        if oom:
            log.error("CUDA out of memory after %d completed experiences: %s", n_done, exc)
            return EXIT_OOM
        raise
    finally:
        _close_loggers(loggers)


def _jsonable(obj):
    """Best-effort conversion of benchmark meta to JSON-serialisable values."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_jsonable(v) for v in obj]
        return str(obj)


_RESULT = {"code": EXIT_OK}


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def _hydra_main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    _RESULT["code"] = run(cfg)


def main() -> None:
    """Console entry point; ``hydra.main`` discards return values, so exit codes go via _RESULT."""
    _hydra_main()
    sys.exit(_RESULT["code"])


if __name__ == "__main__":
    main()
