"""Evaluator (Avalanche EvaluationPlugin) and logger construction.

Evaluation protocol (RESEARCH_LOG 7.2, amendments B1/B3)
------------------------------------------------------
``train.py`` calls ``strategy.eval(test_stream)`` on the FULL test stream once before any
training (random-init row ``b``) and once after every training experience. ``valid_stream`` is
evaluated only if ``train.eval_valid=true`` (after every training experience, written to
``valid_matrix.json``) and is never used for anything automatic.
Class-IL: logits of not-yet-seen classes are masked at evaluation (``train.eval_mask_unseen``,
default true; :class:`clns.evaluation.plugins.EvalClassMaskPlugin`). Per-class counts ->
``per_class_matrix.json``; Avalanche ``class_accuracy_metrics(stream=True)`` is logged too.
``strategy.train`` is called with ``eval_streams=[]`` so Avalanche's PeriodicEval never
evaluates on the training experience.

Metric reset semantics (avalanche v0.6.0 ``GenericPluginMetric``; ``reset_at`` / ``emit_at``)
  * ``Top1_Acc_Exp`` / ``Loss_Exp`` / ``Time_Exp`` (``experience=True``): mode eval, reset at
    ``before_eval_exp`` and emitted at ``after_eval_exp`` -> each value covers exactly one test
    experience of the current eval call; values never mix experiences or eval calls.
  * ``*_Stream`` (``stream=True``): reset at ``before_eval``, emitted at ``after_eval`` ->
    sample-weighted average over the whole test stream (NOT the mean over experiences, and it
    includes not-yet-seen experiences; the paper metrics come from ``matrix_metrics``).
  * ``*_Epoch`` (``epoch=True``): mode train, reset at ``before_training_epoch``, emitted at
    ``after_training_epoch`` (running training accuracy/loss/time of that epoch).
  * ``ExperienceForgetting`` / ``StreamForgetting``: Avalanche stores the value of test exp
    ``k`` from the eval in which ``train_exp_id == eval_exp_id`` (current_experience indices) as
    the baseline, and reports ``baseline - current``; i.e. "first-eval minus current", NOT the
    max-based forgetting of RESEARCH_LOG 7.2 (excluding the last row). The max-based value and
    the CORe50 mapping-aware variants are computed offline by
    :func:`clns.evaluation.matrix_metrics.matrix_metrics` (summary.json). For CORe50 the online
    Avalanche forgetting/BWT/FWT use test-exp index == train-exp index, which is WRONG when the
    test stream is partitioned by first appearance; they are therefore NOT attached when
    ``test_exp_to_train_exp`` is not the identity (nor when the valid stream is evaluated).
  * ``BWT`` = -forgetting (same baseline convention).
  * ``ExperienceForwardTransfer``: needs an evaluation before training (``at_init``) and
    ``strategy.eval_every > -1`` (asserted upstream), so it is only attached when
    ``train.eval_every > -1``; ``train.yaml`` sets 0, which together with ``eval_streams=[]``
    causes no extra evaluation.
Sources:
https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/evaluation/metrics/accuracy.py
https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/evaluation/metrics/forgetting_bwt.py
https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/evaluation/metrics/forward_transfer.py
https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/training/plugins/evaluation.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    bwt_metrics,
    class_accuracy_metrics,
    forgetting_metrics,
    forward_transfer_metrics,
    loss_metrics,
    ram_usage_metrics,
    timing_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

from .plugins import (
    AccuracyMatrixPlugin,
    ClnsTextLogger,
    ClnsWandBLogger,
    EvalClassMaskPlugin,
    JSONLLogger,
)

log = logging.getLogger(__name__)

WANDB_MODES = ("online", "offline", "disabled")
# Seconds wandb.init waits for the W&B service/backend before raising (wandb >= 0.16 reads the
# WANDB_INIT_TIMEOUT environment variable into Settings.init_timeout; "(verify)" on the pinned
# version). Only set when the user did not set it.
DEFAULT_WANDB_INIT_TIMEOUT_S = "120"


def run_name(cfg) -> str:
    return f"{cfg.benchmark.name}/{cfg.model.name}/{cfg.method.name}/seed{cfg.seed}"


def _wandb_run_id(run_dir: Path) -> str:
    """Persistent W&B run id (``wandb_run_id.txt``) so resumes continue the same W&B run."""
    p = run_dir / "wandb_run_id.txt"
    if p.exists():
        rid = p.read_text(encoding="utf-8").strip()
        if rid:
            return rid
    import wandb

    rid = wandb.util.generate_id()
    p.write_text(rid + "\n", encoding="utf-8")
    return rid


def resolve_wandb_mode(configured) -> tuple[str, str]:
    """Return ``(mode, reason)`` for ``logger.mode``.

    ``auto`` (default in configs/logger/wandb.yaml): ``$WANDB_MODE`` if set; otherwise
    ``online`` iff ``$WANDB_API_KEY`` is set; otherwise ``offline`` (with a warning). An explicit
    online/offline/disabled is used as is.
    """
    mode = str(configured if configured is not None else "auto").strip().lower()
    if mode != "auto":
        if mode not in WANDB_MODES:
            raise ValueError(f"logger.mode={configured!r}; expected auto or one of {WANDB_MODES}")
        return mode, "logger.mode"
    env_mode = os.environ.get("WANDB_MODE", "").strip().lower()
    if env_mode:
        if env_mode not in WANDB_MODES:
            raise ValueError(f"WANDB_MODE={env_mode!r}; expected one of {WANDB_MODES}")
        return env_mode, "WANDB_MODE"
    if os.environ.get("WANDB_API_KEY", "").strip():
        return "online", "WANDB_API_KEY set"
    log.warning(
        "W&B: logger.mode=auto, WANDB_MODE unset and WANDB_API_KEY unset -> offline. "
        "Sync later with `wandb sync <run_dir>/wandb/offline-run-*`."
    )
    return "offline", "no WANDB_API_KEY"


def _make_wandb_logger(cfg, run_dir: Path, mode: str):
    from omegaconf import OmegaConf

    lcfg = cfg.logger
    params = {
        "id": _wandb_run_id(run_dir),
        "resume": "allow",
        "mode": mode,
        "entity": lcfg.get("entity", None),
        "group": lcfg.get("group", None),
        "tags": list(lcfg.get("tags", []) or []),
    }
    return ClnsWandBLogger(
        project_name=str(lcfg.project),
        run_name=run_name(cfg),
        dir=str(run_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        params=params,
    )


def _wandb_finish_quietly() -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish(exit_code=1)
    except Exception:  # noqa: BLE001
        pass


def build_loggers(cfg, run_dir: Path, status: dict | None = None) -> list:
    """Interactive (stdout) + ClnsTextLogger(run_dir/log.txt) + JSONL (+ W&B unless logger=none).

    W&B robustness (outage / auth / timeout): ``wandb.init`` (called in the WandBLogger
    constructor) is wrapped; on failure the error is logged, any half-initialised run is
    finished and construction is retried ONCE with ``mode=offline``; if that fails too, the run
    continues without W&B (local metrics.jsonl / log.txt are unaffected). ``status`` (if given)
    is filled with ``wandb_mode`` (resolved), ``wandb_mode_reason``, ``wandb_fallback_offline``,
    ``wandb_error`` and ``wandb_disabled`` -- train.py stores it in progress.json and
    provenance.json. The resolved mode is logged as the fixed line ``W&B mode: <mode>``
    (``none`` for logger=none; ``offline`` after a fallback; ``disabled`` if W&B is unusable).
    """
    run_dir = Path(run_dir)
    status = status if status is not None else {}
    loggers: list = [
        InteractiveLogger(),
        ClnsTextLogger(open(run_dir / "log.txt", "a", encoding="utf-8")),
        JSONLLogger(run_dir / "metrics.jsonl"),
    ]
    lcfg = cfg.logger
    status.update(
        wandb_mode="none", wandb_mode_reason="logger=none", wandb_fallback_offline=False,
        wandb_error=None, wandb_disabled=False,
    )
    if str(lcfg.name) == "none":
        log.info("W&B mode: none")
        return loggers

    mode, reason = resolve_wandb_mode(lcfg.get("mode", "auto"))
    status.update(wandb_mode=mode, wandb_mode_reason=reason)
    os.environ.setdefault("WANDB_INIT_TIMEOUT", DEFAULT_WANDB_INIT_TIMEOUT_S)
    try:
        wb = _make_wandb_logger(cfg, run_dir, mode)
    except Exception as exc:  # noqa: BLE001 - network/auth/timeout must not kill training
        _wandb_finish_quietly()
        status["wandb_error"] = f"{type(exc).__name__}: {exc}"[:2000]
        if mode == "offline":
            wb = None
        else:
            log.warning("W&B init failed in mode=%s (%s); retrying once with mode=offline.",
                        mode, status["wandb_error"])
            status["wandb_fallback_offline"] = True
            mode = "offline"
            status["wandb_mode"] = mode
            try:
                wb = _make_wandb_logger(cfg, run_dir, mode)
            except Exception as exc2:  # noqa: BLE001
                _wandb_finish_quietly()
                status["wandb_error"] += f" | offline retry: {type(exc2).__name__}: {exc2}"[:2000]
                wb = None
        if wb is None:
            status.update(wandb_mode="disabled", wandb_disabled=True)
            mode = "disabled"
            log.error("W&B unusable (%s); continuing WITHOUT W&B (metrics.jsonl/log.txt only).",
                      status["wandb_error"])
    log.info("W&B mode: %s", mode)
    if wb is not None:
        loggers.append(wb)
    return loggers


def build_evaluator(
    cfg,
    loggers: list,
    run_dir: Path,
    scenario: str = "class_il",
    test_exp_to_train_exp: Sequence[int] | None = None,
    n_test: int | None = None,
    train_classes: Sequence[Sequence[int]] | None = None,
    num_classes: int | None = None,
    n_valid: int | None = None,
) -> tuple[EvaluationPlugin, list]:
    """Return ``(EvaluationPlugin, plugins)`` -- the plugins go to the strategy's ``plugins``.

    plugins = [EvalClassMaskPlugin (class-IL and ``train.eval_mask_unseen``, needs
    ``train_classes`` + ``num_classes``), AccuracyMatrixPlugin(test, with per-class counts),
    AccuracyMatrixPlugin(valid -> valid_matrix.json; only if ``n_valid``)].
    The mask plugin comes first so that every later ``after_eval_iteration`` consumer (per-class
    counts, Avalanche accuracy in the EvaluationPlugin, which runs after all user plugins) sees
    the masked logits.
    """
    if n_test is None:
        if test_exp_to_train_exp is None:
            raise ValueError("build_evaluator needs n_test or test_exp_to_train_exp")
        n_test = len(test_exp_to_train_exp)
    if test_exp_to_train_exp is None:
        test_exp_to_train_exp = list(range(n_test))
    identity = [int(m) for m in test_exp_to_train_exp] == list(range(n_test))

    metrics = [
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, experience=True, stream=True),
        class_accuracy_metrics(stream=True),
    ]
    # Avalanche forgetting/BWT/FWT key their state by experience index only (not by stream)
    # and pair test exp k with train exp k. They are wrong when the test stream is partitioned
    # differently (CORe50 first-appearance groups) and would mix test and valid values when
    # the valid stream is evaluated too -> not attached in those cases (summary.json has the
    # correct mapping-aware values from matrix_metrics).
    if not identity:
        log.info("test_exp_to_train_exp is not the identity: Avalanche forgetting/BWT/FWT "
                 "metrics not attached (see summary.json).")
    elif n_valid:
        log.info("train.eval_valid: Avalanche forgetting/BWT/FWT metrics not attached (they "
                 "would mix valid and test experiences).")
    else:
        metrics += [
            forgetting_metrics(experience=True, stream=True),
            bwt_metrics(experience=True, stream=True),
        ]
        eval_every = int(cfg.train.get("eval_every", -1))
        if eval_every > -1:
            metrics.append(forward_transfer_metrics(experience=True, stream=True))
        else:
            log.warning("train.eval_every=-1: Avalanche forward-transfer metrics not attached "
                        "(FWT is still computed offline in summary.json).")
    if bool(cfg.train.get("log_ram", False)):
        metrics.append(ram_usage_metrics(every=1, epoch=True))

    # collect_all=False: the evaluator is pickled into every checkpoint; full metric
    # histories live in metrics.jsonl / W&B instead.
    evaluator = EvaluationPlugin(*metrics, loggers=loggers, collect_all=False)
    plugins: list = []
    if scenario == "class_il" and bool(cfg.train.get("eval_mask_unseen", True)):
        if train_classes is None or num_classes is None:
            raise ValueError("eval_mask_unseen needs train_classes and num_classes")
        plugins.append(EvalClassMaskPlugin(train_classes, num_classes))
    plugins.append(AccuracyMatrixPlugin(run_dir, n_test, test_exp_to_train_exp, scenario))
    if n_valid:
        plugins.append(
            AccuracyMatrixPlugin(
                run_dir, int(n_valid), list(range(int(n_valid))), scenario,
                stream_name="valid", filename="valid_matrix.json", track_per_class=False,
            )
        )
    return evaluator, plugins


__all__ = ["build_evaluator", "build_loggers", "resolve_wandb_mode", "run_name"]
