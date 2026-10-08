"""Custom Avalanche plugins and loggers.

Plugin hooks follow ``avalanche.core.BasePlugin/SupervisedPlugin`` (v0.6.0):
``before/after_training_exp(strategy, **kwargs)``, ``before/after_eval(strategy, **kwargs)``,
``after_eval_exp(strategy, **kwargs)`` -- called in the order of ``strategy.plugins``.
``BaseSGDTemplate.__init__`` appends, after the user plugins: the EvaluationPlugin, PeriodicEval
and finally the Clock. Hence, inside OUR hooks:
  * ``strategy.clock.train_exp_counter`` is still ``i`` during/at the end of training exp ``i``
    (the Clock increments it in its own after_training_exp, which runs last);
  * at ``after_eval`` all per-experience metric values of that eval call are already in
    ``strategy.evaluator.get_last_metrics()`` (emitted by the EvaluationPlugin in each
    ``after_eval_exp``), while the stream-level values are not yet.
https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/core.py
https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/training/templates/base_sgd.py

Logger hooks are called by the EvaluationPlugin as ``hook(strategy, metric_values, **kwargs)``
AFTER ``logger.log_metrics(metric_values)`` for the same callback.
https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/training/plugins/evaluation.py

All plugin objects live in ``strategy.plugins`` and are pickled into checkpoint.pt; after a
resume, ``train.py`` looks them up again with :func:`find_plugin` (the unpickled instances
carry the state of the interrupted run).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from avalanche.core import SupervisedPlugin
from avalanche.logging import BaseLogger, TextLogger, WandBLogger

from clns.utils.checkpoint import atomic_write_json, atomic_write_text

log = logging.getLogger(__name__)

# Metric name of avalanche ExperienceAccuracy: "Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp003"
# (default_metric_name_template; the Task part is omitted when task labels are unavailable).
# https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/evaluation/metric_utils.py
EXP_ACC_RE = re.compile(
    r"^Top1_Acc_Exp/eval_phase/(?P<stream>[^/]+)_stream(?:/Task\d+)?/Exp(?P<exp>\d+)$"
)


def find_plugin(strategy, cls):
    for p in strategy.plugins:
        if isinstance(p, cls):
            return p
    return None


def _cuda_device(strategy) -> torch.device | None:
    dev = getattr(strategy, "device", None)
    if dev is not None and torch.device(dev).type == "cuda" and torch.cuda.is_available():
        return torch.device(dev)
    return None


# --------------------------------------------------------------------------- #
# Accuracy matrix (+ per-class counts)
# --------------------------------------------------------------------------- #
class _TrainingFlagMixin:
    """Tracks whether ``strategy.train()`` is running.

    Avalanche 0.6.0 ``BaseTemplate.eval`` sets ``strategy.is_training = False`` for the duration
    of ANY eval call, including the periodic evals that ``PeriodicEval`` triggers inside
    ``train()`` (the flag is restored only after ``_after_eval``), so ``is_training`` alone cannot
    tell a nested eval from a top-level one. ``before_training``/``after_training`` bracket the
    whole ``train()`` call (avalanche/training/templates/base.py), so this flag can.
    """

    _in_train: bool = False

    def before_training(self, strategy, **kwargs):
        self._in_train = True

    def after_training(self, strategy, **kwargs):
        self._in_train = False


class AccuracyMatrixPlugin(_TrainingFlagMixin, SupervisedPlugin):
    """Builds ``accuracy_matrix.json``: ``rows[i][j]`` = accuracy on test exp ``j`` after training
    exp ``i``; ``random_init[j]`` = accuracy before any training.

    A row is recorded only for a TOP-LEVEL eval call (``not strategy.is_training`` and not
    inside ``strategy.train()``, i.e. never for a periodic eval; see
    :class:`_TrainingFlagMixin`) that covered the FULL stream ``stream_name`` (every experience
    once); evaluations on other streams are ignored (amendment B3). The row index is
    ``clock.train_exp_counter - 1`` at eval time (0 -> random_init).

    With ``track_per_class=True`` it also counts, per (row, test exp), correct predictions and
    samples per TRUE class (argmax of ``strategy.mb_output`` after any eval-time class mask, the
    same prediction Avalanche's accuracy uses) and writes ``per_class_filename``:
    ``{"rows": [[{"classes": [...], "correct": [...], "total": [...]}, ...], ...],
    "random_init": [...]}``. Macro (class-balanced) accuracy is derived offline
    (``matrix_metrics.macro_matrix``).
    """

    def __init__(
        self,
        run_dir: str | Path,
        n_test: int,
        test_exp_to_train_exp: Sequence[int],
        scenario: str,
        stream_name: str = "test",
        filename: str = "accuracy_matrix.json",
        track_per_class: bool = True,
        per_class_filename: str = "per_class_matrix.json",
    ):
        super().__init__()
        self.run_dir = str(run_dir)
        self.n_test = int(n_test)
        self.test_exp_to_train_exp = [int(m) for m in test_exp_to_train_exp]
        self.scenario = scenario
        self.stream_name = stream_name
        self.filename = filename
        self.track_per_class = bool(track_per_class)
        self.per_class_filename = per_class_filename
        self.rows: list[list[float | None]] = []
        self.random_init: list[float | None] | None = None
        self.per_class_rows: list[list[dict] | None] = []
        self.per_class_random_init: list[dict] | None = None
        self._evaluated: list[int] = []
        self._counts: dict[int, dict[int, list[int]]] = {}
        self._in_train = False

    @property
    def path(self) -> Path:
        return Path(self.run_dir) / self.filename

    @property
    def per_class_path(self) -> Path:
        return Path(self.run_dir) / self.per_class_filename

    def _top_level(self, strategy) -> bool:
        return not self._in_train and not bool(getattr(strategy, "is_training", False))

    def _is_mine(self, strategy) -> bool:
        exp = strategy.experience
        return exp is not None and exp.origin_stream.name == self.stream_name

    def before_eval(self, strategy, **kwargs):
        if self._top_level(strategy):
            self._evaluated = []
            self._counts = {}

    def after_eval_iteration(self, strategy, **kwargs):
        if not (self.track_per_class and self._top_level(strategy) and self._is_mine(strategy)):
            return
        out = strategy.mb_output
        if not isinstance(out, torch.Tensor) or out.dim() != 2:
            return
        # Count on CPU: CUDA bincount has no deterministic implementation (strict mode raises).
        pred = out.detach().argmax(dim=1).cpu()
        y = strategy.mb_y.detach().cpu().long()
        if y.numel() == 0:
            return
        correct = (pred == y).to(torch.float64)
        n_cls = int(y.max().item()) + 1
        tot = torch.bincount(y, minlength=n_cls).tolist()
        cor = torch.bincount(y, weights=correct, minlength=n_cls).tolist()
        d = self._counts.setdefault(int(strategy.experience.current_experience), {})
        for c in range(n_cls):
            if tot[c]:
                acc = d.setdefault(c, [0, 0])
                acc[0] += int(round(cor[c]))
                acc[1] += int(tot[c])

    def after_eval_exp(self, strategy, **kwargs):
        if self._top_level(strategy) and self._is_mine(strategy):
            self._evaluated.append(int(strategy.experience.current_experience))

    def after_eval(self, strategy, **kwargs):
        if not self._top_level(strategy):
            return  # nested (periodic) eval inside train(): never a matrix row
        evaluated, self._evaluated = self._evaluated, []
        counts, self._counts = self._counts, {}
        if not evaluated:
            return
        if sorted(evaluated) != list(range(self.n_test)):
            log.warning(
                "AccuracyMatrixPlugin(%s): eval covered experiences %s, not the full stream "
                "(%d); row not recorded.", self.stream_name, sorted(evaluated), self.n_test
            )
            return
        values: dict[int, float] = {}
        for name, val in strategy.evaluator.get_last_metrics().items():
            m = EXP_ACC_RE.match(name)
            if m and m.group("stream") == self.stream_name:
                values[int(m.group("exp"))] = float(val)
        row = [values.get(j) for j in range(self.n_test)]
        if any(v is None for v in row):
            raise RuntimeError(
                "AccuracyMatrixPlugin: Top1_Acc_Exp missing for some experiences; "
                "is accuracy_metrics(experience=True) attached to the evaluator?"
            )
        pc_row = None
        if self.track_per_class:
            pc_row = []
            for j in range(self.n_test):
                cj = counts.get(j, {})
                cls = sorted(cj)
                pc_row.append(
                    {
                        "classes": cls,
                        "correct": [cj[c][0] for c in cls],
                        "total": [cj[c][1] for c in cls],
                    }
                )
        n_trained = int(strategy.clock.train_exp_counter)
        if n_trained == 0:
            self.random_init = row
            self.per_class_random_init = pc_row
        else:
            while len(self.rows) < n_trained:
                self.rows.append([None] * self.n_test)
            while len(self.per_class_rows) < n_trained:
                self.per_class_rows.append(None)
            self.rows[n_trained - 1] = row
            self.per_class_rows[n_trained - 1] = pc_row
        self.write()

    def truncate(self, n_rows: int) -> None:
        """Drop rows beyond ``n_rows`` (rows written after the last checkpoint)."""
        self.rows = self.rows[: int(n_rows)]
        self.per_class_rows = self.per_class_rows[: int(n_rows)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": "Top1_Acc_Exp",
            "stream": self.stream_name,
            "scenario": self.scenario,
            "n_train_rows": len(self.rows),
            "n_test": self.n_test,
            "test_exp_to_train_exp": self.test_exp_to_train_exp,
            "random_init": self.random_init,
            "rows": self.rows,
        }

    def per_class_dict(self) -> dict[str, Any]:
        return {
            "stream": self.stream_name,
            "scenario": self.scenario,
            "n_test": self.n_test,
            "test_exp_to_train_exp": self.test_exp_to_train_exp,
            "prediction": "argmax of the logits after the eval-time class mask (if any)",
            "random_init": self.per_class_random_init,
            "rows": self.per_class_rows,
        }

    def write(self) -> None:
        atomic_write_json(self.path, self.to_dict())
        if self.track_per_class:
            atomic_write_json(self.per_class_path, self.per_class_dict())

    def __setstate__(self, state):
        # Tolerate checkpoints written before per-class tracking existed.
        state.setdefault("per_class_rows", [])
        state.setdefault("per_class_random_init", None)
        state.setdefault("track_per_class", False)
        state.setdefault("per_class_filename", "per_class_matrix.json")
        state.setdefault("filename", "accuracy_matrix.json")
        state.setdefault("_counts", {})
        state["_in_train"] = False
        self.__dict__.update(state)


# --------------------------------------------------------------------------- #
# Class-IL eval-time masking of not-yet-seen classes
# --------------------------------------------------------------------------- #
class EvalClassMaskPlugin(_TrainingFlagMixin, SupervisedPlugin):
    """Class-IL evaluation over "the global head of all classes seen so far" (RESEARCH_LOG 7.2).

    Hook: ``after_eval_forward``. In Avalanche 0.6.0 ``BaseSGDTemplate.eval_epoch`` runs
    ``mb_output = forward()`` -> ``_after_eval_forward`` -> ``criterion()`` ->
    ``_after_eval_iteration`` (where the accuracy metrics are updated), so replacing
    ``strategy.mb_output`` here affects loss and accuracy. Logits of classes NOT in the union of
    the classes of training experiences ``0..n-1`` are set to ``-inf``, with
    n = ``clock.train_exp_counter`` = completed training experiences (during a nested/periodic
    eval inside ``train()`` the experience being trained counts as seen: n + 1).
    Consequences (documented):
      * before any training (random-init row ``b``) NO mask is applied: all classes compete;
      * test experiences made only of unseen classes get accuracy exactly 0 and an infinite
        loss (non-finite values are skipped by JSONLLogger);
      * applies to every method, including L2P: Avalanche 0.6.0 ``LearningToPrompt.forward``
        masks to ``classes_in_this_experience`` only while TRAINING (``use_mask and
        is_training``), never at evaluation.
    Attached only for class-IL and ``train.eval_mask_unseen=true`` (default).
    """

    def __init__(self, train_classes: Sequence[Sequence[int]], num_classes: int):
        super().__init__()
        self.train_classes = [sorted(int(c) for c in cs) for cs in train_classes]
        self.num_classes = int(num_classes)
        self._mask: torch.Tensor | None = None  # bool [num_classes]; True = masked (unseen)
        self._mask_dev: torch.Tensor | None = None
        self._in_train = False

    def seen_classes(self, n_trained: int) -> list[int]:
        seen: set[int] = set()
        for cs in self.train_classes[: max(0, int(n_trained))]:
            seen.update(cs)
        return sorted(seen)

    def before_eval(self, strategy, **kwargs):
        n = int(strategy.clock.train_exp_counter) + (1 if self._in_train else 0)
        self._mask_dev = None
        if n <= 0:
            self._mask = None
            return
        mask = torch.ones(self.num_classes, dtype=torch.bool)
        seen = [c for c in self.seen_classes(n) if 0 <= c < self.num_classes]
        if seen:
            mask[torch.tensor(seen, dtype=torch.long)] = False
        self._mask = mask if bool(mask.any()) else None

    def after_eval_forward(self, strategy, **kwargs):
        if self._mask is None:
            return
        out = strategy.mb_output
        if not isinstance(out, torch.Tensor) or out.dim() != 2 or out.shape[1] != self.num_classes:
            return
        if self._mask_dev is None or self._mask_dev.device != out.device:
            self._mask_dev = self._mask.to(out.device)
        strategy.mb_output = out.masked_fill(self._mask_dev.unsqueeze(0), float("-inf"))

    def after_eval(self, strategy, **kwargs):
        self._mask_dev = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mask_dev"] = None
        state["_in_train"] = False
        return state


# --------------------------------------------------------------------------- #
# Resource / timing plugins -- DEPRECATED, not attached by build_evaluator.
# train.py measures wall clock and CUDA peak memory around strategy.train()/strategy.eval()
# (single source of truth; it includes the EWC Fisher computation, the replay-buffer update,
# the LwF teacher copy and evaluation, which these per-experience hooks missed). Kept only so
# that checkpoints that contain them can still be unpickled.
# --------------------------------------------------------------------------- #
class PeakGPUMemoryPlugin(SupervisedPlugin):
    """DEPRECATED (see section comment). Peak MiB per training experience."""

    def __init__(self):
        super().__init__()
        self.peak_mb: dict[int, float] = {}

    def before_training_exp(self, strategy, **kwargs):
        dev = _cuda_device(strategy)
        if dev is not None:
            torch.cuda.reset_peak_memory_stats(dev)

    def after_training_exp(self, strategy, **kwargs):
        dev = _cuda_device(strategy)
        if dev is not None:
            exp_id = int(strategy.clock.train_exp_counter)
            self.peak_mb[exp_id] = torch.cuda.max_memory_allocated(dev) / 2**20

    def per_exp(self, n: int) -> list[float | None]:
        return [self.peak_mb.get(i) for i in range(n)]


class WallClockPlugin(SupervisedPlugin):
    """DEPRECATED (see section comment). Seconds per training experience."""

    def __init__(self):
        super().__init__()
        self.seconds: dict[int, float] = {}
        self._t0: float | None = None

    def before_training_exp(self, strategy, **kwargs):
        self._t0 = time.perf_counter()

    def after_training_exp(self, strategy, **kwargs):
        if self._t0 is None:
            return
        self.seconds[int(strategy.clock.train_exp_counter)] = time.perf_counter() - self._t0
        self._t0 = None

    def per_exp(self, n: int) -> list[float | None]:
        return [self.seconds.get(i) for i in range(n)]


# --------------------------------------------------------------------------- #
# Loggers
# --------------------------------------------------------------------------- #
def _to_scalar(value) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value if not (isinstance(value, float) and not math.isfinite(value)) else None
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return _to_scalar(value.item())
    return None


class JSONLLogger(BaseLogger):
    """Appends one JSON object per scalar metric value to ``run_dir/metrics.jsonl``.

    Fields: ``name, value, x`` (avalanche x_plot), ``phase`` (train|eval|other),
    ``n_trained`` (= clock.train_exp_counter when logged: index of the experience being trained
    for train-phase values; number of completed experiences for eval values), ``time``.
    The file handle is not pickled (re-opened lazily in append mode after a resume).
    """

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = str(path)
        self._n_trained = 0
        self._fh = None

    # -- state tracking (logger hooks receive (strategy, metric_values)) --
    def before_training(self, strategy, metric_values, **kwargs):
        self._n_trained = int(strategy.clock.train_exp_counter)

    def before_training_exp(self, strategy, metric_values, **kwargs):
        self._n_trained = int(strategy.clock.train_exp_counter)

    def before_eval(self, strategy, metric_values, **kwargs):
        self._n_trained = int(strategy.clock.train_exp_counter)

    # -- writing --
    def _handle(self):
        if self._fh is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        return self._fh

    def _write(self, rec: dict) -> None:
        fh = self._handle()
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    def log_single_metric(self, name, value, x_plot):
        v = _to_scalar(value)
        if v is None:
            return
        if "/train_phase/" in name:
            phase = "train"
        elif "/eval_phase/" in name:
            phase = "eval"
        else:
            phase = "other"
        self._write(
            {
                "name": name,
                "value": v,
                "x": int(x_plot),
                "phase": phase,
                "n_trained": self._n_trained,
                "time": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
        )

    def log_dict(self, values: dict, n_trained: int, phase: str = "other") -> None:
        """Log extra scalars produced outside avalanche metrics (timing, GPU memory, summary)."""
        for k, val in values.items():
            v = _to_scalar(val)
            if v is None:
                continue
            self._write(
                {
                    "name": k,
                    "value": v,
                    "x": None,
                    "phase": phase,
                    "n_trained": int(n_trained),
                    "time": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                }
            )

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fh = None


def truncate_jsonl(path: str | Path, n_completed: int) -> None:
    """Drop metrics.jsonl lines produced after the last checkpoint (resume after a crash).

    ``n_completed == 0`` (no checkpoint: the run restarts from scratch, including the
    random-init evaluation) drops EVERYTHING. Otherwise keeps train-phase lines of experiences
    < n_completed and eval/other lines logged with ``n_trained <= n_completed``.
    """
    path = Path(path)
    if not path.exists():
        return
    if int(n_completed) <= 0:
        atomic_write_text(path, "")
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # partially written line from a crash
        n = int(rec.get("n_trained", 0))
        if rec.get("phase") == "train":
            keep = n < n_completed
        else:
            keep = n <= n_completed
        if keep:
            kept.append(line)
    atomic_write_text(path, "".join(f"{x}\n" for x in kept))


class ClnsTextLogger(TextLogger):
    """TextLogger whose file handle is NOT pickled into checkpoint.pt.

    Upstream ``TextLogger.__getstate__`` stores ``"path:<absolute path>"`` and ``__setstate__``
    re-opens that path in append mode (and prints a resume banner into it), so a checkpoint
    resumed elsewhere would write to (or fail on) a stale path. Here the unpickled logger has
    ``file=None``; train.py replaces it right after loading by a fresh
    ``ClnsTextLogger(run_dir/log.txt)`` (``_swap_loggers``).
    https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/logging/text_logging.py
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        state["file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.file = None


class ClnsWandBLogger(WandBLogger):
    """WandBLogger whose unpickling does NOT call ``wandb.init`` again.

    Upstream ``WandBLogger.__setstate__`` calls ``before_run()`` (i.e. ``wandb.init``) when the
    strategy is restored from a checkpoint
    (https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/logging/wandb_logger.py).
    In ``train.py`` a fresh logger (same run id, ``resume="allow"``) is created before the
    checkpoint is loaded and replaces the unpickled loggers, so a second init must not happen.
    """

    def __setstate__(self, state):
        self.__dict__ = state
        self.wandb = None
        try:
            self.import_wandb()
        except ImportError:  # pragma: no cover
            pass


__all__ = [
    "AccuracyMatrixPlugin",
    "EvalClassMaskPlugin",
    "ClnsTextLogger",
    "PeakGPUMemoryPlugin",
    "WallClockPlugin",
    "JSONLLogger",
    "ClnsWandBLogger",
    "truncate_jsonl",
    "find_plugin",
    "EXP_ACC_RE",
]
