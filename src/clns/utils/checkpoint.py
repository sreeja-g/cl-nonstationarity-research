"""Atomic per-experience checkpointing around Avalanche's checkpoint helpers, plus atomic file I/O.

Avalanche v0.6.0 ``save_checkpoint(strategy, fname, exclude=None)`` dill-pickles
``{"strategy": copy(strategy), "rng_manager": RNGManager, "exp_counter":
strategy.clock.train_exp_counter}``; ``maybe_load_checkpoint(strategy, fname, map_location=None,
unique_objects=None) -> (strategy, exp_counter)`` returns ``(strategy, 0)`` if the file does not
exist, otherwise updates ``strategy.__dict__`` in place (model, optimizer, plugins incl. the
evaluator and its metric state, clock) and restores the global RNG state through
``RNGManager.__setstate__``.
https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/checkpointing/checkpoint.py

The checkpoint therefore contains everything the strategy references (replay buffers, EWC
importances, LwF teacher, evaluator/loggers); its size grows with those.
"""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import Any
from unittest import mock


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temporary file + ``os.replace`` (atomic on POSIX/NTFS)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=False, allow_nan=False) + "\n")


def save_checkpoint_atomic(strategy, path: str | Path) -> None:
    """Save the strategy to ``path`` atomically (``checkpoint.pt.tmp`` then ``os.replace``).

    Must be called outside the strategy's train/eval loops (Avalanche restriction).
    """
    from avalanche.checkpointing.checkpoint import save_checkpoint

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    save_checkpoint(strategy, str(tmp))
    os.replace(tmp, path)


def load_checkpoint(strategy, path: str | Path, device=None, unique_objects=None):
    """Resume ``strategy`` from ``path`` if it exists. Returns ``(strategy, exp_counter)``.

    ``exp_counter`` is the number of completed training experiences (index of the next one).
    ``torch.load`` is called with ``weights_only=False`` (the file holds a dill-pickled strategy,
    which the safe loader cannot read; newer torch versions default to ``weights_only=True``).
    Only load checkpoints this code wrote.
    """
    import torch
    from avalanche.checkpointing import checkpoint as av_ckpt

    path = Path(path)
    if not path.exists():
        return strategy, 0
    map_location = str(device) if device is not None else None
    with mock.patch.object(av_ckpt.torch, "load", partial(torch.load, weights_only=False)):
        strategy, exp_counter = av_ckpt.maybe_load_checkpoint(
            strategy, str(path), map_location=map_location, unique_objects=unique_objects
        )
    return strategy, int(exp_counter)


def reapply_eval_mb_size(strategy, eval_mb_size) -> int | None:
    """Set ``strategy.eval_mb_size`` from the current config after a checkpoint load.

    ``maybe_load_checkpoint`` replaces ``strategy.__dict__`` with the pickled one, so the eval
    mini-batch size of the FIRST attempt would silently win over a resume's
    ``train.eval_mb_size`` override (a volatile key, e.g. lowered after an eval OOM). Avalanche
    v0.6.0 ``BaseSGDTemplate.__init__`` stores it as ``self.eval_mb_size`` and
    ``_make_eval_dataloader`` reads it (training/templates/base_sgd.py). ``train_mb_size`` is
    fingerprinted and therefore deliberately left as restored. Returns the previous value.
    """
    old = getattr(strategy, "eval_mb_size", None)
    strategy.eval_mb_size = int(eval_mb_size)
    return old


__all__ = [
    "atomic_write_text",
    "atomic_write_json",
    "save_checkpoint_atomic",
    "load_checkpoint",
    "reapply_eval_mb_size",
]
