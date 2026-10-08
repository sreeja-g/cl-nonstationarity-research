"""Deterministic seeding.

``CUBLAS_WORKSPACE_CONFIG`` must be set before the first cuBLAS call, so it is set at import
time of this module (``train.py`` imports it before touching CUDA).

Data-order determinism (amendment B4)
-------------------------------------
Avalanche 0.6.0 ``BaseSGDTemplate.make_train_dataloader(num_workers, shuffle, pin_memory,
persistent_workers, drop_last, **kwargs)`` does NOT forward extra kwargs (e.g. ``generator`` or
``worker_init_fn``) to the DataLoader -- only ``ffcv_args`` is picked from ``kwargs``; the same
holds for ``ReplayPlugin.before_training_exp``.
(https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/training/templates/base_sgd.py,
https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/training/plugins/replay.py)
All Avalanche train loaders (plain DataLoader, TaskBalancedDataLoader, ReplayDataLoader) therefore
draw their shuffling and worker base seeds from the *global* torch RNG. We make every training
experience reproducible -- including after a resume -- by re-seeding all global RNGs from
``(seed, exp_id)`` right before ``strategy.train(exp)`` (:func:`seed_experience`). PyTorch seeds
``random``/``torch``/``numpy`` inside each worker from that base seed (torch>=1.9), so no
``worker_init_fn`` is required.
"""

from __future__ import annotations

import os
import random

# Required by torch.use_deterministic_algorithms for cuBLAS (CUDA >= 10.2). Must precede CUDA init.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402

# Stage tags for derive_seed (keep stable: changing them changes every run).
STAGE_MODEL_INIT = 0
STAGE_TRAIN_EXP = 1


def derive_seed(seed: int, stage: int, index: int = 0) -> int:
    """Deterministic 32-bit seed from ``(seed, stage, index)`` via numpy SeedSequence."""
    ss = np.random.SeedSequence([int(seed), int(stage), int(index)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _seed_globals(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        # Seeds python/numpy/torch/torch.cuda generators registered in Avalanche's singleton.
        # https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/training/determinism/rng_manager.py
        from avalanche.training.determinism.rng_manager import RNGManager

        RNGManager.set_random_seeds(seed)
    except ImportError:  # pragma: no cover - avalanche is a hard dependency
        pass


DETERMINISTIC_MODES = ("strict", "warn", "off")


def resolve_deterministic(value) -> str:
    """Normalise config ``deterministic`` to "strict" | "warn" | "off".

    Backward compatible: ``true`` -> "strict", ``false`` -> "off" (also the strings
    "true"/"false"/"1"/"0"/"yes"/"no", case-insensitive).
    """
    if isinstance(value, bool):
        return "strict" if value else "off"
    v = str(value).strip().lower()
    if v in ("true", "1", "yes", "on"):
        return "strict"
    if v in ("false", "0", "no", "none", "null"):
        return "off"
    if v not in DETERMINISTIC_MODES:
        raise ValueError(
            f"deterministic={value!r}; expected one of {DETERMINISTIC_MODES} (or true/false)"
        )
    return v


def seed_everything(seed: int, deterministic="strict") -> str:
    """Seed python, numpy, torch (CPU+CUDA) and Avalanche's RNGManager; configure cuDNN.

    ``deterministic`` (see :func:`resolve_deterministic`):
      * "strict": cudnn.deterministic, no cudnn.benchmark,
        ``torch.use_deterministic_algorithms(True, warn_only=False)`` -> any op WITHOUT a
        deterministic implementation RAISES a RuntimeError naming the op. The ResNet-18 global
        pooling is already replaced by a deterministic mean (models/resnet18.py). If a run
        stops with such an error, rerun with ``deterministic=warn`` (and note it) or replace the
        op.
      * "warn": same, but ``warn_only=True`` (non-deterministic ops only warn).
      * "off": cudnn.benchmark on, no deterministic algorithms.
    Returns the resolved mode. ``PYTHONHASHSEED`` set here only affects child processes (the
    current interpreter's hash seed is fixed at start-up); export it in the launcher for full
    hash determinism.
    """
    import torch

    mode = resolve_deterministic(deterministic)
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _seed_globals(seed)
    if mode in ("strict", "warn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=(mode == "warn"))
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
    return mode


def seed_model_init(seed: int) -> None:
    """Re-seed global RNGs right before model construction (independent of data-loading RNG use)."""
    _seed_globals(derive_seed(seed, STAGE_MODEL_INIT))


def seed_experience(seed: int, exp_id: int) -> None:
    """Re-seed global RNGs from ``(seed, exp_id)`` right before training experience ``exp_id``.

    Makes a resumed experience identical to the same experience in an uninterrupted run (B4).
    """
    _seed_globals(derive_seed(seed, STAGE_TRAIN_EXP, exp_id))


def seed_worker(worker_id: int) -> None:  # pragma: no cover - only used if a loader accepts it
    """Standard worker_init_fn (kept for custom loaders; Avalanche 0.6.0 does not forward it)."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


__all__ = [
    "DETERMINISTIC_MODES",
    "resolve_deterministic",
    "derive_seed",
    "seed_everything",
    "seed_model_init",
    "seed_experience",
    "seed_worker",
]
