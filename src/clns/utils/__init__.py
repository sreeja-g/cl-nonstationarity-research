"""Seeding, provenance, checkpoint, fingerprint and run-lock utilities.

Submodules import torch/avalanche lazily inside functions; import them directly, e.g.
``from clns.utils.seed import seed_everything``.
"""

from .checkpoint import (
    atomic_write_json,
    atomic_write_text,
    load_checkpoint,
    save_checkpoint_atomic,
)
from .fingerprint import check_or_write_fingerprint, config_fingerprint
from .lock import RunDirLock, RunLockError
from .provenance import update_provenance, write_provenance
from .seed import (
    derive_seed,
    resolve_deterministic,
    seed_everything,
    seed_experience,
    seed_model_init,
    seed_worker,
)

__all__ = [
    "atomic_write_json",
    "atomic_write_text",
    "load_checkpoint",
    "save_checkpoint_atomic",
    "check_or_write_fingerprint",
    "config_fingerprint",
    "RunDirLock",
    "RunLockError",
    "write_provenance",
    "update_provenance",
    "derive_seed",
    "resolve_deterministic",
    "seed_everything",
    "seed_experience",
    "seed_model_init",
    "seed_worker",
]
