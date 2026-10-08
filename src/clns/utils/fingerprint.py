"""Config fingerprint of a run directory (``run_dir/fingerprint.json``).

The run-directory path only encodes benchmark/model/method/seed, so two different protocols or
hyper-parameter settings (or a smoke run) would share one directory and a DONE-skip would
silently return the other configuration's results. ``train.py`` therefore stores, at the first
start, the SHA-256 of the resolved config WITHOUT the volatile keys below, and on every later
start (before the DONE check) refuses to run (exit code 4) if the fingerprint differs.

Volatile (excluded) keys -- they may change between attempts without changing results:
``paths``, ``logger``, ``device``, ``hydra``, ``run_dir``, ``deterministic``,
``train.num_workers``, ``train.eval_mb_size``, ``train.stop_after_experiences``.

``deterministic`` is volatile so that the documented recovery (a ``strict`` run stopped by an op
without a deterministic kernel -> rerun with ``deterministic=warn``) resumes instead of being
refused; the mode actually used by every attempt is recorded in ``provenance.json``
(``deterministic_mode`` of the first attempt and of each ``resumes`` entry) and in
``progress.json`` (``deterministic_modes``, one entry per attempt).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .checkpoint import atomic_write_json

FINGERPRINT_FILE = "fingerprint.json"
VOLATILE_TOP_KEYS = frozenset({"paths", "logger", "device", "hydra", "run_dir", "deterministic"})
VOLATILE_TRAIN_KEYS = frozenset({"num_workers", "eval_mb_size", "stop_after_experiences"})


def comparable_config(cfg_container: dict) -> dict:
    """Resolved config (plain dict) minus the volatile keys."""
    out = {k: v for k, v in cfg_container.items() if k not in VOLATILE_TOP_KEYS}
    if isinstance(out.get("train"), dict):
        out["train"] = {k: v for k, v in out["train"].items() if k not in VOLATILE_TRAIN_KEYS}
    return out


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_fingerprint(cfg) -> tuple[str, dict]:
    """Return ``(sha256 hex, comparable config dict)`` for a DictConfig or plain dict."""
    try:
        from omegaconf import OmegaConf

        container = (
            OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
        )
    except ImportError:  # pragma: no cover
        container = dict(cfg)
    comp = comparable_config(container)
    return hashlib.sha256(_canonical(comp).encode("utf-8")).hexdigest(), comp


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out if out or not prefix else {prefix[:-1]: d}
    return {prefix[:-1]: d}


def diff_keys(old: dict, new: dict) -> list[str]:
    """Dotted keys whose values differ (or exist on one side only)."""
    fo, fn = _flatten(old), _flatten(new)
    return sorted(k for k in set(fo) | set(fn) if _canonical(fo.get(k)) != _canonical(fn.get(k))
                  or (k in fo) != (k in fn))


def check_or_write_fingerprint(run_dir: Path, cfg) -> tuple[bool, list[str]]:
    """Compare with ``run_dir/fingerprint.json`` (write it if absent).

    Returns ``(ok, differing_keys)``. ``ok`` is False iff a stored fingerprint differs.
    """
    path = Path(run_dir) / FINGERPRINT_FILE
    digest, comp = config_fingerprint(cfg)
    if not path.exists():
        atomic_write_json(path, {"sha256": digest, "config": comp})
        return True, []
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("sha256") == digest:
        return True, []
    return False, diff_keys(stored.get("config", {}) or {}, comp) or ["<unknown>"]


__all__ = [
    "FINGERPRINT_FILE",
    "VOLATILE_TOP_KEYS",
    "VOLATILE_TRAIN_KEYS",
    "comparable_config",
    "config_fingerprint",
    "diff_keys",
    "check_or_write_fingerprint",
]
