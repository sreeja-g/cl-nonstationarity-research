"""Tuned hyper-parameter files: ``configs/hparams/<benchmark>/<arm>.yaml``.

Written by ``clns.tune --write-best``, read by ``clns.train`` (production runs) and by
``clns.analysis.hypotheses`` (tuned transitions). Format::

    tuned_transitions: [0, 1]            # transitions (k) of the stream used for tuning
    tuning_seed: 0                       # training seed of the tuning runs (task order follows it)
    task_order_sha256: <hex>             # task order hash of the tuning runs (provenance.json)
    methods:
      ewc:
        values: {train.lr: 0.0123, method.ewc_lambda: 4567.0}   # dotted config keys
        provenance: {...}                # n_trials, best_value, db, study, date, ...

``hparams.mode`` (configs/train.yaml): ``required`` (default; a missing file / method entry is
an error -> ``clns.train`` exit 2 before the run dir exists), ``untuned`` (warn, keep the
method defaults, record ``status: untuned``), ``off`` (tuning / smoke runs; nothing applied).

:func:`apply_tuned_hparams` merges the method's ``values`` into the config before the config
fingerprint (the applied values are also written to ``cfg.hparams.applied``, so they are
fingerprinted). ``hparams.mode`` / ``hparams.dir`` are volatile (see fingerprint.py). The file's
sha256 and path are NOT in the config; they go to provenance.json (``hparams``).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("clns.hparams")

MODES = ("required", "untuned", "off")
ALLOWED_PREFIXES = ("train.", "method.")


class HparamsError(ValueError):
    """Missing / invalid tuned-values file (``clns.train`` maps it to exit 2)."""


def default_hparams_dir() -> Path:
    """``<repo>/configs/hparams`` (``configs`` is the Hydra config dir, see train.py)."""
    return Path(__file__).resolve().parents[3] / "configs" / "hparams"


def hparams_path(directory: str | Path | None, benchmark: str, arm: str) -> Path:
    base = Path(directory) if directory else default_hparams_dir()
    return base / benchmark / f"{arm}.yaml"


def load_hparams(path: str | Path) -> dict:
    """Parse a tuned-values file (``{}`` structure if it is empty)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise HparamsError(f"{path}: expected a mapping at the top level")
    return data


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def override_key(override: str) -> str:
    """``"++train.lr=0.1"`` -> ``"train.lr"`` (strips ``+``/``++``/``~`` and the value)."""
    return override.lstrip("+~").split("=", 1)[0].strip()


def apply_tuned_hparams(cfg, cli_overrides: list[str] | None = None) -> dict:
    """Apply the tuned values of ``cfg.benchmark.name`` x ``cfg.model.name`` x ``cfg.method.name``.

    Mutates ``cfg`` (``hparams.applied`` + the value keys) and returns the provenance record
    ``{status, mode, path, sha256, applied, cli_wins}``; ``status`` is ``applied`` | ``untuned`` |
    ``off``. Keys given on the command line win (``cli_wins``). Raises :class:`HparamsError`
    in mode ``required`` when the file or the method entry is missing, or for a file that
    names a key outside ``train.*`` / ``method.*`` or one that does not exist in the config.
    """
    from omegaconf import OmegaConf, open_dict

    hp = cfg.get("hparams", None)
    mode = hp.get("mode", "required") if hp is not None else "required"
    mode = "off" if mode is False else str(mode)  # YAML 1.1: a bare `off` loads as False
    if mode not in MODES:
        raise HparamsError(f"hparams.mode={mode!r}; expected one of {list(MODES)}")
    with open_dict(cfg):
        if cfg.get("hparams", None) is None:
            cfg.hparams = {}
        cfg.hparams.applied = {}
    info: dict[str, Any] = {"status": "off", "mode": mode, "path": None, "sha256": None,
                            "applied": {}, "cli_wins": {}}
    if mode == "off":
        return info

    bench, arm, method = str(cfg.benchmark.name), str(cfg.model.name), str(cfg.method.name)
    path = hparams_path(cfg.hparams.get("dir", None), bench, arm)
    info["path"] = str(path)
    problem = None
    entry = None
    if not path.is_file():
        problem = f"tuned hyper-parameter file {path} not found"
    else:
        info["sha256"] = file_sha256(path)
        entry = (load_hparams(path).get("methods") or {}).get(method)
        if not isinstance(entry, dict) or not isinstance(entry.get("values"), dict):
            problem = f"{path} has no tuned values for method {method!r}"
    if problem is not None:
        if mode == "required":
            raise HparamsError(
                f"{problem} (hparams.mode=required; run clns.tune --write-best, or pass "
                "hparams.mode=untuned to keep the method defaults)")
        log.warning("%s: running with the untuned method defaults (hparams.mode=untuned).",
                    problem)
        info["status"] = "untuned"
        return info

    cli_keys = {override_key(o) for o in (cli_overrides or [])}
    applied: dict[str, Any] = {}
    with open_dict(cfg):
        for key, value in entry["values"].items():
            key = str(key)
            if not key.startswith(ALLOWED_PREFIXES):
                raise HparamsError(f"{path}: key {key!r} is outside train.* / method.*")
            if key not in _flat_keys(cfg):
                raise HparamsError(f"{path}: key {key!r} does not exist in the config")
            if key in cli_keys:
                info["cli_wins"][key] = {"tuned": value, "used": OmegaConf.select(cfg, key)}
                log.info("hparams: command line wins for %s (tuned %r ignored).", key, value)
                continue
            OmegaConf.update(cfg, key, value, merge=False)
            applied[key] = value
        cfg.hparams.applied = applied
    info["applied"] = applied
    info["status"] = "applied"
    log.info("hparams: applied %s from %s", applied, path)
    return info


def _flat_keys(cfg, prefix: str = "") -> set[str]:
    """Dotted keys of all (also null-valued) leaves of a DictConfig."""
    out: set[str] = set()
    for k in cfg.keys():
        v = cfg._get_node(k)
        name = f"{prefix}{k}"
        out.add(name)
        if hasattr(v, "keys"):
            out |= _flat_keys(v, name + ".")
    return out


def tuned_transitions_from_dir(
    directory: str | Path | None,
) -> set[tuple[str, int, int | None, str | None]]:
    """{(benchmark, k, tuning_seed, task_order_sha256)} over ``<dir>/<benchmark>/*.yaml``.

    ``tuning_seed`` / ``task_order_sha256`` are the top-level keys written by ``clns.tune
    --write-best`` (None in a file that lacks them). The analysis flags a run's transition only
    if its task order hash equals the tuning one (or, without hashes, its seed equals
    ``tuning_seed``); see ``clns.analysis.hypotheses.flag_tuned``.
    """
    base = Path(directory) if directory else default_hparams_dir()
    out: set[tuple[str, int, int | None, str | None]] = set()
    if not base.is_dir():
        return out
    for f in sorted(base.glob("*/*.yaml")):
        data = load_hparams(f)
        seed = data.get("tuning_seed")
        sha = data.get("task_order_sha256")
        for k in data.get("tuned_transitions") or []:
            out.add((f.parent.name, int(k), None if seed is None else int(seed),
                     None if sha is None else str(sha)))
    return out


__all__ = [
    "HparamsError",
    "apply_tuned_hparams",
    "default_hparams_dir",
    "file_sha256",
    "hparams_path",
    "load_hparams",
    "override_key",
    "tuned_transitions_from_dir",
]
