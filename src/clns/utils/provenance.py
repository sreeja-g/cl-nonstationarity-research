"""Run provenance: resolved config + environment record.

``write_provenance(run_dir, cfg)`` writes
  * ``config.yaml``     -- the fully resolved Hydra config of the first attempt (kept on resume;
                           a resume with a different config is recorded and warned about),
  * ``provenance.json`` -- git commit + dirty flag, pip freeze, python/torch/CUDA/cuDNN versions,
                           GPU names, hostname, UTC start time, argv. Resumes append a record to
                           ``resumes``. Any probe that fails is stored as ``null``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from .checkpoint import atomic_write_json, atomic_write_text
from .fingerprint import comparable_config

log = logging.getLogger(__name__)




def _repo_root() -> Path:
    # <repo>/src/clns/utils/provenance.py (editable install)
    return Path(__file__).resolve().parents[3]


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 120) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=True
        )
        return out.stdout
    except Exception:  # noqa: BLE001 - provenance must never crash a run
        return None


def git_info() -> dict[str, Any]:
    root = _repo_root()
    commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    status = _run(["git", "status", "--porcelain"], cwd=root)
    return {
        "commit": commit.strip() if commit else None,
        "branch": branch.strip() if branch else None,
        "dirty": (bool(status.strip()) if status is not None else None),
        "repo_root": str(root),
    }


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "executable": sys.executable,
        "argv": list(sys.argv),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        cudnn = torch.backends.cudnn
        info["cudnn"] = cudnn.version() if cudnn.is_available() else None
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        else:
            info["gpus"] = []
    except Exception:  # noqa: BLE001
        info.setdefault("torch", None)
    for mod in ("avalanche", "torchvision", "timm", "hydra", "omegaconf", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    freeze = _run([sys.executable, "-m", "pip", "freeze"])
    info["pip_freeze"] = freeze.splitlines() if freeze is not None else None
    return info


def _attempt_record() -> dict[str, Any]:
    return {
        "start_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git": git_info(),
        "env": env_info(),
    }


def _comparable(cfg_container: dict) -> dict:
    # Same volatile-key filter as the run fingerprint (fingerprint.py), which already refuses
    # (exit 4) to resume with a different non-volatile config; this only records the diff.
    return comparable_config(cfg_container)


def write_provenance(run_dir: Path, cfg, extra: dict | None = None) -> None:
    """Write ``config.yaml`` and ``provenance.json`` in ``run_dir`` (see module docstring)."""
    from omegaconf import OmegaConf

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    record = _attempt_record()
    if extra:
        record.update(extra)

    cfg_path = run_dir / "config.yaml"
    prov_path = run_dir / "provenance.json"
    if not prov_path.exists() or not cfg_path.exists():
        atomic_write_text(cfg_path, OmegaConf.to_yaml(OmegaConf.create(resolved)))
        record["resumes"] = []
        atomic_write_json(prov_path, record)
        return

    # Resume: keep the first attempt's config/provenance, record this attempt.
    prov = _load_json(prov_path)
    resumes = prov.setdefault("resumes", [])
    old = _comparable(OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True))
    new = _comparable(resolved)
    changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    if changed:
        log.warning(
            "Resuming %s with a config that differs from the first attempt in %s. "
            "The checkpoint was produced with the old config; results mix both.",
            run_dir,
            changed,
        )
        atomic_write_text(
            run_dir / f"config_resume_{len(resumes) + 1}.yaml",
            OmegaConf.to_yaml(OmegaConf.create(resolved)),
        )
    record["config_changed_keys"] = changed
    resumes.append(record)
    atomic_write_json(prov_path, prov)


def update_provenance(run_dir: Path, key: str, value: Any) -> None:
    """Set ``key`` in provenance.json (e.g. benchmark meta known only after the data is built)."""
    prov_path = Path(run_dir) / "provenance.json"
    prov = _load_json(prov_path)
    prov[key] = value
    atomic_write_json(prov_path, prov)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


__all__ = ["write_provenance", "update_provenance", "git_info", "env_info"]
