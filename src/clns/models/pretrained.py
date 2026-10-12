"""Pretrained weight loading (Places365 CSAIL checkpoints, torchvision ImageNet weights).

Places365 (``init: places365``): the CSAIL release ``resnet{18,50}_places365.pth.tar`` is a dict
with ``state_dict`` whose keys carry the ``module.`` prefix of ``nn.DataParallel`` and a 365-way
``fc``. It was trained with the torchvision ResNet definition, so after stripping the prefix it
loads with ``strict=True`` into ``torchvision resnetXX(num_classes=365)`` (our
:func:`clns.models.backbones.build_resnet` only swaps the parameter-free pooling layer).
The release (2017) predates PyTorch 0.4.1: it uses the legacy (non-zip) ``torch.save`` format,
the top-level dict also holds ``epoch``/``arch``/``best_prec1``, and its BatchNorm layers have no
``num_batches_tracked`` buffers. Those buffers are excluded from the key comparison and filled by
the ``strict=True`` load (BatchNorm ``_load_from_state_dict`` inserts the model's own counter
when a state dict without version metadata lacks them; verified with torch 2.3.1).
The file is NOT downloaded here: it must already be in ``${paths.weights_dir}`` (fetched by
``scripts/download_data.sh``); its SHA-256 is checked against ``model.weights_sha256``.

ImageNet (``init: imagenet``): pinned torchvision enums ``ResNet18_Weights.IMAGENET1K_V1`` and
``MobileNet_V2_Weights.IMAGENET1K_V1``, fetched by ``Weights.get_state_dict`` into
``weights_dir`` (``check_hash=True`` verifies the hash prefix in torchvision's file name on
download; ``weights_only=True``; a cached file is reused without network access). The full
SHA-256 of the file is checked against ``model.weights_sha256`` on EVERY load (before it is
unpickled when the file is already cached).

Pin ``model.weights_sha256`` BEFORE the first GPU run of a pretrained arm: it is part of the run
fingerprint, so changing it later makes every existing run directory of that arm refuse to
resume/skip (exit 4).

Tests never touch the network: they write fake ``module.``-prefixed checkpoints, or monkeypatch
:func:`imagenet_state_dict`.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# arch -> pinned torchvision weight enum (name resolved lazily: torchvision.models.<Enum>.<member>)
IMAGENET_WEIGHTS = {
    "resnet18": ("ResNet18_Weights", "IMAGENET1K_V1"),
    "mobilenetv2": ("MobileNet_V2_Weights", "IMAGENET1K_V1"),
}
PLACES365_NUM_CLASSES = 365
PLACES365_PREFIX = "module."
# BatchNorm buffer added in PyTorch 0.4.1; absent from the CSAIL Places365 release.
BN_COUNTER_SUFFIX = "num_batches_tracked"
PIN_SHA_HINT = (
    "Pin model.weights_sha256 (sha256sum <file>) BEFORE the first GPU run of this arm: it is part "
    "of the run fingerprint, so changing it later refuses existing runs with exit 4."
)


def places365_missing_hint(path: str | Path) -> str:
    """Error message for a missing Places365 checkpoint."""
    return (
        f"Places365 weights not found at '{path}'. Run `bash scripts/download_data.sh "
        f"places365` (or set CLNS_WEIGHTS_DIR / paths.weights_dir). {PIN_SHA_HINT}"
    )


def sha256_binary(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a binary file (read in chunks; no newline normalisation)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check_sha256(path: str | Path, expected: str | None) -> str:
    """Return the file's SHA-256; raise ValueError if ``expected`` is set and differs.

    ``expected=None`` (hash not pinned yet) only warns and logs the actual hash, which train.py
    records in provenance.json; pin it in ``configs/model/<arm>.yaml`` before real runs.
    """
    actual = sha256_binary(path)
    if expected is None:
        log.warning("No weights_sha256 pinned for %s; actual sha256=%s. %s", path, actual,
                    PIN_SHA_HINT)
    elif actual != str(expected).lower():
        raise ValueError(f"Weights file {path} has sha256 {actual}, expected {expected}")
    return actual


def _torch_load_checkpoint(path: Path, verified: bool):
    """``torch.load(weights_only=True)``; documented fallback for legacy pickles.

    The CSAIL files predate the zip format and hold only tensors/str/int/float, which the
    restricted unpickler accepts. Should a release ever contain another type, the full
    unpickler is used ONLY if the file's SHA-256 matched a pinned hash (``verified``), i.e. the
    exact bytes were reviewed; otherwise the error is re-raised.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        if not verified:
            raise RuntimeError(
                f"{path} cannot be loaded with weights_only=True ({exc}) and its sha256 is not "
                f"pinned; refusing the unrestricted unpickler. Pin model.weights_sha256."
            ) from exc
        log.warning("%s: weights_only=True failed (%s); file sha256 is pinned and matched, "
                    "falling back to weights_only=False.", path, exc)
        return torch.load(path, map_location="cpu", weights_only=False)


def load_places365(model: nn.Module, path: str | Path, sha256: str | None) -> str:
    """Load a CSAIL Places365 checkpoint into a 365-way torchvision ResNet (in place).

    sha256 check -> ``torch.load(weights_only=True)["state_dict"]`` -> strip ``module.`` ->
    assert the key set matches the model (``num_batches_tracked`` excluded, see module
    docstring) -> ``load_state_dict(strict=True)``. Returns the file's actual SHA-256.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(places365_missing_hint(path))
    actual = check_sha256(path, sha256)
    ckpt = _torch_load_checkpoint(path, verified=sha256 is not None)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    # A new plain dict (no ``_metadata``): BatchNorm treats it as pre-0.4.1 and fills missing
    # num_batches_tracked buffers during the strict load.
    state = {
        (k[len(PLACES365_PREFIX):] if k.startswith(PLACES365_PREFIX) else k): v
        for k, v in state.items()
    }
    got = {k for k in state if not k.endswith(BN_COUNTER_SUFFIX)}
    expected = {k for k in model.state_dict() if not k.endswith(BN_COUNTER_SUFFIX)}
    if got != expected:
        missing = sorted(expected - got)[:5]
        extra = sorted(got - expected)[:5]
        raise ValueError(
            f"Places365 checkpoint {path} has {len(got)} tensors (excluding "
            f"{BN_COUNTER_SUFFIX}), model expects {len(expected)} (missing e.g. {missing}, "
            f"unexpected e.g. {extra})"
        )
    model.load_state_dict(state, strict=True)
    return actual


def _imagenet_weights(arch: str):
    import torchvision.models as tvm

    if arch not in IMAGENET_WEIGHTS:
        raise ValueError(f"No pinned ImageNet weights for arch {arch!r}; "
                         f"valid: {sorted(IMAGENET_WEIGHTS)}")
    enum_name, member = IMAGENET_WEIGHTS[arch]
    return getattr(getattr(tvm, enum_name), member)


def imagenet_weights_file(arch: str) -> str:
    """Cache file name of the pinned torchvision weights (must equal ``model.weights_file``)."""
    return Path(_imagenet_weights(arch).url).name


def imagenet_state_dict(arch: str, weights_dir: str | Path,
                        sha256: str | None) -> tuple[dict, Path, str]:
    """(state dict, cache path, actual SHA-256) of the pinned torchvision ImageNet weights.

    May download into ``weights_dir`` on first use (torchvision ``load_state_dict_from_url``,
    ``check_hash=True``, ``weights_only=True``). The file's SHA-256 is checked against
    ``sha256`` (when pinned) on every call: before loading if the file is already cached, right
    after the download otherwise. Tests monkeypatch this function.
    """
    weights = _imagenet_weights(arch)
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    path = weights_dir / Path(weights.url).name
    cached = path.is_file()
    actual = check_sha256(path, sha256) if cached else None
    state = weights.get_state_dict(model_dir=str(weights_dir), progress=False, check_hash=True,
                                   weights_only=True)
    if not cached:
        actual = check_sha256(path, sha256)
    return state, path, actual


def load_imagenet(model: nn.Module, arch: str, weights_dir: str | Path,
                  sha256: str | None) -> tuple[str, Path]:
    """Load the pinned torchvision ImageNet weights into a 1000-way model (in place, strict).

    Returns (the cached file's actual SHA-256, checked against ``sha256`` when pinned; its path).
    """
    state, path, actual = imagenet_state_dict(arch, weights_dir, sha256)
    model.load_state_dict(state, strict=True)
    return actual, path


__all__ = [
    "IMAGENET_WEIGHTS",
    "PLACES365_NUM_CLASSES",
    "sha256_binary",
    "check_sha256",
    "load_places365",
    "places365_missing_hint",
    "imagenet_weights_file",
    "imagenet_state_dict",
    "load_imagenet",
]
