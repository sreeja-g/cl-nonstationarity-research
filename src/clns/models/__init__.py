"""Model arms (``configs/model/<arm>.yaml``; ``model.name`` = arm id, the run-dir model level).

Each arm config gives ``arch`` (resnet18 | resnet50 | mobilenetv2), ``init`` (scratch |
places365 | imagenet), ``trainable`` (all | last_block), ``weights_file`` / ``weights_sha256``
(pretrained arms), ``image_size`` / ``normalization`` (copied into the benchmark config by
``clns.train.apply_model_overrides`` when non-null; every arm keeps image_size null so all arms
run at the benchmark's resolution, e.g. CORe50 128 px) and ``cifar_stem`` (scratch only).
arch/init/trainable must equal the canonical entry of the arm in :data:`ARM_SPECS` (so e.g.
``model=r18_scratch model.init=imagenet`` is refused instead of running under a wrong run dir).

:func:`validate_model_config` is the cheap pre-flight check (arm x benchmark, canonical spec,
Places365 weights file present) that ``clns.train`` runs before touching the run directory.

:func:`build_model` order: validate arm x benchmark + canonical spec -> build the backbone with
its ORIGINAL head -> load weights (strict) -> replace the head by one
``nn.Linear(in, num_classes)`` -> freeze (``last_block``: requires_grad=False +
FrozenBatchNorm2d outside the last block).
"""
from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from clns.models.backbones import (
    ARCHS,
    GlobalMeanPool2d,
    MBV2Wrapper,
    build_backbone,
    replace_head,
)
from clns.models.freezing import LAST_BLOCK, count_params, freeze_except, freeze_last_block
from clns.models.pretrained import (
    PLACES365_NUM_CLASSES,
    imagenet_weights_file,
    load_imagenet,
    load_places365,
    places365_missing_hint,
)

MAIN_BENCHMARKS = ("imagenet100", "clear10", "domainnet", "core50_nic")
# The synthetic benchmark (tests / smoke) accepts every arm.
ALWAYS_ALLOWED_BENCHMARKS = ("synthetic",)
# Arm id -> benchmarks it may run on. ImageNet-initialised arms overlap with ImageNet-100 and
# (via ImageNet/DomainNet/CLEAR content) would confound H2, so they are CORe50-NIC only.
# r18_places_all (H2 confound check) is allowed everywhere; run_grid.sh restricts it to the
# pilot benchmark.
ALLOWED: dict[str, tuple[str, ...]] = {
    "r18_scratch": MAIN_BENCHMARKS,
    "r18_places": MAIN_BENCHMARKS,
    "r50_scratch": MAIN_BENCHMARKS,
    "r50_places": MAIN_BENCHMARKS,
    "r18_imagenet": ("core50_nic",),
    "mbv2_imagenet": ("core50_nic",),
    "r18_places_all": MAIN_BENCHMARKS,
}
# Arm id -> canonical (arch, init, trainable); a config of arm ``name`` must match it exactly.
ARM_SPECS: dict[str, tuple[str, str, str]] = {
    "r18_scratch": ("resnet18", "scratch", "all"),
    "r18_places": ("resnet18", "places365", "last_block"),
    "r50_scratch": ("resnet50", "scratch", "all"),
    "r50_places": ("resnet50", "places365", "last_block"),
    "r18_imagenet": ("resnet18", "imagenet", "last_block"),
    "mbv2_imagenet": ("mobilenetv2", "imagenet", "last_block"),
    "r18_places_all": ("resnet18", "places365", "all"),
}
assert set(ARM_SPECS) == set(ALLOWED)
VALID_MODELS = tuple(ALLOWED)
INITS = ("scratch", "places365", "imagenet")
TRAINABLE = ("all", "last_block")


def check_arm_benchmark(arm: str, benchmark_name: str) -> None:
    """Raise ValueError if model arm ``arm`` may not run on ``benchmark_name``."""
    if arm not in ALLOWED:
        raise ValueError(f"Unknown model.name={arm!r}; valid arms: {list(VALID_MODELS)}")
    if benchmark_name in ALWAYS_ALLOWED_BENCHMARKS:
        return
    allowed = ALLOWED[arm]
    if benchmark_name not in allowed:
        raise ValueError(
            f"model={arm} is not allowed on benchmark={benchmark_name!r}; allowed benchmarks "
            f"for this arm: {list(allowed) + list(ALWAYS_ALLOWED_BENCHMARKS)}"
        )


def check_arm_spec(cfg_model) -> tuple[str, str, str]:
    """Raise ValueError unless arch/init/trainable of ``cfg_model`` equal :data:`ARM_SPECS`.

    Returns the canonical (arch, init, trainable).
    """
    arm = str(cfg_model.name)
    if arm not in ARM_SPECS:
        raise ValueError(f"Unknown model.name={arm!r}; valid arms: {list(VALID_MODELS)}")
    spec = ARM_SPECS[arm]
    got = (
        str(cfg_model.get("arch", None)),
        str(cfg_model.get("init", "scratch")),
        str(cfg_model.get("trainable", "all")),
    )
    diff = [f"model.{k}={g!r} (arm {arm} requires {e!r})"
            for k, g, e in zip(("arch", "init", "trainable"), got, spec) if g != e]
    if diff:
        raise ValueError(
            f"model={arm}: {', '.join(diff)}. Arm ids are fixed (arch, init, trainable) "
            f"combinations (clns.models.ARM_SPECS); select another arm instead of overriding."
        )
    return spec


def validate_model_config(cfg_model, benchmark_name: str,
                          weights_dir: str | Path | None = None) -> None:
    """Pre-flight check without building anything (no weights are read or downloaded).

    Arm x benchmark (:func:`check_arm_benchmark`), canonical spec (:func:`check_arm_spec`) and,
    for ``init: places365``, that ``<weights_dir>/<weights_file>`` exists (FileNotFoundError).
    """
    check_arm_benchmark(str(cfg_model.name), str(benchmark_name))
    _, init, _ = check_arm_spec(cfg_model)
    if init == "places365":
        if weights_dir is None:
            raise ValueError(f"model={cfg_model.name} (init={init}) needs paths.weights_dir")
        path = Path(weights_dir) / str(cfg_model.weights_file)
        if not path.is_file():
            raise FileNotFoundError(places365_missing_hint(path))


def build_model(
    cfg_model,
    num_classes: int,
    benchmark_name: str,
    weights_dir: str | Path | None = None,
    info: dict | None = None,
) -> nn.Module:
    """Build the model of arm ``cfg_model`` (see module docstring).

    :param benchmark_name: ``benchmark.name``; validated against :data:`ALLOWED`.
    :param weights_dir: ``paths.weights_dir``; only read for pretrained arms (scratch arms
        never touch the filesystem).
    :param info: optional dict filled with provenance: arch/init/trainable, ``params``
        ``{"trainable", "total"}`` and, for pretrained arms, ``weights`` ``{"file", "sha256"}``
        (the ACTUAL hash of the loaded file).
    """
    arm = str(cfg_model.name)
    check_arm_benchmark(arm, str(benchmark_name))
    check_arm_spec(cfg_model)
    arch = str(cfg_model.arch)
    init = str(cfg_model.get("init", "scratch"))
    trainable = str(cfg_model.get("trainable", "all"))
    cifar_stem = bool(cfg_model.get("cifar_stem", False))
    if arch not in ARCHS:
        raise ValueError(f"Unknown model.arch={arch!r}; valid: {list(ARCHS)}")
    if init not in INITS:
        raise ValueError(f"Unknown model.init={init!r}; valid: {list(INITS)}")
    if trainable not in TRAINABLE:
        raise ValueError(f"Unknown model.trainable={trainable!r}; valid: {list(TRAINABLE)}")
    if cifar_stem and init != "scratch":
        raise ValueError("model.cifar_stem=true replaces the pretrained 7x7 conv1; scratch only")

    weights: dict | None = None
    if init == "scratch":
        model = build_backbone(arch, num_classes=1000, cifar_stem=cifar_stem)
    else:
        if weights_dir is None:
            raise ValueError(f"model={arm} (init={init}) needs paths.weights_dir")
        sha = cfg_model.get("weights_sha256", None)
        if init == "places365":
            if not arch.startswith("resnet"):
                raise ValueError(f"init=places365 is only available for ResNets, not {arch!r}")
            model = build_backbone(arch, num_classes=PLACES365_NUM_CLASSES)
            path = Path(weights_dir) / str(cfg_model.weights_file)
            actual = load_places365(model, path, sha)
        else:  # imagenet
            wf = cfg_model.get("weights_file", None)
            if wf is not None and str(wf) != imagenet_weights_file(arch):
                raise ValueError(
                    f"model={arm}: weights_file={wf!r} is not the file of the pinned torchvision "
                    f"weights ({imagenet_weights_file(arch)!r})"
                )
            model = build_backbone(arch, num_classes=1000)
            actual, path = load_imagenet(model, arch, weights_dir, sha)
        weights = {"file": str(path), "sha256": actual}
    replace_head(model, arch, int(num_classes))
    if trainable == "last_block":
        freeze_last_block(model, arch)

    if info is not None:
        n_train, n_total = count_params(model)
        info.update(
            {
                "arm": arm,
                "arch": arch,
                "init": init,
                "trainable": trainable,
                "params": {"trainable": n_train, "total": n_total},
                "weights": weights,
            }
        )
    return model


__all__ = [
    "ALLOWED",
    "ARM_SPECS",
    "VALID_MODELS",
    "LAST_BLOCK",
    "build_model",
    "check_arm_benchmark",
    "check_arm_spec",
    "validate_model_config",
    "count_params",
    "freeze_except",
    "freeze_last_block",
    "GlobalMeanPool2d",
    "MBV2Wrapper",
]
