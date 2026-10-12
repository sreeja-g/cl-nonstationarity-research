"""Partial freezing of pretrained backbones (``model.trainable: last_block``).

Avalanche 0.6.0 calls ``model.train()`` at the start of every ``train()`` and rebuilds optimizer
param group 0 from ALL ``model.named_parameters()`` (``reset_optimizer``), so freezing must not
rely on ``eval()`` or on the optimizer's parameter list. Instead:

  * every parameter outside the trainable prefixes gets ``requires_grad=False`` -> its ``.grad``
    stays None and SGD/Adam skip it (including weight decay and momentum);
  * every ``nn.BatchNorm2d`` outside the trainable prefixes is replaced by
    ``torchvision.ops.misc.FrozenBatchNorm2d`` (affine params and running stats as BUFFERS,
    always normalising with the stored statistics) -> ``model.train()`` cannot update them.

BatchNorm layers INSIDE the trainable block stay ordinary (trainable, running stats updated);
this is intended (e.g. MobileNetV2 ``features.18`` BN).
"""
from __future__ import annotations

import logging
from typing import Iterable

import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d

log = logging.getLogger(__name__)

# Pinned module-name prefixes of the "last block" (+ head) per architecture family.
LAST_BLOCK: dict[str, tuple[str, ...]] = {
    "resnet": ("layer4", "fc"),
    "mobilenetv2": ("features.17", "features.18", "classifier"),
}


def arch_family(arch: str) -> str:
    """``resnet18``/``resnet50`` -> ``resnet``; ``mobilenetv2`` -> ``mobilenetv2``."""
    return "resnet" if str(arch).startswith("resnet") else str(arch)


def _under(name: str, prefixes: Iterable[str]) -> bool:
    """True if module/parameter ``name`` is one of ``prefixes`` or nested inside one."""
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _frozen_bn(bn: nn.BatchNorm2d) -> FrozenBatchNorm2d:
    if not bn.track_running_stats or bn.running_mean is None:
        raise ValueError("Cannot freeze a BatchNorm2d without running statistics")
    frozen = FrozenBatchNorm2d(bn.num_features, eps=bn.eps)
    if bn.affine:
        frozen.weight.copy_(bn.weight.detach())
        frozen.bias.copy_(bn.bias.detach())
    frozen.running_mean.copy_(bn.running_mean)
    frozen.running_var.copy_(bn.running_var)
    return frozen.to(bn.running_mean.device)


def freeze_except(model: nn.Module, prefixes: Iterable[str]) -> nn.Module:
    """Freeze everything outside ``prefixes`` (in place; see module docstring). Returns model."""
    prefixes = tuple(prefixes)
    names = {n for n, _ in model.named_modules()}
    unknown = [p for p in prefixes if p not in names]
    if unknown:
        raise ValueError(f"Trainable prefixes {unknown} are not modules of the model")

    # Collect first: replacing children while iterating named_modules() is unsafe.
    to_replace = [
        n for n, m in model.named_modules()
        if isinstance(m, nn.BatchNorm2d) and not _under(n, prefixes)
    ]
    for name in to_replace:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, _frozen_bn(getattr(parent, child)))

    for name, p in model.named_parameters():
        p.requires_grad_(_under(name, prefixes))
    trainable, total = count_params(model)
    log.info("Froze all but %s: %d BatchNorm2d -> FrozenBatchNorm2d; trainable params %d / %d.",
             list(prefixes), len(to_replace), trainable, total)
    return model


def freeze_last_block(model: nn.Module, arch: str) -> nn.Module:
    """``trainable: last_block``: :func:`freeze_except` with the pinned :data:`LAST_BLOCK`."""
    family = arch_family(arch)
    if family not in LAST_BLOCK:
        raise ValueError(f"No last_block definition for arch {arch!r}; "
                         f"valid families: {sorted(LAST_BLOCK)}")
    return freeze_except(model, LAST_BLOCK[family])


def count_params(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts.

    ``total`` also counts the affine ``weight``/``bias`` BUFFERS of FrozenBatchNorm2d (they were
    BatchNorm parameters before freezing), so it is the same for the ``all`` and ``last_block``
    arms of one architecture; running statistics are never counted.
    """
    frozen_affine = sum(
        m.weight.numel() + m.bias.numel()
        for m in model.modules() if isinstance(m, FrozenBatchNorm2d)
    )
    total = sum(p.numel() for p in model.parameters()) + frozen_affine
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


__all__ = ["LAST_BLOCK", "arch_family", "freeze_except", "freeze_last_block", "count_params"]
