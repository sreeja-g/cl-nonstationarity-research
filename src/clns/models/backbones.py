"""CNN backbones (ResNet-18/50, MobileNetV2) with deterministic global pooling.

Every builder returns the torchvision architecture with its ORIGINAL head size (1000-way, or
365-way for the Places365 checkpoints) so that pretrained state dicts load with
``strict=True``; :func:`replace_head` swaps in the single task-agnostic ``nn.Linear`` afterwards
(``clns.models.build_model``). There is no training-time logit masking and no per-task head:
class-IL evaluation masks unseen classes in ``clns.evaluation.plugins.EvalClassMaskPlugin``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, resnet18, resnet50

RESNET_ARCHS = ("resnet18", "resnet50")
ARCHS = RESNET_ARCHS + ("mobilenetv2",)


class GlobalMeanPool2d(nn.Module):
    """Deterministic global average pooling: ``x.mean(dim=(2, 3), keepdim=True)``.

    Replaces torchvision's ``nn.AdaptiveAvgPool2d((1, 1))``, whose CUDA backward
    (``adaptive_avg_pool2d_backward_cuda``) has no deterministic implementation and raises
    under ``torch.use_deterministic_algorithms(True)`` (warn_only=False). The output shape
    (N, C, 1, 1) and values are identical; ResNet.forward flattens it afterwards.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3), keepdim=True)


class MBV2Wrapper(nn.Module):
    """MobileNetV2 as ``features -> GlobalMeanPool2d -> flatten -> classifier``.

    torchvision's ``MobileNetV2.forward`` pools with the functional
    ``nn.functional.adaptive_avg_pool2d`` (non-deterministic CUDA backward), so the module
    cannot simply swap an ``avgpool`` attribute as for ResNet. The submodule names
    ``features`` / ``classifier`` are kept, so torchvision MobileNetV2 state dicts load into
    this wrapper unchanged (``strict=True``).
    """

    def __init__(self, features: nn.Module, classifier: nn.Module):
        super().__init__()
        self.features = features
        self.pool = GlobalMeanPool2d()
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x))
        return self.classifier(torch.flatten(x, 1))


def build_resnet(arch: str, num_classes: int = 1000, cifar_stem: bool = False) -> nn.Module:
    """Randomly initialised torchvision ResNet-18/50 with :class:`GlobalMeanPool2d`.

    ``num_classes`` is the size of the ORIGINAL head (1000, or 365 to load a Places365
    checkpoint). cifar_stem=True swaps the 7x7/stride-2 conv and the max-pool for a
    3x3/stride-1 conv and identity (for small inputs; scratch arms only, since pretrained
    conv1 weights have the 7x7 shape).
    """
    if arch == "resnet18":
        model = resnet18(weights=None, num_classes=num_classes)
    elif arch == "resnet50":
        model = resnet50(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown ResNet arch {arch!r}; valid: {list(RESNET_ARCHS)}")
    if cifar_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.avgpool = GlobalMeanPool2d()
    return model


def build_mobilenetv2(num_classes: int = 1000) -> MBV2Wrapper:
    """Randomly initialised torchvision MobileNetV2 wrapped in :class:`MBV2Wrapper`.

    The classifier stays torchvision's ``Sequential(Dropout(0.2), Linear(1280, num_classes))``
    (part of the architecture the ImageNet weights were trained with).
    """
    tv = mobilenet_v2(weights=None, num_classes=num_classes)
    return MBV2Wrapper(tv.features, tv.classifier)


def build_backbone(arch: str, num_classes: int = 1000, cifar_stem: bool = False) -> nn.Module:
    """Dispatch to :func:`build_resnet` / :func:`build_mobilenetv2`."""
    if arch in RESNET_ARCHS:
        return build_resnet(arch, num_classes, cifar_stem=cifar_stem)
    if arch == "mobilenetv2":
        if cifar_stem:
            raise ValueError("cifar_stem is only supported for ResNet archs")
        return build_mobilenetv2(num_classes)
    raise ValueError(f"Unknown model.arch={arch!r}; valid: {list(ARCHS)}")


def replace_head(model: nn.Module, arch: str, num_classes: int) -> nn.Module:
    """Replace the original head by one freshly initialised ``nn.Linear(in, num_classes)``.

    ResNet: ``model.fc``; MobileNetV2: ``model.classifier[1]`` (the dropout stays).
    """
    if arch in RESNET_ARCHS:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "mobilenetv2":
        old = model.classifier[1]
        model.classifier[1] = nn.Linear(old.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model.arch={arch!r}; valid: {list(ARCHS)}")
    return model


__all__ = [
    "ARCHS",
    "RESNET_ARCHS",
    "GlobalMeanPool2d",
    "MBV2Wrapper",
    "build_resnet",
    "build_mobilenetv2",
    "build_backbone",
    "replace_head",
]
