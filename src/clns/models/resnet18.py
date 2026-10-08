"""ResNet-18 backbone with a single task-agnostic classification head."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class GlobalMeanPool2d(nn.Module):
    """Deterministic global average pooling: ``x.mean(dim=(2, 3), keepdim=True)``.

    Replaces torchvision's ``nn.AdaptiveAvgPool2d((1, 1))``, whose CUDA backward
    (``adaptive_avg_pool2d_backward_cuda``) has no deterministic implementation and raises
    under ``torch.use_deterministic_algorithms(True)`` (warn_only=False). The output shape
    (N, C, 1, 1) and values are identical; ResNet.forward flattens it afterwards.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3), keepdim=True)


def build_resnet18(num_classes: int, cifar_stem: bool = False) -> nn.Module:
    """Randomly initialised torchvision ResNet-18 (no pretrained weights).

    cifar_stem=True swaps the 7x7/stride-2 conv and the max-pool for a 3x3/stride-1
    conv and identity (for small inputs). The global pooling is :class:`GlobalMeanPool2d`
    (deterministic on CUDA). The head is one Linear(512, num_classes); there is no
    training-time logit masking and no per-task head (class-IL evaluation masks unseen classes,
    see ``clns.evaluation.plugins.EvalClassMaskPlugin``).
    """
    model = resnet18(weights=None)
    if cifar_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.avgpool = GlobalMeanPool2d()
    model.fc = nn.Linear(512, num_classes)
    return model
