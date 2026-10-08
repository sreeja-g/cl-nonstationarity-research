from __future__ import annotations

from clns.models.resnet18 import GlobalMeanPool2d, build_resnet18
from clns.models.vit import (
    DEFAULT_TIMM_NAME,
    VIT_IMAGE_SIZE,
    VIT_NORMALIZATION,
    get_timm_name,
)

VALID_MODELS = ("resnet18", "vit_b16_l2p")


def build_model(cfg_model, num_classes: int):
    """Build the backbone. Returns None for vit_b16_l2p (L2P builds its own ViT)."""
    name = cfg_model.name
    if name == "resnet18":
        return build_resnet18(num_classes, cifar_stem=bool(cfg_model.get("cifar_stem", False)))
    if name == "vit_b16_l2p":
        return None
    raise ValueError(f"Unknown model.name={name!r}; valid names: {list(VALID_MODELS)}")


__all__ = [
    "build_model",
    "build_resnet18",
    "GlobalMeanPool2d",
    "get_timm_name",
    "DEFAULT_TIMM_NAME",
    "VIT_IMAGE_SIZE",
    "VIT_NORMALIZATION",
    "VALID_MODELS",
]
