"""ViT helpers for the L2P arm.

L2P (avalanche.training.supervised.LearningToPrompt) builds its own ViT internally via
avalanche.models.vit.create_model, so no nn.Module is constructed here.

Avalanche v0.6.0 create_model supports: vit_{tiny,base,large}_patch16_224 and
vit_{small,base,large}_patch32_224. Do not add timm weight-tag suffixes.
Source: https://raw.githubusercontent.com/ContinualAI/avalanche/v0.6.0/avalanche/models/vit.py

Input pipeline: the checkpoint expects 224 x 224 inputs normalised with mean = std = 0.5
(timm pretrained cfg of vit_base_patch16_224). ``clns.train`` therefore forces
``benchmark.image_size=224`` and ``benchmark.normalization=vit`` when model=vit_b16_l2p
(logged); see :data:`VIT_IMAGE_SIZE` / :data:`VIT_NORMALIZATION`.

timm version: avalanche/models/timm_vit.py indexes ``resolve_pretrained_cfg(...)["url"]`` and
passes ``pretrained_custom_load=`` to ``build_model_with_cfg``; both only work with timm 0.6.x
(timm 0.9 returns a PretrainedCfg dataclass and dropped that kwarg) -> timm pinned to 0.6.12.

OPEN QUESTION (Q1): the timm pretrained weights are ImageNet-21k pretrained and
ImageNet-1k fine-tuned, so they overlap with ImageNet-100 data. Interpret L2P results on
ImageNet-100 accordingly.
"""
from __future__ import annotations

DEFAULT_TIMM_NAME = "vit_base_patch16_224"
VIT_IMAGE_SIZE = 224
VIT_NORMALIZATION = "vit"  # key of clns.data.common.NORMALIZATION (mean = std = 0.5)


def get_timm_name(cfg_model) -> str:
    """Return the timm/Avalanche model name from a model config."""
    return cfg_model.get("timm_name", DEFAULT_TIMM_NAME) or DEFAULT_TIMM_NAME
