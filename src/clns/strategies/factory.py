"""Strategy factory: naive, ewc, lwf, er, l2p.

L2P (avalanche LearningToPrompt, v0.6.0) masks logits of classes outside
``experience.classes_in_this_experience`` only while TRAINING (``use_mask and is_training``); at
evaluation it uses the full head. Class-IL eval-time masking of not-yet-seen classes is done
uniformly for every method by ``clns.evaluation.plugins.EvalClassMaskPlugin``.
"""
from __future__ import annotations

import logging

from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer, ReservoirSamplingBuffer
from avalanche.training.supervised import EWC, LearningToPrompt, LwF, Naive

from clns.models.vit import get_timm_name

log = logging.getLogger(__name__)

METHODS = ("naive", "ewc", "lwf", "er", "l2p")


def _build_optimizer(model, cfg_train):
    name = str(cfg_train.optimizer).lower()
    if name == "sgd":
        return SGD(
            model.parameters(),
            lr=cfg_train.lr,
            momentum=cfg_train.get("momentum", 0.0),
            weight_decay=cfg_train.get("weight_decay", 0.0),
        )
    if name == "adam":
        return Adam(
            model.parameters(),
            lr=cfg_train.lr,
            weight_decay=cfg_train.get("weight_decay", 0.0),
        )
    raise ValueError(f"Unknown train.optimizer={cfg_train.optimizer!r}; valid: ['sgd', 'adam']")


def _build_storage(cfg_method, scenario):
    mem_size = int(cfg_method.mem_size)
    storage = str(cfg_method.get("storage", "reservoir"))
    if storage == "reservoir":
        return ReservoirSamplingBuffer(max_size=mem_size)
    if storage == "class_balanced":
        if scenario == "domain_il":
            log.warning(
                "method.storage=class_balanced with a domain_il scenario: class-balanced "
                "sampling ignores domain composition and may under-represent old domains."
            )
        return ClassBalancedBuffer(max_size=mem_size, adaptive_size=True)
    raise ValueError(f"Unknown method.storage={storage!r}; valid: ['reservoir', 'class_balanced']")


def build_strategy(
    cfg_method,
    cfg_model,
    cfg_train,
    model,
    num_classes,
    evaluator,
    plugins,
    device,
    scenario: str | None = None,
    n_train_experiences: int | None = None,
):
    """Build the Avalanche strategy (see module docstring).

    :param n_train_experiences: length of the training stream; used to refuse EWC
        ``mode=separate`` on long streams (> ``method.max_separate_experiences``, default 20).
    """
    method = cfg_method.name
    model_name = cfg_model.name

    # Validation first.
    if method not in METHODS:
        raise ValueError(f"Unknown method.name={method!r}; valid: {list(METHODS)}")
    if method == "l2p" and model_name != "vit_b16_l2p":
        raise ValueError("l2p requires model.name=vit_b16_l2p")
    if method != "l2p" and model_name == "vit_b16_l2p":
        raise NotImplementedError("arm B for non-L2P methods: planned, not in this phase")

    plugins = list(plugins) if plugins else []
    common = dict(
        train_mb_size=int(cfg_train.mb_size),
        eval_mb_size=int(cfg_train.eval_mb_size),
        train_epochs=int(cfg_train.epochs),
        device=device,
        plugins=plugins,
        evaluator=evaluator,
        eval_every=cfg_train.get("eval_every", -1),
    )

    if method == "l2p":
        # LearningToPrompt owns its model and Adam optimizer: train.optimizer/momentum/
        # weight_decay are ignored; the learning rate is method.lr.
        log.info("l2p: train.optimizer is ignored (L2P uses its own Adam with method.lr).")
        kwargs = OmegaConf.to_container(cfg_method, resolve=True)
        kwargs.pop("name", None)
        return LearningToPrompt(
            model_name=get_timm_name(cfg_model),
            criterion=CrossEntropyLoss(),
            pretrained=bool(cfg_model.get("pretrained", True)),
            num_classes=num_classes,
            **kwargs,
            **common,
        )

    optimizer = _build_optimizer(model, cfg_train)
    criterion = CrossEntropyLoss()
    base = dict(model=model, optimizer=optimizer, criterion=criterion, **common)

    if method == "naive":
        return Naive(**base)
    if method == "ewc":
        mode = str(cfg_method.get("mode", "separate"))
        decay = cfg_method.get("decay_factor", None)
        if mode == "online" and decay is None:
            raise ValueError("method.mode=online requires method.decay_factor to be set")
        max_sep = int(cfg_method.get("max_separate_experiences", 20))
        if mode == "separate" and n_train_experiences is not None and (
            int(n_train_experiences) > max_sep
        ):
            raise ValueError(
                f"EWC mode=separate keeps one importance matrix + one parameter copy per "
                f"experience on the device (~89 MB per experience for ResNet-18); "
                f"{n_train_experiences} training experiences exceed "
                f"method.max_separate_experiences={max_sep}. Use method.mode=online with "
                f"method.decay_factor=<0..1> (e.g. 0.9), or raise max_separate_experiences "
                f"if the memory is available."
            )
        # keep_importance_data is left at Avalanche's default (False); in v0.6.0 it does not
        # bound the memory of mode=separate, so it is not exposed.
        return EWC(
            ewc_lambda=float(cfg_method.ewc_lambda),
            mode=mode,
            decay_factor=decay,
            **base,
        )
    if method == "lwf":
        alpha = cfg_method.alpha
        if OmegaConf.is_config(alpha):
            alpha = OmegaConf.to_container(alpha)
        return LwF(alpha=alpha, temperature=float(cfg_method.temperature), **base)
    # er
    replay = ReplayPlugin(
        mem_size=int(cfg_method.mem_size),
        batch_size=None,
        batch_size_mem=cfg_method.get("batch_size_mem", None),
        storage_policy=_build_storage(cfg_method, scenario),
    )
    base["plugins"] = plugins + [replay]
    return Naive(**base)
