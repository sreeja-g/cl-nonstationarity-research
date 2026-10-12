"""Strategy factory: naive, ewc, lwf, er, derpp (Avalanche 0.6.0 built-in strategies only).

  * naive -> ``Naive``; ewc -> ``EWC``; lwf -> ``LwF`` + :class:`LwFTeacherEvalPlugin`
    (teacher in eval mode, see ``clns.strategies.plugins``);
  * er    -> ``Naive`` + ``ReplayPlugin`` (``method.storage``: class_balanced | reservoir);
  * derpp -> ``DER`` with alpha > 0 and beta > 0 (DER++; Avalanche's DER keeps its own
    class-balanced buffer with logits). er and derpp configs use the same mem_size /
    batch_size_mem.
One optimizer with ONE parameter group over ``model.parameters()``: Avalanche's
``reset_optimizer`` rebuilds group 0 from all ``model.named_parameters()`` before each
experience anyway; frozen parameters (``requires_grad=False``) keep ``grad=None`` and SGD/Adam
skip them. Class-IL eval-time masking of not-yet-seen classes is done uniformly for every method
by ``clns.evaluation.plugins.EvalClassMaskPlugin``.
"""
from __future__ import annotations

import logging

from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer, ReservoirSamplingBuffer
from avalanche.training.supervised import DER, EWC, LwF, Naive
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

from clns.strategies.plugins import LwFTeacherEvalPlugin

log = logging.getLogger(__name__)

METHODS = ("naive", "ewc", "lwf", "er", "derpp")


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
    storage = str(cfg_method.get("storage", "class_balanced"))
    if storage == "reservoir":
        return ReservoirSamplingBuffer(max_size=mem_size)
    if storage == "class_balanced":
        if scenario == "domain_il":
            log.info(
                "method.storage=class_balanced with a domain_il scenario: ClassBalancedBuffer "
                "keeps a per-class reservoir, so old domains stay in the buffer roughly in "
                "proportion to how many of their samples were seen (no explicit per-domain "
                "balancing)."
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

    ``cfg_model`` and ``num_classes`` are accepted for interface stability (every method trains
    the model built by ``clns.models.build_model``).

    :param n_train_experiences: length of the training stream; used to refuse EWC
        ``mode=separate`` on long streams (> ``method.max_separate_experiences``, default 20).
    """
    method = cfg_method.name
    if method not in METHODS:
        raise ValueError(f"Unknown method.name={method!r}; valid: {list(METHODS)}")
    if model is None:
        raise ValueError("build_strategy needs a model (clns.models.build_model)")

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
    optimizer = _build_optimizer(model, cfg_train)
    criterion = CrossEntropyLoss()
    base = dict(model=model, optimizer=optimizer, criterion=criterion, **common)

    if method == "naive":
        return Naive(**base)
    if method == "ewc":
        mode = str(cfg_method.get("mode", "online"))
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
        base["plugins"] = plugins + [LwFTeacherEvalPlugin()]
        return LwF(alpha=alpha, temperature=float(cfg_method.temperature), **base)
    if method == "derpp":
        alpha, beta = float(cfg_method.alpha), float(cfg_method.beta)
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"derpp needs method.alpha > 0 and method.beta > 0 (DER++); got "
                             f"alpha={alpha}, beta={beta}")
        bsm = cfg_method.get("batch_size_mem", None)
        return DER(
            mem_size=int(cfg_method.mem_size),
            batch_size_mem=None if bsm is None else int(bsm),
            alpha=alpha,
            beta=beta,
            **base,
        )
    # er
    bsm = cfg_method.get("batch_size_mem", None)
    replay = ReplayPlugin(
        mem_size=int(cfg_method.mem_size),
        batch_size=None,
        batch_size_mem=None if bsm is None else int(bsm),
        storage_policy=_build_storage(cfg_method, scenario),
    )
    base["plugins"] = plugins + [replay]
    return Naive(**base)
