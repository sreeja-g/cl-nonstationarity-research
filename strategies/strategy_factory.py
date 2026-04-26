# strategies/strategy_factory.py

import torch

from torch.nn import CrossEntropyLoss

from avalanche.training.supervised import Naive
from avalanche.training import JointTraining

from strategies.optimizer_factory import build_optimizer
from strategies.plugin_factory import build_plugins


def build_strategy(
    strategy_name,
    model,
    evaluator,
    optimizer=None,
    criterion=None,
    plugins=None,
    plugin_names=None,
    extra_plugins=None,
    optimizer_name="sgd",
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0,
    train_mb_size=64,
    eval_mb_size=128,
    train_epochs=1,
    device=None,
    **kwargs,
):
    strategy_name = strategy_name.lower()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if optimizer is None:
        optimizer = build_optimizer(
            model=model,
            optimizer_name=optimizer_name,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    if criterion is None:
        criterion = CrossEntropyLoss()

    if plugins is None:
        plugins = build_plugins(
            plugin_names=plugin_names,
            **kwargs,
        )

    if extra_plugins is not None:
        plugins = plugins + extra_plugins

    common_args = dict(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=train_mb_size,
        train_epochs=train_epochs,
        eval_mb_size=eval_mb_size,
        device=device,
        evaluator=evaluator,
        plugins=plugins,
    )

    if strategy_name == "naive":
        return Naive(**common_args)

    if strategy_name == "joint":
        return JointTraining(**common_args)

    raise ValueError(f"Unknown strategy_name: {strategy_name}")