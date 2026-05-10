import os

from avalanche.logging import (
    InteractiveLogger,
    TextLogger,
    TensorboardLogger,
)

from avalanche.training.plugins import EvaluationPlugin

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
)

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name


class ModelSizeMetric(PluginMetric):
    """
    Tracks total model parameter count after each experience.
    """

    def __init__(self):
        super().__init__()

    def reset(self, strategy=None):
        pass

    def result(self, strategy=None):
        return None

    def after_training_exp(self, strategy):
        model_size = sum(
            p.numel() for p in strategy.model.parameters()
        )

        metric_name = get_metric_name(
            self,
            strategy,
            add_experience=True,
            add_task=True,
        )

        return MetricValue(
            self,
            metric_name,
            model_size,
            strategy.clock.train_iterations,
        )


class ReplaySizeMetric(PluginMetric):
    """
    Tracks total replay buffer size after each experience.
    """

    def __init__(self):
        super().__init__()

    def reset(self, strategy=None):
        pass

    def result(self, strategy=None):
        return None

    def after_training_exp(self, strategy):
        replay_size = 0

        for plugin in strategy.plugins:
            if hasattr(plugin, "storage_policy"):
                replay_size += len(plugin.storage_policy.buffer)

        metric_name = get_metric_name(
            self,
            strategy,
            add_experience=True,
            add_task=True,
        )

        return MetricValue(
            self,
            metric_name,
            replay_size,
            strategy.clock.train_iterations,
        )


def build_evaluator(
    log_dir: str,
    use_interactive=False,
    use_text=True,
    use_tensorboard=True,
):
    """
    Builds Avalanche evaluation plugin with:
    - accuracy metrics
    - loss metrics
    - model size tracking
    - replay buffer tracking
    """

    os.makedirs(log_dir, exist_ok=True)

    loggers = []

    if use_interactive:
        loggers.append(InteractiveLogger())

    if use_text:
        loggers.append(
            TextLogger(
                open(
                    os.path.join(log_dir, "log.txt"),
                    "a",
                )
            )
        )

    if use_tensorboard:
        loggers.append(
            TensorboardLogger(
                tb_log_dir=log_dir
            )
        )

    evaluator = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False,
            epoch=False,
            experience=True,
            stream=True,
        ),
        loss_metrics(
            minibatch=False,
            epoch=False,
            experience=True,
            stream=True,
        ),
        ModelSizeMetric(),
        ReplaySizeMetric(),
        loggers=loggers,
    )

    return evaluator