import os

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    bwt_metrics,
    cpu_usage_metrics,
    disk_usage_metrics,
    forgetting_metrics,
    forward_transfer_metrics,
    loss_metrics,
    timing_metrics,
    StreamConfusionMatrix,
)
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin


def build_eval_plugin(
    output_dir: str,
    num_classes: int,
    enable_tensorboard: bool = True,
    enable_text_logging: bool = True,
    enable_interactive_logging: bool = True,
    save_confusion_matrix: bool = False,
):
    """
    Build Avalanche EvaluationPlugin with standard continual learning metrics
    and per-run loggers.

    Parameters
    ----------
    output_dir : str
        Run output directory, e.g. outputs/run_name
    num_classes : int
        Total number of classes in the benchmark
    enable_tensorboard : bool
        Whether to save TensorBoard logs
    enable_text_logging : bool
        Whether to save logs to log.txt
    enable_interactive_logging : bool
        Whether to print metrics to console/notebook
    save_confusion_matrix : bool
        Whether to save confusion matrix images

    Returns
    -------
    eval_plugin : EvaluationPlugin
    text_log_file_handle : file or None
        Return this so caller can close it after training if needed.
    """
    os.makedirs(output_dir, exist_ok=True)

    tb_dir = os.path.join(output_dir, "tb_logs")
    log_path = os.path.join(output_dir, "log.txt")

    loggers = []
    text_log_file_handle = None

    if enable_interactive_logging:
        interactive_logger = InteractiveLogger()
        loggers.append(interactive_logger)

    if enable_text_logging:
        text_log_file_handle = open(log_path, "a")
        text_logger = TextLogger(text_log_file_handle)
        loggers.append(text_logger)

    if enable_tensorboard:
        os.makedirs(tb_dir, exist_ok=True)
        tb_logger = TensorboardLogger(tb_log_dir=tb_dir)
        loggers.append(tb_logger)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        forward_transfer_metrics(experience=True, stream=True),
        timing_metrics(epoch=True),
        cpu_usage_metrics(experience=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        StreamConfusionMatrix(
            num_classes=num_classes,
            save_image=save_confusion_matrix,
        ),
        loggers=loggers,
        strict_checks=False,
    )

    return eval_plugin, text_log_file_handle