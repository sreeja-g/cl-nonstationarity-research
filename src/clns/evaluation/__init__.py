"""Evaluation: Avalanche evaluator/loggers/plugins and offline matrix metrics.

``matrix_metrics`` is pure Python and imported eagerly; everything that needs torch/avalanche
is imported lazily on first attribute access.
"""

from .matrix_metrics import clear_metrics, diagonal, macro_matrix, matrix_metrics

_LAZY = {
    "build_evaluator": "evaluator",
    "build_loggers": "evaluator",
    "resolve_wandb_mode": "evaluator",
    "AccuracyMatrixPlugin": "plugins",
    "EvalClassMaskPlugin": "plugins",
    "PeakGPUMemoryPlugin": "plugins",
    "WallClockPlugin": "plugins",
    "JSONLLogger": "plugins",
    "ClnsTextLogger": "plugins",
    "ClnsWandBLogger": "plugins",
    "truncate_jsonl": "plugins",
    "find_plugin": "plugins",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        mod = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["matrix_metrics", "clear_metrics", "diagonal", "macro_matrix"]  # + lazy names
