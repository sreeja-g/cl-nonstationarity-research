"""Small SupervisedPlugins adding behaviour to Avalanche's built-in strategies.

LwF teacher mode (avalanche 0.6.0): ``LearningWithoutForgetting.post_adapt`` (called from
``LwFPlugin.after_training_exp``; training/regularization.py) stores
``copy.deepcopy(strategy.model)`` while the model is in TRAIN mode and never calls ``eval()``.
The teacher forward in ``_lwf_penalty`` runs under ``no_grad`` but would still use dropout and
batch statistics (and update the teacher's BN running stats). :class:`LwFTeacherEvalPlugin`
puts the teacher in eval mode with ``requires_grad=False``.

Hooks: ``after_training`` (runs after EVERY plugin's ``after_training_exp``, so after LwF's
copy regardless of plugin order; the LwF strategy appends its LwFPlugin after user plugins) and
``before_training_exp`` (re-applied before the teacher is used, so it also holds after a resume
from checkpoint.pt, whose unpickled teacher keeps whatever mode was pickled).
"""
from __future__ import annotations

from avalanche.core import SupervisedPlugin
from avalanche.training.plugins import LwFPlugin


def _set_teacher_eval(strategy) -> None:
    for p in strategy.plugins:
        if isinstance(p, LwFPlugin):
            teacher = getattr(p.lwf, "prev_model", None)
            if teacher is not None:
                teacher.eval()
                teacher.requires_grad_(False)


class LwFTeacherEvalPlugin(SupervisedPlugin):
    """Keep every LwF teacher (``LwFPlugin.lwf.prev_model``) in eval mode, frozen."""

    def before_training_exp(self, strategy, **kwargs):
        _set_teacher_eval(strategy)

    def after_training(self, strategy, **kwargs):
        _set_teacher_eval(strategy)


__all__ = ["LwFTeacherEvalPlugin"]
