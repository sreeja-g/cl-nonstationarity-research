"""Post-hoc analyses: rank inversion, shift-forgetting correlation, H1/H2 regressions, CKA.

Names are re-exported lazily (PEP 562 ``__getattr__``) so that ``python -m
clns.analysis.<module>`` does not import the module twice (runpy RuntimeWarning).
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "aggregate_transitions": "transitions",
    "align_forgetting_with_transitions": "transitions",
    "check_shift_matches_run": "transitions",
    "forgetting": "transitions",
    "load_accuracy_matrices": "transitions",
    "load_shift": "transitions",
    "one_step_forgetting": "transitions",
    "run_transition_table": "transitions",
    "shift_transitions": "transitions",
    "all_pairs_table": "transitions",
    "select_shift": "transitions",
    "transition_id": "transitions",
    "partial_spearman": "correlation_study",
    "spearman_study": "correlation_study",
    "two_way_bootstrap_ci": "correlation_study",
    "bootstrap_flip_stability": "rank_inversion",
    "compute_rankings": "rank_inversion",
    "find_flips": "rank_inversion",
    "kendall_tau": "rank_inversion",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{mod}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
