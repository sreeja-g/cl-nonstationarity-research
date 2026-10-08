"""Post-hoc analyses: rank inversion and shift-forgetting correlation."""

from clns.analysis.correlation_study import (
    aggregate_transitions,
    align_forgetting_with_transitions,
    check_shift_matches_run,
    forgetting,
    load_accuracy_matrices,
    load_shift,
    one_step_forgetting,
    partial_spearman,
    run_transition_table,
    shift_transitions,
    spearman_study,
    two_way_bootstrap_ci,
)
from clns.analysis.rank_inversion import (
    bootstrap_flip_stability,
    compute_rankings,
    find_flips,
    kendall_tau,
)

__all__ = [
    "aggregate_transitions",
    "align_forgetting_with_transitions",
    "bootstrap_flip_stability",
    "check_shift_matches_run",
    "compute_rankings",
    "find_flips",
    "forgetting",
    "kendall_tau",
    "load_accuracy_matrices",
    "load_shift",
    "one_step_forgetting",
    "partial_spearman",
    "run_transition_table",
    "shift_transitions",
    "spearman_study",
    "two_way_bootstrap_ci",
]
