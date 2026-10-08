"""Offline continual-learning metrics from an accuracy matrix (RESEARCH_LOG 7.2 / 7.2.1).

Pure Python (no torch), so it is unit-testable anywhere and usable in analysis code.

Notation (0-based): ``a[i][j]`` = accuracy on TEST experience ``j`` after training through TRAIN
experience ``i``; ``T`` = number of rows (training experiences completed); ``m[j]`` =
``test_exp_to_train_exp[j]`` = the train experience that introduced test experience ``j``
(identity for ImageNet-100 / CLEAR / DomainNet; first-appearance groups for CORe50, where only
non-empty groups exist as test experiences, so the test stream can be shorter than the train
stream). ``b[j]`` = accuracy of the randomly initialised model on test experience ``j``.

* diagonal          d_j = a[m_j][j]
* acc_final         mean_{j : m_j <= T-1} a[T-1][j]                      (normalised by |J|)
* acc_avg_incremental  mean_i ( mean_{j : m_j <= i} a[i][j] )   over rows with >= 1 seen group
* bwt               mean_{j : m_j <= T-2} (a[T-1][j] - a[m_j][j])        (|J ∩ {..T-2}|)
* forgetting_max    mean_{j : m_j <= T-2} (max_{l = m_j..T-2} a[l][j] - a[T-1][j])
                    -- the max EXCLUDES the final row, so it can be negative.
* fwt               mean_{j : m_j >= 1} (a[m_j - 1][j] - b[j]); ``fwt_degenerate`` is True for
                    class-IL (unseen classes give a_{j-1,j} ~ 0).
* macro_matrix       class-balanced accuracy per (i, j) from per-class counts (DomainNet macro).
* CLEAR metrics (square N x N matrix, iid protocol; Supp. §3 of Lin et al.), reported under
  their own names, never mixed with the GEM-style metrics above:
  clear_accuracy (lower triangle incl. diagonal), clear_bwt (strict lower triangle, raw),
  clear_fwt (strict upper triangle, raw), clear_in_domain_acc (diagonal),
  clear_next_domain_acc (super-diagonal).

Undefined quantities (empty index sets) are returned as ``None``.
"""

from __future__ import annotations

from typing import Sequence


def _mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def _check(a: Sequence[Sequence[float]], mapping: Sequence[int]) -> tuple[int, int]:
    T = len(a)
    if T == 0:
        raise ValueError("Accuracy matrix has no rows")
    n_test = len(a[0])
    if any(len(r) != n_test for r in a):
        raise ValueError("Accuracy matrix rows have different lengths")
    if len(mapping) != n_test:
        raise ValueError(
            f"test_exp_to_train_exp has {len(mapping)} entries for {n_test} test experiences"
        )
    if any(int(m) < 0 for m in mapping):
        raise ValueError("test_exp_to_train_exp entries must be >= 0")
    return T, n_test


def diagonal(a, test_exp_to_train_exp=None) -> list[float | None]:
    """a[m_j][j] for every test experience j (None if m_j has not been trained yet)."""
    mapping = list(range(len(a[0]))) if test_exp_to_train_exp is None else test_exp_to_train_exp
    T, n_test = _check(a, mapping)
    return [a[int(m)][j] if int(m) < T else None for j, m in enumerate(mapping)]


def clear_metrics(R: Sequence[Sequence[float]]) -> dict[str, float | None]:
    """CLEAR's own metrics on a square N x N matrix (RESEARCH_LOG 7.2, last bullet)."""
    N = len(R)
    if N == 0 or any(len(r) != N for r in R):
        raise ValueError("CLEAR metrics need a square N x N accuracy matrix")
    lower_incl = [R[i][j] for i in range(N) for j in range(N) if i >= j]
    lower_strict = [R[i][j] for i in range(N) for j in range(N) if i > j]
    upper_strict = [R[i][j] for i in range(N) for j in range(N) if i < j]
    return {
        "clear_accuracy": _mean(lower_incl),
        "clear_bwt": _mean(lower_strict),
        "clear_fwt": _mean(upper_strict),
        "clear_in_domain_acc": _mean([R[i][i] for i in range(N)]),
        "clear_next_domain_acc": _mean([R[i][i + 1] for i in range(N - 1)]),
    }


def macro_matrix(per_class_rows: Sequence[Sequence[dict]]) -> list[list[float | None]]:
    """Class-balanced (macro) accuracy matrix from per-class counts (``per_class_matrix.json``).

    ``per_class_rows[i][j] = {"classes": [...], "correct": [...], "total": [...]}``;
    entry (i, j) = mean over classes c with total > 0 of correct_c / total_c (None if the test
    experience had no samples). Feed the result to :func:`matrix_metrics` for macro variants
    (e.g. ``acc_final`` -> macro final accuracy; RESEARCH_LOG 7.2, DomainNet).
    """
    out: list[list[float | None]] = []
    for row in per_class_rows:
        r: list[float | None] = []
        for cell in row:
            accs = [
                float(c) / float(t)
                for c, t in zip(cell.get("correct", []), cell.get("total", []))
                if t
            ]
            r.append(_mean(accs))
        out.append(r)
    return out


def matrix_metrics(
    a: Sequence[Sequence[float]],
    b: Sequence[float] | None = None,
    scenario: str = "class_il",
    test_exp_to_train_exp: Sequence[int] | None = None,
    clear: bool = False,
) -> dict:
    """Compute the summary metrics described in the module docstring.

    :param a: T_train x T_test accuracy matrix (rows = after each training experience).
    :param b: random-init accuracy per test experience (needed for FWT), or None.
    :param scenario: "class_il" | "domain_il" (sets ``fwt_degenerate``).
    :param test_exp_to_train_exp: introducing train experience per test experience (default:
        identity, which requires T_test <= T_train for a complete run).
    :param clear: also compute CLEAR's own metrics (requires a square matrix).
    """
    a = [[float(x) for x in row] for row in a]
    mapping = (
        list(range(len(a[0]))) if test_exp_to_train_exp is None
        else [int(m) for m in test_exp_to_train_exp]
    )
    T, n_test = _check(a, mapping)
    last = T - 1

    seen_final = [j for j in range(n_test) if mapping[j] <= last]
    acc_final = _mean([a[last][j] for j in seen_final])

    row_means = []
    for i in range(T):
        seen = [a[i][j] for j in range(n_test) if mapping[j] <= i]
        if seen:
            row_means.append(sum(seen) / len(seen))
    acc_avg_incremental = _mean(row_means)

    old = [j for j in range(n_test) if mapping[j] <= last - 1]
    bwt = _mean([a[last][j] - a[mapping[j]][j] for j in old])
    forgetting_max = _mean(
        [max(a[l][j] for l in range(mapping[j], last)) - a[last][j] for j in old]  # noqa: E741
    )

    fwt = None
    if b is not None:
        if len(b) != n_test:
            raise ValueError(f"random-init row has {len(b)} entries, expected {n_test}")
        fwt_idx = [j for j in range(n_test) if 1 <= mapping[j] <= last]
        fwt = _mean([a[mapping[j] - 1][j] - float(b[j]) for j in fwt_idx])

    out = {
        "acc_final": acc_final,
        "acc_avg_incremental": acc_avg_incremental,
        "bwt": bwt,
        "forgetting_max": forgetting_max,
        "fwt": fwt,
        "fwt_degenerate": scenario == "class_il",
    }
    if clear:
        out.update(clear_metrics(a))
    return out


__all__ = ["matrix_metrics", "clear_metrics", "diagonal", "macro_matrix"]
