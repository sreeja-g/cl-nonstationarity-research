"""Shift-forgetting correlation study.

Accuracy matrix convention: ``a[i][j]`` = accuracy on TEST experience j after
training on TRAIN experience i (shape T_train x T_test). ``test_exp_to_train_exp[j]``
= m_j, the train experience that introduced test experience j (identity for most
benchmarks; shorter than T_train for CORe50, RESEARCH_LOG 7.2.1).

Shift: ``shift_transitions(M)[k] = M[k, k+1]`` is the (marginal MMD^2) shift of the
transition INTO train experience k+1; M comes from ``clns.shift.measure_stream``, which must be
run ONCE PER RUN SEED (``--seed k``): the task order follows the training seed (CLEAR-10's time
order does not, so its seeds share one order). A run is matched to ANY shift result of its
benchmark (``<shift-dir>/<benchmark>/seed*/``) whose ``shift_meta.json`` describes the same
benchmark instance as the run's ``provenance.json`` ``benchmark_meta``: same class order (or no
class order on both sides, domain-IL), same split hashes including ``task_order_sha256`` (a run
without one never pairs with a shift that has one) and T == T_train. If several shift results
match, the one measured with the run's own seed is preferred; several matches without that
tie-break, or none at all, skip that run with a warning (counted per benchmark in
``n_runs_shift_mismatch``; ``--strict`` raises instead, :func:`select_shift`).

Forgetting, two pairings with transition k = m_j:

* ``one_step`` (DEFAULT, primary): d_j = a[m_j][j] - a[m_j+1][j], the accuracy drop on
  j caused by the very next transition m_j -> m_j+1. Each value is attributable to
  exactly one transition.
* ``cumulative`` (secondary): f_j = max_{m_j <= i <= T_train-2} a[i][j] - a[T_train-1][j]
  (RESEARCH_LOG 7.2 average forgetting). POSITION CONFOUND: f_j accumulates over all
  T_train-1-m_j later transitions, so early experiences get more forgetting
  regardless of the shift at m_j. Reported together with a partial Spearman that
  controls for the position covariate (T_train - 1 - m_j).

Unit of analysis = the TRANSITION (avoids pseudo-replication: the runs of one
stratum share the same T-1 transitions, so pooling run-level pairs inflates n and
gave a false-positive rate of ~0.8 under the null in review). Primary analysis, per
stratum (benchmark, model, method): forgetting is averaged over the seeds that share
a transition (transition identity = the pair of tasks it moves between, see
:func:`clns.analysis.transitions.transition_id`; seeds with different task orders make different
transitions, CLEAR-10 seeds share them), the shift of a transition is averaged over
the matched shift matrices of those seeds, and Spearman is computed over transitions. The
permutation p-value permutes the shift values across transitions (n = n_transitions); the
bootstrap CI resamples transitions (optionally two-way: runs, then transitions). Pooling
models/methods is only done with ``--scope pooled`` and is flagged in the output.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# The transition table (loading, shift matching, forgetting, transition identity) lives in
# clns.analysis.transitions so that hypotheses.py pairs forgetting with shift identically; the
# names stay importable from here.
from clns.analysis.transitions import (  # noqa: F401
    FORGETTING_MODES,
    aggregate_transitions,
    align_forgetting_with_transitions,
    check_shift_matches_run,
    forgetting,
    load_accuracy_matrices,
    load_shift,
    load_shift_candidates,
    one_step_forgetting,
    run_transition_table,
    select_shift,
    shift_mismatch,
    shift_transitions,
    transition_id,
)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1])


def _residualize(r: np.ndarray, rz: np.ndarray) -> np.ndarray:
    """Residual of r after least-squares regression on [1, rz]."""
    design = np.column_stack([np.ones_like(rz), rz])
    beta, *_ = np.linalg.lstsq(design, r, rcond=None)
    return r - design @ beta


def partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y controlling for z (ranks residualized on rank(z))."""
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    if len(x) < 4 or np.ptp(z) == 0:
        return float("nan")
    rz = stats.rankdata(z)
    ex = _residualize(stats.rankdata(x), rz)
    ey = _residualize(stats.rankdata(y), rz)
    if np.ptp(ex) < 1e-9 or np.ptp(ey) < 1e-9:  # fully explained by z (float tolerance)
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def spearman_study(
    shift,
    forget,
    n_boot: int = 1000,
    n_perm: int = 10000,
    seed: int = 0,
    covariate=None,
) -> dict:
    """Spearman rho over INDEPENDENT units (transitions), one value per unit.

    Do not pass run-level pairs that share transitions (pseudo-replication);
    aggregate them first (``aggregate_transitions``). Returns dict(rho, ci_low,
    ci_high, p_perm, n, n_dropped_nan[, partial_rho, partial_p_perm]). CI: percentile
    bootstrap over units. p_perm: two-sided permutation p-value, permuting shift
    across units, (1 + #{|rho_perm| >= |rho|}) / (1 + n_perm). With ``covariate``
    (e.g. position T_train-1-m_j), also the partial Spearman controlling for it, with
    a permutation p-value obtained by permuting the residualized shift ranks.
    """
    x = np.asarray(shift, dtype=float)
    y = np.asarray(forget, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    z = None
    if covariate is not None:
        z = np.asarray(covariate, dtype=float)
        ok &= np.isfinite(z)
    n_drop = int((~ok).sum())
    x, y = x[ok], y[ok]
    z = None if z is None else z[ok]
    n = len(x)
    rho = _spearman(x, y)
    rng = np.random.default_rng(seed)

    boots = []
    if np.isfinite(rho):
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            r = _spearman(x[idx], y[idx])
            if np.isfinite(r):
                boots.append(r)
    ci_low, ci_high = _percentile_ci(boots)

    p_perm = float("nan")
    if np.isfinite(rho):
        rx, ry = stats.rankdata(x), stats.rankdata(y)
        count = sum(
            abs(np.corrcoef(rng.permutation(rx), ry)[0, 1]) >= abs(rho) - 1e-12
            for _ in range(n_perm)
        )
        p_perm = (1.0 + count) / (1.0 + n_perm)
    out = {
        "rho": rho, "ci_low": ci_low, "ci_high": ci_high, "p_perm": p_perm,
        "n": int(n), "n_dropped_nan": n_drop,
    }
    if z is not None:
        prho = partial_spearman(x, y, z)
        pp = float("nan")
        if np.isfinite(prho):
            rz = stats.rankdata(z)
            ex = _residualize(stats.rankdata(x), rz)
            ey = _residualize(stats.rankdata(y), rz)
            count = sum(
                abs(np.corrcoef(rng.permutation(ex), ey)[0, 1]) >= abs(prho) - 1e-12
                for _ in range(n_perm)
            )
            pp = (1.0 + count) / (1.0 + n_perm)
        out.update(partial_rho=prho, partial_p_perm=pp)
    return out


def _percentile_ci(vals) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def two_way_bootstrap_ci(
    table: pd.DataFrame, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI of the transition-level Spearman with two-way resampling:
    runs with replacement, re-aggregate to transitions, then transitions with
    replacement. ``table`` is run-level (columns run, unit, shift, forget, position)."""
    rng = np.random.default_rng(seed)
    runs = np.array(sorted(table["run"].unique()))
    by_run = {r: table[table["run"] == r] for r in runs}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(runs, size=len(runs), replace=True)
        parts = [by_run[r].assign(run=f"{r}#{b}") for b, r in enumerate(pick)]
        agg = aggregate_transitions(pd.concat(parts, ignore_index=True))
        agg = agg[np.isfinite(agg["shift"]) & np.isfinite(agg["forget"])]
        n = len(agg)
        if n < 3:
            continue
        idx = rng.integers(0, n, n)
        r = _spearman(agg["shift"].to_numpy()[idx], agg["forget"].to_numpy()[idx])
        if np.isfinite(r):
            boots.append(r)
    return _percentile_ci(boots)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _note(mode: str, scope: str) -> str:
    parts = ["unit = transition (forgetting averaged over seeds sharing it); "
             "perm p permutes shift across transitions; CI resamples transitions",
             "shift = marginal MMD^2, not comparable across benchmarks"]
    if mode == "one_step":
        parts.append("forgetting = one-step drop a[m_j][j]-a[m_j+1][j] paired with M[m_j,m_j+1]")
    else:
        parts.append("forgetting = cumulative max-based (POSITION CONFOUND: early experiences "
                     "accumulate more transitions); see partial_rho controlling for T_train-1-m_j")
    if scope == "pooled":
        parts.append("WARNING: POOLED over models/methods - heterogeneous forgetting scales; "
                     "exploratory only, not a primary result")
    return "; ".join(parts)


def build_run_table(recs, shift_dir, mode: str, strict: bool = False
                    ) -> tuple[pd.DataFrame, dict]:
    """Run-level transition table for all records with matching shift matrices.

    Each run is paired with the shift result of its benchmark chosen by :func:`select_shift`
    (any seed dir with the same class order / split hashes / T; the run's own seed preferred).
    Returns (table, counts). A benchmark without any shift result is skipped with a warning. A
    run whose shift results exist but none (or several, conflicting) match is skipped with a
    warning and counted per benchmark (``counts["n_runs_shift_mismatch"]``); ``strict`` raises
    the ValueError instead.
    """
    frames = []
    counts = {"runs_missing_shift": 0, "pairs_nan": 0, "pairs_total": 0,
              "n_runs_shift_mismatch": {}}
    cache: dict = {}
    for r in recs:
        if r["benchmark"] not in cache:
            cache[r["benchmark"]] = load_shift_candidates(shift_dir, r["benchmark"])
        candidates = cache[r["benchmark"]]
        if not candidates:
            warnings.warn(
                f"no shift matrix for benchmark {r['benchmark']} under {shift_dir}", stacklevel=2
            )
            counts["runs_missing_shift"] += 1
            continue
        try:
            shift = select_shift(candidates, r)
        except ValueError as exc:
            if strict:
                raise
            warnings.warn(f"skipping run (shift mismatch): {exc}", stacklevel=2)
            mism = counts["n_runs_shift_mismatch"]
            mism[r["benchmark"]] = mism.get(r["benchmark"], 0) + 1
            continue
        t = run_transition_table(r, shift_transitions(shift["matrix"]), mode)
        counts["pairs_total"] += len(t)
        counts["pairs_nan"] += int((~(np.isfinite(t["shift"]) & np.isfinite(t["forget"]))).sum())
        run_id = f"{r['benchmark']}/{r['model']}/{r['method']}/seed{r['seed']}"
        t = t.assign(
            benchmark=r["benchmark"], model=r["model"], method=r["method"], seed=r["seed"],
            run=run_id, unit=[transition_id(r, int(k)) for k in t["k"]],
        )
        frames.append(t)
    cols = ["k", "forget", "shift", "position", "a_tt", "t", "benchmark", "model", "method",
            "seed", "run", "unit"]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    return table, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Shift-forgetting Spearman study (unit = transition).")
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--shift-dir", required=True,
                    help="root written by clns.shift.measure_stream: "
                    "<benchmark>/seed<k>/shift_matrix.npz; each run uses any seed dir of its "
                    "benchmark with the same class order / split hashes / T (own seed preferred)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--forgetting", choices=FORGETTING_MODES, default="one_step",
                    help="one_step (default, primary) or cumulative "
                    "(position-confounded, secondary)")
    ap.add_argument("--scope", choices=("stratified", "pooled"), default="stratified",
                    help="stratified per (model, method) (default); pooled mixes models/methods")
    ap.add_argument("--two-way-bootstrap", action="store_true",
                    help="CI resamples runs then transitions (default: transitions only)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict", action="store_true",
                    help="raise on a run whose shift result does not match (default: skip, count)")
    args = ap.parse_args(argv)

    recs = load_accuracy_matrices(args.runs_dir)
    if not recs:
        print(f"error: no finished runs (with DONE) under {args.runs_dir}", file=sys.stderr)
        return 1
    table, counts = build_run_table(recs, args.shift_dir, args.forgetting, args.strict)
    if table.empty:
        print(f"error: no run could be matched to a shift matrix under {args.shift_dir}",
              file=sys.stderr)
        return 1
    if args.scope == "pooled":
        warnings.warn("--scope pooled mixes models/methods in one Spearman; exploratory only",
                      stacklevel=1)
        print("WARNING: pooled scope mixes models and methods; NOT a primary analysis.",
              file=sys.stderr)
        strata = [((b, "*", "*"), sub) for b, sub in table.groupby("benchmark")]
    else:
        strata = list(table.groupby(["benchmark", "model", "method"]))

    rows = []
    for (bench, model, method), sub in strata:
        agg = aggregate_transitions(sub)
        cov = agg["position"] if args.forgetting == "cumulative" else None
        res = spearman_study(agg["shift"], agg["forget"], args.n_boot, args.n_perm, args.seed,
                             covariate=cov)
        if args.two_way_bootstrap:
            res["ci_low"], res["ci_high"] = two_way_bootstrap_ci(sub, args.n_boot, args.seed)
        res["n_transitions"] = res.pop("n")
        res.update(
            benchmark=bench, model=model, method=method, scope=args.scope,
            forgetting=args.forgetting, n_runs=int(sub["run"].nunique()),
            n_run_pairs=len(sub),
            n_run_pairs_nan=int((~(np.isfinite(sub["shift"]) & np.isfinite(sub["forget"]))).sum()),
            measure="marginal_mmd", cross_benchmark_comparable=False,
            ci_method="two_way" if args.two_way_bootstrap else "transitions",
            n_runs_shift_mismatch=int(counts["n_runs_shift_mismatch"].get(bench, 0)),
            note=_note(args.forgetting, args.scope),
        )
        rows.append(res)
    print(
        f"runs: {len(recs)} loaded, {counts['runs_missing_shift']} without shift matrix; "
        f"run-level pairs: {counts['pairs_total']} total, {counts['pairs_nan']} NaN (dropped); "
        f"shift mismatch skipped: {counts['n_runs_shift_mismatch'] or 0}"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
