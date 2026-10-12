"""H1 / H2: does forgetting co-vary more with shift for some methods / arms than for others?

Associational analysis only: nothing here identifies a causal effect of shift (the shift of a
transition is not randomised, and tasks differ in more than their shift).

* **H1** (per benchmark; per arm, plus all arms together): larger shift between two tasks makes
  regularisation methods (EWC, LwF) forget more, replay methods (ER, DER++) less so.
  ``delta = slope_regularization - slope_replay`` (``delta > 0`` means "as predicted"). Naive is
  a separate descriptive stratum (``H1_naive``: its shift slope only), never in the contrast.
* **H2** (per benchmark, on matched architecture pairs r18_scratch/r18_places and
  r50_scratch/r50_places; per pair and pooled over pairs): a pretrained, partly frozen model is
  less affected by shift than the same network from scratch.
  ``delta = slope_scratch - slope_pretrained`` (``delta > 0`` = as predicted).

``shift_z`` = shift z-scored within the benchmark (shift is a marginal MMD^2 that is not
comparable across benchmarks), ``a_tt`` = accuracy on the new task right after training on it
(plasticity; a model that never learns the new task cannot show forgetting from it), ``t`` =
stream index of the transition (k; NOT the table column ``position`` = T-1-k). ``a_tt`` is
measured after the transition, so it may be post-treatment: every fit is repeated without it
(``variant = no_a_tt``); the column ``covariates`` names the covariates of each row.
**CORe50 NIC**: ``a_tt`` is undefined for the batches that introduce no new class (most of
them), so CORe50 is fitted WITHOUT ``a_tt`` (keeps ``t``) instead of dropping those rows.

**Primary (confirmatory) analysis: two-stage run-level test** (``analysis = two_stage``).
A run = one (arm, method, seed). Different runs have different shift sensitivities, so the run,
not the row, is the unit of replication:

1. *Stage 1*, per run: OLS ``forget ~ 1 + shift_z + a_tt + t`` (CORe50: without ``a_tt``) on
   the run's rows -> run slope ``b_r`` (column of the shift_z coefficient), its OLS SE and the
   number of rows (:func:`run_slopes`). Runs with fewer than ``MIN_RUN_ROWS`` rows or a rank
   deficient design are skipped and counted (``n_runs_skipped``).
2. *Stage 2*, on the run slopes of one benchmark:
   * H1 per arm (primary) and ``all_arms`` (all run slopes of the benchmark, + arm fixed
     effects), **term ``delta`` = primary test**: OLS of the run slopes on
     ``1 + replay + seed FE [+ arm FE]`` (:func:`seed_fe_delta`), ``delta = -coef(replay)`` =
     regularisation minus replay slope adjusted for the seed (the seed fixes the task order; runs
     of one seed share its transitions, whose slope noise the seed FE remove), classical OLS SE,
     t CI with the residual df, two-sided p (``ci_method = ols_seed_fe(run_slopes)``).
     **term ``delta_welch`` = secondary**: Welch two-sample t-test of the same run slopes
     ignoring seeds (``ci_method = welch_t(run_slopes)``; conservative when seeds differ in
     their transitions). ``n_runs_ref`` = regularisation runs, ``n_runs_other`` = replay runs.
     ``slope_regularization`` / ``slope_replay`` and ``H1_naive`` ``slope``: group mean of run
     slopes with a one-sample t CI.
   * H2 per architecture pair: runs paired by (method, seed) (same task order),
     ``delta = mean(b_scratch - b_pretrained)``, paired t CI (``paired_t(run_slopes)``,
     ``n_pairs``); ``pooled_pairs`` pools the differences of both pairs; per family
     (``pair=r18|family=regularization`` / ``replay``) as secondary scopes. Runs without a
     partner are dropped (noted).
   Inference generalises over runs (seeds, methods) GIVEN the benchmark's transitions; it does
   not generalise over tasks, and a transition shock shared by the runs of one family is
   treated as part of that family's slope for these tasks.

**Secondary / descriptive: pooled transition-level fit** (``analysis = pooled_transition``,
always ``descriptive_only``)::

    H1: forget ~ 1 + shift_z + replay + shift_z:replay + a_tt + t + method FE [+ arm FE]
    H2: forget ~ 1 + shift_z + pretrained + shift_z:pretrained + a_tt + t + method FE [+ pair FE]

Reference levels: regularisation / scratch (``interaction = -delta``). Method fixed effects are
within-family dummies for H1. 95% CIs from a *wild cluster bootstrap-t* (Webb weights, CR1),
clustered by TRANSITION (benchmark, task pair). It ignores per-run slope heterogeneity, which
makes its CIs too narrow when runs differ in shift sensitivity (review-05: false-positive rate
0.35-0.50 at a per-run slope sd of 0.05-0.10), hence descriptive only. ``n_clusters`` is
reported; fewer than ``MIN_CLUSTERS`` clusters -> no CI.

**Scopes.** ``pairs = consecutive`` (primary): the transition k -> k+1 with its shift
``M[k, k+1]``. ``pairs = all_pairs``: every (old task j, later transition) pair with shift
``M[m_j, t+1]`` (:mod:`clns.analysis.transitions`); always ``descriptive_only`` (rows of the same
old task j are dependent and that dependence is not modelled); needs shift measured with
``measure_stream --pairs full`` (``--pairs all`` refuses otherwise; ``--pairs both`` skips
all_pairs for that benchmark with a warning and one ``analysis = accounting`` row
``term = all_pairs_skipped``).

**Flags.** ``near_chance``: ``a_tt`` below ``chance + margin``. Chance is 1 / (classes the
model can predict at that point): class-IL with eval-time masking (default) -> 1 / |classes
seen in train experiences 0..k+1|; class-IL without masking and domain-IL -> 1 / number of
classes. ``tuned``: transitions on which hyper-parameters were tuned
(``--tuned-transitions clear10:0,clear10:1``). Flagged rows are kept and counted
(``n_near_chance``, ``n_tuned``) unless ``--exclude-flagged`` (applied before stage 1).

**Run accounting.** A run whose shift result does not match (:func:`select_shift` raises) is
skipped with a warning; every benchmark gets an ``analysis = accounting`` row
(``term = n_runs_shift_mismatch``, ``n_runs_loaded``) even if it yields no fit, and a WARNING
summary is printed. Without ``--allow-partial`` the CLI writes the outputs and then exits 3 when
any run was skipped this way (confirmatory output must cover every run); ``--strict`` raises at
the first mismatching run instead.

Usage::

    python -m clns.analysis.hypotheses --runs-dir <out>/runs --shift-dir <shift> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from clns.analysis.transitions import (
    all_pairs_table,
    load_accuracy_matrices,
    load_shift_candidates,
    run_transition_table,
    scenario_kind,
    select_shift,
    shift_is_full,
    shift_transitions,
    transition_id,
)

log = logging.getLogger(__name__)

FAMILY = {
    "ewc": "regularization",
    "lwf": "regularization",
    "er": "replay",
    "derpp": "replay",
    "naive": "naive",
}
H2_PAIRS = {"r18": ("r18_scratch", "r18_places"), "r50": ("r50_scratch", "r50_places")}
MIN_CLUSTERS = 5
DESCRIPTIVE_CLUSTERS = 20
MIN_RUN_ROWS = 4
CI_METHOD = "wild_cluster_bootstrap_t(webb)"
CI_SEED_FE = "ols_seed_fe(run_slopes)"
CI_WELCH = "welch_t(run_slopes)"
CI_PAIRED = "paired_t(run_slopes)"
CI_ONE = "one_sample_t(run_slopes)"
TWO_STAGE = "two_stage"
POOLED = "pooled_transition"
ACCOUNTING = "accounting"
PAIR_MODES = ("consecutive", "all", "both")
EXIT_PARTIAL = 3
ASSOCIATIONAL_NOTE = (
    "associational: shift is not randomised; slopes describe co-variation of forgetting with "
    "shift, not its causal effect"
)
CORE50_NOTE = ("CORe50: a_tt undefined for batches without a new class; fitted without a_tt "
               "(keeps t)")
POOLED_NOTE = ("secondary/descriptive: pooled transition-clustered fit ignores per-run "
               "slope heterogeneity (primary = analysis two_stage)")
TWO_STAGE_NOTE = ("primary: run slopes (stage 1 OLS per run) compared across runs; inference "
                  "over runs given this benchmark's transitions")
ALL_PAIRS_NOTE = ("all_pairs: descriptive only (rows sharing the old task j are dependent; "
                  "that dependence is not modelled)")
OUT_COLUMNS = (
    "analysis", "hypothesis", "benchmark", "pairs", "scope", "variant", "covariates", "term",
    "estimate", "se", "ci_lo", "ci_hi", "df", "p_value", "descriptive_only", "n_rows",
    "n_clusters", "n_runs", "n_runs_ref", "n_runs_other", "n_pairs", "n_runs_skipped",
    "n_flagged", "n_near_chance", "n_tuned", "n_dropped_nan", "n_runs_loaded",
    "n_runs_shift_mismatch", "all_pairs_skipped", "n_boot", "n_boot_ok", "ci_method", "note",
)
# Webb (2014) six-point distribution: mean 0, variance 1, suitable for few clusters.
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


class AllPairsUnavailable(ValueError):
    """``--pairs all`` on a shift result measured without ``measure_stream --pairs full``."""


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def _class_sets(bmeta: dict):
    tc = bmeta.get("train_classes")
    return [set(int(c) for c in cs) for cs in tc] if tc else None


def _chance(rec: dict, k: int) -> float:
    """Chance accuracy on the new task of transition k under the run's evaluation protocol.

    * class-IL (incl. CORe50) with eval-time masking of unseen classes (``train.eval_mask_unseen``,
      default): the model chooses among the classes seen in train experiences 0..k+1 ->
      1 / |union of train_classes[0..k+1]|;
    * class-IL without masking, and domain-IL: 1 / number of classes (union over all train
      experiences; ``classes_per_experience`` if the classes are not recorded).

    NaN if the run's ``benchmark_meta`` does not say.
    """
    bmeta = rec.get("benchmark_meta") or {}
    sets = _class_sets(bmeta)
    kind = scenario_kind(rec)
    if sets:
        if kind != "domain_il" and rec.get("eval_mask_unseen", True):
            seen = set().union(*sets[: k + 2])
        else:
            seen = set().union(*sets)
        return 1.0 / len(seen) if seen else float("nan")
    cpe = bmeta.get("classes_per_experience")
    return 1.0 / float(cpe) if cpe else float("nan")


TABLE_COLUMNS = [
    "benchmark", "model", "method", "seed", "run", "k", "t", "forget", "shift", "a_tt",
    "position", "cluster", "chance", "family", "kind",
]


def load_table(runs_dir, shift_dir, pairs: str = "consecutive", strict: bool = False,
               not_full: str = "raise") -> tuple[pd.DataFrame, dict]:
    """Transition table over all finished runs, each paired with its matching shift matrix.

    ``pairs``: ``consecutive`` or ``all`` (see :mod:`clns.analysis.transitions`). The ``cluster``
    column is the bootstrap cluster: the identity of the transition (task pair) the row belongs
    to; ``kind`` = :func:`scenario_kind` (``core50`` rows are fitted without ``a_tt``).

    Benchmarks without any shift result are skipped (``runs_missing_shift``). A run whose shift
    candidates exist but none (or several, conflicting) match is skipped, logged and counted per
    benchmark (``counts["n_runs_shift_mismatch"]``), or raises ValueError with ``strict``;
    ``counts["runs_per_benchmark"]`` = finished runs found per benchmark.
    ``pairs="all"`` needs shift measured with ``--pairs full``: ``not_full="raise"`` raises
    :class:`AllPairsUnavailable`, ``"skip"`` drops the whole benchmark from the table with a
    warning (``counts["all_pairs_skipped"]``).
    """
    if pairs not in ("consecutive", "all"):
        raise ValueError(f"pairs must be 'consecutive' or 'all', got {pairs!r}")
    if not_full not in ("raise", "skip"):
        raise ValueError(f"not_full must be 'raise' or 'skip', got {not_full!r}")
    recs = load_accuracy_matrices(runs_dir)
    counts: dict = {"runs_loaded": len(recs), "runs_missing_shift": 0,
                    "n_runs_shift_mismatch": {}, "all_pairs_skipped": [],
                    "runs_per_benchmark": {}}
    cache: dict = {}
    frames = []
    not_full_benches: set[str] = set()
    for r in recs:
        bench = r["benchmark"]
        per_b = counts["runs_per_benchmark"]
        per_b[bench] = per_b.get(bench, 0) + 1
        if bench not in cache:
            cache[bench] = load_shift_candidates(shift_dir, bench)
        if not cache[bench]:
            counts["runs_missing_shift"] += 1
            continue
        try:
            shift = select_shift(cache[bench], r)
        except ValueError as exc:
            if strict:
                raise
            log.warning("WARNING: skipping run %s: %s", r["path"], exc)
            mism = counts["n_runs_shift_mismatch"]
            mism[bench] = mism.get(bench, 0) + 1
            continue
        if pairs == "consecutive":
            t = run_transition_table(r, shift_transitions(shift["matrix"]), "one_step")
        else:
            if not shift_is_full(shift):
                msg = (f"all_pairs needs the full shift matrix, but {shift['dir']} was measured "
                       f"with --pairs {shift['meta'].get('pairs', 'consecutive')!s}; rerun "
                       "python -m clns.shift.measure_stream ... --pairs full")
                if not_full == "raise":
                    raise AllPairsUnavailable(msg)
                if bench not in not_full_benches:
                    log.warning("WARNING: skipping all_pairs for benchmark %s: %s", bench, msg)
                not_full_benches.add(bench)
                continue
            t = all_pairs_table(r, shift["matrix"])
        if t.empty:
            continue
        run_id = f"{bench}/{r['model']}/{r['method']}/seed{r['seed']}"
        t = t.assign(
            benchmark=bench, model=r["model"], method=r["method"], seed=r["seed"],
            run=run_id, cluster=[transition_id(r, int(k)) for k in t["k"]],
            chance=[_chance(r, int(k)) for k in t["k"]],
            family=FAMILY.get(r["method"], "other"), kind=scenario_kind(r),
        )
        frames.append(t)
    counts["all_pairs_skipped"] = sorted(not_full_benches)
    frames = [f for f in frames if f["benchmark"].iloc[0] not in not_full_benches]
    if not frames:
        return pd.DataFrame(columns=TABLE_COLUMNS), counts
    return pd.concat(frames, ignore_index=True), counts


def parse_tuned(spec: str | None) -> set[tuple[str, int]]:
    """``"clear10:0,clear10:1"`` -> {("clear10", 0), ("clear10", 1)}."""
    out: set[tuple[str, int]] = set()
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        bench, sep, k = tok.rpartition(":")
        if not sep or not bench or not k.lstrip("-").isdigit():
            raise ValueError(f"bad --tuned-transitions entry {tok!r}; expected benchmark:k")
        out.add((bench, int(k)))
    return out


def flag_near_chance(table: pd.DataFrame, margin: float = 0.05,
                     chance_default: float = float("nan")) -> pd.Series:
    """True where the new task was not learned: ``a_tt < chance + margin`` (chance from the
    run's metadata, else ``chance_default``; NaN chance or NaN a_tt -> not flagged)."""
    chance = table["chance"].fillna(chance_default)
    return ((table["a_tt"] < chance + margin) & chance.notna()).fillna(False)


def flag_tuned(table: pd.DataFrame, tuned: set[tuple[str, int]]) -> pd.Series:
    """True for rows whose (benchmark, transition index k) was used for hyper-parameter tuning."""
    return pd.Series(
        [(b, int(k)) in tuned for b, k in zip(table["benchmark"], table["k"])],
        index=table.index, dtype=bool,
    )


# --------------------------------------------------------------------------- #
# OLS + wild cluster bootstrap-t
# --------------------------------------------------------------------------- #
def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Least-squares coefficients; None if X is rank deficient (the fit is not identified)."""
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    return None if rank < X.shape[1] else beta


def _cluster_index(clusters) -> tuple[np.ndarray, int]:
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    return inv.ravel(), int(inv.max()) + 1


def cr1_se(X: np.ndarray, resid: np.ndarray, clusters, contrasts: np.ndarray) -> np.ndarray:
    """CR1 cluster-robust standard errors of ``contrasts @ beta`` (one per contrast row).

    V = G/(G-1) * (N-1)/(N-K) * (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1.
    """
    g, n_g = _cluster_index(clusters)
    return _se_batch(X, resid[None, :], g, n_g, np.linalg.inv(X.T @ X),
                     np.atleast_2d(contrasts))[0]


def _se_batch(X, E, g, n_g, xtx_inv, C) -> np.ndarray:
    """CR1 standard errors of the contrasts for a batch of residual vectors E (B x N)."""
    n, k = X.shape
    ind = np.zeros((n, n_g))
    ind[np.arange(n), g] = 1.0
    V = C @ xtx_inv  # (n_c, K): Var(c'b) = sum_g (s_g . v_c)^2
    proj = X @ V.T   # (N, n_c): s_g . v_c = sum_{i in g} e_i * proj_i
    adj = n_g / (n_g - 1) * (n - 1) / max(n - k, 1)
    out = np.empty((E.shape[0], C.shape[0]))
    for c in range(C.shape[0]):
        per_cluster = (E * proj[:, c]) @ ind  # (B, G)
        out[:, c] = np.sqrt(adj * (per_cluster ** 2).sum(axis=1))
    return out


def wild_cluster_bootstrap_t(X: np.ndarray, y: np.ndarray, clusters, contrasts,
                             n_boot: int = 2000, seed: int = 0, level: float = 0.95,
                             chunk: int = 250) -> dict:
    """Unrestricted wild cluster bootstrap-t with Webb weights and CR1 standard errors.

    For each draw b, one Webb weight w_g per cluster: y* = X b_hat + w_g(i) * u_hat_i, refit,
    t*_c = (c'b* - c'b_hat) / se*_c (CR1 of the bootstrap residuals). CI of c'b:
    [c'b_hat - q_{1-a/2}(t*) se_c, c'b_hat - q_{a/2}(t*) se_c]. X must have full column rank.

    Returns dict(est, se, lo, hi, n_ok) with arrays of the number of contrasts (``n_ok``:
    smallest number of finite t* over contrasts).
    """
    C = np.atleast_2d(np.asarray(contrasts, dtype=float))
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    g, n_g = _cluster_index(clusters)
    xtx_inv = np.linalg.inv(X.T @ X)
    A = xtx_inv @ X.T              # (K, N): beta = A y
    beta = A @ y
    u = y - X @ beta
    est = C @ beta
    se = _se_batch(X, u[None, :], g, n_g, xtx_inv, C)[0]
    rng = np.random.default_rng(seed)
    t_draws = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        w = rng.choice(WEBB, size=(b, n_g))
        E = u[None, :] * w[:, g]           # bootstrap errors; y* - X b_hat
        dbeta = E @ A.T                    # (b, K): b* - b_hat
        resid = E - dbeta @ X.T            # residuals of the bootstrap fit
        se_b = _se_batch(X, resid, g, n_g, xtx_inv, C)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_draws.append((dbeta @ C.T) / se_b)
    T = np.concatenate(t_draws) if t_draws else np.empty((0, len(C)))
    alpha = 1.0 - level
    lo = np.full(len(C), np.nan)
    hi = np.full(len(C), np.nan)
    n_ok = n_boot
    for c in range(len(C)):
        tc = T[:, c][np.isfinite(T[:, c])]
        n_ok = min(n_ok, len(tc))
        if len(tc) == 0 or not np.isfinite(se[c]) or se[c] <= 0:
            continue
        q_lo, q_hi = np.quantile(tc, [alpha / 2, 1 - alpha / 2])
        lo[c] = est[c] - q_hi * se[c]
        hi[c] = est[c] - q_lo * se[c]
    return {"est": est, "se": se, "lo": lo, "hi": hi, "n_ok": int(n_ok)}


class Spec:
    """Design of one regression: columns, and the contrasts reported as terms.

    Column order: intercept, shift_z, [group, shift_z*group], [a_tt], t, fixed-effect dummies.
    """

    def __init__(self, ref: str | None, other: str | None, use_att: bool = True,
                 fe: dict | None = None):
        self.ref, self.other, self.use_att = ref, other, use_att
        self.grouped = ref is not None
        self.fe = {k: list(v) for k, v in (fe or {}).items() if len(v)}
        self.n_fe = sum(len(v) for v in self.fe.values())
        i_cov = 4 if self.grouped else 2
        self.n_cols = i_cov + (2 if use_att else 1) + self.n_fe
        terms: list[tuple[str, dict]] = []
        if self.grouped:
            terms += [(f"slope_{ref}", {1: 1.0}), (f"slope_{other}", {1: 1.0, 3: 1.0}),
                      ("interaction", {3: 1.0}), ("delta", {3: -1.0})]
        else:
            terms += [("slope", {1: 1.0})]
        if use_att:
            terms += [("plasticity", {i_cov: 1.0})]
        # the stream index t (= k); named "t", not "position" (table column position = T-1-k)
        terms += [("t", {i_cov + (1 if use_att else 0): 1.0})]
        self.terms = tuple(t for t, _ in terms)
        self.contrasts = np.zeros((len(terms), self.n_cols))
        for r, (_, coef) in enumerate(terms):
            for col, v in coef.items():
                self.contrasts[r, col] = v

    @property
    def covariates(self) -> str:
        parts = ["shift_z"]
        if self.grouped:
            parts += [self.other, f"shift_z:{self.other}"]
        if self.use_att:
            parts.append("a_tt")
        parts.append("t")
        parts += [f"{k}_fe" for k in self.fe]
        return "+".join(parts)

    def build(self, df: pd.DataFrame, group: np.ndarray | None) -> np.ndarray:
        cols = [np.ones(len(df)), df["shift_z"].to_numpy(float)]
        if self.grouped:
            cols += [group, group * df["shift_z"].to_numpy(float)]
        if self.use_att:
            cols.append(df["a_tt"].to_numpy(float))
        cols.append(df["t"].to_numpy(float))
        for v in self.fe.values():
            cols += list(v)
        return np.column_stack(cols)

    def extract(self, X: np.ndarray, y: np.ndarray):
        beta = ols(X, y)
        return None if beta is None else self.contrasts @ beta


def _dummies(values: pd.Series, levels=None) -> list[np.ndarray]:
    """0/1 columns for every level but the first (sorted) present in ``values``."""
    levels = sorted(values.unique()) if levels is None else levels
    return [(values == lv).to_numpy(float) for lv in levels[1:]]


def _method_fe_within_family(sub: pd.DataFrame) -> list[np.ndarray]:
    """Method dummies within each family (first method of a family = its reference)."""
    out: list[np.ndarray] = []
    for _, fam in sub.groupby("family", sort=True):
        levels = sorted(fam["method"].unique())
        out += [(sub["method"] == lv).to_numpy(float) for lv in levels[1:]]
    return out


def _fit_rows(df, spec: Spec, group, meta: dict, n_boot: int, seed: int,
              variant: str = "primary", extra_notes=()) -> list[dict]:
    """Result rows (one per term) of one pooled transition-level regression on the already
    filtered ``df`` (``analysis = pooled_transition``: secondary, always descriptive_only)."""
    X = spec.build(df, group)
    y = df["forget"].to_numpy(float)
    n_clusters = int(df["cluster"].nunique())
    descriptive = n_clusters < DESCRIPTIVE_CLUSTERS
    base = {
        **meta, "analysis": POOLED, "variant": variant, "covariates": spec.covariates,
        "n_rows": len(df), "n_clusters": n_clusters, "descriptive_only": True,
        "n_flagged": int((df["near_chance"] | df["tuned"]).sum()),
        "n_near_chance": int(df["near_chance"].sum()), "n_tuned": int(df["tuned"].sum()),
        "n_boot": n_boot, "ci_method": CI_METHOD,
    }
    nterm = len(spec.terms)
    notes = [ASSOCIATIONAL_NOTE, POOLED_NOTE, *extra_notes]
    nan = np.full(nterm, np.nan)
    est, se, lo, hi, n_ok = nan, nan, nan, nan, 0
    beta = ols(X, y)
    if beta is None:
        notes.append("rank-deficient design: not identified")
    else:
        est = spec.contrasts @ beta
        if n_clusters < 2:
            notes.append("single cluster: no SE / CI")
        else:
            se = cr1_se(X, y - X @ beta, df["cluster"].to_numpy(), spec.contrasts)
        if n_clusters < MIN_CLUSTERS:
            notes.append(f"only {n_clusters} clusters (< {MIN_CLUSTERS}): no CI reported")
        else:
            res = wild_cluster_bootstrap_t(X, y, df["cluster"].to_numpy(), spec.contrasts,
                                           n_boot, seed)
            lo, hi, n_ok = res["lo"], res["hi"], res["n_ok"]
            if n_ok < n_boot:
                notes.append(f"{n_boot - n_ok} bootstrap draws without a finite t skipped")
        if descriptive and n_clusters >= MIN_CLUSTERS:
            notes.append(f"{n_clusters} clusters (< {DESCRIPTIVE_CLUSTERS}): too few for "
                         "inference over transitions")
    return [
        {**base, "term": term, "estimate": float(est[i]), "se": float(se[i]),
         "ci_lo": float(lo[i]), "ci_hi": float(hi[i]), "n_boot_ok": n_ok,
         "note": "; ".join(notes)}
        for i, term in enumerate(spec.terms)
    ]


def _is_core50(df: pd.DataFrame) -> pd.Series:
    if "kind" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["kind"].eq("core50")


def prepare(table: pd.DataFrame, tuned=frozenset(), margin: float = 0.05,
            chance_default: float = float("nan")) -> tuple[pd.DataFrame, dict]:
    """Flags + within-benchmark z-scored shift; drops rows with non-finite forget / shift and,
    except for CORe50 (fitted without it), non-finite ``a_tt``.

    Returns (frame, {benchmark: rows dropped}).
    """
    df = table.copy()
    df["near_chance"] = flag_near_chance(df, margin, chance_default)
    df["tuned"] = flag_tuned(df, set(tuned))
    ok = np.isfinite(df[["forget", "shift"]].to_numpy(float)).all(axis=1)
    ok &= (np.isfinite(df["a_tt"].to_numpy(float)) | _is_core50(df).to_numpy())
    dropped = df.loc[~ok].groupby("benchmark").size().to_dict()
    df = df.loc[ok].copy()
    z = df.groupby("benchmark")["shift"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0.0
    )
    df["shift_z"] = z
    return df, {str(k): int(v) for k, v in dropped.items()}


# --------------------------------------------------------------------------- #
# Two-stage run-level test (primary)
# --------------------------------------------------------------------------- #
RUN_KEYS = ["benchmark", "model", "method", "family", "seed", "run"]
SLOPE_COLUMNS = [*RUN_KEYS, "slope", "slope_se", "n_rows", "n_near_chance", "n_tuned",
                 "n_flagged"]


def run_slopes(df: pd.DataFrame, use_att: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 1: per run, OLS ``forget ~ 1 + shift_z [+ a_tt] + t`` on that run's rows.

    Returns ``(slopes, skipped)``: one row per fitted run (``slope`` = shift_z coefficient,
    ``slope_se`` = its classical OLS SE, NaN without residual degrees of freedom, ``n_rows``,
    flag counts), and one row per skipped run with ``reason`` (``few_rows``: fewer than
    ``MIN_RUN_ROWS`` rows; ``rank_deficient``).
    """
    fitted, skipped = [], []
    for _, r in df.groupby("run", sort=True):
        r = r.sort_values("t")
        keys = {k: r[k].iloc[0] for k in RUN_KEYS}
        if len(r) < MIN_RUN_ROWS:
            skipped.append({**keys, "n_rows": len(r), "reason": "few_rows"})
            continue
        cols = [np.ones(len(r)), r["shift_z"].to_numpy(float)]
        if use_att:
            cols.append(r["a_tt"].to_numpy(float))
        cols.append(r["t"].to_numpy(float))
        X = np.column_stack(cols)
        y = r["forget"].to_numpy(float)
        beta = ols(X, y)
        if beta is None:
            skipped.append({**keys, "n_rows": len(r), "reason": "rank_deficient"})
            continue
        dof = len(r) - X.shape[1]
        se = float("nan")
        if dof > 0:
            s2 = float(((y - X @ beta) ** 2).sum()) / dof
            se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]))
        fitted.append({
            **keys, "slope": float(beta[1]), "slope_se": se, "n_rows": len(r),
            "n_near_chance": int(r["near_chance"].sum()), "n_tuned": int(r["tuned"].sum()),
            "n_flagged": int((r["near_chance"] | r["tuned"]).sum()),
        })
    return (pd.DataFrame(fitted, columns=SLOPE_COLUMNS),
            pd.DataFrame(skipped, columns=[*RUN_KEYS, "n_rows", "reason"]))


def _t_ci(est: float, se: float, dof: float, level: float = 0.95) -> dict:
    from scipy import stats

    out = {"estimate": float(est), "se": float(se), "df": float(dof), "ci_lo": float("nan"),
           "ci_hi": float("nan"), "p_value": float("nan")}
    if not (np.isfinite(est) and np.isfinite(se) and np.isfinite(dof)) or dof <= 0 or se <= 0:
        return out
    q = float(stats.t.ppf(0.5 + level / 2, dof))
    out.update(ci_lo=float(est - q * se), ci_hi=float(est + q * se),
               p_value=float(2 * stats.t.sf(abs(est / se), dof)))
    return out


def one_sample_t(x) -> dict:
    """Mean of ``x`` with a one-sample t CI (df = n-1); NaN CI for n < 2 or zero variance."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return _t_ci(float("nan"), float("nan"), float("nan"))
    if n < 2:
        return _t_ci(float(x.mean()), float("nan"), float("nan"))
    return _t_ci(float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)), n - 1)


def welch_t(a, b) -> dict:
    """``mean(a) - mean(b)`` with Welch SE, Welch-Satterthwaite df, t CI and two-sided p (NaN
    CI if a group has fewer than 2 values or both variances are 0)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    est = float(a.mean() - b.mean()) if len(a) and len(b) else float("nan")
    if len(a) < 2 or len(b) < 2:
        return _t_ci(est, float("nan"), float("nan"))
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se2 = va + vb
    if se2 == 0:
        return _t_ci(est, 0.0, float("nan"))
    dof = se2 ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return _t_ci(est, float(np.sqrt(se2)), float(dof))


def seed_fe_delta(runs: pd.DataFrame, arm_fe: bool = False) -> tuple[dict, str]:
    """Primary H1 stage 2: OLS ``slope ~ 1 + replay + seed FE [+ arm FE]`` over run slopes.

    ``delta = -coef(replay)`` = regularisation minus replay slope, adjusted for the seed (= task
    order; for CLEAR-10 the seed only changes the init, the FE then absorb init-level shifts)
    [and arm]. Classical OLS SE, t CI with the residual df = n_runs - n_columns, two-sided p.
    Returns ``(stat, why)``; ``why`` explains a NaN CI (rank deficiency, no residual df).
    """
    y = runs["slope"].to_numpy(float)
    cols = [np.ones(len(runs)), (runs["family"] == "replay").to_numpy(float)]
    cols += _dummies(runs["seed"])
    if arm_fe:
        cols += _dummies(runs["model"])
    X = np.column_stack(cols)
    beta = ols(X, y)
    if beta is None:
        return (_t_ci(float("nan"), float("nan"), float("nan")),
                "seed-FE design rank deficient (a seed without both families?): not identified")
    est = float(-beta[1])
    dof = len(y) - X.shape[1]
    if dof <= 0:
        return _t_ci(est, float("nan"), float("nan")), "no residual df: no CI"
    s2 = float(((y - X @ beta) ** 2).sum()) / dof
    se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]))
    stat = _t_ci(est, se, float(dof))
    return stat, ("" if np.isfinite(stat["ci_lo"]) else "zero residual variance: no CI")


def _flag_sums(runs: pd.DataFrame) -> dict:
    return {"n_rows": int(runs["n_rows"].sum()), "n_flagged": int(runs["n_flagged"].sum()),
            "n_near_chance": int(runs["n_near_chance"].sum()),
            "n_tuned": int(runs["n_tuned"].sum())}


def _ts_row(meta, variant, covariates, term, stat, ci_method, sums, n_runs, notes,
            descriptive, **extra) -> dict:
    return {**meta, "analysis": TWO_STAGE, "variant": variant, "covariates": covariates,
            "term": term, **stat, "ci_method": ci_method, "descriptive_only": bool(descriptive),
            **sums, "n_runs": int(n_runs), "note": "; ".join(notes), **extra}


def _skip_note(k: int) -> list[str]:
    return [f"{k} run(s) skipped in stage 1 (< {MIN_RUN_ROWS} rows or rank deficient)"] if k \
        else []


def analyze_two_stage(df: pd.DataFrame, pairs_label: str, dropped: dict,
                      sensitivity: bool = True) -> list[dict]:
    """Primary H1/H2 two-stage run-level analysis of a prepared frame (one ``pairs`` scope).

    Rows have ``analysis = two_stage``; ``variant`` ``primary`` (stage 1 with ``a_tt``, except
    CORe50) and, with ``sensitivity``, ``no_a_tt``.
    """
    out: list[dict] = []
    all_pairs = pairs_label == "all_pairs"
    for bench, bdf in df.groupby("benchmark", sort=True):
        core50 = bool(_is_core50(bdf).any())
        variants = [("primary", not core50)]
        if sensitivity and not core50:
            variants.append(("no_a_tt", False))
        meta0 = {"benchmark": bench, "pairs": pairs_label,
                 "n_dropped_nan": int(dropped.get(str(bench), 0))}
        for variant, use_att in variants:
            slopes, skipped = run_slopes(bdf, use_att)
            cov = "stage1 per run: " + "+".join(["shift_z", *(["a_tt"] if use_att else []), "t"])
            base = [ASSOCIATIONAL_NOTE, TWO_STAGE_NOTE]
            if core50 and variant == "primary":
                base.append(CORE50_NOTE)
            if all_pairs:
                base.append(ALL_PAIRS_NOTE)

            def n_skip(mask, skipped=skipped) -> int:
                return int(mask(skipped).sum()) if len(skipped) else 0

            # ---- H1: seed-FE OLS (primary) + Welch (secondary) on run slopes; naive descr. ----
            arms = sorted(set(slopes["model"]) | set(skipped["model"]))
            fam2 = ["regularization", "replay"]
            scopes = [(f"arm={a}", (lambda d, a=a: d["model"] == a), False) for a in arms]
            if len(arms) > 1:
                scopes.append(("all_arms", lambda d: d["model"].notna(), True))
            for scope, sel, arm_fe in scopes:
                reg = slopes[sel(slopes) & (slopes["family"] == "regularization")]
                rep = slopes[sel(slopes) & (slopes["family"] == "replay")]
                if len(reg) == 0 or len(rep) == 0:
                    continue
                k = n_skip(lambda s, sel=sel: sel(s) & s["family"].isin(fam2))
                notes = base + _skip_note(k)
                m = {**meta0, "hypothesis": "H1", "scope": scope}
                ex = {"n_runs_ref": len(reg), "n_runs_other": len(rep), "n_runs_skipped": k}
                both = pd.concat([reg, rep])
                stat, why = seed_fe_delta(both, arm_fe=arm_fe)
                fe_notes = notes + ["primary H1 test: OLS of run slopes on replay + seed FE"
                                    + (" + arm FE" if arm_fe else "")]
                if why:
                    fe_notes.append(why)
                out.append(_ts_row(m, variant, cov, "delta", stat, CI_SEED_FE,
                                   _flag_sums(both), len(both), fe_notes, all_pairs, **ex))
                stat = welch_t(reg["slope"], rep["slope"])
                w_notes = notes + ["secondary: Welch t ignoring seeds (primary = term delta)"]
                if not np.isfinite(stat["ci_lo"]):
                    w_notes.append("fewer than 2 runs in a group or zero variance: no CI")
                out.append(_ts_row(m, variant, cov, "delta_welch", stat, CI_WELCH,
                                   _flag_sums(both), len(both), w_notes, all_pairs, **ex))
                for term, g in (("slope_regularization", reg), ("slope_replay", rep)):
                    out.append(_ts_row(m, variant, cov, term, one_sample_t(g["slope"]), CI_ONE,
                                       _flag_sums(g), len(g), notes, all_pairs, **ex))
            for a in arms:
                g = slopes[(slopes["family"] == "naive") & (slopes["model"] == a)]
                if len(g) == 0:
                    continue
                k = n_skip(lambda s, a=a: (s["family"] == "naive") & (s["model"] == a))
                m = {**meta0, "hypothesis": "H1_naive", "scope": f"arm={a}"}
                out.append(_ts_row(m, variant, cov, "slope", one_sample_t(g["slope"]), CI_ONE,
                                   _flag_sums(g), len(g), base + _skip_note(k), all_pairs,
                                   n_runs_skipped=k))

            # ---- H2: paired t on run slopes, scratch - pretrained, paired by (method, seed) --
            present = []
            for name, (scratch, pre) in H2_PAIRS.items():
                a_ = slopes[slopes["model"] == scratch].set_index(["method", "seed"])
                b_ = slopes[slopes["model"] == pre].set_index(["method", "seed"])
                if len(a_) == 0 or len(b_) == 0:
                    continue
                j = a_.join(b_, how="inner", lsuffix="_s", rsuffix="_p").reset_index()
                n_unp = len(a_) + len(b_) - 2 * len(j)
                k = n_skip(lambda s, sc=scratch, pr=pre: s["model"].isin([sc, pr]))
                present.append((name, j, n_unp, k))
            scopes2 = []
            for name, j, n_unp, k in present:
                scopes2.append((f"pair={name}", j, n_unp, k, False))
                for fam in fam2:
                    jf = j[j["family_s"] == fam]
                    scopes2.append((f"pair={name}|family={fam}", jf, 0, None, True))
            if len(present) > 1:
                scopes2.append(("pooled_pairs", pd.concat([p[1] for p in present]),
                                sum(p[2] for p in present), sum(p[3] for p in present), False))
            for scope, j, n_unp, k, secondary in scopes2:
                if len(j) == 0:
                    continue
                notes = list(base)
                if secondary:
                    notes.append("secondary: one method family only")
                if n_unp:
                    notes.append(f"{n_unp} run(s) without a (method, seed) partner dropped")
                notes += _skip_note(k or 0)
                stat = one_sample_t(j["slope_s"] - j["slope_p"])
                if not np.isfinite(stat["ci_lo"]):
                    notes.append("fewer than 2 pairs or zero variance: no CI")
                sums = {c: int(j[f"{c}_s"].sum() + j[f"{c}_p"].sum())
                        for c in ("n_rows", "n_flagged", "n_near_chance", "n_tuned")}
                m = {**meta0, "hypothesis": "H2", "scope": scope}
                ex = {"n_pairs": len(j), "n_runs_ref": len(j), "n_runs_other": len(j),
                      "n_runs_skipped": np.nan if k is None else k}
                out.append(_ts_row(m, variant, cov, "delta", stat, CI_PAIRED, sums, 2 * len(j),
                                   notes, all_pairs, **ex))
                for term, col in (("slope_scratch", "slope_s"), ("slope_pretrained", "slope_p")):
                    out.append(_ts_row(m, variant, cov, term, one_sample_t(j[col]), CI_ONE,
                                       sums, 2 * len(j), notes, all_pairs, **ex))
    return out


def analyze(df: pd.DataFrame, pairs_label: str, dropped: dict, n_boot: int = 2000,
            seed: int = 0, exclude_flagged: bool = False, sensitivity: bool = True,
            pooled: bool = True, two_stage: bool = True) -> list[dict]:
    """All H1/H2 analyses of a prepared frame (one ``pairs`` scope), as result rows.

    ``two_stage`` (primary, :func:`analyze_two_stage`) rows first, then the secondary
    ``pooled_transition`` regressions (:func:`analyze_pooled`). ``exclude_flagged`` drops
    near-chance / tuned rows before either analysis.
    """
    if exclude_flagged:
        df = df.loc[~(df["near_chance"] | df["tuned"])]
    out: list[dict] = []
    if two_stage:
        out += analyze_two_stage(df, pairs_label, dropped, sensitivity)
    if pooled:
        out += analyze_pooled(df, pairs_label, dropped, n_boot, seed, sensitivity)
    return out


def analyze_pooled(df: pd.DataFrame, pairs_label: str, dropped: dict, n_boot: int = 2000,
                   seed: int = 0, sensitivity: bool = True) -> list[dict]:
    """Secondary pooled transition-level H1/H2 regressions (wild cluster bootstrap-t by
    transition), ``analysis = pooled_transition``, always ``descriptive_only``.

    Every fit has a ``primary`` variant (with ``a_tt``, except CORe50) and, with
    ``sensitivity``, a ``no_a_tt`` variant on the same rows.
    """
    out: list[dict] = []
    ap_note = (ALL_PAIRS_NOTE,) if pairs_label == "all_pairs" else ()
    for bench, bdf in df.groupby("benchmark", sort=True):
        n_drop = int(dropped.get(str(bench), 0))
        core50 = bool(_is_core50(bdf).any())
        notes = ((CORE50_NOTE,) if core50 else ()) + ap_note

        def meta(h, scope, bench=bench, n_drop=n_drop):
            return {"hypothesis": h, "benchmark": bench, "pairs": pairs_label, "scope": scope,
                    "n_dropped_nan": n_drop}

        def fits(h, scope, sub, ref, other, group, fe, meta=meta, core50=core50, notes=notes):
            rows = _fit_rows(sub, Spec(ref, other, not core50, fe), group, meta(h, scope),
                             n_boot, seed, "primary", notes)
            if sensitivity and not core50:
                rows += _fit_rows(sub, Spec(ref, other, False, fe), group, meta(h, scope),
                                  n_boot, seed, "no_a_tt", ap_note)
            return rows

        # ---- H1: regularisation vs replay; naive as its own descriptive stratum ----
        arms = sorted(bdf["model"].unique())
        h1 = bdf[bdf["family"].isin(["regularization", "replay"])]
        scopes = [(f"arm={a}", h1[h1["model"] == a], False) for a in arms]
        if len(arms) > 1:
            scopes.append(("all_arms", h1, True))
        for scope, sub, with_arm in scopes:
            if sub["family"].nunique() < 2:
                continue
            group = (sub["family"] == "replay").to_numpy(float)
            fe = {"method": _method_fe_within_family(sub)}
            if with_arm:
                fe["arm"] = _dummies(sub["model"])
            out += fits("H1", scope, sub, "regularization", "replay", group, fe)
        for a in arms:
            sub = bdf[(bdf["family"] == "naive") & (bdf["model"] == a)]
            if len(sub) > 0:
                out += _fit_rows(sub, Spec(None, None, not core50), None,
                                 meta("H1_naive", f"arm={a}"), n_boot, seed, "primary", notes)

        # ---- H2: matched architecture pairs, scratch (reference) vs pretrained ----
        present = []
        for name, (scratch, pre) in H2_PAIRS.items():
            sub = bdf[bdf["model"].isin([scratch, pre])]
            if sub["model"].nunique() < 2:
                continue
            present.append((name, sub, scratch))
            group = (sub["model"] == pre).to_numpy(float)
            out += fits("H2", f"pair={name}", sub, "scratch", "pretrained", group,
                        {"method": _dummies(sub["method"])})
        if len(present) > 1:
            sub = pd.concat([p[1] for p in present])
            pre_arms = {pre for _, pre in H2_PAIRS.values()}
            group = sub["model"].isin(pre_arms).to_numpy(float)
            fe = {"method": _dummies(sub["method"]),
                  "pair": [sub["model"].isin(H2_PAIRS[n]).to_numpy(float)
                           for n, _, _ in present[1:]]}
            out += fits("H2", "pooled_pairs", sub, "scratch", "pretrained", group, fe)
    return out


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="H1/H2 (associational): primary two-stage run-level test of shift slopes; "
        "secondary pooled regressions with transition-clustered wild bootstrap-t CIs."
    )
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--shift-dir", required=True,
                    help="root written by clns.shift.measure_stream (<benchmark>/seed<k>/...; "
                    "measure once per run seed; --pairs full is needed for all_pairs)")
    ap.add_argument("--out", required=True, help="output directory (hypotheses.csv / .json)")
    ap.add_argument("--pairs", choices=PAIR_MODES, default="consecutive",
                    help="consecutive (primary), all (old task j x later transition; needs "
                    "measure_stream --pairs full), or both (all_pairs skipped with a warning for "
                    "benchmarks without the full matrix)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-naive", action="store_true",
                    help="drop the naive method everywhere (H2 pools all methods by default)")
    ap.add_argument("--chance-margin", type=float, default=0.05,
                    help="a_tt < chance + margin flags a near-chance transition (default 0.05)")
    ap.add_argument("--chance-default", type=float, default=float("nan"),
                    help="chance accuracy when a run's benchmark_meta does not give its classes "
                    "(default: such rows are not flagged)")
    ap.add_argument("--tuned-transitions", default="",
                    help="benchmark:k,... transitions used for hyper-parameter tuning (flagged)")
    ap.add_argument("--exclude-flagged", action="store_true",
                    help="drop near-chance and tuned transitions (default: keep, count)")
    ap.add_argument("--strict", action="store_true",
                    help="raise on a run whose shift result does not match (default: skip, count)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="exit 0 even if runs were skipped for a shift mismatch (default: write "
                    f"the outputs, then exit {EXIT_PARTIAL})")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tuned = parse_tuned(args.tuned_transitions)
    modes = ["consecutive", "all"] if args.pairs == "both" else [args.pairs]
    not_full = "skip" if args.pairs == "both" else "raise"
    rows: list[dict] = []
    acct: list[dict] = []
    counts: dict = {}
    per_mode: dict = {}
    skipped: list[str] = []
    any_fit_input = False
    for mode in modes:
        label = "consecutive" if mode == "consecutive" else "all_pairs"
        try:
            table, counts = load_table(args.runs_dir, args.shift_dir, mode, args.strict,
                                       not_full)
        except AllPairsUnavailable as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        acct += accounting_rows(counts, label, mismatch=mode == modes[0])
        if mode == "all":
            skipped = counts["all_pairs_skipped"]
        if args.exclude_naive:
            table = table[table["method"] != "naive"]
        if table.empty:
            per_mode[label] = {"n_rows_used": {}, "n_dropped_nan": {}}
            continue
        any_fit_input = True
        df, dropped = prepare(table, tuned, args.chance_margin, args.chance_default)
        used = {str(b): int(n) for b, n in df.groupby("benchmark").size().items()}
        per_mode[label] = {"n_rows_used": used, "n_dropped_nan": dropped}
        for b, n in sorted(used.items()):
            kind = "core50, no a_tt" if _is_core50(df[df["benchmark"] == b]).any() else "a_tt"
            print(f"[{label}] {b}: {n} rows used ({dropped.get(b, 0)} dropped NaN; {kind})")
        rows += analyze(df, label, dropped, args.n_boot, args.seed, args.exclude_flagged)
    mism = {b: n for b, n in counts.get("n_runs_shift_mismatch", {}).items() if n}
    if not any_fit_input and not mism:
        print(f"error: no run could be paired with a shift matrix under {args.shift_dir} "
              f"({counts})", file=sys.stderr)
        return 1
    res = pd.DataFrame(rows + acct, columns=list(OUT_COLUMNS))
    per_b = counts.get("runs_per_benchmark", {})
    res["n_runs_shift_mismatch"] = [int(mism.get(b, 0)) for b in res["benchmark"]]
    res["n_runs_loaded"] = [int(per_b.get(b, 0)) for b in res["benchmark"]]
    res["all_pairs_skipped"] = res["all_pairs_skipped"].eq(True)
    out = Path(args.out)
    _atomic_text(out / "hypotheses.csv", res.to_csv(index=False))
    cli = {k: (None if isinstance(v, float) and v != v else v) for k, v in vars(args).items()}
    blob = {"note": ASSOCIATIONAL_NOTE, "args": cli, "counts": counts, "per_pairs": per_mode,
            "results": json.loads(res.to_json(orient="records"))}
    _atomic_text(out / "hypotheses.json", json.dumps(blob, indent=2, allow_nan=False))
    print(f"runs: {counts.get('runs_loaded', 0)} loaded, "
          f"{counts.get('runs_missing_shift', 0)} without shift matrix, "
          f"shift mismatch skipped: {mism or 0}")
    for b, n in sorted(mism.items()):
        print(f"WARNING: benchmark {b}: {n} of {per_b.get(b, '?')} runs skipped (shift result "
              "does not match the run's task order / split hashes)")
    if skipped:
        print(f"WARNING: all_pairs skipped (shift not --pairs full): {skipped}")
    print(f"wrote {out / 'hypotheses.csv'} ({len(res)} rows)")
    if mism and not args.allow_partial:
        print(f"error: {sum(mism.values())} run(s) skipped for shift mismatch; the output is "
              f"partial (exit {EXIT_PARTIAL}). Fix the shift results or pass --allow-partial.",
              file=sys.stderr)
        return EXIT_PARTIAL
    if not any_fit_input:
        print("error: no run left to analyse after the shift-mismatch skips", file=sys.stderr)
        return 1
    return 0


def accounting_rows(counts: dict, pairs_label: str, mismatch: bool = True) -> list[dict]:
    """``analysis = accounting`` rows: per benchmark the runs loaded / skipped for a shift
    mismatch (``term = n_runs_shift_mismatch``, emitted even if the benchmark yields no fit),
    and one ``term = all_pairs_skipped`` placeholder per benchmark whose all_pairs scope was
    skipped (the only rows with ``all_pairs_skipped = True``). ``mismatch=False`` omits the
    mismatch rows (``--pairs both`` writes them once, under the first scope)."""
    out = []
    mism = counts.get("n_runs_shift_mismatch", {})
    per_b = counts.get("runs_per_benchmark", {}) if mismatch else {}
    for b, n_loaded in sorted(per_b.items()):
        n = int(mism.get(b, 0))
        out.append({
            "analysis": ACCOUNTING, "hypothesis": ACCOUNTING, "benchmark": b,
            "pairs": pairs_label, "term": "n_runs_shift_mismatch", "estimate": float(n),
            "n_runs_loaded": int(n_loaded), "n_runs_shift_mismatch": n,
            "all_pairs_skipped": False, "descriptive_only": True,
            "note": (f"{n} of {n_loaded} runs skipped: shift result does not match the run"
                     if n else f"all {n_loaded} runs paired with a shift result"),
        })
    if pairs_label == "all_pairs":
        for b in counts.get("all_pairs_skipped", []):
            out.append({
                "analysis": ACCOUNTING, "hypothesis": ACCOUNTING, "benchmark": b,
                "pairs": "all_pairs", "term": "all_pairs_skipped", "estimate": float("nan"),
                "all_pairs_skipped": True, "descriptive_only": True,
                "note": "all_pairs skipped for this benchmark: shift not measured with "
                        "--pairs full",
            })
    return out

if __name__ == "__main__":
    sys.exit(main())
