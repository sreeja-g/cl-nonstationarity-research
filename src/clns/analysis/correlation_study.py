"""Shift-forgetting correlation study.

Accuracy matrix convention: ``a[i][j]`` = accuracy on TEST experience j after
training on TRAIN experience i (shape T_train x T_test). ``test_exp_to_train_exp[j]``
= m_j, the train experience that introduced test experience j (identity for most
benchmarks; shorter than T_train for CORe50, RESEARCH_LOG 7.2.1).

Shift: ``shift_transitions(M)[k] = M[k, k+1]`` is the (marginal MMD^2) shift of the
transition INTO train experience k+1; M comes from ``clns.shift.measure_stream``. A run is
matched to ANY shift result of its benchmark (``<shift-dir>/<benchmark>/seed*/``) whose
``shift_meta.json`` describes the same benchmark instance as the run's ``provenance.json``
``benchmark_meta``: same class order (or no class order on both sides, domain-IL), same split
hashes and T == T_train. The class order is fixed by ``benchmark.class_order_seed``, NOT by the
training seed, so one shift measurement per benchmark normally serves every training seed. If
several shift results match, the one measured with the run's own seed is preferred; several
matches without that tie-break, or none at all, are an error (:func:`select_shift`).

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
a transition (same class order => same transition; different class orders, e.g. a changed
``class_order_seed``, make different transitions), the shift of a transition is averaged over
the matched shift matrices of those seeds, and Spearman is computed over transitions. The permutation p-value
permutes the shift values across transitions (n = n_transitions); the bootstrap CI
resamples transitions (optionally two-way: runs, then transitions). Pooling
models/methods is only done with ``--scope pooled`` and is flagged in the output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FORGETTING_MODES = ("one_step", "cumulative")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_accuracy_matrices(runs_dir) -> list[dict]:
    """Load finished runs under ``runs_dir/<benchmark>/<model>/<method>/seed<k>/``.

    Runs without a ``DONE`` marker are skipped with a warning. Each record has keys
    benchmark, model, method, seed, path, rows (np.ndarray T_train x T_test),
    test_exp_to_train_exp (list[int]), scenario, random_init, benchmark_meta (from
    ``provenance.json``; None if absent).
    """
    runs_dir = Path(runs_dir)
    records = []
    for p in sorted(runs_dir.glob("*/*/*/seed*/accuracy_matrix.json")):
        run = p.parent
        m = re.fullmatch(r"seed(\d+)", run.name)
        if m is None:
            continue
        if not (run / "DONE").exists():
            warnings.warn(f"skipping unfinished run (no DONE): {run}", stacklevel=2)
            continue
        with open(p, encoding="utf-8") as fh:
            blob = json.load(fh)
        rows = np.asarray(blob["rows"], dtype=float)
        mapping = blob.get("test_exp_to_train_exp")
        if mapping is None:
            mapping = list(range(rows.shape[1]))
        bmeta = None
        prov = run / "provenance.json"
        if prov.exists():
            with open(prov, encoding="utf-8") as fh:
                bmeta = json.load(fh).get("benchmark_meta")
        records.append(
            {
                "benchmark": run.parent.parent.parent.name,
                "model": run.parent.parent.name,
                "method": run.parent.name,
                "seed": int(m.group(1)),
                "path": str(p),
                "rows": rows,
                "test_exp_to_train_exp": [int(v) for v in mapping],
                "scenario": blob.get("scenario"),
                "random_init": blob.get("random_init"),
                "benchmark_meta": bmeta,
            }
        )
    return records


def load_shift(shift_dir, benchmark: str, seed: int) -> dict | None:
    """Shift matrix + meta of ``<shift_dir>/<benchmark>/seed<seed>/``; None if not present.

    Layout written by ``clns.shift.measure_stream``:
    ``<shift_dir>/<benchmark>/seed<k>/{shift_matrix.npz, shift_meta.json}``.
    """
    d = Path(shift_dir) / benchmark / f"seed{seed}"
    npz, meta = d / "shift_matrix.npz", d / "shift_meta.json"
    if not npz.exists() or not meta.exists():
        return None
    with open(meta, encoding="utf-8") as fh:
        m = json.load(fh)
    return {"matrix": np.load(npz)["matrix"], "meta": m, "dir": str(d), "seed": int(seed)}


def load_shift_candidates(shift_dir, benchmark: str) -> list[dict]:
    """All shift results of ``benchmark`` (every ``seed<k>`` directory), sorted by seed."""
    root = Path(shift_dir) / benchmark
    if not root.is_dir():
        return []
    out = []
    for d in root.glob("seed*"):
        m = re.fullmatch(r"seed(\d+)", d.name)
        if m is None:
            continue
        sh = load_shift(shift_dir, benchmark, int(m.group(1)))
        if sh is not None:
            out.append(sh)
    return sorted(out, key=lambda sh: sh["seed"])


def _order(o):
    return None if o is None else [int(c) for c in o]


def shift_mismatch(shift: dict, rec: dict) -> str | None:
    """Why ``shift`` does NOT describe the run's benchmark instance; None if it does.

    Compared against the run's ``provenance.json`` ``benchmark_meta``: T (shift) == T_train
    (run); equal class order (absent on both sides, domain-IL, counts as equal); every split
    hash the shift records equals the run's value where the run records it. The benchmark
    seed is NOT compared: the class order is fixed by ``benchmark.class_order_seed``. A run
    without ``benchmark_meta`` cannot be verified and never matches.
    """
    meta = shift["meta"]
    t_shift = int(meta.get("T", np.asarray(shift["matrix"]).shape[0]))
    t_train = int(np.asarray(rec["rows"]).shape[0])
    if t_shift != t_train:
        return f"shift T={t_shift} != run T_train={t_train}"
    bmeta = rec.get("benchmark_meta")
    if bmeta is None:
        return "run has no provenance benchmark_meta; cannot verify class order / split hashes"
    if _order(meta.get("class_order")) != _order(bmeta.get("class_order")):
        return "class_order mismatch between run and shift matrix"
    for k, v in (meta.get("split_hashes") or {}).items():
        if k in bmeta and str(bmeta[k]) != str(v):
            return f"split hash {k} mismatch between run and shift matrix"
    return None


def check_shift_matches_run(shift: dict, rec: dict) -> None:
    """Raise ValueError unless the shift matrix belongs to the run's benchmark instance."""
    why = shift_mismatch(shift, rec)
    if why is not None:
        raise ValueError(f"{why} ({rec['path']} vs {shift['dir']})")


def select_shift(candidates: list[dict], rec: dict) -> dict:
    """The shift result to pair with run ``rec`` (matching rule: :func:`shift_mismatch`).

    Exactly one match -> it. Several matches -> the one measured with the run's own seed
    (``seed<k>`` dir == run seed), else ValueError (conflicting). No match -> ValueError
    listing why each candidate was rejected.
    """
    ok, rejected = [], []
    for sh in candidates:
        why = shift_mismatch(sh, rec)
        if why is None:
            ok.append(sh)
        else:
            rejected.append(f"{sh['dir']}: {why}")
    if len(ok) == 1:
        return ok[0]
    if len(ok) > 1:
        same = [sh for sh in ok if sh["seed"] == int(rec["seed"])]
        if len(same) == 1:
            return same[0]
        raise ValueError(
            f"conflicting shift results for {rec['path']}: {[sh['dir'] for sh in ok]} all "
            "match its class order / split hashes / T and none was measured with the run's "
            "seed; keep only one of them"
        )
    raise ValueError(f"no shift result matches {rec['path']}: " + "; ".join(rejected))


# --------------------------------------------------------------------------- #
# Forgetting / alignment
# --------------------------------------------------------------------------- #
def _mapping(a: np.ndarray, test_exp_to_train_exp) -> np.ndarray:
    t_train, t_test = a.shape
    mapping = np.arange(t_test) if test_exp_to_train_exp is None else np.asarray(
        test_exp_to_train_exp, dtype=int
    )
    if mapping.shape[0] != t_test:
        raise ValueError(
            f"mapping length {mapping.shape[0]} != number of test experiences {t_test}"
        )
    if mapping.size and (mapping.min() < 0 or mapping.max() >= t_train):
        raise ValueError("mapping entries must lie in [0, T_train)")
    return mapping


def forgetting(a, test_exp_to_train_exp=None) -> tuple[np.ndarray, np.ndarray]:
    """CUMULATIVE forgetting per test experience (secondary; position-confounded).

    f_j = max_{m_j <= i <= T_train-2} a[i][j] - a[T_train-1][j]; NaN where
    m_j >= T_train - 1. Returns (f, mapping).
    """
    a = np.asarray(a, dtype=float)
    t_train, t_test = a.shape
    mapping = _mapping(a, test_exp_to_train_exp)
    f = np.full(t_test, np.nan)
    for j in range(t_test):
        m_j = int(mapping[j])
        if m_j < t_train - 1:
            f[j] = a[m_j : t_train - 1, j].max() - a[t_train - 1, j]
    return f, mapping


def one_step_forgetting(a, test_exp_to_train_exp=None) -> tuple[np.ndarray, np.ndarray]:
    """ONE-STEP forgetting per test experience (primary pairing).

    d_j = a[m_j][j] - a[m_j+1][j], the drop right after the next transition
    m_j -> m_j+1; NaN where m_j >= T_train - 1. Returns (d, mapping).
    """
    a = np.asarray(a, dtype=float)
    t_train, t_test = a.shape
    mapping = _mapping(a, test_exp_to_train_exp)
    d = np.full(t_test, np.nan)
    for j in range(t_test):
        m_j = int(mapping[j])
        if m_j < t_train - 1:
            d[j] = a[m_j, j] - a[m_j + 1, j]
    return d, mapping


def shift_transitions(shift_matrix) -> np.ndarray:
    """First off-diagonal of the shift matrix: M[k, k+1] for k = 0..T-2."""
    return np.diag(np.asarray(shift_matrix, dtype=float), k=1)


def align_forgetting_with_transitions(f, mapping, transitions) -> tuple[np.ndarray, np.ndarray]:
    """Pair forgetting of test exp j with transition index m_j (into train exp m_j+1).

    Returns (shift_vec, forget_vec) for all j with finite f_j and m_j < len(transitions).
    """
    f = np.asarray(f, dtype=float)
    mapping = np.asarray(mapping, dtype=int)
    transitions = np.asarray(transitions, dtype=float)
    keep = np.isfinite(f) & (mapping < len(transitions))
    return transitions[mapping[keep]], f[keep]


def run_transition_table(rec: dict, transitions, mode: str = "one_step") -> pd.DataFrame:
    """Per-run table: one row per transition k with forgetting (mean over test exps j
    with m_j = k), the shift M[k, k+1] and the position covariate T_train-1-k.

    Rows with non-finite forgetting or shift are kept (NaN) so callers can count drops.
    """
    if mode not in FORGETTING_MODES:
        raise ValueError(f"mode must be one of {FORGETTING_MODES}, got {mode!r}")
    fn = one_step_forgetting if mode == "one_step" else forgetting
    a = np.asarray(rec["rows"], dtype=float)
    f, mp = fn(a, rec["test_exp_to_train_exp"])
    transitions = np.asarray(transitions, dtype=float)
    t_train = a.shape[0]
    rows = []
    for k in sorted(set(int(m) for m in mp if m < t_train - 1)):
        vals = f[mp == k]
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "k": k,
                "forget": float(vals.mean()) if vals.size else float("nan"),
                "shift": float(transitions[k]) if k < len(transitions) else float("nan"),
                "position": t_train - 1 - k,
            }
        )
    return pd.DataFrame(rows, columns=["k", "forget", "shift", "position"])


def transition_id(rec: dict, k: int) -> str:
    """Identity of transition k of a run: runs share it iff they share the class order.

    Domain-IL benchmarks (no class order) share transition k across all seeds.
    """
    bmeta = rec.get("benchmark_meta") or {}
    order = bmeta.get("class_order")
    tag = "fixed" if order is None else ",".join(str(c) for c in order)
    return f"{tag}|{k}"


def aggregate_transitions(table: pd.DataFrame) -> pd.DataFrame:
    """Collapse a run-level table (columns unit, forget, shift, position, run) to one
    row per transition unit: mean forgetting and mean shift over runs."""
    g = table.groupby("unit", sort=True)
    return pd.DataFrame(
        {
            "shift": g["shift"].mean(),
            "forget": g["forget"].mean(),
            "position": g["position"].first(),
            "n_runs": g["run"].nunique(),
        }
    ).reset_index()


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


def build_run_table(recs, shift_dir, mode: str) -> tuple[pd.DataFrame, dict]:
    """Run-level transition table for all records with matching shift matrices.

    Each run is paired with the shift result of its benchmark chosen by :func:`select_shift`
    (any seed dir with the same class order / split hashes / T; the run's own seed preferred).
    Returns (table, counts). A benchmark without any shift result is skipped with a warning;
    raises ValueError if shift results exist but none (or several, conflicting) match a run.
    """
    frames = []
    counts = {"runs_missing_shift": 0, "pairs_nan": 0, "pairs_total": 0}
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
        shift = select_shift(candidates, r)
        t = run_transition_table(r, shift_transitions(shift["matrix"]), mode)
        counts["pairs_total"] += len(t)
        counts["pairs_nan"] += int((~(np.isfinite(t["shift"]) & np.isfinite(t["forget"]))).sum())
        run_id = f"{r['benchmark']}/{r['model']}/{r['method']}/seed{r['seed']}"
        t = t.assign(
            benchmark=r["benchmark"], model=r["model"], method=r["method"], seed=r["seed"],
            run=run_id, unit=[transition_id(r, int(k)) for k in t["k"]],
        )
        frames.append(t)
    cols = ["k", "forget", "shift", "position", "benchmark", "model", "method", "seed", "run",
            "unit"]
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
    args = ap.parse_args(argv)

    recs = load_accuracy_matrices(args.runs_dir)
    if not recs:
        print(f"error: no finished runs (with DONE) under {args.runs_dir}", file=sys.stderr)
        return 1
    table, counts = build_run_table(recs, args.shift_dir, args.forgetting)
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
            note=_note(args.forgetting, args.scope),
        )
        rows.append(res)
    print(
        f"runs: {len(recs)} loaded, {counts['runs_missing_shift']} without shift matrix; "
        f"run-level pairs: {counts['pairs_total']} total, {counts['pairs_nan']} NaN (dropped)"
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
