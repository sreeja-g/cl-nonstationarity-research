"""Rank-inversion analysis: do method rankings flip across benchmarks / shift levels?

Input is a long-format DataFrame with one row per (condition, model, method, seed
[, experience]) and a metric column. ``condition_col`` (default "benchmark") may be
replaced by any grouping column, e.g. a shift-level bin; rankings are then computed
per (condition, model), averaging over everything else in that cell.

Caveats (read before reporting flips):

* Effect size: a flip between two methods whose scores differ by less than the
  seed-to-seed spread is not meaningful. Every comparison therefore reports the
  score gaps and the pooled seed SD in both conditions and ``small_gap=True`` when
  |gap| < pooled seed SD in either condition.
* Seed bootstrap: with fewer than 3 seeds the resampling distribution is degenerate;
  stability is then NaN with ``stability_status="insufficient_seeds"``.
* Multiplicity: the number of comparisons is (#condition pairs) x (#method pairs) x
  (#models); no correction is applied, so some flips are expected by chance.
  TODO: Benjamini-Hochberg FDR over flip tests (needs a per-comparison p-value design).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MIN_SEEDS_FOR_STABILITY = 3


def compute_rankings(
    df: pd.DataFrame,
    metric: str,
    higher_is_better: bool = True,
    condition_col: str = "benchmark",
    model_col: str = "model",
    method_col: str = "method",
    seed_col: str = "seed",
) -> pd.DataFrame:
    """Mean metric over experiences then seeds per (condition, model, method); rank 1 = best.

    Grouping is always (condition_col, model_col), so the result has exactly one row
    per (condition, model, method). Ties share the minimum rank. Returns columns
    [condition_col, model_col, method_col, "score", "seed_sd", "n_seeds", "rank"],
    where seed_sd is the SD (ddof=1) of the per-seed means (NaN for one seed).
    """
    group_cols = [condition_col, model_col]
    keys = [*group_cols, method_col]
    per_seed = df.groupby([*keys, seed_col], as_index=False)[metric].mean()
    g = per_seed.groupby(keys)[metric]
    out = pd.DataFrame(
        {"score": g.mean(), "seed_sd": g.std(ddof=1), "n_seeds": g.count()}
    ).reset_index()
    out["rank"] = out.groupby(group_cols)["score"].rank(
        ascending=not higher_is_better, method="min"
    )
    return out


def _cell_lookup(sub: pd.DataFrame, condition_col: str, method_col: str) -> dict:
    """{(method, condition): row} for one model; raises on duplicate cells."""
    dup = sub.duplicated([method_col, condition_col])
    if dup.any():
        bad = sub.loc[dup, [method_col, condition_col]].head(3).to_dict("records")
        raise ValueError(
            f"rankings contain several rows per (method, {condition_col}), e.g. {bad}; "
            f"compute them with compute_rankings(..., condition_col={condition_col!r})"
        )
    return {(r[method_col], r[condition_col]): r for _, r in sub.iterrows()}


def find_flips(
    rankings: pd.DataFrame,
    condition_col: str = "benchmark",
    model_col: str = "model",
    method_col: str = "method",
) -> pd.DataFrame:
    """All (condition pair x method pair) comparisons within each model.

    A flip means the relative order of two methods differs between the conditions
    (strictly opposite signs of the rank difference). Per comparison: rank diffs,
    score gaps ``gap_a``/``gap_b`` (method_1 - method_2), pooled seed SDs
    ``seed_sd_a``/``seed_sd_b`` = sqrt((sd_1^2 + sd_2^2)/2), and ``small_gap`` = True
    when |gap| < pooled seed SD in either condition (NaN SDs never flag).
    """
    has_sd = "seed_sd" in rankings.columns and "score" in rankings.columns
    rows = []
    for model, sub in rankings.groupby(model_col):
        cell = _cell_lookup(sub, condition_col, method_col)
        conds = sorted(sub[condition_col].unique())
        for ca, cb in itertools.combinations(conds, 2):
            methods = sorted(
                m for m in sub[method_col].unique() if (m, ca) in cell and (m, cb) in cell
            )
            for m1, m2 in itertools.combinations(methods, 2):
                da = float(cell[(m1, ca)]["rank"] - cell[(m2, ca)]["rank"])
                db = float(cell[(m1, cb)]["rank"] - cell[(m2, cb)]["rank"])
                rec = {
                    model_col: model,
                    "condition_a": ca,
                    "condition_b": cb,
                    "method_1": m1,
                    "method_2": m2,
                    "rank_diff_a": da,
                    "rank_diff_b": db,
                    "flip": bool(np.sign(da) * np.sign(db) < 0),
                }
                small = False
                for tag, c in (("a", ca), ("b", cb)):
                    if has_sd:
                        gap = float(cell[(m1, c)]["score"] - cell[(m2, c)]["score"])
                        sd = float(np.sqrt(
                            (cell[(m1, c)]["seed_sd"] ** 2 + cell[(m2, c)]["seed_sd"] ** 2) / 2.0
                        ))
                    else:
                        gap, sd = float("nan"), float("nan")
                    rec[f"gap_{tag}"] = gap
                    rec[f"seed_sd_{tag}"] = sd
                    if np.isfinite(sd) and abs(gap) < sd:
                        small = True
                rec["small_gap"] = small
                rows.append(rec)
    cols = [
        model_col, "condition_a", "condition_b", "method_1", "method_2",
        "rank_diff_a", "rank_diff_b", "flip",
        "gap_a", "seed_sd_a", "gap_b", "seed_sd_b", "small_gap",
    ]
    return pd.DataFrame(rows, columns=cols)


def kendall_tau(rank_a, rank_b) -> float:
    """Kendall tau-b between two rank vectors (same method order)."""
    tau, _ = stats.kendalltau(np.asarray(rank_a), np.asarray(rank_b))
    return float(tau)


def bootstrap_flip_stability(
    df: pd.DataFrame,
    metric: str,
    higher_is_better: bool = True,
    n_boot: int = 200,
    seed: int = 0,
    condition_col: str = "benchmark",
    model_col: str = "model",
    method_col: str = "method",
    seed_col: str = "seed",
) -> pd.DataFrame:
    """Resample seeds with replacement and recompute flips.

    Returns the flips of the full data plus ``flip_freq`` (fraction of resamples in
    which the pair flips), ``stability`` (fraction agreeing with the original flip
    status) and ``stability_status``. Seeds are resampled jointly across all
    conditions. With fewer than ``MIN_SEEDS_FOR_STABILITY`` (3) distinct seeds the
    bootstrap is not run: stability/flip_freq are NaN and the status is
    "insufficient_seeds". See the module docstring for multiplicity caveats.
    """

    def flips_of(d: pd.DataFrame) -> pd.DataFrame:
        r = compute_rankings(d, metric, higher_is_better, condition_col, model_col,
                             method_col, seed_col)
        return find_flips(r, condition_col, model_col, method_col)

    base = flips_of(df)
    key = [model_col, "condition_a", "condition_b", "method_1", "method_2"]
    seeds = np.array(sorted(df[seed_col].unique()))
    out = base.copy()
    if len(seeds) < MIN_SEEDS_FOR_STABILITY:
        out["flip_freq"] = np.nan
        out["stability"] = np.nan
        out["stability_status"] = "insufficient_seeds"
        return out
    rng = np.random.default_rng(seed)
    counts = pd.Series(0.0, index=pd.MultiIndex.from_frame(base[key]))
    for _ in range(n_boot):
        draw = rng.choice(seeds, size=len(seeds), replace=True)
        parts = [df[df[seed_col] == s].assign(**{seed_col: k}) for k, s in enumerate(draw)]
        f = flips_of(pd.concat(parts, ignore_index=True))
        flipped = f[f["flip"]]
        for idx in pd.MultiIndex.from_frame(flipped[key]):
            if idx in counts.index:
                counts[idx] += 1.0
    freq = counts.to_numpy() / max(n_boot, 1)
    out["flip_freq"] = freq
    out["stability"] = np.where(out["flip"], freq, 1.0 - freq)
    out["stability_status"] = "ok"
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Rank-inversion analysis from a long-format CSV.")
    ap.add_argument("--results-csv", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--lower-is-better", action="store_true")
    ap.add_argument("--condition-col", default="benchmark")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.results_csv)
    hib = not args.lower_is_better
    cond = args.condition_col
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rankings = compute_rankings(df, args.metric, hib, condition_col=cond)
    rankings.to_csv(out / "rankings.csv", index=False)
    stab = bootstrap_flip_stability(
        df, args.metric, hib, args.n_boot, args.seed, condition_col=cond
    )
    stab.to_csv(out / "flips.csv", index=False)

    taus = []
    for model, sub in rankings.groupby("model"):
        piv = sub.set_index(["method", cond])["rank"].unstack(cond)
        for ca, cb in itertools.combinations(piv.columns, 2):
            pair = piv[[ca, cb]].dropna()
            if len(pair) >= 2:
                taus.append(
                    {
                        "model": model,
                        "condition_a": ca,
                        "condition_b": cb,
                        "n_methods": len(pair),
                        "kendall_tau": kendall_tau(pair[ca], pair[cb]),
                    }
                )
    pd.DataFrame(taus).to_csv(out / "kendall.csv", index=False)
    n_flip = int(stab["flip"].sum()) if len(stab) else 0
    print(
        f"wrote rankings.csv, flips.csv, kendall.csv to {out} "
        f"({n_flip} flips in {len(stab)} comparisons; no multiplicity correction)"
    )


if __name__ == "__main__":
    main()
