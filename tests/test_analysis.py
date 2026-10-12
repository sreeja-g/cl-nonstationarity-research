import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from clns.analysis.correlation_study import (  # noqa: E402
    aggregate_transitions,
    align_forgetting_with_transitions,
    check_shift_matches_run,
    forgetting,
    load_accuracy_matrices,
    one_step_forgetting,
    partial_spearman,
    run_transition_table,
    shift_transitions,
    spearman_study,
)
from clns.analysis.correlation_study import main as correlation_main  # noqa: E402
from clns.analysis.rank_inversion import (  # noqa: E402
    bootstrap_flip_stability,
    compute_rankings,
    find_flips,
    kendall_tau,
)


def _results(b2_scores):
    rows = []
    b1 = {"A": 0.9, "B": 0.8, "C": 0.5}
    for bench, scores in (("b1", b1), ("b2", b2_scores)):
        for method, base in scores.items():
            for seed in range(4):
                for exp in range(3):
                    rows.append(
                        {
                            "benchmark": bench,
                            "model": "m",
                            "method": method,
                            "seed": seed,
                            "experience": exp,
                            "acc": base + 0.001 * seed - 0.001 * exp,
                        }
                    )
    return pd.DataFrame(rows)


def test_rankings_and_single_flip():
    df = _results({"A": 0.6, "B": 0.7, "C": 0.4})
    r = compute_rankings(df, "acc")
    b1 = r[r.benchmark == "b1"].set_index("method")["rank"]
    assert (b1["A"], b1["B"], b1["C"]) == (1, 2, 3)
    flips = find_flips(r)
    assert len(flips) == 3
    assert flips["flip"].sum() == 1
    row = flips[flips["flip"]].iloc[0]
    assert {row["method_1"], row["method_2"]} == {"A", "B"}


def test_lower_is_better_reverses_ranks():
    df = _results({"A": 0.6, "B": 0.7, "C": 0.4})
    r = compute_rankings(df, "acc", higher_is_better=False)
    b1 = r[r.benchmark == "b1"].set_index("method")["rank"]
    assert b1["C"] == 1


def test_kendall_tau_reversed():
    df = _results({"A": 0.3, "B": 0.4, "C": 0.6})
    r = compute_rankings(df, "acc")
    p = r.pivot(index="method", columns="benchmark", values="rank")
    assert kendall_tau(p["b1"], p["b2"]) == pytest.approx(-1.0)


def test_bootstrap_flip_stability_full():
    df = _results({"A": 0.6, "B": 0.7, "C": 0.4})
    out = bootstrap_flip_stability(df, "acc", n_boot=30, seed=0)
    flipped = out[out["flip"]]
    assert len(flipped) == 1
    assert flipped["flip_freq"].iloc[0] == pytest.approx(1.0)
    assert (out["stability"] == 1.0).all()


def test_forgetting_with_mapping():
    a = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.9, 0.1],
            [0.7, 0.6, 0.2],
            [0.5, 0.4, 0.95],
        ]
    )
    f, mp = forgetting(a, [0, 1, 3])
    assert list(mp) == [0, 1, 3]
    assert f[0] == pytest.approx(0.9 - 0.5)
    assert f[1] == pytest.approx(0.9 - 0.4)
    assert np.isnan(f[2])  # introduced in the last train experience
    trans = np.array([0.1, 0.2, 0.3])
    sv, fv = align_forgetting_with_transitions(f, mp, trans)
    assert list(sv) == [0.1, 0.2]
    assert fv == pytest.approx([0.4, 0.5])


def test_forgetting_identity_default_and_bad_mapping():
    a = np.array([[0.9, 0.0], [0.5, 0.8]])
    f, mp = forgetting(a)
    assert list(mp) == [0, 1]
    assert f[0] == pytest.approx(0.4) and np.isnan(f[1])
    with pytest.raises(ValueError):
        forgetting(a, [0])
    with pytest.raises(ValueError):
        forgetting(a, [0, 2])


def test_shift_transitions():
    m = np.arange(16, dtype=float).reshape(4, 4)
    assert list(shift_transitions(m)) == [1.0, 6.0, 11.0]


def test_spearman_monotone():
    res = spearman_study(
        [1, 2, 3, 4, 5], [0.1, 0.2, 0.4, 0.5, 0.9], n_boot=100, n_perm=2000, seed=0
    )
    assert res["rho"] == pytest.approx(1.0)
    assert res["p_perm"] < 0.05
    assert res["n"] == 5


def test_load_accuracy_matrices(tmp_path):
    def make(seed, done, mapping=True):
        d = tmp_path / "bm" / "resnet18" / "ewc" / f"seed{seed}"
        d.mkdir(parents=True)
        blob = {"rows": [[0.9, 0.1], [0.5, 0.8]], "scenario": "class_il", "random_init": [0.1, 0.1]}
        if mapping:
            blob["test_exp_to_train_exp"] = [0, 1]
        (d / "accuracy_matrix.json").write_text(json.dumps(blob))
        if done:
            (d / "DONE").write_text("")

    make(0, True)
    make(1, False)
    make(2, True, mapping=False)
    with pytest.warns(UserWarning):
        recs = load_accuracy_matrices(tmp_path)
    assert [r["seed"] for r in recs] == [0, 2]
    r = recs[0]
    assert (r["benchmark"], r["model"], r["method"]) == ("bm", "resnet18", "ewc")
    assert r["rows"].shape == (2, 2)
    assert recs[1]["test_exp_to_train_exp"] == [0, 1]


# --------------------------------------------------------------------------- #
# Rank inversion: effect size, seeds, condition column
# --------------------------------------------------------------------------- #
def _overlapping_results():
    # A and B differ by ~0.01 while the seed SD is ~0.06: the flip is not meaningful.
    vals = {
        "b1": {"A": [0.80, 0.70, 0.60, 0.75, 0.65], "B": [0.62, 0.78, 0.71, 0.66, 0.74]},
        "b2": {"A": [0.72, 0.64, 0.80, 0.69, 0.70], "B": [0.75, 0.66, 0.70, 0.68, 0.71]},
    }
    rows = []
    for bench, by_m in vals.items():
        by_m = {**by_m, "C": [0.3] * 5}
        for method, seeds in by_m.items():
            for seed, v in enumerate(seeds):
                rows.append(
                    {"benchmark": bench, "model": "m", "method": method, "seed": seed, "acc": v}
                )
    return pd.DataFrame(rows)


def test_flip_with_overlapping_seeds_is_small_gap_and_unstable():
    df = _overlapping_results()
    out = bootstrap_flip_stability(df, "acc", n_boot=200, seed=0)
    flipped = out[out["flip"]]
    assert len(flipped) == 1
    row = flipped.iloc[0]
    assert {row["method_1"], row["method_2"]} == {"A", "B"}
    assert bool(row["small_gap"])
    assert abs(row["gap_a"]) < row["seed_sd_a"]
    assert row["stability"] < 1.0
    assert row["stability_status"] == "ok"
    # A/C and B/C are large-gap non-flips.
    other = out[~out["flip"]]
    assert not other["small_gap"].any()


def test_flip_stability_insufficient_seeds():
    df = _results({"A": 0.6, "B": 0.7, "C": 0.4})
    out = bootstrap_flip_stability(df[df["seed"] < 2], "acc", n_boot=10)
    assert (out["stability_status"] == "insufficient_seeds").all()
    assert out["stability"].isna().all()


def test_condition_col_other_than_benchmark(tmp_path):
    df = _results({"A": 0.6, "B": 0.7, "C": 0.4})
    df3 = df[df["benchmark"] == "b1"].assign(benchmark="b3")
    df = pd.concat([df, df3], ignore_index=True)
    df["shift_level"] = df["benchmark"].map({"b1": "low", "b2": "high", "b3": "low"})
    r = compute_rankings(df, "acc", condition_col="shift_level")
    assert len(r) == 6  # 2 levels x 3 methods
    flips = find_flips(r, condition_col="shift_level")
    assert flips["flip"].sum() == 1
    # Rankings grouped by benchmark cannot be compared on shift_level: clear error.
    rb = compute_rankings(df, "acc").merge(
        df[["benchmark", "shift_level"]].drop_duplicates(), on="benchmark"
    )
    with pytest.raises(ValueError, match="several rows"):
        find_flips(rb, condition_col="shift_level")
    csv = tmp_path / "res.csv"
    df.to_csv(csv, index=False)
    from clns.analysis.rank_inversion import main as rank_main

    rank_main([
        "--results-csv", str(csv), "--metric", "acc", "--condition-col", "shift_level",
        "--n-boot", "5", "--out-dir", str(tmp_path / "out"),
    ])
    assert (tmp_path / "out" / "flips.csv").exists()
    assert (tmp_path / "out" / "kendall.csv").exists()


# --------------------------------------------------------------------------- #
# Correlation study: one-step pairing, transition units, null calibration
# --------------------------------------------------------------------------- #
def test_one_step_forgetting():
    a = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.9, 0.1],
            [0.7, 0.6, 0.2],
            [0.5, 0.4, 0.95],
        ]
    )
    d, mp = one_step_forgetting(a, [0, 1, 3])
    assert d[0] == pytest.approx(0.9 - 0.8)
    assert d[1] == pytest.approx(0.9 - 0.6)
    assert np.isnan(d[2])
    rec = {"rows": a, "test_exp_to_train_exp": [0, 1, 3]}
    t = run_transition_table(rec, [0.1, 0.2, 0.3], mode="one_step")
    assert list(t["k"]) == [0, 1]
    assert list(t["shift"]) == [0.1, 0.2]
    assert list(t["position"]) == [3, 2]
    assert t["forget"].to_numpy() == pytest.approx([0.1, 0.3])


def test_aggregate_transitions_means_over_runs():
    tab = pd.DataFrame(
        {
            "unit": ["u0", "u0", "u1", "u1"],
            "run": ["r0", "r1", "r0", "r1"],
            "shift": [1.0, 3.0, 2.0, 2.0],
            "forget": [0.1, 0.3, 0.5, np.nan],
            "position": [2, 2, 1, 1],
        }
    )
    agg = aggregate_transitions(tab).set_index("unit")
    assert agg.loc["u0", "shift"] == pytest.approx(2.0)
    assert agg.loc["u0", "forget"] == pytest.approx(0.2)
    assert agg.loc["u1", "forget"] == pytest.approx(0.5)
    assert agg.loc["u1", "n_runs"] == 2


def test_partial_spearman_removes_position_confound():
    # Both track position; their deviations from position are unrelated
    # (x swaps the first two pairs, y the last two), so rho is high but partial ~ 0.
    pos = np.arange(10.0)
    x = np.array([1, 0, 3, 2, 4, 5, 6, 7, 8, 9], dtype=float)
    y = np.array([0, 1, 2, 3, 4, 5, 7, 6, 9, 8], dtype=float)
    res = spearman_study(x, y, n_boot=0, n_perm=200, covariate=pos)
    assert res["rho"] > 0.9
    assert abs(res["partial_rho"]) < 0.1
    assert res["partial_rho"] == pytest.approx(partial_spearman(x, y, pos))
    assert res["partial_p_perm"] > 0.1


def _null_replicate(rng, n_trans=9, n_runs=35):
    """Reviewer null (sp.py): shared per-transition effect u_k, shift x_k independent."""
    x = rng.random(n_trans)
    u = rng.standard_normal(n_trans)
    y = u[None, :] + 0.3 * rng.standard_normal((n_runs, n_trans))
    return x, y


def test_transition_level_permutation_test_is_calibrated_under_null():
    rng = np.random.default_rng(12345)
    reps, alpha = 200, 0.05
    sig_unit = sig_pooled = 0
    for rep in range(reps):
        x, y = _null_replicate(rng)
        # Correct analysis: one value per transition (mean over runs).
        res = spearman_study(x, y.mean(axis=0), n_boot=0, n_perm=199, seed=rep)
        sig_unit += res["p_perm"] < alpha
        if rep < 40:  # anti-conservative reference: pseudo-replicated pooling
            xp = np.tile(x, y.shape[0])
            res_p = spearman_study(xp, y.ravel(), n_boot=0, n_perm=199, seed=rep)
            sig_pooled += res_p["p_perm"] < alpha
    assert sig_unit / reps <= 0.1
    assert sig_pooled / 40 > 0.3  # the pooled analysis flagged in review


def _write_run(root, bench, model, method, seed, rows, order, sha="abc"):
    d = root / bench / model / method / f"seed{seed}"
    d.mkdir(parents=True)
    blob = {
        "rows": rows,
        "scenario": "class_il",
        "random_init": [0.1] * len(rows[0]),
        "test_exp_to_train_exp": list(range(len(rows[0]))),
    }
    (d / "accuracy_matrix.json").write_text(json.dumps(blob))
    prov = {"benchmark_meta": {"class_order": order, "classes_sha256": sha}}
    (d / "provenance.json").write_text(json.dumps(prov))
    (d / "DONE").write_text("")


def _write_shift(root, bench, seed, matrix, order, sha="abc"):
    d = root / bench / f"seed{seed}"
    d.mkdir(parents=True)
    np.savez(d / "shift_matrix.npz", matrix=np.asarray(matrix, dtype=float))
    meta = {
        "T": len(matrix),
        "benchmark_seed": seed,
        "class_order": order,
        "split_hashes": {"classes_sha256": sha},
        "measure": "marginal_mmd",
    }
    (d / "shift_meta.json").write_text(json.dumps(meta))


def _acc(seed, method_bias):
    t = 4
    a = np.zeros((t, t))
    for i in range(t):
        for j in range(i + 1):
            a[i, j] = 0.9 - 0.1 * (i - j) * (1 + 0.1 * seed) - method_bias * (i - j)
    return a.tolist()


def test_correlation_main_end_to_end(tmp_path):
    runs, shifts = tmp_path / "runs", tmp_path / "shift"
    for seed in range(3):
        order = [int(c) for c in np.roll(np.arange(8), seed)]
        m = np.zeros((4, 4))
        m[0, 1], m[1, 2], m[2, 3] = 0.1 + seed, 0.3 + seed, 0.2 + seed
        m = m + m.T
        _write_shift(shifts, "bm", seed, m, order)
        for method, bias in (("ewc", 0.0), ("er", 0.02)):
            _write_run(runs, "bm", "resnet18", method, seed, _acc(seed, bias), order)
    out = tmp_path / "corr.csv"
    code = correlation_main([
        "--runs-dir", str(runs), "--shift-dir", str(shifts), "--out", str(out),
        "--n-boot", "50", "--n-perm", "99",
    ])
    assert code == 0
    res = pd.read_csv(out)
    assert set(res["method"]) == {"ewc", "er"}  # stratified, never pooled by default
    # Class orders differ per seed here (as with different class_order_seed values): each run
    # matches only the shift result with its order -> 3 orders x 3 transitions are distinct units.
    assert (res["n_transitions"] == 9).all()
    assert (res["n_runs"] == 3).all()
    assert (res["forgetting"] == "one_step").all()
    assert not res["cross_benchmark_comparable"].astype(bool).any()

    code = correlation_main([
        "--runs-dir", str(runs), "--shift-dir", str(shifts), "--out", str(out),
        "--n-boot", "20", "--n-perm", "20", "--forgetting", "cumulative",
        "--two-way-bootstrap",
    ])
    assert code == 0
    res = pd.read_csv(out)
    assert "partial_rho" in res.columns
    assert (res["ci_method"] == "two_way").all()


def test_correlation_guards(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert correlation_main(
        ["--runs-dir", str(empty), "--shift-dir", str(empty), "--out", str(tmp_path / "o.csv")]
    ) == 1

    runs, shifts = tmp_path / "runs", tmp_path / "shift"
    _write_run(runs, "bm", "resnet18", "ewc", 0, _acc(0, 0.0), [0, 1, 2, 3])
    rec = load_accuracy_matrices(runs)[0]
    good = {
        "matrix": np.zeros((4, 4)),
        "dir": "x",
        "meta": {
            "T": 4,
            "benchmark_seed": 0,
            "class_order": [0, 1, 2, 3],
            "split_hashes": {"classes_sha256": "abc"},
        },
    }
    check_shift_matches_run(good, rec)
    # the benchmark seed is not compared (class order is fixed by class_order_seed)
    check_shift_matches_run({**good, "meta": {**good["meta"], "benchmark_seed": 5}}, rec)
    for bad_meta in (
        {"T": 5},
        {"class_order": [3, 2, 1, 0]},
        {"split_hashes": {"classes_sha256": "zzz"}},
    ):
        bad = {**good, "meta": {**good["meta"], **bad_meta}}
        with pytest.raises(ValueError):
            check_shift_matches_run(bad, rec)

    _write_shift(shifts, "bm", 0, np.zeros((5, 5)), [0, 1, 2, 3])  # T mismatch
    args = ["--runs-dir", str(runs), "--shift-dir", str(shifts), "--out", str(tmp_path / "o.csv")]
    with pytest.raises(ValueError, match="T_train"):
        correlation_main([*args, "--strict"])
    # default: the mismatched run is skipped and counted; nothing left -> clean error
    with pytest.warns(UserWarning, match="shift mismatch"):
        assert correlation_main(args) == 1


def test_correlation_skips_and_counts_shift_mismatch(tmp_path):
    runs, shifts = tmp_path / "runs", tmp_path / "shift"
    order = [0, 1, 2, 3, 4, 5, 6, 7]
    m = np.zeros((4, 4))
    m[0, 1], m[1, 2], m[2, 3] = 0.1, 0.3, 0.2
    _write_shift(shifts, "bm", 0, m + m.T, order)
    for seed in range(2):
        _write_run(runs, "bm", "resnet18", "ewc", seed, _acc(seed, 0.0), order)
    _write_run(runs, "bm", "resnet18", "ewc", 2, _acc(2, 0.0), order[::-1])  # other order
    out = tmp_path / "corr.csv"
    with pytest.warns(UserWarning, match="shift mismatch"):
        assert correlation_main([
            "--runs-dir", str(runs), "--shift-dir", str(shifts), "--out", str(out),
            "--n-boot", "10", "--n-perm", "10",
        ]) == 0
    res = pd.read_csv(out)
    assert (res["n_runs"] == 2).all() and (res["n_runs_shift_mismatch"] == 1).all()


def test_shift_matched_by_class_order_not_seed(tmp_path):
    """One shift measurement (seed0) serves runs of every training seed with the same order."""
    runs, shifts = tmp_path / "runs", tmp_path / "shift"
    order = [3, 1, 0, 2, 7, 5, 4, 6]
    m = np.zeros((4, 4))
    m[0, 1], m[1, 2], m[2, 3] = 0.1, 0.3, 0.2
    _write_shift(shifts, "bm", 0, m + m.T, order)
    for seed in range(3):
        _write_run(runs, "bm", "resnet18", "ewc", seed, _acc(seed, 0.0), order)
    out = tmp_path / "corr.csv"
    assert correlation_main([
        "--runs-dir", str(runs), "--shift-dir", str(shifts), "--out", str(out),
        "--n-boot", "10", "--n-perm", "10",
    ]) == 0
    res = pd.read_csv(out)
    assert (res["n_runs"] == 3).all()
    assert (res["n_transitions"] == 3).all()  # shared order -> the same 3 transitions


def test_select_shift_preference_and_conflicts(tmp_path):
    from clns.analysis.correlation_study import load_shift_candidates, select_shift

    runs, shifts = tmp_path / "runs", tmp_path / "shift"
    order = [0, 1, 2, 3]
    for s in (0, 1):
        _write_shift(shifts, "bm", s, np.zeros((4, 4)), order)
    _write_shift(shifts, "bm", 2, np.zeros((4, 4)), [3, 2, 1, 0])  # other order: never matches
    _write_run(runs, "bm", "resnet18", "ewc", 1, _acc(1, 0.0), order)
    _write_run(runs, "bm", "resnet18", "ewc", 7, _acc(7, 0.0), order)
    recs = {r["seed"]: r for r in load_accuracy_matrices(runs)}
    cands = load_shift_candidates(shifts, "bm")
    assert [c["seed"] for c in cands] == [0, 1, 2]
    assert select_shift(cands, recs[1])["seed"] == 1  # own seed preferred among matches
    # seeds 0 and 1 both match, neither is seed 7: identical matrices -> accepted (lowest seed)
    assert select_shift(cands, recs[7])["seed"] == 0
    differ = [cands[0], {**cands[1], "matrix": np.ones((4, 4))}, cands[2]]
    with pytest.raises(ValueError, match="conflicting"):
        select_shift(differ, recs[7])  # ... but differing matches conflict
    assert select_shift(cands[:1] + cands[2:], recs[7])["seed"] == 0  # single match, any seed
    with pytest.raises(ValueError, match="no shift result matches"):
        select_shift(cands[2:], recs[7])


def test_domain_il_shift_without_class_order(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "dn", "resnet18", "ewc", 2, _acc(2, 0.0), None)
    rec = load_accuracy_matrices(runs)[0]
    sh = {"matrix": np.zeros((4, 4)), "dir": "x", "seed": 0,
          "meta": {"T": 4, "benchmark_seed": 0, "class_order": None,
                   "split_hashes": {"classes_sha256": "abc"}}}
    check_shift_matches_run(sh, rec)  # absent on both sides -> match (seed irrelevant)
    with pytest.raises(ValueError, match="class_order"):
        check_shift_matches_run({**sh, "meta": {**sh["meta"], "class_order": [0, 1]}}, rec)


def test_correlation_units_follow_actual_task_pairs(tmp_path):
    """DomainNet-like: seeds with different domain orders must NOT share transition units
    (v1 mapped transition k of every seed to ``fixed|k``); equal orders share them."""
    runs, shifts = tmp_path / "runs", tmp_path / "shift"
    doms = ["clipart", "real", "sketch", "painting"]
    orders = {0: doms, 1: doms[::-1], 2: doms}  # seeds 0 and 2 share an order
    m = np.zeros((4, 4))
    m[0, 1], m[1, 2], m[2, 3] = 0.1, 0.3, 0.2
    for seed, order in orders.items():
        sha = ",".join(order)
        d = shifts / "dn" / f"seed{seed}"
        d.mkdir(parents=True)
        np.savez(d / "shift_matrix.npz", matrix=m + m.T)
        (d / "shift_meta.json").write_text(json.dumps({
            "T": 4, "class_order": None, "task_order_sha256": sha,
            "split_hashes": {"task_order_sha256": sha}}))
        _write_run(runs, "dn", "resnet18", "ewc", seed, _acc(seed, 0.0), None)
        prov = runs / "dn" / "resnet18" / "ewc" / f"seed{seed}" / "provenance.json"
        prov.write_text(json.dumps({"benchmark_meta": {
            "class_order": None, "task_order": order, "task_order_sha256": sha}}))
        acc = prov.parent / "accuracy_matrix.json"
        acc.write_text(json.dumps({**json.loads(acc.read_text()), "scenario": "domain_il"}))
    out = tmp_path / "corr.csv"
    assert correlation_main([
        "--runs-dir", str(runs), "--shift-dir", str(shifts), "--out", str(out),
        "--n-boot", "10", "--n-perm", "10",
    ]) == 0
    res = pd.read_csv(out)
    # 3 transitions of order A (seeds 0, 2) + 3 transitions of the reversed order B (seed 1)
    # = 6 units, not the 3 that "fixed|k" would give.
    assert (res["n_transitions"] == 6).all()
