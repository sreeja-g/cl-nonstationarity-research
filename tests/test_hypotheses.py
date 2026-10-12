"""H1/H2 analysis: planted effects, calibration under the null, clustering, flags, CLI."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from clns.analysis import hypotheses as _hyp  # noqa: E402
from clns.analysis.hypotheses import (  # noqa: E402
    AllPairsUnavailable,
    _chance,
    analyze,
    flag_near_chance,
    flag_tuned,
    load_table,
    ols,
    parse_tuned,
    prepare,
    wild_cluster_bootstrap_t,
)
from clns.analysis.transitions import (  # noqa: E402
    all_pairs_table,
    shift_mismatch,
    transition_id,
)

hyp_main = _hyp.main
REPO = Path(__file__).resolve().parents[1]
T = 12


def _world_matrix(rng, t=T):
    """Symmetric shift matrix |x_i - x_j| from a latent position per task."""
    x = np.cumsum(rng.uniform(0.1, 1.0, t)) * rng.choice([-1, 1], t)
    return np.abs(x[:, None] - x[None, :])


def _write_shift(root, bench, seed, m, sha="order-sha"):
    d = root / bench / f"seed{seed}"
    d.mkdir(parents=True)
    np.savez(d / "shift_matrix.npz", matrix=m)
    meta = {"T": len(m), "seed": seed, "class_order": None, "task_order_sha256": sha,
            "split_hashes": {"task_order_sha256": sha}}
    (d / "shift_meta.json").write_text(json.dumps(meta))


def _write_run(root, bench, arm, method, seed, rows, bmeta, scenario="domain_il", mapping=None):
    d = root / bench / arm / method / f"seed{seed}"
    d.mkdir(parents=True)
    t = len(rows[0])
    (d / "accuracy_matrix.json").write_text(json.dumps({
        "rows": rows, "scenario": scenario, "random_init": [0.1] * t,
        "test_exp_to_train_exp": list(range(t)) if mapping is None else mapping,
    }))
    (d / "provenance.json").write_text(json.dumps({"benchmark_meta": bmeta}))
    (d / "DONE").write_text("")


def _acc(m, slope, noise_u, rng, row_noise=0.01, a_tt=0.8):
    """a[t+1, j] = a[t, j] - (slope * M[j, t+1] + u_t + eps): one-step drop = planted effect."""
    t = len(m)
    a = np.zeros((t, t))
    for i in range(t):
        a[i, i] = a_tt + 0.01 * rng.standard_normal()  # plasticity varies (else = intercept)
        if i > 0:
            for j in range(i):
                drop = slope * m[j, i] + noise_u[i - 1] + row_noise * rng.standard_normal()
                a[i, j] = a[i - 1, j] - drop
    return a.tolist()


def _build_world(tmp_path, slopes, arms, seeds=(0, 1, 2), u_sd=0.0, seed=0, bench="bm",
                 a_tt=0.8):
    """slopes: {(arm, method): slope}. Shared task order and shift matrix across all runs."""
    rng = np.random.default_rng(seed)
    runs, shifts = tmp_path / "runs", tmp_path / "shift"
    m = _world_matrix(rng)
    bmeta = {"task_order": [f"b{i}" for i in range(T)], "task_order_sha256": "order-sha",
             "train_classes": [list(range(10))] * T}
    u = u_sd * rng.standard_normal(T)  # transition-level noise shared by every run
    for s in seeds:
        _write_shift(shifts, bench, s, m)
    for arm in arms:
        for method in ("naive", "ewc", "lwf", "er", "derpp"):
            for s in seeds:
                a = _acc(m, slopes[(arm, method)], u, rng, a_tt=a_tt)
                _write_run(runs, bench, arm, method, s, a, bmeta)
    return runs, shifts


def _slopes(arms, reg, rep, pre=None):
    out = {}
    for arm in arms:
        for method in ("naive", "ewc", "lwf", "er", "derpp"):
            base = reg if method in ("ewc", "lwf", "naive") else rep
            if pre is not None and arm.endswith("_places"):
                base = base * pre
            out[(arm, method)] = base
    return out


def _result(rows, hypothesis, scope, term, pairs="consecutive", variant="primary",
            analysis="pooled_transition"):
    r = [x for x in rows if x["hypothesis"] == hypothesis and x["scope"] == scope
         and x["term"] == term and x["pairs"] == pairs and x["variant"] == variant
         and x["analysis"] == analysis]
    assert len(r) == 1, (hypothesis, scope, term, analysis, len(r))
    return r[0]


def _run_analysis(runs, shifts, pairs="consecutive", n_boot=300, **kw):
    table, _ = load_table(runs, shifts, pairs)
    df, dropped = prepare(table, **kw)
    label = "consecutive" if pairs == "consecutive" else "all_pairs"
    return analyze(df, label, dropped, n_boot=n_boot, seed=0)


# --------------------------------------------------------------------------- #
def test_ols_recovers_coefficients_and_flags_rank_deficiency():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(50), rng.standard_normal(50), rng.standard_normal(50)])
    y = X @ np.array([1.0, 2.0, -3.0])
    assert ols(X, y) == pytest.approx([1.0, 2.0, -3.0])
    assert ols(np.column_stack([X, X[:, 1]]), y) is None


def test_h1_planted_effect_has_positive_delta(tmp_path):
    arms = ["r18_scratch"]
    runs, shifts = _build_world(tmp_path, _slopes(arms, reg=0.10, rep=0.02), arms, u_sd=0.01)
    rows = _run_analysis(runs, shifts)
    d = _result(rows, "H1", "arm=r18_scratch", "delta")
    assert d["estimate"] == pytest.approx(0.08 * _shift_sd(runs, shifts), abs=0.01)
    assert d["ci_lo"] > 0
    assert d["n_clusters"] == T - 1
    # reference coding: slope_regularization - slope_replay = delta, interaction = -delta
    assert _result(rows, "H1", "arm=r18_scratch", "interaction")["estimate"] == pytest.approx(
        -d["estimate"])
    # naive is a separate descriptive stratum, never in the contrast
    assert _result(rows, "H1_naive", "arm=r18_scratch", "slope")["estimate"] > 0
    assert d["n_boot_ok"] > 0


def _shift_sd(runs, shifts):
    """Std of the consecutive shift over the table rows (shift_z slope = raw slope * sd)."""
    table, _ = load_table(runs, shifts, "consecutive")
    return float(table["shift"].std(ddof=0))


def _null_world(rng, k, v_sd=0.3):
    """In-memory H1 table under the null (equal slopes): transition-level shocks u (shared)
    and v (per family), i.e. errors correlated within a transition."""
    x = rng.standard_normal(k)
    u = 0.5 * rng.standard_normal(k)
    rows = []
    for fam, meths in (("regularization", ("ewc", "lwf")), ("replay", ("er", "derpp"))):
        v = v_sd * rng.standard_normal(k)
        for m in meths:
            for s in range(3):
                for t in range(k):
                    rows.append(dict(
                        benchmark="b", model="r18_scratch", method=m, seed=s, run=f"{m}{s}",
                        k=t, t=t, forget=0.1 * x[t] + u[t] + v[t] + 0.05 * rng.standard_normal(),
                        shift=x[t], a_tt=0.8 + 0.05 * rng.standard_normal(), position=k - 1 - t,
                        cluster=f"c{t}", chance=0.1, family=fam))
    return pd.DataFrame(rows)


def test_h1_null_is_calibrated_wild_cluster_bootstrap_t():
    """Equal slopes, transition-correlated errors, G = 40 transitions: the 95% wild cluster
    bootstrap-t CI of delta excludes 0 in about 5% of 120 replicates (fixed seeds: the counts
    are deterministic; observed 9/120 by transition vs 15/120 when rows are the clusters)."""
    rng = np.random.default_rng(7)
    reps, rejected, rejected_rows = 120, 0, 0
    for r in range(reps):
        world = _null_world(rng, 40)
        df, dropped = prepare(world)
        rows = analyze(df, "consecutive", dropped, n_boot=199, seed=r, sensitivity=False)
        d = _result(rows, "H1", "arm=r18_scratch", "delta")
        assert d["ci_method"] == "wild_cluster_bootstrap_t(webb)" and d["n_clusters"] == 40
        assert d["descriptive_only"] is True  # pooled fit: secondary / descriptive
        rejected += not (d["ci_lo"] < 0 < d["ci_hi"])
        df_r, _ = prepare(world.assign(cluster=[f"r{i}" for i in range(len(world))]))
        d_r = _result(analyze(df_r, "consecutive", dropped, n_boot=199, seed=r,
                              sensitivity=False), "H1", "arm=r18_scratch", "delta")
        rejected_rows += not (d_r["ci_lo"] < 0 < d_r["ci_hi"])
    assert 1 <= rejected <= 12, rejected  # nominal 6 of 120
    assert rejected_rows > rejected


def test_descriptive_only_below_20_clusters():
    df, dropped = prepare(_null_world(np.random.default_rng(0), 9))
    rows = analyze(df, "consecutive", dropped, n_boot=99)
    d = _result(rows, "H1", "arm=r18_scratch", "delta")
    assert d["n_clusters"] == 9 and d["descriptive_only"] is True
    assert np.isfinite(d["ci_lo"]) and "too few for inference" in d["note"]


def test_h2_pairs_scratch_minus_pretrained_positive(tmp_path):
    arms = ["r18_scratch", "r18_places", "r50_scratch", "r50_places"]
    runs, shifts = _build_world(
        tmp_path, _slopes(arms, reg=0.10, rep=0.10, pre=0.3), arms, u_sd=0.01
    )
    rows = _run_analysis(runs, shifts)
    for scope in ("pair=r18", "pair=r50", "pooled_pairs"):
        d = _result(rows, "H2", scope, "delta")
        assert d["ci_lo"] > 0, scope
        assert _result(rows, "H2", scope, "slope_scratch")["estimate"] > _result(
            rows, "H2", scope, "slope_pretrained")["estimate"]
    assert _result(rows, "H1", "all_arms", "delta")  # pooled-over-arms H1 exists


def test_all_pairs_scope_runs_and_has_more_rows(tmp_path):
    arms = ["r18_scratch"]
    runs, shifts = _build_world(tmp_path, _slopes(arms, reg=0.10, rep=0.02), arms)
    rows = _run_analysis(runs, shifts, pairs="all")
    d = _result(rows, "H1", "arm=r18_scratch", "delta", pairs="all_pairs")
    cons = _run_analysis(runs, shifts)
    assert d["n_rows"] > _result(cons, "H1", "arm=r18_scratch", "delta")["n_rows"]
    assert d["ci_lo"] > 0


def test_transition_cluster_ci_wider_than_row_ci_under_correlated_noise():
    """Rows of one transition share its noise: treating rows as clusters understates the
    uncertainty; the wild cluster bootstrap-t by transition does not."""
    rng = np.random.default_rng(1)
    n_cl, per = 10, 40
    cl = np.repeat(np.arange(n_cl), per)
    x = np.repeat(rng.standard_normal(n_cl), per)
    y = 0.0 * x + np.repeat(rng.standard_normal(n_cl), per) + 0.1 * rng.standard_normal(n_cl * per)
    X = np.column_stack([np.ones_like(x), x])
    c = np.array([[0.0, 1.0]])
    by_cl = wild_cluster_bootstrap_t(X, y, cl, c, 400, 0)
    by_row = wild_cluster_bootstrap_t(X, y, np.arange(len(y)), c, 400, 0)
    assert (by_cl["hi"] - by_cl["lo"])[0] > 2 * (by_row["hi"] - by_row["lo"])[0]
    assert by_cl["lo"][0] < by_cl["est"][0] < by_cl["hi"][0]
    assert by_cl["n_ok"] == 400


def test_cr1_se_matches_textbook_formula():
    from clns.analysis.hypotheses import cr1_se

    rng = np.random.default_rng(3)
    n, g = 60, 6
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = X @ np.array([0.5, 1.0]) + rng.standard_normal(n)
    cl = np.arange(n) % g
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    bread = np.linalg.inv(X.T @ X)
    meat = sum(np.outer(X[cl == c].T @ u[cl == c], X[cl == c].T @ u[cl == c]) for c in range(g))
    v = g / (g - 1) * (n - 1) / (n - 2) * bread @ meat @ bread
    assert cr1_se(X, u, cl, np.eye(2)) == pytest.approx(np.sqrt(np.diag(v)))


def test_few_clusters_report_no_ci(tmp_path):
    arms = ["r18_scratch"]
    runs, shifts = _build_world(tmp_path, _slopes(arms, reg=0.1, rep=0.02), arms)
    table, _ = load_table(runs, shifts)
    table = table[table["k"] < 3]  # 3 transitions
    df, dropped = prepare(table)
    rows = analyze(df, "consecutive", dropped, n_boot=20)
    d = _result(rows, "H1", "arm=r18_scratch", "delta")
    assert np.isnan(d["ci_lo"]) and "no CI" in d["note"] and d["n_clusters"] == 3


def test_flags_near_chance_tuned_and_parse():
    tab = pd.DataFrame({
        "benchmark": ["a", "a", "b"], "k": [0, 1, 0],
        "a_tt": [0.21, 0.9, np.nan], "chance": [0.2, 0.2, np.nan],
    })
    assert list(flag_near_chance(tab, 0.05)) == [True, False, False]
    assert list(flag_near_chance(tab, 0.05, chance_default=0.5)) == [True, False, False]
    assert list(flag_tuned(tab, parse_tuned("a:0, b:3"))) == [True, False, False]
    with pytest.raises(ValueError):
        parse_tuned("clear10")


def test_flags_counted_in_results(tmp_path):
    arms = ["r18_scratch"]
    runs, shifts = _build_world(tmp_path, _slopes(arms, reg=0.1, rep=0.02), arms, a_tt=0.04)
    table, _ = load_table(runs, shifts)
    assert np.allclose(table["chance"], 0.1)  # 10 classes per task
    df, dropped = prepare(table, tuned={("bm", 0)}, margin=0.05)
    rows = analyze(df, "consecutive", dropped, n_boot=20)
    d = _result(rows, "H1", "arm=r18_scratch", "delta")
    assert d["n_near_chance"] == d["n_rows"] and d["n_tuned"] > 0
    kept = analyze(df, "consecutive", dropped, n_boot=20, exclude_flagged=True)
    assert not [r for r in kept if r["hypothesis"] == "H1"]  # everything flagged -> nothing left


def test_cli_end_to_end(tmp_path):
    arms = ["r18_scratch", "r18_places"]
    runs, shifts = _build_world(tmp_path, _slopes(arms, reg=0.10, rep=0.02, pre=0.5), arms)
    out = tmp_path / "res"
    code = hyp_main(["--runs-dir", str(runs), "--shift-dir", str(shifts), "--out", str(out),
                     "--n-boot", "50", "--pairs", "both", "--tuned-transitions", "bm:0"])
    assert code == 0
    res = pd.read_csv(out / "hypotheses.csv")
    for col in ("ci_lo", "ci_hi", "n_clusters", "estimate", "n_flagged", "n_boot_ok"):
        assert col in res.columns
    assert set(res["pairs"]) == {"consecutive", "all_pairs"}
    assert {"H1", "H2", "H1_naive"} <= set(res["hypothesis"])
    blob = json.loads((out / "hypotheses.json").read_text())
    assert blob["results"] and "associational" in blob["note"]
    # no matching shift -> clean error
    assert hyp_main(["--runs-dir", str(runs), "--shift-dir", str(tmp_path / "none"),
                     "--out", str(tmp_path / "x")]) == 1


# --------------------------------------------------------------------------- #
# Transition identity (amendment 4) and task-order matching (amendment 5)
# --------------------------------------------------------------------------- #
def _rec(order, **extra):
    bmeta = {"task_order": order, **extra}
    return {"benchmark_meta": bmeta, "rows": np.zeros((len(order), len(order))), "path": "p"}


def test_domainnet_like_orders_give_distinct_ids_same_order_same_id():
    doms = ["clipart", "real", "sketch", "painting"]
    a = _rec(doms)
    b = _rec(doms[::-1])
    c = _rec(list(doms))
    ids = lambda r: [transition_id(r, k) for k in range(3)]  # noqa: E731
    assert ids(a) == ids(c)
    assert not set(ids(a)) & set(ids(b))
    assert transition_id(a, 0) == "clipart->real"


def test_class_il_and_core50_ids_use_class_sets():
    tc = [[0, 1], [2, 3], [4, 5]]
    r = _rec([5, 4, 3, 2, 1, 0], train_classes=tc)
    assert transition_id(r, 1) == "2,3->4,5"
    core = _rec([2, tc], train_classes=tc, run=2, protocol="nic")
    assert transition_id(core, 0) == "run2:b0[0,1]->run2:b1[2,3]"
    legacy = {"benchmark_meta": {"class_order": [2, 0, 1]}}
    assert transition_id(legacy, 1) == "2,0,1|1"
    assert transition_id({"benchmark_meta": None}, 0) == "fixed|0"


def test_domain_il_ids_use_task_order_even_if_a_domain_misses_a_class():
    """Domain-IL: a domain without one class must still be identified by its domain name (class
    sets would separate 'real' in one seed from 'real' in another / merge distinct domains)."""
    doms = ["clipart", "real", "sketch"]
    full = list(range(5))
    tc = [full, full[:-1], full]  # 'real' lacks class 4
    r = _rec(doms, train_classes=tc)
    r["scenario"] = "domain_il"
    assert [transition_id(r, k) for k in range(2)] == ["clipart->real", "real->sketch"]
    # another seed visiting clipart -> sketch (both complete) must not share an id with any
    # other complete-domain pair, as class-set ids "0..4->0..4" would
    r2 = _rec(["sketch", "clipart", "real"], train_classes=[full, full, full[:-1]])
    r2["scenario"] = "domain_il"
    assert transition_id(r2, 0) == "sketch->clipart" != transition_id(r, 0)


def test_core50_batches_with_same_classes_in_different_positions_never_merge():
    """CORe50 NIC: two batches (sessions) with the same class set at different positions are
    different tasks; ids carry the run and the batch index."""
    batches = [[0, 1], [2, 3], [0, 1], [2, 3]]
    core = _rec([0, batches], train_classes=batches, run=0, protocol="nic")
    ids = [transition_id(core, k) for k in range(3)]
    assert ids[0] != ids[2]  # (b0 [0,1] -> b1 [2,3]) vs (b2 [0,1] -> b3 [2,3])
    assert len(set(ids)) == 3
    other_run = _rec([1, batches], train_classes=batches, run=1, protocol="nic")
    assert transition_id(other_run, 0) != ids[0]


def test_shift_mismatch_requires_task_order_sha_on_run():
    rec = {"rows": np.zeros((3, 3)), "path": "p",
           "benchmark_meta": {"class_order": None}}
    sh = {"matrix": np.zeros((3, 3)), "dir": "d",
          "meta": {"T": 3, "class_order": None, "task_order_sha256": "aa"}}
    assert "task_order_sha256" in shift_mismatch(sh, rec)
    rec["benchmark_meta"]["task_order_sha256"] = "bb"
    assert "mismatch" in shift_mismatch(sh, rec)
    rec["benchmark_meta"]["task_order_sha256"] = "aa"
    assert shift_mismatch(sh, rec) is None
    # a shift without the hash (legacy) still matches a legacy run
    sh["meta"].pop("task_order_sha256")
    assert shift_mismatch(sh, {**rec, "benchmark_meta": {"class_order": None}}) is None


def test_all_pairs_table_values_and_core50_mapping():
    a = np.array([[0.9, 0.0, 0.0], [0.7, 0.8, 0.0], [0.5, 0.6, 0.9]])
    m = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    rec = {"rows": a, "test_exp_to_train_exp": [0, 1, 2]}
    t = all_pairs_table(rec, m)
    got = {(int(r.j), int(r.k)): (r.forget, r.shift, r.a_tt) for r in t.itertuples()}
    assert got[(0, 0)] == pytest.approx((0.2, 1.0, 0.8))
    assert got[(0, 1)] == pytest.approx((0.2, 2.0, 0.9))  # shift M[m_0=0, t+1=2]
    assert got[(1, 1)] == pytest.approx((0.2, 3.0, 0.9))
    assert len(t) == 3
    # CORe50-style: 2 test experiences for 4 train experiences, mapping [0, 2]
    a4 = np.array([[0.9, 0.0], [0.8, 0.0], [0.6, 0.7], [0.5, 0.6]])
    t4 = all_pairs_table({"rows": a4, "test_exp_to_train_exp": [0, 2]}, np.ones((4, 4)))
    assert sorted(zip(t4["j"], t4["k"])) == [(0, 0), (0, 1), (0, 2), (1, 2)]
    # plasticity of the transition INTO train exp 1 has no first-appearance group -> NaN
    assert np.isnan(t4[(t4.j == 0) & (t4.k == 0)]["a_tt"].iloc[0])
    assert t4[(t4.j == 0) & (t4.k == 1)]["a_tt"].iloc[0] == pytest.approx(0.7)


def test_run_grid_script_syntax():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    for name in ("run_grid.sh", "smoke_test.sh", "pod_start.sh"):
        r = subprocess.run([bash, "-n", str(REPO / "scripts" / name)], capture_output=True,
                           text=True)
        assert r.returncode == 0, (name, r.stderr)


def test_plots_write_files(tmp_path):
    pytest.importorskip("matplotlib")
    from clns.analysis.plots import plot_accuracy_heatmap, plot_forgetting_vs_shift

    arms = ["r18_scratch"]
    runs, shifts = _build_world(tmp_path, _slopes(arms, reg=0.1, rep=0.02), arms)
    table, _ = load_table(runs, shifts)
    assert plot_forgetting_vs_shift(table, tmp_path / "s.png").stat().st_size > 0
    assert plot_accuracy_heatmap(np.eye(3), tmp_path / "h.png").stat().st_size > 0


# --------------------------------------------------------------------------- #
# Review-03 fixes: all_pairs needs --pairs full, chance under masking, CORe50 without a_tt,
# method fixed effects + no-a_tt sensitivity, shift-mismatch robustness
# --------------------------------------------------------------------------- #
def _make_consecutive_only(shifts, bench):
    """Turn the shift results of ``bench`` into what measure_stream --pairs consecutive writes."""
    for d in (shifts / bench).glob("seed*"):
        m = np.load(d / "shift_matrix.npz")["matrix"].astype(float)
        n = len(m)
        keep = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) <= 1
        np.savez(d / "shift_matrix.npz", matrix=np.where(keep, m, np.nan))
        meta = json.loads((d / "shift_meta.json").read_text())
        (d / "shift_meta.json").write_text(json.dumps({**meta, "pairs": "consecutive"}))


def test_all_pairs_requires_full_shift_matrix(tmp_path):
    arms = ["r18_scratch"]
    runs, shifts = _build_world(tmp_path, _slopes(arms, reg=0.1, rep=0.02), arms)
    _build_world(tmp_path, _slopes(arms, reg=0.1, rep=0.02), arms, bench="full")
    _make_consecutive_only(shifts, "bm")
    with pytest.raises(AllPairsUnavailable, match="--pairs full"):
        load_table(runs, shifts, "all")
    table, counts = load_table(runs, shifts, "all", not_full="skip")
    assert set(table["benchmark"]) == {"full"} and counts["all_pairs_skipped"] == ["bm"]
    out = tmp_path / "res"
    base = ["--runs-dir", str(runs), "--shift-dir", str(shifts), "--n-boot", "50"]
    assert hyp_main([*base, "--out", str(tmp_path / "x"), "--pairs", "all"]) == 1
    assert hyp_main([*base, "--out", str(out), "--pairs", "both"]) == 0
    res = pd.read_csv(out / "hypotheses.csv")
    ap = res[(res["pairs"] == "all_pairs") & (res["analysis"] != "accounting")]
    assert set(ap["benchmark"]) == {"full"}  # never a silent duplicate of consecutive
    flagged = res[res["all_pairs_skipped"]]
    assert len(flagged) == 1 and flagged.iloc[0]["benchmark"] == "bm"
    assert flagged.iloc[0]["pairs"] == "all_pairs" and flagged.iloc[0]["analysis"] == "accounting"
    assert flagged.iloc[0]["term"] == "all_pairs_skipped"
    assert "--pairs full" in flagged.iloc[0]["note"]
    # consecutive rows of bm carry neither the flag nor the note
    cons_bm = res[(res["benchmark"] == "bm") & (res["pairs"] == "consecutive")]
    assert len(cons_bm) and not cons_bm["all_pairs_skipped"].any()
    assert not cons_bm["note"].fillna("").str.contains("--pairs full").any()
    blob = json.loads((out / "hypotheses.json").read_text())
    assert blob["counts"]["all_pairs_skipped"] == ["bm"]


def test_chance_under_eval_masking_class_il_core50_and_domain():
    # ImageNet-100-like: 20 tasks x 5 classes, masked eval -> 1 / classes seen up to k+1
    tc = [list(range(5 * i, 5 * i + 5)) for i in range(20)]
    inet = {"scenario": "class_il", "benchmark": "imagenet100",
            "benchmark_meta": {"train_classes": tc}}
    assert _chance(inet, 0) == pytest.approx(1 / 10)
    assert _chance(inet, 3) == pytest.approx(1 / 25)
    assert _chance({**inet, "eval_mask_unseen": False}, 0) == pytest.approx(1 / 100)
    # CORe50-like NIC: later batches repeat classes; chance = 1 / |union of batches 0..k+1|
    batches = [list(range(10)), list(range(10)), [10, 11], list(range(12)), [12]]
    core = {"scenario": "class_il", "benchmark": "core50_nic",
            "benchmark_meta": {"protocol": "nic", "train_classes": batches}}
    assert _chance(core, 0) == pytest.approx(1 / 10)
    assert _chance(core, 1) == pytest.approx(1 / 12)  # new task has 2 classes, model sees 12
    assert _chance(core, 3) == pytest.approx(1 / 13)
    # domain-IL: every class is always predictable
    dom = {"scenario": "domain_il", "benchmark_meta": {"train_classes": [list(range(7))] * 3}}
    assert _chance(dom, 0) == pytest.approx(1 / 7)
    # a_tt 0.3 on a 2-class new task is not near chance when 12 classes are predictable
    tab = pd.DataFrame({"benchmark": ["c"], "k": [1], "a_tt": [0.3],
                        "chance": [_chance(core, 1)]})
    assert not flag_near_chance(tab, 0.05).iloc[0]


def _build_core50_world(tmp_path, seeds=(0, 1, 2), t=12):
    """CORe50-like runs: test groups only for some batches -> a_tt NaN for the others."""
    rng = np.random.default_rng(5)
    runs, shifts = tmp_path / "runs", tmp_path / "shift"
    intro = [0, 1, 3, 4, 6, 7, 9, 10]  # train exps that introduce a first-appearance group
    for s in seeds:
        m = _world_matrix(rng, t)
        sha = f"core-{s}"
        _write_shift(shifts, "core50_nic", s, m, sha=sha)
        batches = [[i % 4, (i % 4) + 4] for i in range(t)]
        bmeta = {"protocol": "nic", "run": s, "task_order": [s, batches],
                 "task_order_sha256": sha, "train_classes": batches}
        for method in ("ewc", "lwf", "er", "derpp"):
            a = np.zeros((t, len(intro)))
            for j, mj in enumerate(intro):
                a[mj, j] = 0.8 + 0.02 * rng.standard_normal()
                for i in range(mj + 1, t):
                    a[i, j] = a[i - 1, j] - (0.05 * m[mj, i] + 0.01 * rng.standard_normal())
            _write_run(runs, "core50_nic", "r18_scratch", method, s, a.tolist(), bmeta,
                       scenario="class_il", mapping=intro)
    return runs, shifts


def test_core50_fitted_without_a_tt_keeps_all_rows(tmp_path):
    runs, shifts = _build_core50_world(tmp_path)
    table, _ = load_table(runs, shifts)
    assert (table["kind"] == "core50").all()
    assert table["a_tt"].isna().sum() > 0  # batches without a new class
    df, dropped = prepare(table)
    assert len(df) == len(table) and dropped == {}
    rows = analyze(df, "consecutive", dropped, n_boot=99)
    d = _result(rows, "H1", "arm=r18_scratch", "delta")
    assert d["n_rows"] == len(table) == 4 * 3 * 8
    assert "a_tt" not in d["covariates"] and "+t" in d["covariates"]
    assert "CORe50" in d["note"]
    assert not [r for r in rows if r["variant"] == "no_a_tt"]  # primary already has no a_tt
    assert not [r for r in rows if r["term"] == "plasticity"]


def test_method_fixed_effects_and_no_a_tt_sensitivity(tmp_path):
    arms = ["r18_scratch", "r18_places"]
    runs, shifts = _build_world(tmp_path, _slopes(arms, reg=0.10, rep=0.02, pre=0.5), arms,
                                u_sd=0.01)
    rows = _run_analysis(runs, shifts, n_boot=99)
    h1 = _result(rows, "H1", "arm=r18_scratch", "delta")
    assert "method_fe" in h1["covariates"] and "a_tt" in h1["covariates"]
    assert "arm_fe" in _result(rows, "H1", "all_arms", "delta")["covariates"]
    h2 = _result(rows, "H2", "pair=r18", "delta")
    assert "method_fe" in h2["covariates"]
    for hyp, scope in (("H1", "arm=r18_scratch"), ("H1", "all_arms"), ("H2", "pair=r18")):
        sens = _result(rows, hyp, scope, "delta", variant="no_a_tt")
        assert "a_tt" not in sens["covariates"] and "method_fe" in sens["covariates"]
        assert sens["n_rows"] == _result(rows, hyp, scope, "delta")["n_rows"]
        assert sens["estimate"] > 0
    assert np.isfinite(h1["estimate"]) and np.isfinite(h2["ci_lo"])


def test_method_fixed_effects_absorb_method_intercepts():
    """A method-specific forgetting level (unrelated to shift) must not move delta."""
    rng = np.random.default_rng(11)
    df = _null_world(rng, 30, v_sd=0.0)
    level = {"ewc": 0.0, "lwf": 0.3, "er": -0.2, "derpp": 0.1}
    a, d = prepare(df)
    b, d2 = prepare(df.assign(forget=df["forget"] + df["method"].map(level)))
    ra = _result(analyze(a, "consecutive", d, n_boot=49), "H1", "arm=r18_scratch", "delta")
    rb = _result(analyze(b, "consecutive", d2, n_boot=49), "H1", "arm=r18_scratch", "delta")
    assert rb["estimate"] == pytest.approx(ra["estimate"], abs=1e-9)


def test_shift_mismatch_skipped_counted_and_strict(tmp_path):
    arms = ["r18_scratch"]
    runs, shifts = _build_world(tmp_path, _slopes(arms, reg=0.1, rep=0.02), arms)
    bmeta = {"task_order": [f"x{i}" for i in range(T)], "task_order_sha256": "other-order"}
    rng = np.random.default_rng(0)
    _write_run(runs, "bm", "r18_scratch", "ewc", 9,
               _acc(_world_matrix(rng), 0.1, np.zeros(T), rng), bmeta)
    table, counts = load_table(runs, shifts)
    assert counts["n_runs_shift_mismatch"] == {"bm": 1}
    assert "seed9" not in " ".join(table["run"])
    with pytest.raises(ValueError, match="no shift result matches"):
        load_table(runs, shifts, strict=True)
    out = tmp_path / "res"
    base = ["--runs-dir", str(runs), "--shift-dir", str(shifts), "--out", str(out),
            "--n-boot", "20"]
    assert hyp_main(base) == 3  # partial confirmatory output: written, but non-zero exit
    assert hyp_main([*base, "--allow-partial"]) == 0
    res = pd.read_csv(out / "hypotheses.csv")
    assert (res["n_runs_shift_mismatch"] == 1).all()
    acc = res[(res["analysis"] == "accounting") & (res["term"] == "n_runs_shift_mismatch")]
    assert len(acc) == 1 and acc.iloc[0]["estimate"] == 1
    assert acc.iloc[0]["n_runs_loaded"] == 5 * 3 + 1
    with pytest.raises(ValueError):
        hyp_main(["--runs-dir", str(runs), "--shift-dir", str(shifts), "--out", str(out),
                  "--n-boot", "20", "--strict"])


# --------------------------------------------------------------------------- #
# Review-05: two-stage run-level test (primary), all_pairs descriptive, term t
# --------------------------------------------------------------------------- #
def _runslope_world(rng, slope_sd, planted=0.0, k_tr=19, seeds=3):
    """Reviewer's sim_runslope world: distinct transitions per seed, per-run random slope."""
    rows = []
    for s in range(seeds):
        x = rng.standard_normal(k_tr)
        u = rng.standard_normal(k_tr) * 0.3
        for fam, meths in (("regularization", ("ewc", "lwf")), ("replay", ("er", "derpp"))):
            extra = planted if fam == "regularization" else 0.0
            for m in meths:
                b_run = rng.standard_normal() * slope_sd
                a_run = rng.standard_normal() * 0.1
                for k in range(k_tr):
                    rows.append(dict(
                        benchmark="b", model="r18_scratch", method=m, seed=s, run=f"{m}{s}",
                        k=k, t=k, forget=a_run + (0.1 + extra + b_run) * x[k] + u[k]
                        + 0.05 * rng.standard_normal(), shift=x[k],
                        a_tt=0.8 + 0.05 * rng.standard_normal(), position=k_tr - 1 - k,
                        cluster=f"s{s}c{k}", chance=0.01, family=fam))
    return pd.DataFrame(rows)


def _ts(rows, hypothesis, scope, term, **kw):
    return _result(rows, hypothesis, scope, term, analysis="two_stage", **kw)


def test_two_stage_null_calibrated_under_run_slope_heterogeneity():
    """Per-run slope sd 0.05 under the null (review-05 sim: pooled transition fit FPR 0.35+).
    Full run (400 sims): primary seed-FE test FPR 0.043 / 0.048 / 0.052 at sd 0 / 0.05 / 0.10;
    secondary Welch 0.003 / 0.010 / 0.018. Here 100 sims (fixed seed, loose bounds)."""
    rng = np.random.default_rng(7)
    reps, rej_fe, rej_w = 100, 0, 0
    for _ in range(reps):
        df, dropped = prepare(_runslope_world(rng, 0.05))
        rows = _hyp.analyze_two_stage(df, "consecutive", dropped, sensitivity=False)
        d = _ts(rows, "H1", "arm=r18_scratch", "delta")
        w = _ts(rows, "H1", "arm=r18_scratch", "delta_welch")
        assert d["ci_method"] == "ols_seed_fe(run_slopes)" and d["n_runs"] == 12
        assert d["df"] == 12 - 4  # intercept + replay + 2 seed dummies
        assert w["ci_method"] == "welch_t(run_slopes)" and np.isfinite(w["df"])
        assert d["n_runs_ref"] == 6 and d["n_runs_other"] == 6
        assert d["estimate"] == pytest.approx(w["estimate"])  # balanced within seed
        rej_fe += not (d["ci_lo"] < 0 < d["ci_hi"])
        rej_w += not (w["ci_lo"] < 0 < w["ci_hi"])
    assert rej_fe <= 11, rej_fe  # nominal 5 of 100
    assert rej_w <= 8, rej_w


def test_seed_fe_equals_welch_without_seed_structure_and_df():
    """One seed, equal group sizes: the seed-FE OLS is the pooled two-sample t, whose estimate
    and SE equal Welch's (df n-2 instead of Welch-Satterthwaite)."""
    rng = np.random.default_rng(4)
    runs = pd.DataFrame({"slope": rng.standard_normal(10), "seed": 0, "model": "a",
                         "family": ["regularization"] * 5 + ["replay"] * 5})
    fe, why = _hyp.seed_fe_delta(runs)
    w = _hyp.welch_t(runs["slope"][:5], runs["slope"][5:])
    assert why == "" and fe["df"] == 8
    assert fe["estimate"] == pytest.approx(w["estimate"]) and fe["se"] == pytest.approx(w["se"])
    # a seed-level shift moves Welch's SE, not the seed-FE estimate / SE
    runs3 = pd.concat([runs.assign(seed=s) for s in range(3)], ignore_index=True)
    shifted = runs3.assign(slope=runs3["slope"] + runs3["seed"] * 5.0)
    a, _ = _hyp.seed_fe_delta(runs3)
    b, _ = _hyp.seed_fe_delta(shifted)
    assert b["estimate"] == pytest.approx(a["estimate"]) and b["se"] == pytest.approx(a["se"])
    assert b["df"] == 30 - 4
    wb = _hyp.welch_t(shifted.loc[shifted.family == "regularization", "slope"],
                      shifted.loc[shifted.family == "replay", "slope"])
    assert wb["se"] > 3 * b["se"]
    # arm FE adds columns: 2 arms x 3 seeds -> df = n - (2 + 2 + 1)
    two = pd.concat([runs3, runs3.assign(model="b")], ignore_index=True)
    c, _ = _hyp.seed_fe_delta(two, arm_fe=True)
    assert c["df"] == 60 - 5
    # a seed with only one family is fine (absorbed by its dummy); no residual df -> no CI
    tiny = runs.iloc[[0, 5]]
    t, why = _hyp.seed_fe_delta(tiny)
    assert np.isnan(t["ci_lo"]) and "no residual df" in why


def test_two_stage_detects_planted_effect(tmp_path):
    arms = ["r18_scratch", "r18_places"]
    runs, shifts = _build_world(tmp_path, _slopes(arms, reg=0.10, rep=0.02), arms, u_sd=0.01)
    rows = _run_analysis(runs, shifts, n_boot=49)
    d = _ts(rows, "H1", "arm=r18_scratch", "delta")
    assert d["ci_method"] == "ols_seed_fe(run_slopes)" and d["df"] == 12 - 4
    assert d["ci_lo"] > 0 and d["p_value"] < 0.05 and d["descriptive_only"] is False
    assert "primary H1 test" in d["note"]
    assert d["estimate"] == pytest.approx(0.08 * _shift_sd(runs, shifts), abs=0.01)
    reg = _ts(rows, "H1", "arm=r18_scratch", "slope_regularization")
    rep = _ts(rows, "H1", "arm=r18_scratch", "slope_replay")
    assert reg["estimate"] - rep["estimate"] == pytest.approx(d["estimate"])
    assert reg["ci_method"] == "one_sample_t(run_slopes)" and reg["df"] == 5
    assert _ts(rows, "H1_naive", "arm=r18_scratch", "slope")["estimate"] > 0
    assert _ts(rows, "H1", "arm=r18_scratch", "delta", variant="no_a_tt")["estimate"] > 0
    allarms = _ts(rows, "H1", "all_arms", "delta")
    assert allarms["ci_lo"] > 0 and allarms["df"] == 24 - 5  # + arm FE
    # two-stage rows come first; the pooled fit is still there, labelled descriptive
    assert rows[0]["analysis"] == "two_stage"
    assert _result(rows, "H1", "arm=r18_scratch", "delta")["descriptive_only"] is True
    # the secondary Welch row agrees with scipy
    w = _ts(rows, "H1", "arm=r18_scratch", "delta_welch")
    assert "secondary" in w["note"] and w["ci_lo"] > 0
    from scipy import stats
    df_, _ = prepare(load_table(runs, shifts)[0])
    sl, _ = _hyp.run_slopes(df_)
    sl = sl[sl.model == "r18_scratch"]
    a = sl[sl.family == "regularization"]["slope"]
    b = sl[sl.family == "replay"]["slope"]
    assert w["p_value"] == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue)


def _h2_frame(rng, drop=None):
    """Run slopes of scratch / pretrained share a big per-(method, seed) offset; scratch is
    exactly 0.1 steeper (on raw shift)."""
    x = rng.standard_normal(10)
    rows = []
    for m in ("ewc", "lwf", "er", "derpp"):
        for s in range(3):
            off = rng.standard_normal()
            for arm, extra in (("r18_scratch", 0.1), ("r18_places", 0.0)):
                if drop == (arm, m, s):
                    continue
                for k in range(10):
                    rows.append(dict(
                        benchmark="b", model=arm, method=m, seed=s, run=f"{arm}/{m}/{s}", k=k,
                        t=k, forget=(off + extra) * x[k] + 0.001 * rng.standard_normal(),
                        shift=x[k], a_tt=0.8 + 0.05 * rng.standard_normal(), position=9 - k,
                        cluster=f"c{k}", chance=0.01, family=_hyp.FAMILY[m]))
    return pd.DataFrame(rows), float(np.std(x))


def test_two_stage_h2_pairs_by_method_and_seed():
    df, sd = _h2_frame(np.random.default_rng(3))
    pdf, dropped = prepare(df)
    rows = _hyp.analyze_two_stage(pdf, "consecutive", dropped)
    d = _ts(rows, "H2", "pair=r18", "delta")
    assert d["ci_method"] == "paired_t(run_slopes)" and d["n_pairs"] == 12 and d["n_runs"] == 24
    assert d["estimate"] == pytest.approx(0.1 * sd, abs=2e-3)
    assert d["ci_lo"] > 0 and d["ci_hi"] - d["ci_lo"] < 0.01  # pairing removes the offsets
    # the same slopes unpaired (Welch) would not be distinguishable from 0
    sl, _ = _hyp.run_slopes(pdf)
    w = _hyp.welch_t(sl[sl.model == "r18_scratch"]["slope"], sl[sl.model == "r18_places"]["slope"])
    assert w["ci_lo"] < 0 < w["ci_hi"]
    for fam in ("regularization", "replay"):
        f = _ts(rows, "H2", f"pair=r18|family={fam}", "delta")
        assert f["n_pairs"] == 6 and "secondary" in f["note"]
    # a run without its partner is dropped and noted
    df2, _ = _h2_frame(np.random.default_rng(3), drop=("r18_places", "ewc", 0))
    pdf2, dropped2 = prepare(df2)
    d2 = _ts(_hyp.analyze_two_stage(pdf2, "consecutive", dropped2), "H2", "pair=r18", "delta")
    assert d2["n_pairs"] == 11 and "without a (method, seed) partner" in d2["note"]


def test_two_stage_skips_short_and_rank_deficient_runs():
    rng = np.random.default_rng(0)
    df = _runslope_world(rng, 0.0, k_tr=8)
    df = df[~((df["run"] == "ewc0") & (df["k"] >= 3))].copy()  # 3 rows left -> few_rows
    deg = df["run"] == "er1"
    df.loc[deg, "a_tt"] = df.loc[deg, "t"] * 0.01 + 0.5  # a_tt collinear with t
    pdf, dropped = prepare(df)
    slopes, skipped = _hyp.run_slopes(pdf)
    assert dict(zip(skipped["run"], skipped["reason"])) == {
        "ewc0": "few_rows", "er1": "rank_deficient"}
    assert len(slopes) == 10
    d = _ts(_hyp.analyze_two_stage(pdf, "consecutive", dropped, sensitivity=False),
            "H1", "arm=r18_scratch", "delta")
    assert d["n_runs_skipped"] == 2 and d["n_runs_ref"] == 5 and d["n_runs_other"] == 5
    assert "2 run(s) skipped in stage 1" in d["note"]


def test_all_pairs_always_descriptive_and_term_t(tmp_path):
    arms = ["r18_scratch"]
    runs, shifts = _build_world(tmp_path, _slopes(arms, reg=0.10, rep=0.02), arms)
    rows = _run_analysis(runs, shifts, pairs="all", n_boot=49)
    ap_rows = [r for r in rows if r["pairs"] == "all_pairs"]
    assert ap_rows and all(r["descriptive_only"] for r in ap_rows)
    assert all("all_pairs: descriptive only" in r["note"] for r in ap_rows)
    terms = {r["term"] for r in rows if r["analysis"] == "pooled_transition"}
    assert "t" in terms and "position" not in terms


def test_select_shift_accepts_identical_candidates(tmp_path):
    from clns.analysis.transitions import load_shift_candidates, select_shift

    m = _world_matrix(np.random.default_rng(0))
    for s in (0, 1, 2):
        _write_shift(tmp_path, "clear10", s, m)
    rec = {"rows": np.zeros((T, T)), "path": "p", "seed": 7,
           "benchmark_meta": {"class_order": None, "task_order_sha256": "order-sha"}}
    cands = load_shift_candidates(tmp_path, "clear10")
    assert select_shift(cands, rec)["seed"] == 0  # identical -> lowest seed
    m2 = m.copy()
    m2[0, 1] += 1.0
    np.savez(tmp_path / "clear10" / "seed2" / "shift_matrix.npz", matrix=m2)
    with pytest.raises(ValueError, match="conflicting"):
        select_shift(load_shift_candidates(tmp_path, "clear10"), rec)
