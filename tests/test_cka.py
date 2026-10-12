"""clns.analysis.cka: linear CKA between consecutive probe rows of synthetic runs."""

import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from clns.analysis.cka import MIN_CKA_N, collect, main, run_cka  # noqa: E402


def _write_probe_run(root, method, seed, rows, labels, test_exp, mapping=None, done=True,
                     train_classes=None):
    d = root / "synthetic" / "r18_scratch" / method / f"seed{seed}"
    (d / "probe").mkdir(parents=True)
    p = len(labels)
    np.savez(d / "probe" / "meta.npz", labels=np.asarray(labels, dtype=np.int64),
             test_exp=np.asarray(test_exp, dtype=np.int64), index=np.arange(p, dtype=np.int64),
             image_id=np.asarray([f"img{i}" for i in range(p)]), probe_seed=np.int64(1))
    for r, f in enumerate(rows):
        np.savez(d / "probe" / f"row_{r:04d}.npz", features=np.asarray(f, dtype=np.float16),
                 exp_index=np.int64(r))
    (d / "provenance.json").write_text(json.dumps(
        {"benchmark_meta": {"train_classes": train_classes}}))
    if mapping is not None:
        (d / "accuracy_matrix.json").write_text(json.dumps(
            {"rows": [[0.0] * (max(test_exp) + 1)] * len(rows),
             "test_exp_to_train_exp": list(mapping)}))
    if done:
        (d / "DONE").write_text("")
    return d


def test_cka_class_il_old_classes(tmp_path):
    """Class-IL: test exp e = classes of train exp e; old after k = classes of exps 0..k."""
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(6), 30)  # 180 probe images, 6 classes, 2 per experience
    test_exp = labels // 2
    x0 = rng.standard_normal((180, 16))
    q, _ = np.linalg.qr(rng.standard_normal((16, 16)))
    x1 = x0 @ q * 2.0                      # orthogonal transform + scaling: CKA = 1
    x2 = x1.copy()
    x2[labels >= 2] = rng.standard_normal((120, 16))  # only classes 2..5 change
    runs = tmp_path / "runs"
    tc = [[0, 1], [2, 3], [4, 5]]
    _write_probe_run(runs, "ewc", 0, [x0, x1, x2], labels, test_exp, [0, 1, 2],
                     train_classes=tc)
    _write_probe_run(runs, "er", 0, [x0, x1], labels, test_exp, [0, 1, 2], done=False)
    table, counts = collect(runs)
    assert counts["runs_used"] == 1 and counts["skipped_unfinished"] == 1
    assert list(table.columns[:7]) == ["benchmark", "model", "method", "seed", "k", "cka_all",
                                       "cka_old"]
    r0 = table[table["k"] == 0].iloc[0]
    assert r0["cka_all"] == pytest.approx(1.0, abs=1e-2)
    assert r0["n_old"] == 60 and r0["reason"] == ""  # classes of experience 0 = {0, 1}
    r1 = table[table["k"] == 1].iloc[0]
    # old after experience 1 = classes {0, 1, 2, 3}: classes 2, 3 changed -> cka_old < 1
    assert r1["n_old"] == 120 and r1["cka_old"] < 0.9 and r1["cka_all"] < 0.9
    assert r1["old_def"] == "test_exp_to_train_exp<=k"


def test_cka_domain_il_old_is_by_task_not_class(tmp_path):
    """Domain-IL: every class occurs in every domain; old = domains (test exps) 0..k, so
    cka_old differs from cka_all (hub Q17: a class-based definition made them equal)."""
    rng = np.random.default_rng(2)
    labels = np.tile(np.arange(5), 3 * 10)       # 150 images, 5 classes in every domain
    test_exp = np.repeat(np.arange(3), 50)        # 3 domains x 50 images
    x0 = rng.standard_normal((150, 12))
    x1 = x0.copy()
    x1[test_exp == 1] = rng.standard_normal((50, 12))  # only domain 1 drifts at k=0
    runs = tmp_path / "runs"
    d = _write_probe_run(runs, "lwf", 1, [x0, x1], labels, test_exp, [0, 1, 2],
                         train_classes=[list(range(5))] * 3)
    r = run_cka(d).iloc[0]
    assert r["n_old"] == 50                        # domain 0 only (not all 150 images)
    assert r["cka_old"] == pytest.approx(1.0, abs=1e-2)
    assert r["cka_all"] < 0.9


def test_cka_core50_mapping_and_min_n(tmp_path):
    """CORe50-like: test groups introduced at train exps [0, 2]; below 25 images -> NaN."""
    rng = np.random.default_rng(3)
    test_exp = np.repeat([0, 1], [30, 30])
    labels = np.repeat([0, 1], [30, 30])
    rows = [rng.standard_normal((60, 8)) for _ in range(4)]
    runs = tmp_path / "runs"
    d = _write_probe_run(runs, "naive", 0, rows, labels, test_exp, [0, 2])
    t = run_cka(d)
    assert list(t["n_old"]) == [30, 30, 60]       # group 1 is old only from k = 2
    small = _write_probe_run(runs, "er", 0, [r[:20] for r in rows[:2]], labels[:20],
                             test_exp[:20], [0, 2])
    s = run_cka(small).iloc[0]
    assert np.isnan(s["cka_all"]) and np.isnan(s["cka_old"]) and MIN_CKA_N == 25
    assert "n_probe=20 < 25" in s["reason"] and "n_old=20 < 25" in s["reason"]


def test_cka_cli_writes_csv(tmp_path):
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(4), 10)
    rows = [rng.standard_normal((40, 8)) for _ in range(3)]
    runs = tmp_path / "runs"
    _write_probe_run(runs, "naive", 2, rows, labels, labels // 2)  # no accuracy_matrix.json
    out = tmp_path / "res"
    assert main(["--runs-dir", str(runs), "--out", str(out)]) == 0
    res = pd.read_csv(out / "cka.csv")
    assert len(res) == 2 and set(res["seed"]) == {2} and res["cka_all"].notna().all()
    assert res["old_def"].str.contains("identity").all()
    assert main(["--runs-dir", str(tmp_path / "none"), "--out", str(out)]) == 1
