"""Unit tests for clns.evaluation.matrix_metrics (pure Python, no torch needed)."""

import pytest

from clns.evaluation.matrix_metrics import clear_metrics, diagonal, macro_matrix, matrix_metrics

A3 = [
    [0.9, 0.1, 0.0],
    [0.7, 0.8, 0.2],
    [0.6, 0.5, 0.9],
]
B3 = [0.1, 0.1, 0.1]


def test_identity_mapping_gem_metrics():
    m = matrix_metrics(A3, b=B3, scenario="class_il")
    assert m["acc_final"] == pytest.approx((0.6 + 0.5 + 0.9) / 3)
    # rows: [0.9], [0.7, 0.8], [0.6, 0.5, 0.9]
    assert m["acc_avg_incremental"] == pytest.approx((0.9 + 0.75 + 2.0 / 3) / 3)
    assert m["bwt"] == pytest.approx(((0.6 - 0.9) + (0.5 - 0.8)) / 2)
    # max over rows j..T-2 (final row excluded)
    assert m["forgetting_max"] == pytest.approx(((0.9 - 0.6) + (0.8 - 0.5)) / 2)
    assert m["fwt"] == pytest.approx(((0.1 - 0.1) + (0.2 - 0.1)) / 2)
    assert m["fwt_degenerate"] is True
    assert "clear_accuracy" not in m


def test_forgetting_max_excludes_final_row_and_can_be_negative():
    a = [
        [0.5, 0.0],
        [0.7, 0.9],  # final row better than when exp 0 was learned
    ]
    m = matrix_metrics(a, scenario="domain_il")
    assert m["forgetting_max"] == pytest.approx(0.5 - 0.7)
    assert m["bwt"] == pytest.approx(0.7 - 0.5)
    assert m["fwt"] is None  # no random-init row
    assert m["fwt_degenerate"] is False


def test_forgetting_uses_max_not_first():
    a = [
        [0.5, 0.0, 0.0],
        [0.8, 0.6, 0.0],  # exp 0 improves after exp 1 (e.g. domain-IL)
        [0.4, 0.6, 0.7],
    ]
    m = matrix_metrics(a)
    assert m["forgetting_max"] == pytest.approx(((0.8 - 0.4) + (0.6 - 0.6)) / 2)
    assert m["bwt"] == pytest.approx(((0.4 - 0.5) + (0.6 - 0.6)) / 2)


def test_single_experience_undefined_metrics_are_none():
    m = matrix_metrics([[0.8]], b=[0.1])
    assert m["acc_final"] == pytest.approx(0.8)
    assert m["acc_avg_incremental"] == pytest.approx(0.8)
    assert m["bwt"] is None
    assert m["forgetting_max"] is None
    assert m["fwt"] is None


def test_core50_style_mapping():
    """T_train=4, T_test=3; test exps introduced by train exps [0, 1, 3] (train exp 2 adds no
    new class -> empty first-appearance group, dropped)."""
    a = [
        [0.8, 0.0, 0.0],
        [0.6, 0.9, 0.1],
        [0.5, 0.7, 0.2],
        [0.4, 0.6, 0.95],
    ]
    b = [0.1, 0.05, 0.1]
    mapping = [0, 1, 3]
    m = matrix_metrics(a, b=b, scenario="class_il", test_exp_to_train_exp=mapping)

    assert diagonal(a, mapping) == [0.8, 0.9, 0.95]
    assert m["acc_final"] == pytest.approx((0.4 + 0.6 + 0.95) / 3)
    # per-row means over introduced groups: {0}, {0,1}, {0,1}, {0,1,2}
    assert m["acc_avg_incremental"] == pytest.approx((0.8 + 0.75 + 0.6 + 0.65) / 4)
    # BWT / forgetting over J ∩ {0..T-2} = test exps {0, 1}, normalised by 2 (not T-1 = 3)
    assert m["bwt"] == pytest.approx(((0.4 - 0.8) + (0.6 - 0.9)) / 2)
    assert m["forgetting_max"] == pytest.approx(((0.8 - 0.4) + (0.9 - 0.6)) / 2)
    # FWT over test exps with m_j >= 1: j=1 uses row 0, j=2 uses row 2 (= m_j - 1)
    assert m["fwt"] == pytest.approx(((0.0 - 0.05) + (0.2 - 0.1)) / 2)


def test_mapping_length_mismatch_raises():
    with pytest.raises(ValueError):
        matrix_metrics(A3, test_exp_to_train_exp=[0, 1])


def test_clear_metrics():
    R = [
        [0.6, 0.5, 0.4],
        [0.7, 0.8, 0.6],
        [0.65, 0.75, 0.9],
    ]
    c = clear_metrics(R)
    assert c["clear_accuracy"] == pytest.approx((0.6 + 0.7 + 0.8 + 0.65 + 0.75 + 0.9) / 6)
    assert c["clear_bwt"] == pytest.approx((0.7 + 0.65 + 0.75) / 3)
    assert c["clear_fwt"] == pytest.approx((0.5 + 0.4 + 0.6) / 3)
    assert c["clear_in_domain_acc"] == pytest.approx((0.6 + 0.8 + 0.9) / 3)
    assert c["clear_next_domain_acc"] == pytest.approx((0.5 + 0.6) / 2)
    m = matrix_metrics(R, scenario="domain_il", clear=True)
    assert m["clear_accuracy"] == pytest.approx(c["clear_accuracy"])
    with pytest.raises(ValueError):
        clear_metrics([[0.1, 0.2]])


def test_macro_matrix():
    cell_a = {"classes": [0, 1], "correct": [9, 1], "total": [10, 10]}  # micro 0.5, macro 0.5
    cell_b = {"classes": [0, 1], "correct": [9, 0], "total": [9, 1]}  # micro 0.9, macro 0.5
    cell_c = {"classes": [2], "correct": [0], "total": [0]}  # no samples -> None
    m = macro_matrix([[cell_a, cell_b], [cell_b, cell_c]])
    assert m[0] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert m[1][0] == pytest.approx(0.5) and m[1][1] is None
    full = macro_matrix([[cell_a, cell_b], [cell_b, cell_a]])
    assert matrix_metrics(full, scenario="domain_il")["acc_final"] == pytest.approx(0.5)
