import math

import pytest

torch = pytest.importorskip("torch")

from clns.shift.metrics import (  # noqa: E402
    linear_cka,
    median_bandwidth,
    mmd2_unbiased,
    mmd_permutation_test,
    mmd_subsample_ci,
)


def _sample(n, d=5, shift=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64)
    x[:, 0] += shift
    return x


def test_mmd_identical_distribution_near_zero():
    x = _sample(300, seed=1)
    y = _sample(300, seed=2)
    assert abs(mmd2_unbiased(x, y)) < 0.05


def test_mmd_increases_with_shift():
    x = _sample(300, seed=1)
    vals = [
        mmd2_unbiased(x, _sample(300, shift=s, seed=2), bandwidth=3.0) for s in (0.0, 1.0, 3.0)
    ]
    assert vals[0] < vals[1] < vals[2]


def test_mmd_median_bandwidth_detects_large_shift():
    x = _sample(300, seed=1)
    same = mmd2_unbiased(x, _sample(300, seed=2))
    far = mmd2_unbiased(x, _sample(300, shift=3.0, seed=3))
    assert far > same + 0.05


def test_mmd_blocked_equals_unblocked():
    x = _sample(50, d=4, seed=1)
    y = _sample(37, d=4, shift=0.5, seed=2)
    a = mmd2_unbiased(x, y, bandwidth=1.3, block_size=7, dtype=torch.float64)
    b = mmd2_unbiased(x, y, bandwidth=1.3, block_size=1000, dtype=torch.float64)
    assert a == pytest.approx(b, rel=1e-6)
    c = mmd2_unbiased(x, y, bandwidth="median", block_size=7, dtype=torch.float64)
    d = mmd2_unbiased(x, y, bandwidth="median", block_size=1000, dtype=torch.float64)
    assert c == pytest.approx(d, rel=1e-6)


def test_mmd_tiny_hand_example():
    # x = {0, 1}, y = {0, 1}, sigma = 1: kxx = kyy = k = exp(-0.5),
    # kxy mean = (2 + 2k)/4 -> mmd2 = 2k - (1 + k) = k - 1.
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    val = mmd2_unbiased(x, x.clone(), bandwidth=1.0, dtype=torch.float64)
    k = math.exp(-0.5)
    assert val == pytest.approx(k - 1.0, rel=1e-9)


def test_mmd_bandwidth_list_sum_vs_mean():
    x = _sample(40, seed=1)
    y = _sample(40, shift=1.0, seed=2)
    s = mmd2_unbiased(x, y, bandwidth=[1.0, 2.0], sum_bandwidths=True)
    m = mmd2_unbiased(x, y, bandwidth=[1.0, 2.0], sum_bandwidths=False)
    assert s == pytest.approx(2 * m, rel=1e-6)


def test_mmd_input_validation():
    x = _sample(10)
    with pytest.raises(ValueError):
        mmd2_unbiased(x, x, kernel="linear")
    with pytest.raises(ValueError):
        mmd2_unbiased(x[:1], x)


def test_median_bandwidth_floor_on_duplicates():
    x = torch.ones(10, 3)
    assert median_bandwidth(x, x) == pytest.approx(1e-6)


def test_permutation_test_shifted_and_same():
    x = _sample(100, seed=1)
    _, p_shift = mmd_permutation_test(x, _sample(100, shift=1.5, seed=2), n_permutations=200)
    assert p_shift < 0.05
    ps = []
    for s in range(5):
        _, p = mmd_permutation_test(
            _sample(60, seed=10 + s), _sample(60, seed=20 + s), n_permutations=100, seed=s
        )
        assert 0.0 < p <= 1.0
        ps.append(p)
    assert sum(ps) / len(ps) > 0.2


@pytest.mark.parametrize("debiased", [True, False])
def test_cka_identity_and_invariances(debiased):
    x = _sample(200, d=8, seed=1)
    assert linear_cka(x, x, debiased=debiased) == pytest.approx(1.0, abs=1e-9)
    q, _ = torch.linalg.qr(_sample(8, d=8, seed=5))
    assert linear_cka(x, x @ q, debiased=debiased) == pytest.approx(1.0, abs=1e-9)
    assert linear_cka(x, 5.0 * x, debiased=debiased) == pytest.approx(1.0, abs=1e-9)


def test_cka_independent_is_small_and_unequal_dims_ok():
    x = _sample(1000, d=10, seed=1)
    y = _sample(1000, d=6, seed=2)
    assert linear_cka(x, y) < 0.2
    with pytest.raises(ValueError):
        linear_cka(x, y[:10])


def test_cka_debiased_independent_n_less_than_d():
    # n < d: biased CKA saturates toward 1 for independent data; debiased stays near 0.
    x = _sample(60, d=600, seed=11)
    y = _sample(60, d=600, seed=12)
    assert abs(linear_cka(x, y, debiased=True)) < 0.1
    assert linear_cka(x, y, debiased=False) > 0.5
    assert linear_cka(x, x, debiased=True) == pytest.approx(1.0, abs=1e-9)
    assert linear_cka(x, x, debiased=False) == pytest.approx(1.0, abs=1e-9)


def test_cka_biased_gram_form_matches_feature_form():
    # d > n triggers the n x n Gram form; compare with a direct feature-form computation.
    x = _sample(20, d=50, seed=3)
    y = _sample(20, d=30, seed=4)
    xc = x - x.mean(0, keepdim=True)
    yc = y - y.mean(0, keepdim=True)
    ref = (torch.linalg.matrix_norm(yc.T @ xc) ** 2) / (
        torch.linalg.matrix_norm(xc.T @ xc) * torch.linalg.matrix_norm(yc.T @ yc)
    )
    assert linear_cka(x, y, debiased=False) == pytest.approx(float(ref), rel=1e-9)


def test_cka_rejects_conv_maps_and_tiny_n():
    with pytest.raises(ValueError, match="pooling"):
        linear_cka(torch.randn(10, 3, 4, 4), torch.randn(10, 5))
    with pytest.raises(ValueError):
        linear_cka(torch.randn(3, 5), torch.randn(3, 5), debiased=True)


def test_bandwidth_accepts_numpy_and_torch_scalars():
    np = pytest.importorskip("numpy")
    x = _sample(30, seed=1)
    y = _sample(30, shift=1.0, seed=2)
    ref = mmd2_unbiased(x, y, bandwidth=1.5)
    for bw in (np.float32(1.5), np.float64(1.5), np.array(1.5), torch.tensor(1.5), 3 / 2):
        assert mmd2_unbiased(x, y, bandwidth=bw) == pytest.approx(ref, rel=1e-6)
    assert mmd2_unbiased(x, y, bandwidth=np.array([1.5])) == pytest.approx(ref, rel=1e-6)
    with pytest.raises(ValueError):
        mmd2_unbiased(x, y, bandwidth=-1.0)


def test_median_bandwidth_pooled_is_label_invariant():
    x = _sample(40, seed=1)
    y = _sample(30, shift=1.0, seed=2)
    assert median_bandwidth(x, y) == pytest.approx(median_bandwidth(y, x), rel=1e-12)
    assert median_bandwidth(torch.cat([x, y])) == pytest.approx(median_bandwidth(x, y), rel=1e-12)


def test_subsample_ci_covers_zero_for_identical_distributions():
    # Reviewer repro (boot2.py): with-replacement bootstrap excluded 0 here.
    for trial in range(5):
        x = _sample(100, d=20, seed=100 + trial)
        y = _sample(100, d=20, seed=200 + trial)
        sigma = median_bandwidth(x, y)
        lo, hi = mmd_subsample_ci(x, y, n_subsamples=100, seed=trial, bandwidth=sigma)
        assert lo <= 0.0 <= hi
        stat, _, null = mmd_permutation_test(
            x, y, n_permutations=100, seed=trial, bandwidth=sigma, return_null=True
        )
        assert null.shape == (100,)
        assert float(torch.quantile(null, 0.95)) > 0.0


def test_subsample_ci_and_null_band_detect_shift():
    x = _sample(100, d=5, seed=1)
    y = _sample(100, d=5, shift=2.0, seed=2)
    sigma = median_bandwidth(x, y)
    lo, hi = mmd_subsample_ci(x, y, n_subsamples=100, seed=0, bandwidth=sigma)
    stat, p, null = mmd_permutation_test(
        x, y, n_permutations=100, seed=0, bandwidth=sigma, return_null=True
    )
    assert 0.0 < lo <= hi
    assert stat > float(torch.quantile(null, 0.95))
    assert p < 0.05


def test_subsample_ci_validation():
    x = _sample(3)
    with pytest.raises(ValueError):
        mmd_subsample_ci(x, x, n_subsamples=5)  # n//2 = 1 < 2
