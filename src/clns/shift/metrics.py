"""Distribution-shift measures on fixed-encoder features (RESEARCH_LOG 7.3).

Two different quantities are provided; do not conflate them:

* MMD (``mmd2_unbiased``): compares the *input distributions* of two experiences
  E_t and E_{t+1}, embedded by ONE frozen reference encoder (non-CLIP, see
  RESEARCH_LOG 7.4 Q10). It measures data shift, independent of any trained model.
  This is the MARGINAL MMD on P(X); it is not comparable across benchmarks
  (label-block change vs P(X|Y) change, RESEARCH_LOG 7.3).
  TODO(RESEARCH_LOG 7.3): class-conditional MMD on P(X | Y=y) for shared classes
  (and the CORe50 cumulative-reference variant) is deferred to a later phase.
* Linear CKA (``linear_cka``): compares two representations of the SAME paired
  probe inputs, e.g. the model after experience t versus after t+1. It measures
  representation drift, not data shift. Default estimator: debiased HSIC
  (Song et al. 2007), as required by RESEARCH_LOG 7.3.

Uncertainty for MMD (RESEARCH_LOG 7.3 asks for an effect size with a CI):
``mmd_subsample_ci`` (m-out-of-n subsampling WITHOUT replacement) and the
permutation null band from ``mmd_permutation_test(..., return_null=True)``.
A with-replacement bootstrap is NOT used: duplicated points enter the
off-diagonal kernel sums as k(x, x) = 1 and bias the unbiased MMD^2 upward, so
the CI would exclude 0 even for identical distributions.
"""

from __future__ import annotations

import math
import numbers

import torch

_MIN_BANDWIDTH = 1e-6


def _as_2d(x, what: str = "input") -> torch.Tensor:
    """Tensor view of shape (n, d). 1-d inputs become (n, 1).

    Inputs with ndim > 2 (e.g. conv feature maps (n, c, h, w)) are rejected rather
    than silently flattened: pool them first (global average pooling over spatial
    positions, RESEARCH_LOG 7.3), otherwise d explodes and the measure changes meaning.
    """
    t = torch.as_tensor(x)
    if t.ndim == 1:
        t = t.unsqueeze(1)
    elif t.ndim != 2:
        raise ValueError(
            f"{what} must be 1-d or 2-d (n, d); got shape {tuple(t.shape)}. "
            "Apply global average pooling (or select a token) before calling."
        )
    if not t.is_floating_point():
        t = t.float()
    return t


def _sqdist(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Pairwise squared Euclidean distances via |a|^2 + |b|^2 - 2ab, clamped at 0."""
    if dtype is not None:
        a = a.to(dtype)
        b = b.to(dtype)
    aa = (a * a).sum(dim=1).unsqueeze(1)
    bb = (b * b).sum(dim=1).unsqueeze(0)
    d2 = aa + bb - 2.0 * (a @ b.T)
    return d2.clamp_min(0.0)


def median_bandwidth(x, y=None, max_n: int = 3000, seed: int = 0) -> float:
    """Median pairwise Euclidean distance of the POOLED sample (median heuristic).

    ``x`` and ``y`` (optional) are concatenated first and the pooled set is then
    subsampled (seeded) to at most ``max_n`` points. Because the subsample is drawn
    from the pooled set, the result does not depend on which points are labelled
    x or y, i.e. it is invariant to permutations of the pooled data (required for
    a valid permutation test). Memory: ~max_n^2 float64 (72 MB at 3000).
    Floored at 1e-6 so that duplicate-only data does not give sigma = 0.
    """
    z = _as_2d(x)
    if y is not None:
        z = torch.cat([z, _as_2d(y).to(z.device)], dim=0)
    if z.shape[0] > max_n:
        gen = torch.Generator().manual_seed(int(seed))
        idx = torch.randperm(z.shape[0], generator=gen)[:max_n].to(z.device)
        z = z[idx]
    z = z.to(torch.float64)
    n = z.shape[0]
    if n < 2:
        return 1.0
    d2 = _sqdist(z, z)
    iu = torch.triu_indices(n, n, offset=1, device=z.device)
    vals = d2[iu[0], iu[1]].sqrt()
    return max(float(vals.median()), _MIN_BANDWIDTH)


def _scalar_or_none(b):
    """float(b) if ``b`` is a real scalar (python/numpy number, 0-d array/tensor), else None."""
    if isinstance(b, bool):
        return None
    if isinstance(b, numbers.Real):  # python int/float, numpy floating/integer scalars
        return float(b)
    if torch.is_tensor(b) and b.ndim == 0:
        return float(b.item())
    if getattr(b, "ndim", None) == 0 and hasattr(b, "item"):  # 0-d numpy array
        return float(b.item())
    return None


def _resolve_sigmas(bandwidth, x: torch.Tensor, y: torch.Tensor, seed: int = 0) -> list[float]:
    """Bandwidth spec -> list of positive floats.

    Accepts "median", any real scalar (python/numpy number, 0-d numpy array or 0-d
    torch tensor) or a 1-d sequence/array/tensor of such scalars.
    """
    if isinstance(bandwidth, str):
        if bandwidth != "median":
            raise ValueError(f"bandwidth must be 'median', a number or a list; got {bandwidth!r}")
        return [median_bandwidth(x, y, seed=seed)]
    scalar = _scalar_or_none(bandwidth)
    if scalar is not None:
        sigmas = [scalar]
    else:
        try:
            items = list(bandwidth)
        except TypeError:
            raise ValueError(f"unsupported bandwidth spec: {bandwidth!r}") from None
        sigmas = []
        for s in items:
            v = _scalar_or_none(s)
            if v is None:
                raise ValueError(f"unsupported bandwidth entry: {s!r}")
            sigmas.append(v)
    if not sigmas or any((not math.isfinite(s)) or s <= 0 for s in sigmas):
        raise ValueError(f"bandwidths must be positive and finite; got {sigmas}")
    return sigmas


def _block_kernel_sum(
    a: torch.Tensor,
    b: torch.Tensor,
    sigmas: list[float],
    block_size: int,
    dtype: torch.dtype | None,
    same: bool,
) -> torch.Tensor:
    """Sum of RBF kernel entries over all (a_i, b_j) pairs, per bandwidth.

    If ``same`` (a is b), the diagonal is excluded. Tiles of at most
    ``block_size`` x ``block_size`` are materialised; sums are accumulated in float64.
    """
    sums = torch.zeros(len(sigmas), dtype=torch.float64, device=a.device)
    na, nb = a.shape[0], b.shape[0]
    for i0 in range(0, na, block_size):
        ai = a[i0 : i0 + block_size]
        for j0 in range(0, nb, block_size):
            bj = b[j0 : j0 + block_size]
            d2 = _sqdist(ai, bj, dtype)
            for s_idx, sigma in enumerate(sigmas):
                k = torch.exp(d2 * (-1.0 / (2.0 * sigma * sigma)))
                if same and i0 == j0:
                    k.fill_diagonal_(0.0)
                sums[s_idx] += k.sum(dtype=torch.float64)
    return sums


def mmd2_unbiased(
    x,
    y,
    kernel: str = "rbf",
    bandwidth="median",
    block_size: int = 1024,
    dtype: torch.dtype | None = None,
    sum_bandwidths: bool = True,
) -> float:
    """Unbiased squared MMD between samples x (n, d) and y (m, d), RBF kernel.

    MMD^2_u = sum_{i!=j} k(x_i,x_j)/(n(n-1)) + sum_{i!=j} k(y_i,y_j)/(m(m-1))
              - 2 sum_{i,j} k(x_i,y_j)/(nm).
    Can be slightly negative (unbiased estimator). ``bandwidth``: "median"
    (median heuristic on the pooled sample), a real scalar, or a list of scalars
    whose per-bandwidth statistics are summed (default) or averaged. Kernel
    matrices are processed in ``block_size`` tiles to bound memory
    (~block_size^2 entries). Unequal n and m are supported; both must be >= 2.
    Computation runs on ``x.device`` (``y`` is moved there).
    """
    if kernel != "rbf":
        raise ValueError(f"only the 'rbf' kernel is supported, got {kernel!r}")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    x = _as_2d(x, "x")
    y = _as_2d(y, "y")
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        raise ValueError(f"need at least 2 samples in each set, got n={n}, m={m}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"feature dims differ: {x.shape[1]} vs {y.shape[1]}")
    y = y.to(x.device)
    sigmas = _resolve_sigmas(bandwidth, x, y)
    kxx = _block_kernel_sum(x, x, sigmas, block_size, dtype, same=True)
    kyy = _block_kernel_sum(y, y, sigmas, block_size, dtype, same=True)
    kxy = _block_kernel_sum(x, y, sigmas, block_size, dtype, same=False)
    per_sigma = kxx / (n * (n - 1)) + kyy / (m * (m - 1)) - 2.0 * kxy / (n * m)
    out = per_sigma.sum() if sum_bandwidths else per_sigma.mean()
    return float(out)


def mmd_subsample_ci(
    x,
    y,
    n_subsamples: int = 200,
    subsample_size: int | None = None,
    seed: int = 0,
    level: float = 0.95,
    bandwidth="median",
    block_size: int = 1024,
    dtype: torch.dtype | None = None,
    sum_bandwidths: bool = True,
) -> tuple[float, float]:
    """Percentile interval of MMD^2_u over m-out-of-n subsamples WITHOUT replacement.

    Each subsample draws ``subsample_size`` points without replacement from each set
    (default m = n // 2 for x and m = len(y) // 2 for y; an int value is used for
    both sets, capped at their sizes). Sampling without replacement keeps the
    estimator unbiased in every subsample (no duplicated points), so under
    identical distributions the subsample statistics are centred at 0 and the
    interval covers 0. The interval is NOT rescaled to the full sample size: a
    size-m statistic has larger variance than the size-n one, so the interval is
    conservative (wider), which is acceptable for an effect-size band. The
    bandwidth is resolved ONCE on the full data (pass the benchmark's pooled
    bandwidth for comparability along the stream). Returns (lo, hi).
    """
    x = _as_2d(x, "x")
    y = _as_2d(y, "y").to(x.device)
    n, m = x.shape[0], y.shape[0]
    if subsample_size is None:
        mx, my = n // 2, m // 2
    else:
        mx, my = min(int(subsample_size), n), min(int(subsample_size), m)
    if mx < 2 or my < 2:
        raise ValueError(f"subsample sizes must be >= 2, got {mx}, {my} (n={n}, m={m})")
    sigmas = _resolve_sigmas(bandwidth, x, y, seed=seed)
    gen = torch.Generator().manual_seed(int(seed))
    vals = []
    for _ in range(int(n_subsamples)):
        ix = torch.randperm(n, generator=gen)[:mx].to(x.device)
        iy = torch.randperm(m, generator=gen)[:my].to(x.device)
        vals.append(
            mmd2_unbiased(
                x[ix], y[iy], bandwidth=sigmas, block_size=block_size, dtype=dtype,
                sum_bandwidths=sum_bandwidths,
            )
        )
    if not vals:
        return float("nan"), float("nan")
    t = torch.tensor(vals, dtype=torch.float64)
    alpha = (1.0 - level) / 2.0
    q = torch.quantile(t, torch.tensor([alpha, 1.0 - alpha], dtype=torch.float64))
    return float(q[0]), float(q[1])


def mmd_permutation_test(
    x,
    y,
    n_permutations: int = 200,
    seed: int = 0,
    kernel: str = "rbf",
    bandwidth="median",
    dtype: torch.dtype | None = None,
    sum_bandwidths: bool = True,
    max_pooled: int = 8000,
    return_null: bool = False,
):
    """Permutation test for MMD^2_u.

    Returns (observed statistic, p-value), or (statistic, p-value, null) with
    ``return_null=True`` where ``null`` is a float64 CPU tensor of the permuted
    statistics (use e.g. its 95th percentile as a null band beside the CI).

    The bandwidth is resolved ONCE, from a subsample of the POOLED data
    (``median_bandwidth``), so it is invariant to the permutation and is not
    re-estimated per permutation. The pooled (n+m)^2 kernel matrix is built once on
    ``x.device``: memory is about 2 * (n+m)^2 * bytes-per-element (distances +
    kernel), i.e. ~512 MB in float32 / ~1 GB in float64 at the default
    ``max_pooled=8000``; larger inputs raise (subsample first).
    p = (1 + #{perm stat >= observed}) / (1 + n_permutations).
    """
    if kernel != "rbf":
        raise ValueError(f"only the 'rbf' kernel is supported, got {kernel!r}")
    x = _as_2d(x, "x")
    y = _as_2d(y, "y").to(x.device)
    n, m = x.shape[0], y.shape[0]
    if n < 2 or m < 2:
        raise ValueError(f"need at least 2 samples in each set, got n={n}, m={m}")
    if n + m > max_pooled:
        raise ValueError(f"pooled size {n + m} > max_pooled={max_pooled}; subsample first")
    sigmas = _resolve_sigmas(bandwidth, x, y, seed=seed)
    z = torch.cat([x, y], dim=0)
    big_n = n + m
    d2 = _sqdist(z, z, dtype)
    kmat = None
    for sigma in sigmas:
        k = torch.exp(d2 * (-1.0 / (2.0 * sigma * sigma)))
        kmat = k if kmat is None else kmat + k
    del d2
    if not sum_bandwidths:
        kmat = kmat / len(sigmas)
    kmat.fill_diagonal_(0.0)

    def stat(ix: torch.Tensor, iy: torch.Tensor) -> float:
        rows_x = kmat.index_select(0, ix)
        kxx = rows_x.index_select(1, ix).sum(dtype=torch.float64)
        kxy = rows_x.index_select(1, iy).sum(dtype=torch.float64)
        kyy = kmat.index_select(0, iy).index_select(1, iy).sum(dtype=torch.float64)
        return float(kxx / (n * (n - 1)) + kyy / (m * (m - 1)) - 2.0 * kxy / (n * m))

    dev = kmat.device
    obs = stat(torch.arange(n, device=dev), torch.arange(n, big_n, device=dev))
    gen = torch.Generator().manual_seed(int(seed))
    null = []
    count = 0
    for _ in range(int(n_permutations)):
        perm = torch.randperm(big_n, generator=gen).to(dev)
        s = stat(perm[:n], perm[n:])
        null.append(s)
        if s >= obs:
            count += 1
    p = (1.0 + count) / (1.0 + n_permutations)
    if return_null:
        return obs, p, torch.tensor(null, dtype=torch.float64)
    return obs, p


def _unbiased_hsic(k: torch.Tensor, gl: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator of Song et al. (2007) from two n x n Gram matrices.

    HSIC_u = [tr(K~ L~) + (1'K~1)(1'L~1)/((n-1)(n-2)) - 2/(n-2) 1'K~L~1] / (n(n-3)),
    where K~, L~ are the Gram matrices with zeroed diagonals. Requires n >= 4.
    """
    n = k.shape[0]
    k = k.clone()
    gl = gl.clone()
    k.fill_diagonal_(0.0)
    gl.fill_diagonal_(0.0)
    ones = torch.ones(n, 1, dtype=k.dtype, device=k.device)
    tr = (k * gl).sum()  # tr(K L) for symmetric L
    term2 = (ones.T @ k @ ones) * (ones.T @ gl @ ones) / ((n - 1) * (n - 2))
    term3 = 2.0 / (n - 2) * (ones.T @ k @ gl @ ones)
    return (tr + term2.squeeze() - term3.squeeze()) / (n * (n - 3))


def linear_cka(
    x,
    y,
    debiased: bool = True,
    dtype: torch.dtype = torch.float64,
    eps: float = 1e-12,
) -> float:
    """Linear CKA between representations x (n, d1) and y (n, d2) of the SAME n inputs.

    ``debiased=True`` (default, RESEARCH_LOG 7.3): CKA from the unbiased HSIC
    estimator of Song et al. (2007),
        CKA = HSIC_u(K, L) / sqrt(HSIC_u(K, K) * HSIC_u(L, L)),  K = XX', L = YY',
    computed on n x n Gram matrices; requires n >= 4. Unlike the biased estimator
    it does not saturate toward 1 when n < d (independent representations give
    values near 0). Returns NaN if HSIC_u(K,K) or HSIC_u(L,L) is not positive
    (can happen for tiny n).
    ``debiased=False``: biased Eq. 3 of Kornblith et al.,
        ||Y'X||_F^2 / (||X'X||_F * ||Y'Y||_F) after column-centering,
    computed in the d x d feature form when d <= n and in the equivalent n x n
    Gram form tr(KL) / (||K||_F ||L||_F) when max(d1, d2) > n (bounded memory).

    Inputs must be 1-d or 2-d; conv maps (ndim > 2) are rejected: global-average-pool
    them first (RESEARCH_LOG 7.3). Invariant to orthogonal transforms and isotropic
    scaling. Intended for paired probe inputs under model t vs t+1 (representation
    drift), whereas MMD compares different input distributions under a frozen
    encoder (data shift).
    """
    x = _as_2d(x, "x").to(dtype)
    y = _as_2d(y, "y").to(dtype).to(x.device)
    n = x.shape[0]
    if n != y.shape[0]:
        raise ValueError(f"CKA needs equal n, got {n} vs {y.shape[0]}")
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    if debiased:
        if n < 4:
            raise ValueError(f"debiased CKA needs n >= 4, got n={n}")
        k = x @ x.T
        gl = y @ y.T
        hxy = _unbiased_hsic(k, gl)
        hxx = _unbiased_hsic(k, k)
        hyy = _unbiased_hsic(gl, gl)
        if float(hxx) <= 0.0 or float(hyy) <= 0.0:
            return float("nan")
        return float(hxy / torch.sqrt(hxx * hyy).clamp_min(eps))
    if max(x.shape[1], y.shape[1]) > n:
        k = x @ x.T
        gl = y @ y.T
        num = (k * gl).sum()
        den = torch.linalg.matrix_norm(k, ord="fro") * torch.linalg.matrix_norm(gl, ord="fro")
        return float(num / den.clamp_min(eps))
    hsic = torch.linalg.matrix_norm(y.T @ x, ord="fro") ** 2
    nx = torch.linalg.matrix_norm(x.T @ x, ord="fro")
    ny = torch.linalg.matrix_norm(y.T @ y, ord="fro")
    return float(hsic / (nx * ny).clamp_min(eps))
