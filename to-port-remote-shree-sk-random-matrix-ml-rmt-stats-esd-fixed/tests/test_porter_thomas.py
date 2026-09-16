"""Calibration tests for rmt.porter_thomas.

The defining property of a correct test is that the p-value is Uniform(0,1)
under the null, at every dimension. A raw KS distance compared against a fixed
threshold has no such property, which is what these tests pin down.
"""
import numpy as np
import pytest

from rmt import porter_thomas as PT
from rmt import scalars as SC
from rmt.config import TOL


def _haar(N, k, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    return Q[:, :k]


@pytest.mark.parametrize("N", [128, 512])
def test_pvalue_uniform_under_null(rng, N):
    """Haar (Porter-Thomas) vectors -> mean p ~ 0.5 and ~alpha rejection rate."""
    Cbar, C, _ = PT.pt_test_statistic(N, n_samples=3000, rng=0)
    V = _haar(N, 200, rng)
    p = np.array([PT.pt_pvalue(V[:, i], C, Cbar) for i in range(V.shape[1])])
    assert abs(p.mean() - 0.5) < TOL["pt_p_uniform"], p.mean()
    assert 0.88 < np.mean(p > 0.05) < 1.0, np.mean(p > 0.05)


def test_localized_vector_rejected(rng):
    N = 512
    Cbar, C, _ = PT.pt_test_statistic(N, n_samples=3000, rng=0)
    v = np.zeros(N)
    v[:20] = rng.standard_normal(20)
    assert PT.pt_pvalue(v, C, Cbar) < 0.01


@pytest.mark.parametrize("N", [512, 2048])
def test_heavy_tailed_vector_rejected_at_every_dimension(rng, N):
    """Regression for the old ``D < 0.1`` rule: a Student-t(3) vector is clearly
    not Porter-Thomas, yet its KS distance stays below 0.1 at every N because
    the null scale of D shrinks like 1/sqrt(N) while the threshold did not."""
    Cbar, C, _ = PT.pt_test_statistic(N, n_samples=3000, rng=0)
    ps, Ds = [], []
    for _ in range(5):
        v = rng.standard_t(df=3, size=N)
        v = v / np.linalg.norm(v)
        x = np.sort(v)
        Ds.append(np.max(np.abs(Cbar(x) - np.arange(N) / (N - 1))))
        ps.append(PT.pt_pvalue(v, C, Cbar))
    assert np.median(ps) < 0.05, ps            # correct test rejects
    assert np.median(Ds) < 0.1                 # old rule would have accepted


def test_null_ks_distance_scales_like_inv_sqrt_N():
    _, _, D256 = PT.pt_test_statistic(256, n_samples=1500, rng=1)
    _, _, D1024 = PT.pt_test_statistic(1024, n_samples=1500, rng=1)
    ratio = np.median(D256) / np.median(D1024)
    assert 1.7 < ratio < 2.3, ratio          # expect ~sqrt(4) = 2


def test_cdf_interpolates_monotonically():
    c = PT.CDF(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    x = np.linspace(0, 1, 101)
    assert np.all(np.diff(c(x)) >= -1e-12)


def test_summary_on_random_matrix(rng):
    W = rng.standard_normal((512, 512)) / np.sqrt(512)
    _, s, Vh = np.linalg.svd(W)
    out = SC.porter_thomas_ks(Vh, n_vectors=120, n_samples=2000)
    assert abs(out["pt_p_mean"] - 0.5) < 0.12, out
    assert out["pt_frac_random"] > 0.85, out


def test_summary_on_localized_basis():
    N = 256
    Vh = np.eye(N)                       # maximally localized: every entry 0 or 1
    out = SC.porter_thomas_ks(Vh, n_vectors=50, n_samples=1500)
    assert out["pt_frac_random"] < 0.05, out
    assert out["pt_p_mean"] < 0.05, out


def test_porter_thomas_pvalues_axis(rng):
    V = _haar(128, 30, rng)
    p_cols = PT.porter_thomas_pvalues(V, axis=0, n_samples=1200)
    p_rows = PT.porter_thomas_pvalues(V.T, axis=1, n_samples=1200)
    assert p_cols.shape == p_rows.shape == (30,)
    assert np.allclose(p_cols, p_rows)


def test_pooled_statistic_runs():
    Cbar, C, D = PT.pt_test_statistic_pooled(64, n_samples=200, pooling_window=2)
    assert D.size == 200 and np.all(np.diff(D) >= 0)


def test_null_statistic_is_cached_and_deterministic():
    """The null depends only on (N, n_samples, seed); rebuilding it per matrix
    costs ~12 s at N=4096 for no information gain."""
    import time
    PT._pt_test_statistic_cached.cache_clear()
    t0 = time.perf_counter()
    _, _, D1 = PT.pt_test_statistic(512, n_samples=1500, rng=3)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, _, D2 = PT.pt_test_statistic(512, n_samples=1500, rng=3)
    warm = time.perf_counter() - t0
    assert np.array_equal(D1, D2)
    assert warm < 0.05 * cold + 1e-3


def test_generator_bypasses_cache():
    rng = np.random.default_rng(9)
    _, _, D1 = PT.pt_test_statistic(128, n_samples=400, rng=rng)
    _, _, D2 = PT.pt_test_statistic(128, n_samples=400, rng=rng)
    assert not np.array_equal(D1, D2)
