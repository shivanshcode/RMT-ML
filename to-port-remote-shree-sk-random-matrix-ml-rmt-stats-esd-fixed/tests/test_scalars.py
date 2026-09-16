import numpy as np
import pytest

from rmt import scalars as S


def test_stable_rank_identity():
    assert abs(S.stable_rank(np.eye(20)) - 20) < 1e-9


def test_stable_rank_rank_one():
    v = np.random.default_rng(0).standard_normal((30, 1))
    W = v @ v.T
    assert abs(S.stable_rank(W) - 1.0) < 1e-6


def test_stable_rank_scale_invariant(rng):
    W = rng.standard_normal((40, 25))
    assert abs(S.stable_rank(W) - S.stable_rank(3.7 * W)) < 1e-6


def test_stable_rank_bounds(rng):
    W = rng.standard_normal((40, 25))
    sr = S.stable_rank(W)
    assert 1.0 <= sr <= min(W.shape) + 1e-9


def test_spectral_entropy_uniform_is_log_k():
    s = np.ones(16)
    assert abs(S.spectral_entropy(s) - np.log(16)) < 1e-9


def test_spectral_entropy_dominant_is_zero():
    s = np.array([10.0, 1e-8, 1e-8])
    assert S.spectral_entropy(s) < 1e-4


def test_row_wise_entropy_uniform_and_onehot():
    W_uniform = np.ones((4, 8))
    assert abs(S.row_wise_entropy(W_uniform) - np.log(8)) < 1e-9
    W_onehot = np.zeros((4, 8)); W_onehot[:, 0] = 1.0
    assert S.row_wise_entropy(W_onehot) < 1e-9


def test_ipr_onehot_and_uniform():
    v = np.zeros(10); v[3] = 5.0
    assert abs(float(S.ipr(v)) - 1.0) < 1e-9
    u = np.ones(10)
    assert abs(float(S.ipr(u)) - 1.0 / 10) < 1e-9


def test_ipr_summary_keys(rng):
    from rmt import mp
    W = rng.standard_normal((200, 120))
    _, s, Vh = np.linalg.svd(W, full_matrices=False)
    sigma = mp.estimate_sigma_gd_median(s=s, n=200, m=120)
    d = S.ipr_summary(Vh, s, 200, 120, sigma)
    assert set(d) == {"ipr_top10_mean", "ipr_bulk_mean"}
    for v in d.values():
        assert 0 < v <= 1.0 + 1e-9


def test_porter_thomas_random_is_random(rng):
    from scipy.stats import ortho_group
    Q = ortho_group.rvs(200, random_state=1234)
    res = S.porter_thomas_ks(Q)
    assert res["pt_frac_random"] > 0.8
    eye = np.eye(200)
    res2 = S.porter_thomas_ks(eye)
    assert res2["pt_frac_random"] < res["pt_frac_random"]


def test_decile_index_ranges_partition():
    rngs = S.decile_index_ranges(95, 10, ascending=True)
    assert rngs[0][0] == 0 and rngs[-1][1] == 95
    for i in range(len(rngs) - 1):
        assert rngs[i][1] == rngs[i + 1][0]      # disjoint, contiguous


def test_per_decile_keys_and_partition(rng):
    s = np.sort(np.abs(rng.standard_normal(500)))
    d = S.per_decile(s, 10)
    for i in range(1, 11):
        assert f"entropy_decile_{i}" in d and f"srk_decile_{i}" in d
    # decile masses (Σν² per decile) sum to total
    ranges = S.decile_index_ranges(s.size, 10, ascending=True)
    masses = [np.sum(np.sort(s)[lo:hi] ** 2) for lo, hi in ranges]
    assert abs(sum(masses) - np.sum(s**2)) < 1e-6


def test_mp_softrank_and_bulk_mass():
    s = np.array([5.0, 2.0, 1.0, 0.5])
    assert abs(S.mp_softrank(s, nu_plus=2.5) - 2.5 / 5.0) < 1e-9
    bm = S.bulk_mass_frac(s, nu_plus=2.5)
    assert 0 <= bm <= 1
