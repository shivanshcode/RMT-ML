"""Regressions for the exact finite-L laws and the corrected unfolding.

Pins every claim made in the fix report, including the negative results.
"""
import numpy as np
import pytest

from rmt import reference as REF
from rmt import spacing as SP
from rmt import nulls as NU
from rmt import controls as CT


# --------------------------------------------------------------------------- #
# Exact finite-L laws                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("beta", [1, 2, 4])
def test_cluster_function_is_unity_at_zero(beta):
    """Y_2(0) = 1 for every Wigner-Dyson class: complete level repulsion."""
    assert REF.two_level_cluster(np.array([0.0]), beta)[0] == pytest.approx(1.0, abs=1e-9)


def test_poisson_is_exact_in_closed_form():
    assert REF.sigma2_exact(37.0, 0) == pytest.approx(37.0)
    assert REF.delta3_exact(15.0, 0) == pytest.approx(1.0)


@pytest.mark.parametrize("L,expect", [(3, 0.13214), (5, 0.17382), (10, 0.23570),
                                      (20, 0.30142), (50, 0.39140)])
def test_goe_delta3_exact_values(L, expect):
    assert REF.delta3_exact(L, 1) == pytest.approx(expect, rel=2e-4)


def test_delta3_asymptote_is_badly_wrong_at_small_L_for_beta_one():
    """Review defect F1, quantified.  This is WHY rmt.reference exists."""
    assert REF.delta3_asymptote(3, 1) / REF.delta3_exact(3, 1) - 1 < -0.20
    assert REF.delta3_asymptote(5, 1) / REF.delta3_exact(5, 1) - 1 < -0.10
    # beta = 2 does NOT have this problem -- the defect is specific to beta = 1.
    assert abs(REF.delta3_asymptote(3, 2) / REF.delta3_exact(3, 2) - 1) < 0.01


def test_sigma2_asymptote_is_already_accurate():
    """Sigma^2 needs no correction, so every Sigma^2 benchmark error is real."""
    for L in (3, 5, 10, 20, 50):
        assert abs(REF.sigma2_asymptote(L, 1) / REF.sigma2_exact(L, 1) - 1) < 0.002


def test_exact_laws_converge_to_the_asymptotes_at_large_L():
    for beta in (1, 2, 4):
        assert REF.delta3_exact(200, beta) == pytest.approx(
            REF.delta3_asymptote(200, beta), rel=0.02)


# --------------------------------------------------------------------------- #
# Coordinate-transform unfolding                                               #
# --------------------------------------------------------------------------- #
def _square_svals(n=512, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(np.linalg.svd(rng.standard_normal((n, n)) / np.sqrt(n),
                                 compute_uv=False))


def test_lambda_domain_picks_sqrt_and_stays_on_the_global_fit():
    """The lambda-domain hard edge is what forced the gauss branch before."""
    xi, branch, tx, ratio = SP.unfold_transform_auto(_square_svals() ** 2)
    assert branch == "cheb"
    assert tx == "sqrt"
    assert abs(ratio - 1.0) < 0.3


def test_nu_domain_is_not_gratuitously_transformed():
    _, branch, tx, _ = SP.unfold_transform_auto(_square_svals())
    assert (branch, tx) == ("cheb", "identity")


def test_transform_search_fixes_lambda_domain_sigma2():
    """Was +10.8 % at L=10 / +19.7 % at L=20 on the gauss branch.

    Averaged over replicas: the PER-MATRIX null sd of Sigma^2(20) is ~9 %
    (review §3/F1), so a single matrix cannot resolve an 8 % bias.
    """
    got = {L: [] for L in (5.0, 10.0, 20.0)}
    for seed in range(4):
        xi, _, _, _ = SP.unfold_transform_auto(_square_svals(1024, seed=seed) ** 2)
        xs = np.sort(xi)
        for L in got:
            got[L].append(SP.sigma2(None, L, unfolded=xs, n_windows=100_000, rng=seed))
    for L, vals in got.items():
        assert abs(np.mean(vals) / REF.sigma2_exact(L, 1) - 1) < 0.08


# --------------------------------------------------------------------------- #
# Adaptive outlier stripping                                                   #
# --------------------------------------------------------------------------- #
def test_strips_equal_amplitude_spike_cluster():
    """Equal-amplitude spikes detach as one tight group -> the gap rule."""
    rng = np.random.default_rng(0)
    n, m = 3584, 1024
    W = rng.standard_normal((n, m)) / np.sqrt(n)
    u = rng.standard_normal((n, 20)); u /= np.linalg.norm(u, axis=0)
    v = rng.standard_normal((m, 20)); v /= np.linalg.norm(v, axis=0)
    sv = np.sort(np.linalg.svd(W + 3.0 * u @ v.T, compute_uv=False))
    _, lo, hi = SP.strip_edge_outliers(sv)
    assert 15 <= lo + hi <= 40


def test_strips_graded_spike_tail_and_recovers_sigma2():
    """Spikes of DECAYING amplitude never detach as a cluster -- they thin out
    into the bulk -- so the gap rule alone finds only the topmost one.  This is
    the repo benchmark's own generator, amp*(i+1)**-0.7."""
    def spiked(seed):
        rng = np.random.default_rng(seed)
        n, m = 3584, 1024
        W = rng.standard_normal((n, m)) / np.sqrt(m)
        for i in range(20):
            a = rng.standard_normal(n); b = rng.standard_normal(m)
            W += 4.0 * (i + 1) ** -0.7 * np.outer(a / np.linalg.norm(a),
                                                  b / np.linalg.norm(b))
        return np.sort(np.linalg.svd(W, compute_uv=False))

    vals = []
    for seed in (1000, 1001, 1002):
        s = spiked(seed)
        kept, lo, hi = SP.strip_edge_outliers(s)
        assert 2 <= lo + hi <= 30
        vals.append(SP.sigma2(kept, 20))
        # untrimmed, the spikes wreck the global fit outright
        assert SP.sigma2(s, 20) > 2.0
    assert abs(np.mean(vals) / REF.sigma2_exact(20, 1) - 1) < 0.08


def test_clean_spectra_lose_nothing():
    for lv in (_square_svals(1024), _square_svals(1024, seed=5)):
        _, lo, hi = SP.strip_edge_outliers(lv)
        assert lo + hi <= 2


def test_does_not_strip_a_poisson_tail():
    """Gap size ALONE would strip ~20 % of a Poisson sample -- the detached-count
    condition is what prevents it."""
    rng = np.random.default_rng(0)
    lv = np.sort(np.cumsum(rng.exponential(1.0, 6000)))
    _, lo, hi = SP.strip_edge_outliers(lv)
    assert lo + hi == 0


# --------------------------------------------------------------------------- #
# Pre-unfolded detection                                                       #
# --------------------------------------------------------------------------- #
def test_pre_unfolded_detection_is_a_flatness_test():
    rng = np.random.default_rng(0)
    assert SP.looks_pre_unfolded(np.sort(np.cumsum(rng.exponential(1.0, 6000))))
    assert not SP.looks_pre_unfolded(_square_svals())
    A = rng.standard_normal((800, 800))
    assert not SP.looks_pre_unfolded(np.linalg.eigvalsh((A + A.T) / np.sqrt(1600)))


# --------------------------------------------------------------------------- #
# Sigma^2 reliable-L rule                                                      #
# --------------------------------------------------------------------------- #
def test_narrow_gauss_window_certifies_nothing():
    """The old rule was max(3.0, win/3), which certified L=3 at win=5 -- where
    the measured Sigma^2 error is -15.3 %."""
    assert SP.sigma2_reliable_lmax("gauss", 5) == 0.0
    assert SP.sigma2_reliable_lmax("gauss", 10) == 0.0
    assert SP.sigma2_reliable_lmax("gauss", 15) == pytest.approx(5.0)
    assert SP.sigma2_reliable_lmax("gauss", 30) == pytest.approx(10.0)
    assert SP.sigma2_reliable_lmax("gauss", 60) == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# t-quantile critical value                                                    #
# --------------------------------------------------------------------------- #
def test_critical_value_is_t_not_normal():
    assert NU.critical_value(20, 40) == pytest.approx(3.88, abs=0.01)
    assert NU.critical_value(12, 40) == pytest.approx(4.48, abs=0.01)
    # The reported |z| = 4.10 on a 12-replica null was never a false positive.
    assert NU.critical_value(12, 40) > 4.10
    assert NU.critical_value(400, 40) == pytest.approx(3.23, abs=0.05)


# --------------------------------------------------------------------------- #
# Variance-profile null                                                        #
# --------------------------------------------------------------------------- #
def test_variance_profile_gaussian_preserves_both_margins():
    rng = np.random.default_rng(0)
    a = np.exp(rng.normal(0, 0.7, (400, 1)))
    b = np.exp(rng.normal(0, 0.4, (1, 400)))
    W = a * b * rng.standard_normal((400, 400))
    cv = lambda x: float(x.std() / x.mean())
    G = CT.variance_profile_gaussian(W, rng=1)
    assert cv(G.std(1)) == pytest.approx(cv(W.std(1)), rel=0.15)
    assert cv(G.std(0)) == pytest.approx(cv(W.std(0)), rel=0.15)
    # ...where every shuffle control flattens at least one of them.
    assert cv(CT.entry_shuffle(W, rng=1).std(1)) < 0.3 * cv(W.std(1))
    assert cv(CT.row_shuffle(W, rng=1).std(0)) < 0.5 * cv(W.std(0))
