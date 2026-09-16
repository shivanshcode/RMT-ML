"""Tests for rmt.nulls — the matched-null calibration layer.

The defining property being pinned down here is that a p-value is Uniform(0,1)
under the null and that a null band is a *measurement of the pipeline*, not a
restatement of an asymptotic constant.
"""
import numpy as np
import pytest
import scipy.stats

from rmt import nulls as NU
from rmt import spacing as SP

# NumPy compatibility: np.trapezoid is 2.0+; np.trapz is the 1.x spelling and is
# deprecated in 2.0. Bind whichever exists so the suite runs on both. Many HPC
# Anaconda stacks still ship NumPy 1.26.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- #
# beta-general theory                                                          #
# --------------------------------------------------------------------------- #
def test_beta1_theory_matches_shipped_goe_formulas():
    """nulls must not silently disagree with rmt.spacing for beta = 1."""
    for L in (5.0, 10.0, 20.0, 50.0):
        assert abs(NU.sigma2_theory(L, 1) - SP.sigma2_goe_theory(L)) < 1e-12
        assert abs(NU.delta3_theory(L, 1) - SP.delta3_goe_theory(L)) < 1e-12


def test_poisson_theory():
    assert NU.sigma2_theory(7.0, 0) == 7.0
    assert abs(NU.delta3_theory(30.0, 0) - 2.0) < 1e-12


@pytest.mark.parametrize("beta", [1, 2, 4])
def test_rigidity_decreases_with_beta(beta):
    """More level repulsion => a more rigid spectrum at every L."""
    if beta == 1:
        return
    assert NU.delta3_theory(20.0, beta) < NU.delta3_theory(20.0, 1)
    assert NU.sigma2_theory(20.0, beta) < NU.sigma2_theory(20.0, 1)


@pytest.mark.parametrize("beta", [1, 2, 4])
def test_surmise_cdf_is_a_cdf_with_unit_mean(beta):
    s = np.linspace(0, 8, 40001)
    F = NU.wigner_surmise_cdf(s, beta)
    assert abs(F[0]) < 1e-12
    assert F[-1] > 1 - 1e-6
    assert np.all(np.diff(F) >= -1e-12)
    p = np.gradient(F, s)
    assert abs(_trapz(s * p, s) - 1.0) < 2e-3


def test_beta1_surmise_matches_shipped_wigner():
    s = np.linspace(0, 5, 500)
    assert np.allclose(NU.wigner_surmise_cdf(s, 1), SP.wigner_goe_cdf(s))


# --------------------------------------------------------------------------- #
# ensembles                                                                    #
# --------------------------------------------------------------------------- #
def test_gse_levels_are_kramers_collapsed(rng):
    ev = np.linalg.eigvalsh(NU.gse(60, rng))
    pairs = ev.reshape(-1, 2)
    assert np.allclose(pairs[:, 0], pairs[:, 1], atol=1e-8)
    assert NU.gse_levels(60, rng).size == 60


@pytest.mark.parametrize("beta", [1, 2, 4])
def test_circular_levels_are_exactly_unfolded(rng, beta):
    """The whole point of the circular ensembles: no unfolding is required."""
    xi = NU.circular_levels(400, beta, rng)
    d = np.diff(np.sort(xi))
    assert abs(d.mean() - 1.0) < 0.02
    assert np.all(d > 0)


@pytest.mark.parametrize("beta", [1, 2, 4])
def test_circular_r_statistic_matches_atas(rng, beta):
    r = np.mean([SP.r_statistic(NU.circular_levels(500, beta, rng))
                 for _ in range(4)])
    assert abs(r - NU.R_THEORY[beta]) < 0.012, (beta, r)


def test_circular_delta3_recovers_beta2_and_beta4_asymptotes(rng):
    """beta = 2 and beta = 4 track the asymptote at L = 10 to a few percent --
    which is the evidence that the estimator is unbiased, and hence that the
    beta = 1 discrepancy at small L is the asymptotic formula, not the code."""
    for beta in (2, 4):
        xs = [NU.circular_levels(500, beta, rng) for _ in range(5)]
        d3 = np.mean([SP.delta3(None, 10, unfolded=x) for x in xs])
        assert abs(d3 / NU.delta3_theory(10, beta) - 1) < 0.06, (beta, d3)


def test_beta1_delta3_asymptote_is_low_at_small_L(rng):
    """Regression pinning the defect that motivates the whole module: with an
    EXACT unfolding, measured Delta_3 at L = 5 exceeds the beta = 1 asymptote by
    ~10%. Comparing a real matrix to that constant manufactures a deviation."""
    xs = [NU.circular_levels(600, 1, rng) for _ in range(6)]
    d3 = np.mean([SP.delta3(None, 5, unfolded=x) for x in xs])
    assert d3 / NU.delta3_theory(5, 1) - 1 > 0.05


# --------------------------------------------------------------------------- #
# statistics vector / null bands                                               #
# --------------------------------------------------------------------------- #
def test_spacing_statistics_matches_the_pipeline_path(rng):
    lv = NU.wishart_levels(400, 400, rng)
    st = NU.spacing_statistics(lv, seed=0)
    xi, used = SP.unfold_auto(np.sort(lv))
    assert st["branch"] == used
    assert abs(st["delta3_L10"] - SP.delta3(None, 10, unfolded=np.sort(xi), rng=1)) < 1e-9
    assert abs(st["r_statistic_mean"] - SP.r_statistic(lv)) < 1e-12


def test_spacing_statistics_nan_on_short_spectrum():
    st = NU.spacing_statistics(np.sort(np.random.default_rng(0).random(30)))
    assert np.isnan(st["delta3_L10"])
    assert st["branch"] == ""


def test_null_band_shape_and_zscore():
    band = NU.null_band((300, 300), n_reps=8, seed=1, brody=False)
    e = band["delta3_L10"]
    assert e["p2.5"] <= e["mean"] <= e["p97.5"]
    assert e["sd"] > 0
    assert abs(NU.zscore(e["mean"], e)) < 1e-9
    assert NU.zscore(e["mean"] + 2 * e["sd"], e) == pytest.approx(2.0, abs=1e-6)


def test_null_band_brackets_its_own_realisations():
    band = NU.null_band((300, 300), n_reps=12, seed=2, brody=False)
    v = band["delta3_L10"]["values"]
    assert np.mean(np.abs((v - band["delta3_L10"]["mean"])
                          / band["delta3_L10"]["sd"]) < 2.5) > 0.8


def test_null_band_asymptote_is_outside_the_delta3_L5_band():
    """The headline defect, as a regression test."""
    band = NU.null_band((512, 512), n_reps=12, seed=3, brody=False)
    e = band["delta3_L5"]
    assert NU.delta3_theory(5, 1) < e["p2.5"], (NU.delta3_theory(5, 1), e)


# --------------------------------------------------------------------------- #
# calibrated KS p-value                                                        #
# --------------------------------------------------------------------------- #
def test_ks_pvalue_mc_uniform_under_null(rng):
    """The property scipy.stats.kstest does NOT have here."""
    NU.clear_null_cache()
    n = m = 256
    ps = np.array([NU.ks_pvalue_mc(NU.wishart_levels(n, m, rng), (n, m),
                                   n_null=120, seed=11)["p_mc"]
                   for _ in range(25)])
    assert 0.3 < ps.mean() < 0.7, ps.mean()
    assert scipy.stats.kstest(ps, "uniform").pvalue > 0.02


def test_scipy_kstest_pvalue_is_not_uniform(rng):
    """Documents why ks_pvalue_mc exists: the shipped p-value is inflated and
    essentially never rejects, so it has no power."""
    ps = np.array([SP.nn_spacing_ks(NU.wishart_levels(256, 256, rng))["nn_KS_GOE_p"]
                   for _ in range(15)])
    assert ps.mean() > 0.55, ps.mean()
    assert np.mean(ps < 0.05) == 0.0


def test_ks_pvalue_mc_detects_a_poisson_admixture(rng):
    NU.clear_null_cache()
    n = m = 256
    s = NU.wishart_levels(n, m, rng)
    k = int(0.2 * s.size)
    mixed = np.sort(np.concatenate([s[:s.size - k],
                                    rng.uniform(s.min(), s.max(), k)]))
    assert NU.ks_pvalue_mc(mixed, (n, m), n_null=120, seed=11)["p_mc"] < 0.05


def test_ks_pvalue_mc_never_returns_zero(rng):
    NU.clear_null_cache()
    v = np.eye(200)[0] * 0 + np.sort(rng.random(200))   # wildly non-RMT
    out = NU.ks_pvalue_mc(v, (200,), kind="goe", n_null=40, seed=5)
    assert out["p_mc"] >= 1.0 / 41.0


def test_ks_null_is_cached_and_deterministic():
    NU.clear_null_cache()
    a = NU.ks_pvalue_mc(NU.wishart_levels(200, 200, 0), (200, 200),
                        n_null=40, seed=3)
    b = NU.ks_pvalue_mc(NU.wishart_levels(200, 200, 0), (200, 200),
                        n_null=40, seed=3)
    assert a == b


# --------------------------------------------------------------------------- #
# the unfolding-validity fix                                                   #
# --------------------------------------------------------------------------- #
def test_unfolding_validity_tolerates_a_single_dip():
    """A strict all(d > 0) was a knife-edge on ~10^3 spacings and produced
    branch-dependent NaNs in the CSV."""
    xi = np.cumsum(np.ones(2000))
    xi[1000] = xi[999]                      # one non-increasing step
    assert SP.unfolding_is_valid(xi)
    assert not SP.unfolding_is_valid(xi, nonmono_tol=0.0)


def test_unfolding_validity_still_rejects_a_broken_fit(rng):
    sv = np.linalg.svd(rng.standard_normal((512, 512)) / 32, compute_uv=False)
    assert not SP.unfolding_is_valid(SP._unfold_cheb(np.sort(sv), 0))
