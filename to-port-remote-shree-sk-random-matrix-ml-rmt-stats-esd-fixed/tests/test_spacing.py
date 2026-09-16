import numpy as np
import pytest

from rmt import spacing as SP
from rmt import ensembles as E
from rmt.config import TOL

# NumPy compatibility: np.trapezoid is 2.0+; np.trapz is the 1.x spelling and is
# deprecated in 2.0. Bind whichever exists so the suite runs on both. Many HPC
# Anaconda stacks still ship NumPy 1.26.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _goe_eigs(n, rng):
    return np.linalg.eigvalsh(E.goe(n, rng))


def _wishart_svals(n, rng):
    """Singular values of a square Gaussian matrix: a Marchenko-Pastur bulk with
    hard spectral edges. This is the regime real weight matrices live in and the
    regime where global-polynomial unfolding fails."""
    W = rng.standard_normal((n, n)) / np.sqrt(n)
    return np.linalg.svd(W, compute_uv=False)


# --------------------------------------------------------------------------- #
# r-statistic                                                                  #
# --------------------------------------------------------------------------- #
def test_r_statistic_goe(rng):
    r = SP.r_statistic(_goe_eigs(800, rng))
    mid, tol = TOL["r_goe"]
    assert abs(r - mid) < tol, r


def test_r_statistic_poisson(rng):
    r = SP.r_statistic(E.poisson_levels(4000, rng))
    mid, tol = TOL["r_poisson"]
    assert abs(r - mid) < tol, r


def test_r_statistic_ordering(rng):
    assert SP.r_statistic(_goe_eigs(800, rng)) > \
        SP.r_statistic(E.poisson_levels(4000, rng))


# --------------------------------------------------------------------------- #
# Unfolding                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gen", ["goe", "wishart", "poisson"])
def test_unfolding_is_monotone_and_unit_mean(rng, gen):
    """A valid unfolding is strictly increasing and has <s> == 1 with NO
    rescaling. Regression test for the global-polynomial unfolding, which
    produced negative spacings on MP-type spectra."""
    lv = {"goe": lambda: _goe_eigs(800, rng),
          "wishart": lambda: _wishart_svals(1024, rng),
          "poisson": lambda: E.poisson_levels(4000, rng)}[gen]()
    d = SP.unfold_diagnostics(lv)
    assert d["unfold_frac_nonpositive"] == 0.0, d
    assert abs(d["unfold_mean_spacing"] - 1.0) < TOL["unfold_mean_spacing"], d


def test_poly_unfolding_fails_on_eigenvalue_domain(rng):
    """Documents *why* the default changed. ``per_matrix`` feeds the eigenvalue
    domain lambda = s^2/N to the spacing block; a global degree-7 polynomial
    cannot follow that density and leaves a large-scale modulation that inflates
    the number variance by more than an order of magnitude."""
    lam = _wishart_svals(1024, rng) ** 2 / 1024.0
    th = SP.sigma2_goe_theory(10)
    err_poly = abs(SP.sigma2(lam, 10, method="poly") - th) / th
    err_gauss = abs(SP.sigma2(lam, 10, method="gauss") - th) / th
    assert err_gauss < 0.25, err_gauss
    assert err_poly > 5.0, err_poly


def test_gauss_unfolding_is_reparametrisation_invariant(rng):
    """A local-kernel unfolding removes the density exactly, so statistics on s
    and on lambda = s^2/N agree. This is the property the raw np.polyfit path
    violated, which is how the pipeline result came to depend on an arbitrary
    choice of domain."""
    sv = _wishart_svals(1024, rng)
    lam = sv ** 2 / 1024.0
    for L in (5, 10):
        a, b = SP.sigma2(sv, L, method="gauss"), SP.sigma2(lam, L, method="gauss")
        assert abs(a - b) / a < 0.15, (L, a, b)
        a, b = SP.delta3(sv, L, method="gauss"), SP.delta3(lam, L, method="gauss")
        assert abs(a - b) / a < 0.10, (L, a, b)


def test_cheb_unfolding_is_domain_dependent(rng):
    """A *global* fit is not reparametrisation-invariant: a degree-7 Chebyshev
    staircase fit in nu is not one in lambda. It buys long-range accuracy at the
    price of caring about the domain, which is why cfg.spacing_domain defaults
    to 'sval' (Thamm's convention) and must not be flipped casually."""
    sv = _wishart_svals(1024, rng)
    lam = sv ** 2 / 1024.0
    th = SP.sigma2_goe_theory(50)
    err_nu = abs(SP.sigma2(sv, 50, method="cheb") - th) / th
    err_lam = abs(SP.sigma2(lam, 50, method="cheb") - th) / th
    assert err_nu < err_lam


def test_auto_falls_back_when_global_fit_is_invalid(rng):
    """Degree 0 cannot represent any staircase, so 'auto' must not return it."""
    sv = _wishart_svals(1024, rng)
    assert not SP.unfolding_is_valid(SP._unfold_cheb(sv, 0))
    xi = SP.unfold(sv, deg=0, method="auto")
    assert SP.unfolding_is_valid(xi)
    assert np.allclose(xi, SP.unfold(sv, method="gauss"))


def test_cheb_beats_gauss_at_long_range(rng):
    """The reason 'auto' prefers the global fit: a 15-level kernel cannot carry
    count fluctuations at L=50, and Sigma^2 collapses toward zero."""
    sv = _wishart_svals(2048, rng)
    th = SP.sigma2_goe_theory(50)
    assert abs(SP.sigma2(sv, 50, method="cheb") - th) / th < 0.25
    assert SP.sigma2(sv, 50, method="gauss") < 0.75 * th


def test_gauss_unfold_raises_on_too_few_levels():
    with pytest.raises(ValueError):
        SP.unfold(np.sort(np.random.rand(20)), method="gauss", win_size=15)


# --------------------------------------------------------------------------- #
# NN spacing distribution                                                      #
# --------------------------------------------------------------------------- #
def test_nn_spacing_goe_closer_to_wigner(rng):
    d = SP.nn_spacing_ks(_goe_eigs(800, rng))
    assert d["nn_KS_GOE"] < d["nn_KS_Poisson"]
    assert d["nn_KS_GOE_p"] > d["nn_KS_Poisson_p"]


def test_nn_spacing_poisson_closer_to_poisson(rng):
    d = SP.nn_spacing_ks(E.poisson_levels(4000, rng))
    assert d["nn_KS_Poisson"] < d["nn_KS_GOE"]
    assert d["nn_KS_Poisson_p"] > d["nn_KS_GOE_p"]


def test_nn_spacing_ks_pvalue_rejects_wrong_law(rng):
    d = SP.nn_spacing_ks(_goe_eigs(1200, rng))
    assert d["nn_KS_Poisson_p"] < 1e-6
    assert d["nn_KS_GOE_p"] > 0.01


def test_wigner_poisson_cdf_shapes():
    s = np.linspace(0, 6, 100)
    for cdf in (SP.wigner_goe_cdf, SP.poisson_cdf):
        F = cdf(s)
        assert np.all(np.diff(F) >= -1e-12)
        assert abs(F[0]) < 1e-9
        assert F[-1] > 0.99


def test_wigner_pdf_integrates_to_one():
    s = np.linspace(0, 12, 20001)
    assert abs(_trapz(SP.wigner_goe_pdf(s), s) - 1.0) < 1e-4


def test_brody_cdf_limits():
    s = np.linspace(0.01, 5, 200)
    assert np.allclose(SP.brody_cdf(s, 0.0), SP.poisson_cdf(s), atol=1e-9)
    assert np.allclose(SP.brody_cdf(s, 1.0), SP.wigner_goe_cdf(s), atol=1e-9)


def test_brody_beta_goe_vs_poisson(rng):
    bg = SP.brody_beta(_goe_eigs(1500, rng), n_bootstrap=60)["brody_beta"]
    bp = SP.brody_beta(E.poisson_levels(4000, rng), n_bootstrap=60)["brody_beta"]
    mid, tol = TOL["brody_goe"]
    assert abs(bg - mid) < tol, bg
    mid, tol = TOL["brody_poisson"]
    assert abs(bp - mid) < tol, bp
    assert bg > bp


# --------------------------------------------------------------------------- #
# Delta_3 and Sigma^2                                                          #
# --------------------------------------------------------------------------- #
def test_delta3_window_matches_dense_quadrature(rng):
    """The exact Bohigas-Giannoni form must agree with a brute-force least
    squares fit of the staircase under the L2 integral norm."""
    xi = np.sort(rng.uniform(0, 2000, 2000))
    for L in (5.0, 10.0, 50.0):
        a = 500.0
        pts = np.sort(xi[(xi >= a) & (xi < a + L)])
        exact = SP._delta3_window(pts - (a + L / 2), L)
        t = np.linspace(a, a + L, 20001)
        Nt = np.searchsorted(pts, t, side="right").astype(float)
        A = np.vstack([np.ones_like(t), t]).T
        coef, *_ = np.linalg.lstsq(A, Nt, rcond=None)
        dense = _trapz((Nt - A @ coef) ** 2, t) / L
        assert abs(exact - dense) < 1e-3 * max(1.0, dense), (L, exact, dense)


@pytest.mark.parametrize("L", [5, 10, 20, 50])
def test_delta3_goe_log_law(rng, L):
    d3 = np.mean([SP.delta3(_goe_eigs(1200, rng), L) for _ in range(3)])
    th = SP.delta3_goe_theory(L)
    assert abs(d3 - th) / th < TOL["delta3_rtol"], (L, d3, th)


@pytest.mark.parametrize("L", [5, 10])
def test_sigma2_goe_log_law(rng, L):
    s2 = np.mean([SP.sigma2(_goe_eigs(1500, rng), L) for _ in range(3)])
    th = SP.sigma2_goe_theory(L)
    assert abs(s2 - th) / th < TOL["sigma2_rtol"], (L, s2, th)


def test_sigma2_degrades_gracefully_at_L20(rng):
    """Sigma^2 is the unfolding-sensitive statistic; at the edge of its validity
    range it is still the right order of magnitude, unlike the GOE-vs-Poisson
    gap it has to resolve."""
    s2 = np.mean([SP.sigma2(_goe_eigs(1500, rng), 20) for _ in range(3)])
    th = SP.sigma2_goe_theory(20)
    assert abs(s2 - th) / th < 0.30, (s2, th)
    assert s2 < 0.25 * 20


@pytest.mark.parametrize("L", [5, 10, 20])
def test_rigidity_on_wishart_bulk(rng, L):
    """The MP-spectrum case that the old polynomial unfolding got wrong by up
    to a factor 18 (Sigma^2) and 7 (Delta_3)."""
    sv = _wishart_svals(2048, rng)
    s2, d3 = SP.sigma2(sv, L), SP.delta3(sv, L)
    assert abs(s2 - SP.sigma2_goe_theory(L)) / SP.sigma2_goe_theory(L) < 0.25
    assert abs(d3 - SP.delta3_goe_theory(L)) / SP.delta3_goe_theory(L) < 0.25


def test_delta3_sigma2_goe_below_poisson(rng):
    ev = _goe_eigs(1200, rng)
    lv = E.poisson_levels(4000, rng)
    assert SP.sigma2(ev, 10) < SP.sigma2(lv, 10)
    assert SP.delta3(ev, 20) < SP.delta3(lv, 20)


def test_delta3_poisson_linear_law(rng):
    lv = E.poisson_levels(8000, rng)
    for L in (10, 20):
        d3 = SP.delta3(lv, L)
        assert abs(d3 - L / 15.0) / (L / 15.0) < 0.25, (L, d3)


def test_sigma2_warns_outside_validity(rng):
    ev = _goe_eigs(1200, rng)
    with pytest.warns(RuntimeWarning):
        SP.sigma2(ev, 60, method="cheb")
    with pytest.warns(RuntimeWarning):        # gauss caps out much earlier
        SP.sigma2(ev, 30, method="gauss")


def test_rigidity_nan_when_window_exceeds_span(rng):
    lv = np.sort(rng.random(200))
    assert np.isnan(SP.delta3(lv, 10_000))
    assert np.isnan(SP.sigma2(lv, 10_000))


# --------------------------------------------------------------------------- #
# Complex spacing ratio                                                        #
# --------------------------------------------------------------------------- #
def test_complex_spacing_ginue(rng):
    z = SP.complex_spacing_ratio(E.ginue(400, rng))
    mid, tol = TOL["ginue_abs"]
    assert abs(z["abs_mean"] - mid) < tol, z
    lo, hi = TOL["ginue_cos"]
    assert lo <= z["cos_mean"] <= hi, z


def test_complex_spacing_poisson2d(rng):
    z = SP.complex_spacing_ratio(np.diag(E.poisson_points_2d(2000, rng)))
    mid, tol = TOL["poi2d_abs"]
    assert abs(z["abs_mean"] - mid) < tol, z
    assert abs(z["cos_mean"]) <= TOL["poi2d_cos_abs"], z


def test_complex_spacing_ordering(rng):
    g = SP.complex_spacing_ratio(E.ginue(400, rng))
    p = SP.complex_spacing_ratio(np.diag(E.poisson_points_2d(2000, rng)))
    assert g["abs_mean"] > p["abs_mean"]
    assert g["cos_mean"] < p["cos_mean"]


def test_unfolding_kernel_chunking_is_exact(rng, monkeypatch):
    """The broadening kernel is n x n (2.1 GB at n=16384), so it is evaluated in
    row blocks; blocking must not change the result."""
    lv = np.sort(rng.random(2000))
    full = SP.GaussBroadening(15).unfold_spectrum(lv)
    monkeypatch.setattr(SP, "_KERNEL_BLOCK", 20_000)
    chunked = SP.GaussBroadening(15).unfold_spectrum(lv)
    assert np.allclose(full, chunked, rtol=0, atol=1e-9)


# --------------------------------------------------------------------------- #
# auto-dispatch cross-check                                                    #
# --------------------------------------------------------------------------- #
def test_auto_picks_global_fit_when_the_basis_fits(rng):
    sv = _wishart_svals(2048, rng)
    _, used = SP.unfold_auto(sv)
    assert used == "cheb"


def test_auto_falls_back_on_square_eigenvalue_domain(rng):
    """lambda = nu^2/N of a *square* matrix has a hard edge at 0 and spans ~10
    decades; the global fit stays monotone and unit-mean there yet leaves a huge
    density modulation, so monotonicity alone is not a sufficient validity test.
    The short-scale cross-check against the local kernel catches it."""
    lam = _wishart_svals(1024, rng) ** 2 / 1024.0
    xi, used = SP.unfold_auto(lam)
    # UPDATED.  Was: falls back to "gauss".  That fallback was the defect, not
    # the cure -- it put the matrix on the local kernel, whose window-reach bias
    # costs ~11 % on Sigma^2(10).  unfold_transform_auto now removes the
    # lambda^(-1/2) hard-edge singularity exactly via x -> sqrt(x), so the global
    # fit is recovered.  What the test still pins is the ORIGINAL diagnosis: in
    # the identity coordinate the global fit really does leave a huge modulation.
    assert used == "cheb"
    assert SP.unfold_transform_auto(lam)[2] == "sqrt"
    assert SP.unfolding_crosscheck(SP._unfold_cheb(lam),
                                   SP.unfold(lam, method="gauss")) > 2.0
    assert SP.unfolding_crosscheck(SP._unfold_cheb(lam, 7),
                                   SP.unfold(lam, method="gauss")) > 2.0
    th = SP.sigma2_goe_theory(10)
    assert abs(SP.sigma2(lam, 10, unfolded=xi) - th) / th < 0.25


def test_auto_recovers_goe_law_in_every_domain(rng):
    """The whole point of the dispatch: the answer must not depend on whether
    the caller passed singular values or eigenvalues, square or rectangular."""
    for n, m in ((1024, 1024), (3584, 1024)):
        W = rng.standard_normal((n, m)) / np.sqrt(m)
        sv = np.linalg.svd(W, compute_uv=False)
        for levels in (sv, sv ** 2 / m):
            xi, _ = SP.unfold_auto(levels)
            th = SP.sigma2_goe_theory(10)
            assert abs(SP.sigma2(levels, 10, unfolded=xi) - th) / th < 0.30, (n, m)
            th = SP.delta3_goe_theory(10)
            assert abs(SP.delta3(levels, 10, unfolded=xi) - th) / th < 0.25, (n, m)


def test_precomputed_unfolding_matches_recomputed(rng):
    sv = _wishart_svals(1024, rng)
    xi = SP.unfold(sv)
    for L in (5, 10):
        assert abs(SP.delta3(sv, L, unfolded=xi) - SP.delta3(sv, L)) < 1e-9
    a = SP.nn_spacing_ks(sv, unfolded=xi)
    b = SP.nn_spacing_ks(sv)
    assert abs(a["nn_KS_GOE"] - b["nn_KS_GOE"]) < 1e-12
