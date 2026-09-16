import numpy as np
import pytest

from rmt import mp
from rmt import ensembles as E
from rmt.config import TOL


GAMMAS = [(500, 2000), (2000, 500), (2000, 2000), (1500, 750)]


def _gauss_legendre_integral(f, a, b, npts=400):
    nodes, wts = np.polynomial.legendre.leggauss(npts)
    xm = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    return 0.5 * (b - a) * np.sum(wts * f(xm))


def test_mp_pdf_integrates_to_one():
    for (n, m) in [(400, 1600), (800, 800), (1000, 250)]:
        for sigma in [0.5, 1.0, 2.0]:
            lo, hi = mp.mp_bounds(n, m, sigma)
            I = _gauss_legendre_integral(lambda x: mp.mp_pdf(x, n, m, sigma), lo, hi)
            assert abs(I - 1.0) < TOL["mp_integral"], (n, m, sigma, I)


def test_mp_pdf_zero_outside_support():
    n, m, sigma = 600, 1200, 1.0
    lo, hi = mp.mp_bounds(n, m, sigma)
    x = np.array([lo * 0.5, hi * 1.5])
    assert np.allclose(mp.mp_pdf(x, n, m, sigma), 0.0)


def test_mp_bounds_ordering():
    lo, hi = mp.mp_bounds(500, 2000, 1.0)
    assert 0 <= lo < hi
    lo2, hi2 = mp.mp_bounds(1000, 1000, 1.0)
    assert abs(lo2) < 1e-9       # gamma=1 => nu_minus ~ 0


def test_mp_cdf_monotone_and_endpoints():
    n, m, sigma = 800, 1600, 1.0
    lo, hi = mp.mp_bounds(n, m, sigma)
    xs = np.linspace(lo, hi, 50)
    F = mp.mp_cdf(xs, n, m, sigma)
    assert np.all(np.diff(F) >= -1e-9)
    assert mp.mp_cdf(lo - 1, n, m, sigma) == 0.0
    assert abs(mp.mp_cdf(hi + 1, n, m, sigma) - 1.0) < 1e-9


def test_mp_median_in_support():
    n, m, sigma = 900, 1800, 1.3
    lo, hi = mp.mp_bounds(n, m, sigma)
    med = mp.mp_median(n, m, sigma)
    assert lo < med < hi


def test_estimate_sigma_recovers_sigma(rng):
    for (n, m) in GAMMAS:
        for sigma in [0.5, 1.0, 2.0]:
            W = E.wishart_factor(n, m, sigma=sigma, rng=rng)
            sig = mp.estimate_sigma_gd_median(W)
            assert abs(sig - sigma) / sigma < TOL["sigma_rel"], (n, m, sigma, sig)


def test_estimate_sigma_accepts_precomputed_s(rng):
    W = E.wishart_factor(1500, 750, sigma=1.0, rng=rng)
    s = np.linalg.svd(W, compute_uv=False)
    a = mp.estimate_sigma_gd_median(W)
    b = mp.estimate_sigma_gd_median(s=s, n=1500, m=750)
    assert abs(a - b) < 1e-9


def test_estimate_sigma_alias():
    assert mp.estimate_sigma_med is mp.estimate_sigma_gd_median


def test_wishart_esd_matches_mp_density(rng):
    n, m, sigma = 1000, 3000, 1.0
    W = E.wishart_factor(n, m, sigma=sigma, rng=rng)
    s = np.linalg.svd(W, compute_uv=False)
    lo, hi = mp.mp_bounds(n, m, sigma)
    bulk = s[(s >= lo) & (s <= hi)]
    f_bulk = bulk.size / s.size
    # design.md §3.6: histogram the FULL spectrum (density=True), then overlay
    # the MP curve MULTIPLIED by the bulk-mass fraction so areas match.
    nb = _fd_bins_like(bulk, lo, hi)
    hist, edges = np.histogram(s, bins=nb, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    theo = mp.mp_pdf(centers, n, m, sigma) * f_bulk
    l1 = np.mean(np.abs(hist - theo) * np.diff(edges))
    assert l1 <= 0.06, l1


def _fd_bins_like(x, lo, hi):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 2 * iqr * x.size ** (-1 / 3)
    if h <= 0:
        return int(np.ceil(np.sqrt(x.size)))
    return int(np.clip((hi - lo) / h, 50, 200))


def test_eigenvalue_api_consistency():
    n, m, sigma = 800, 1600, 1.2
    W = E.wishart_factor(n, m, sigma=sigma)
    s = np.linalg.svd(W, compute_uv=False)
    lam = mp.eigenvalues_of_cov(s=s, n=n, m=m, N=m)
    assert np.allclose(lam, s**2 / m)
    be = mp.mp_bounds_eig(n, m, sigma, m)
    bn = mp.mp_bounds(n, m, sigma)
    assert np.allclose(be, tuple(b**2 / m for b in bn))
    lo, hi = be
    I = _gauss_legendre_integral(lambda x: mp.mp_pdf_eig(x, n, m, sigma, m), lo, hi)
    assert abs(I - 1.0) < 1e-3, I


def test_estimate_sigma_robust_to_outliers(rng):
    n, m, sigma = 2000, 500, 1.0
    W = E.wishart_factor(n, m, sigma=sigma, rng=rng)
    s = np.sort(np.linalg.svd(W, compute_uv=False))[::-1]
    s[:5] *= 10.0                            # plant spikes
    sig = mp.estimate_sigma_gd_median(s=s, n=n, m=m)
    assert abs(sig - sigma) / sigma < TOL["sigma_rel_outliers"]


def test_refined_sigma_noop_on_pure_wishart(rng):
    n, m, sigma = 1500, 750, 1.0
    W = E.wishart_factor(n, m, sigma=sigma, rng=rng)
    s0, sr, nit = mp.estimate_sigma_med_refined(W)
    assert abs(sr - s0) / s0 < 0.03
    assert nit <= 3


def test_refined_sigma_removes_spikes(rng):
    n, m, sigma = 1500, 750, 1.0
    W = E.wishart_factor(n, m, sigma=sigma, rng=rng)
    s = np.sort(np.linalg.svd(W, compute_uv=False))[::-1]
    s[:10] *= 8.0
    s0, sr, _ = mp.estimate_sigma_med_refined(s=s, n=n, m=m)
    # the median estimator is already spike-robust; refinement must still
    # recover the clean sigma to within the outlier tolerance and not blow up.
    assert abs(sr - sigma) / sigma < TOL["sigma_rel_outliers"]


def test_usvt_hard_threshold_scaling():
    t1 = mp.usvt_hard_threshold(1000, 1000, 1.0)
    t2 = mp.usvt_hard_threshold(1000, 1000, 2.0)
    assert abs(t2 / t1 - 2.0) < 1e-6                 # ∝ sigma
    # square optimal ≈ (4/√3) σ √n
    assert abs(t1 - (4 / np.sqrt(3)) * np.sqrt(1000)) / t1 < 0.05
    # ∝ √max
    ta = mp.usvt_hard_threshold(500, 2000, 1.0)
    tb = mp.usvt_hard_threshold(500, 8000, 1.0)
    assert abs(tb / ta - np.sqrt(8000 / 2000)) < 1e-6


def test_small_sv_deviation_keys_and_signs(rng):
    n, m, sigma = 2000, 500, 1.0
    W = E.wishart_factor(n, m, sigma=sigma, rng=rng)
    s = np.sort(np.linalg.svd(W, compute_uv=False))
    d = mp.small_sv_deviation(s, n, m, sigma)
    assert set(d) == {"ks_lower", "n_below_minus", "frac_mass_below_minus", "excess_small_sv"}
    assert abs(d["excess_small_sv"]) < 0.05 * len(s)
    assert d["frac_mass_below_minus"] < 0.05
    # push a block of mid-spectrum SVs down into the small-SV region
    s2 = s.copy()
    mid = len(s2) // 2
    s2[mid:mid + 40] = s2[:1] * 0.5          # well below nu_minus
    s2 = np.sort(s2)
    d2 = mp.small_sv_deviation(s2, n, m, sigma)
    assert d2["excess_small_sv"] > d["excess_small_sv"]
    assert d2["n_below_minus"] > d["n_below_minus"]
