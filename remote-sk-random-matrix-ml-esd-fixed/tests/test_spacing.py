import numpy as np
import pytest

from rmt import spacing as SP
from rmt import ensembles as E
from rmt.config import TOL


def _goe_eigs(n, rng):
    return np.linalg.eigvalsh(E.goe(n, rng))


def test_r_statistic_goe(rng):
    ev = _goe_eigs(800, rng)
    r = SP.r_statistic(ev)
    mid, tol = TOL["r_goe"]
    assert abs(r - mid) < tol, r


def test_r_statistic_poisson(rng):
    lv = E.poisson_levels(4000, rng)
    r = SP.r_statistic(lv)
    mid, tol = TOL["r_poisson"]
    assert abs(r - mid) < tol, r


def test_r_statistic_ordering(rng):
    rg = SP.r_statistic(_goe_eigs(800, rng))
    rp = SP.r_statistic(E.poisson_levels(4000, rng))
    assert rg > rp


def test_unfold_mean_spacing_unit(rng):
    ev = _goe_eigs(800, rng)
    s = SP.nn_spacing(ev)
    assert abs(np.mean(s) - 1.0) < 0.05


def test_nn_spacing_goe_closer_to_wigner(rng):
    ev = _goe_eigs(800, rng)
    d = SP.nn_spacing_ks(ev)
    assert d["nn_KS_GOE"] < d["nn_KS_Poisson"]


def test_nn_spacing_poisson_closer_to_poisson(rng):
    lv = E.poisson_levels(4000, rng)
    d = SP.nn_spacing_ks(lv)
    assert d["nn_KS_Poisson"] < d["nn_KS_GOE"]


def test_wigner_poisson_cdf_shapes():
    s = np.linspace(0, 6, 100)
    for cdf in (SP.wigner_goe_cdf, SP.poisson_cdf):
        F = cdf(s)
        assert np.all(np.diff(F) >= -1e-12)
        assert abs(F[0]) < 1e-9
        assert F[-1] > 0.99


def test_delta3_sigma2_goe_log_law(rng):
    s2 = {10: [], 50: []}
    d3 = {10: [], 50: []}
    for _ in range(4):
        ev = _goe_eigs(1000, rng)
        for L in (10, 50):
            s2[L].append(SP.sigma2(ev, L))
            d3[L].append(SP.delta3(ev, L))
    for L in (10, 50):
        s2m = np.mean(s2[L]); d3m = np.mean(d3[L])
        s2t = SP.sigma2_goe_theory(L); d3t = SP.delta3_goe_theory(L)
        assert abs(s2m - s2t) / s2t < TOL["sigma2_rtol"], (L, s2m, s2t)
        assert abs(d3m - d3t) / d3t < TOL["delta3_rtol"], (L, d3m, d3t)


def test_delta3_sigma2_goe_below_poisson(rng):
    ev = _goe_eigs(1000, rng)
    lv = E.poisson_levels(4000, rng)
    assert SP.sigma2(ev, 20) < SP.sigma2(lv, 20)
    assert SP.delta3(ev, 20) < SP.delta3(lv, 20)


def test_complex_spacing_ginue(rng):
    z = SP.complex_spacing_ratio(E.ginue(400, rng))
    mid, tol = TOL["ginue_abs"]
    assert abs(z["abs_mean"] - mid) < tol, z
    lo, hi = TOL["ginue_cos"]
    assert lo <= z["cos_mean"] <= hi, z


def test_complex_spacing_poisson2d(rng):
    pts = E.poisson_points_2d(2000, rng)
    z = SP.complex_spacing_ratio(np.diag(pts))
    mid, tol = TOL["poi2d_abs"]
    assert abs(z["abs_mean"] - mid) < tol, z
    assert abs(z["cos_mean"]) <= TOL["poi2d_cos_abs"], z


def test_complex_spacing_ordering(rng):
    g = SP.complex_spacing_ratio(E.ginue(400, rng))
    p = SP.complex_spacing_ratio(np.diag(E.poisson_points_2d(2000, rng)))
    assert g["abs_mean"] > p["abs_mean"]
    assert g["cos_mean"] < p["cos_mean"]
