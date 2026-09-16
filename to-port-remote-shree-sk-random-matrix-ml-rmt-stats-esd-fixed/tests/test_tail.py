import numpy as np
import pytest

from rmt import tail
from rmt import ensembles as E
from rmt.config import TOL


def test_csn_recovers_pareto_alpha(rng):
    x = E.pareto(20000, 3.0, rng=rng)
    res = tail.fit_powerlaw_csn(x)
    lo, hi = TOL["csn_alpha"]
    assert lo <= res["alpha"] <= hi, res
    assert res["ks_D"] < 0.05


def test_csn_xmin_reasonable(rng):
    x = E.pareto(20000, 3.0, rng=rng)
    res = tail.fit_powerlaw_csn(x)
    assert res["xmin"] >= 1.0
    assert res["n_tail"] >= 50


def test_csn_requires_min_tail():
    x = np.array([1.0, 2.0, 3.0])
    res = tail.fit_powerlaw_csn(x, min_tail=50)
    assert np.isnan(res["alpha"])
    assert res["n_tail"] == 0


def test_hill_on_pareto_is_survival_exponent(rng):
    x = E.pareto(40000, 3.0, rng=rng)
    a = tail.hill_alpha_at(x, k=1000)
    lo, tol = TOL["hill_alpha"]
    assert abs(a - lo) < tol, a


def test_hill_windowed_recovers_and_plateaus(rng):
    x = E.pareto(40000, 3.0, rng=rng)
    ks, aloc = tail.hill_estimator_windowed(x, window=20)
    assert ks.size == aloc.size
    pl = tail.hill_plateau(x, window=20)
    assert pl["hill_is_powerlaw"] is True
    lo, tol = TOL["hill_windowed"]
    assert abs(pl["hill_plateau_alpha"] - lo) < tol, pl


def test_hill_windowed_rejects_mp_bulk(rng):
    W = E.wishart_factor(1500, 1500, sigma=1.0, rng=rng)
    s = np.linalg.svd(W, compute_uv=False)
    pl = tail.hill_plateau(s, window=20)
    assert pl["hill_is_powerlaw"] is False


def test_csn_minus_hill_is_one(rng):
    x = E.pareto(40000, 3.0, rng=rng)
    csn = tail.fit_powerlaw_csn(x)["alpha"]
    hill = tail.hill_alpha_at(x, k=1000)
    assert abs((csn - hill) - 1.0) < 0.3, (csn, hill)


def test_hill_lambda_is_half_of_nu(rng):
    x = E.pareto(40000, 3.0, rng=rng)
    a_nu = tail.hill_alpha_at(x, k=1000)
    a_lam = tail.hill_alpha_at(x**2, k=1000)
    assert abs(a_lam / a_nu - 0.5) < 0.1, (a_nu, a_lam)


def test_mp_bulk_is_not_power_law(rng):
    W = E.wishart_factor(1500, 1500, sigma=1.0, rng=rng)
    s = np.linalg.svd(W, compute_uv=False)
    res = tail.fit_powerlaw_csn(s**2)
    assert res["alpha"] > 5.0
    assert res["ks_D"] > 0.05


def test_hill_monotone_arrays(rng):
    x = E.pareto(2000, 3.0, rng=rng)
    ks, inv = tail.hill_estimator(x)
    assert ks.size == inv.size
    assert ks.max() <= x.size // 2
    ksw, alw = tail.hill_estimator_windowed(x)
    assert ksw.size == alw.size


def test_alpha_estimator_switch(rng):
    x = E.pareto(20000, 3.0, rng=rng)
    rc = tail.select_alpha(x, estimator="csn")
    assert rc["source"] == "csn" and 2.8 <= rc["alpha"] <= 3.2
    rw = tail.select_alpha(x, estimator="hill_windowed")
    assert rw["source"] == "hill_windowed" and 1.7 <= rw["alpha"] <= 2.3
    ra = tail.select_alpha(x, estimator="all")
    for key in ("csn_alpha", "hill_alpha", "hill_windowed_alpha", "hill_is_powerlaw"):
        assert key in ra
