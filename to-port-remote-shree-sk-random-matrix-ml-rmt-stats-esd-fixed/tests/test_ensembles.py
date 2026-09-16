import numpy as np
import pytest

from rmt import ensembles as E


def test_goe_symmetric(rng):
    M = E.goe(50, rng)
    assert M.shape == (50, 50)
    assert np.allclose(M, M.T)


def test_gue_hermitian(rng):
    M = E.gue(40, rng)
    assert M.shape == (40, 40)
    assert np.allclose(M, M.conj().T)


def test_ginue_shape_complex(rng):
    M = E.ginue(30, rng)
    assert M.shape == (30, 30)
    assert np.iscomplexobj(M)


def test_wishart_factor_shape_and_dtype(rng):
    W = E.wishart_factor(100, 60, sigma=2.0, rng=rng)
    assert W.shape == (100, 60)
    assert W.dtype == np.float64


def test_wishart_entry_variance(rng):
    sigma = 1.7
    W = E.wishart_factor(400, 400, sigma=sigma, rng=rng)
    assert abs(np.var(W) - sigma**2) / sigma**2 < 0.05


def test_pareto_tail_exponent(rng):
    alpha = 3.0
    x = E.pareto(200000, alpha, xmin=1.0, rng=rng)
    xs = np.sort(x)
    surv = 1.0 - (np.arange(1, xs.size + 1) - 0.5) / xs.size
    # log-log survival slope ~ -(alpha-1) in the tail
    hi = xs > np.quantile(xs, 0.5)
    slope = np.polyfit(np.log(xs[hi]), np.log(surv[hi]), 1)[0]
    assert abs(slope - (-(alpha - 1.0))) < 0.2


def test_poisson_levels_uniform(rng):
    from scipy import stats
    x = E.poisson_levels(5000, rng)
    assert np.all(np.diff(x) >= 0)
    D, _ = stats.kstest(x, "uniform")
    assert D < 0.05


def test_reproducible_with_seed():
    a = E.wishart_factor(20, 20, rng=7)
    b = E.wishart_factor(20, 20, rng=7)
    assert np.allclose(a, b)
