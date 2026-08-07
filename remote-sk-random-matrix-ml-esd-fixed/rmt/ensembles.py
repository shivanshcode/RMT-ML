"""rmt.ensembles — synthetic RMT matrix/level generators with known laws.

Pure numpy. Used by the test-suite and selftest as analytic ground truth.
All generators accept ``rng: np.random.Generator | int | None``.
"""
from __future__ import annotations

import numpy as np


def _rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def goe(n, rng=None) -> np.ndarray:
    """Gaussian Orthogonal Ensemble: real symmetric n×n.

    Diagonal ~ N(0, 2/n), off-diagonal ~ N(0, 1/n).  (Semicircle on [-2, 2].)
    """
    g = _rng(rng)
    A = g.standard_normal((n, n)) / np.sqrt(n)
    M = (A + A.T) / np.sqrt(2.0)
    # Above gives diag var 2/n, off-diag var 1/n.
    return M


def gue(n, rng=None) -> np.ndarray:
    """Gaussian Unitary Ensemble: complex Hermitian n×n."""
    g = _rng(rng)
    A = (g.standard_normal((n, n)) + 1j * g.standard_normal((n, n))) / np.sqrt(2.0 * n)
    M = (A + A.conj().T) / np.sqrt(2.0)
    return M


def ginue(n, rng=None) -> np.ndarray:
    """Ginibre Unitary Ensemble: complex iid n×n, entries CN(0, 1/n)."""
    g = _rng(rng)
    return (g.standard_normal((n, n)) + 1j * g.standard_normal((n, n))) / np.sqrt(2.0 * n)


def wishart_factor(n, m, sigma=1.0, rng=None) -> np.ndarray:
    """Real iid N(0, sigma²) factor of shape (n, m) — the factor W, not WWᵀ."""
    g = _rng(rng)
    return sigma * g.standard_normal((n, m))


def poisson_levels(n, rng=None) -> np.ndarray:
    """n iid U(0, 1) sorted ascending — an uncorrelated (Poisson) spectrum."""
    g = _rng(rng)
    return np.sort(g.uniform(0.0, 1.0, size=n))


def poisson_points_2d(n, rng=None) -> np.ndarray:
    """n iid complex points uniform in the unit square [0,1]²."""
    g = _rng(rng)
    return g.uniform(0, 1, size=n) + 1j * g.uniform(0, 1, size=n)


def pareto(n, alpha, xmin=1.0, rng=None) -> np.ndarray:
    """Samples with density ∝ x^{-alpha}, x ≥ xmin.  Survival exponent = alpha-1.

    Inverse-CDF: x = xmin * (1 - u)^{-1/(alpha-1)}.
    """
    g = _rng(rng)
    u = g.uniform(0.0, 1.0, size=n)
    return xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))
