"""Exact finite-L spectral-rigidity laws for beta = 1, 2, 4 and Poisson.

Why this module exists
----------------------
``spacing.delta3_goe_theory`` / ``sigma2_goe_theory`` are the **L -> infinity**
Dyson-Mehta asymptotes.  Review item F1 established that comparing a measured
Delta_3(L) against them manufactures a deviation at small L.  This module
supplies the *exact* finite-L laws instead, so the reference is right and the
z-score machinery is only correcting for finite-sample scatter.

Everything follows from the two-level cluster function Y_2(r) (Mehta, *Random
Matrices*, 3rd ed., Ch. 6, 7, 11 and 16):

    Sigma^2(L) = L - 2 * int_0^L (L - r) Y_2(r) dr                    (16.1.4)
    Delta_3(L) = (2/L^4) int_0^L (L^3 - 2 L^2 r + r^3) Sigma^2(r) dr  (16.1.7)

with, writing s(r) = sin(pi r)/(pi r),

    beta = 1 (GOE):  Y_2(r) = s(r)^2 + s'(r) * int_r^inf s(t) dt
    beta = 2 (GUE):  Y_2(r) = s(r)^2
    beta = 4 (GSE):  Y_2(r) = s(2r)^2 - s'(2r) * int_0^{2r} s(t) dt
    Poisson:         Y_2(r) = 0   =>   Sigma^2 = L, Delta_3 = L/15

Validation (``tests/test_reference.py``): the exact curves reproduce measured
circular-ensemble Delta_3 -- where the unfolding is exact by construction -- to
better than 1 % at every L, where the asymptote is off by +27 % at L = 3.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Union

import numpy as np
from scipy.integrate import quad
from scipy.special import sici

__all__ = [
    "sigma2_exact", "delta3_exact", "sigma2_asymptote", "delta3_asymptote",
    "two_level_cluster", "r_statistic_reference", "BETA_OF",
]

EULER_GAMMA = 0.5772156649015329

#: Map a human-readable ensemble label onto its Dyson index.
BETA_OF = {"goe": 1, "gue": 2, "gse": 4, "coe": 1, "cue": 2, "cse": 4,
           "orthogonal": 1, "unitary": 2, "symplectic": 4}

#: Large-N <r~> = <min(s_n,s_n+1)/max(s_n,s_n+1)>.  Atas, Bogomolny, Giraud &
#: Roux, PRL 110, 084101 (2013); these are the numerical large-N values, not
#: the 3x3 surmise (which gives 0.5359 for beta = 1).
R_STATISTIC = {0: 2.0 * np.log(2.0) - 1.0, 1: 0.5307, 2: 0.5996, 4: 0.6744}


def r_statistic_reference(beta: int) -> float:
    """<r~> for the given Dyson index (beta = 0 means Poisson)."""
    return float(R_STATISTIC[int(beta)])


# --------------------------------------------------------------------------- #
# Two-level cluster function                                                   #
# --------------------------------------------------------------------------- #
def _s(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    out = np.ones_like(r)
    nz = r != 0.0
    out[nz] = np.sin(np.pi * r[nz]) / (np.pi * r[nz])
    return out


def _dsdr(r: np.ndarray) -> np.ndarray:
    """d/dr of sin(pi r)/(pi r)."""
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r)
    nz = r != 0.0
    x = np.pi * r[nz]
    out[nz] = np.pi * (x * np.cos(x) - np.sin(x)) / x ** 2
    return out


def _si_over_pi(x: np.ndarray) -> np.ndarray:
    """Si(pi x) / pi = int_0^x s(t) dt."""
    si, _ = sici(np.pi * np.asarray(x, dtype=np.float64))
    return si / np.pi


def two_level_cluster(r, beta: int) -> np.ndarray:
    """Y_2(r) for beta in {0, 1, 2, 4}.  Y_2(0) = 1 for every non-Poisson beta."""
    r = np.asarray(r, dtype=np.float64)
    beta = int(beta)
    if beta == 0:
        return np.zeros_like(r)
    if beta == 2:
        return _s(r) ** 2
    if beta == 1:
        # int_r^inf s = 1/2 - Si(pi r)/pi
        return _s(r) ** 2 + _dsdr(r) * (0.5 - _si_over_pi(r))
    if beta == 4:
        x = 2.0 * r
        return _s(x) ** 2 - _dsdr(x) * _si_over_pi(x)
    raise ValueError(f"beta must be 0, 1, 2 or 4; got {beta}")


# --------------------------------------------------------------------------- #
# Exact Sigma^2 and Delta_3                                                    #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4096)
def sigma2_exact(L: float, beta: int = 1) -> float:
    """Exact number variance Sigma^2(L).  Poisson (beta = 0) returns L."""
    L = float(L)
    beta = int(beta)
    if L <= 0.0:
        return 0.0
    if beta == 0:
        return L
    # Y_2 oscillates with unit period; split the range so quad resolves it.
    n_break = max(2, min(400, int(np.ceil(L)) + 1))
    pts = np.linspace(0.0, L, n_break + 1)
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        val, _ = quad(lambda r: (L - r) * float(two_level_cluster(np.array([r]), beta)[0]),
                      a, b, limit=200, epsabs=1e-11, epsrel=1e-11)
        total += val
    return float(L - 2.0 * total)


@lru_cache(maxsize=4096)
def delta3_exact(L: float, beta: int = 1) -> float:
    """Exact Dyson-Mehta spectral rigidity Delta_3(L).

    Poisson (beta = 0) returns the exact L/15.  Otherwise the Sigma^2 kernel
    integral is evaluated on a Gauss-Legendre grid; Sigma^2 itself is smooth in
    r, so 96 nodes is converged to ~1e-9 and far cheaper than nested quad.
    """
    L = float(L)
    beta = int(beta)
    if L <= 0.0:
        return 0.0
    if beta == 0:
        return L / 15.0
    nodes, weights = np.polynomial.legendre.leggauss(96)
    r = 0.5 * L * (nodes + 1.0)
    w = 0.5 * L * weights
    kern = L ** 3 - 2.0 * L ** 2 * r + r ** 3
    s2 = np.array([sigma2_exact(float(ri), beta) for ri in r])
    return float(2.0 * np.sum(w * kern * s2) / L ** 4)


# --------------------------------------------------------------------------- #
# The asymptotes, kept for plots and for regression against the old behaviour  #
# --------------------------------------------------------------------------- #
def sigma2_asymptote(L: float, beta: int = 1) -> float:
    """L -> infinity Sigma^2.  Accurate to <0.2 % even at L = 3 for all beta."""
    L, g = float(L), EULER_GAMMA
    if beta == 0:
        return L
    if beta == 1:
        return (2.0 / np.pi ** 2) * (np.log(2 * np.pi * L) + g + 1.0 - np.pi ** 2 / 8.0)
    if beta == 2:
        return (1.0 / np.pi ** 2) * (np.log(2 * np.pi * L) + g + 1.0)
    if beta == 4:
        return (1.0 / (2 * np.pi ** 2)) * (np.log(4 * np.pi * L) + g + 1.0 + np.pi ** 2 / 8.0)
    raise ValueError(beta)


def delta3_asymptote(L: float, beta: int = 1) -> float:
    """L -> infinity Delta_3.  **Off by +27 % at L = 3 for beta = 1** -- this is
    review defect F1.  Use :func:`delta3_exact` for anything below L ~ 30."""
    L, g = float(L), EULER_GAMMA
    if beta == 0:
        return L / 15.0
    if beta == 1:
        return (1.0 / np.pi ** 2) * (np.log(2 * np.pi * L) + g - 1.25 - np.pi ** 2 / 8.0)
    if beta == 2:
        return (1.0 / (2 * np.pi ** 2)) * (np.log(2 * np.pi * L) + g - 1.25)
    if beta == 4:
        return (1.0 / (4 * np.pi ** 2)) * (np.log(4 * np.pi * L) + g - 1.25 + np.pi ** 2 / 8.0)
    raise ValueError(beta)
