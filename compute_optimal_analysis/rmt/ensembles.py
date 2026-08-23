"""Synthetic random-matrix ensembles with explicit finite-size scaling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np


RandomState: TypeAlias = np.random.Generator | int | None


def _rng(rng: RandomState) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _positive_int(value: int, name: str) -> int:
    result = int(value)
    if result < 1 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def goe(n: int, rng: RandomState = None) -> np.ndarray:
    """Return an ``n x n`` GOE matrix with semicircle support near ``[-2, 2]``."""

    n = _positive_int(n, "n")
    generator = _rng(rng)
    upper = generator.normal(0.0, 1.0 / np.sqrt(n), size=(n, n))
    matrix = np.triu(upper, 1)
    matrix = matrix + matrix.T
    diagonal = generator.normal(0.0, np.sqrt(2.0 / n), size=n)
    matrix[np.diag_indices(n)] = diagonal
    return matrix


def gue(n: int, rng: RandomState = None) -> np.ndarray:
    """Return a complex Hermitian GUE matrix with conventional scaling."""

    n = _positive_int(n, "n")
    generator = _rng(rng)
    real = generator.normal(size=(n, n))
    imag = generator.normal(size=(n, n))
    raw = (real + 1j * imag) / np.sqrt(2.0 * n)
    matrix = (raw + raw.conj().T) / np.sqrt(2.0)
    matrix[np.diag_indices(n)] = generator.normal(0.0, 1.0 / np.sqrt(n), size=n)
    return matrix


def ginue(n: int, rng: RandomState = None) -> np.ndarray:
    """Return a complex Ginibre matrix with entry variance ``1/n``."""

    n = _positive_int(n, "n")
    generator = _rng(rng)
    real = generator.normal(size=(n, n))
    imag = generator.normal(size=(n, n))
    return (real + 1j * imag) / np.sqrt(2.0 * n)


def wishart_factor(
    n: int,
    m: int,
    sigma: float = 1.0,
    rng: RandomState = None,
) -> np.ndarray:
    """Return an i.i.d. Gaussian factor with entry variance ``sigma**2``."""

    n = _positive_int(n, "n")
    m = _positive_int(m, "m")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    return _rng(rng).normal(0.0, sigma, size=(n, m)).astype(np.float64)


def wishart(
    n: int,
    m: int,
    sigma: float = 1.0,
    rng: RandomState = None,
    *,
    covariance: bool = False,
) -> np.ndarray:
    """Return a Gaussian factor or its reduced, canonically normalized covariance."""

    factor = wishart_factor(n, m, sigma=sigma, rng=rng)
    if not covariance:
        return factor
    if n <= m:
        return factor @ factor.T / m
    return factor.T @ factor / n


def spiked_covariance(
    n: int,
    m: int,
    spikes: np.ndarray | Sequence[float],
    *,
    noise_variance: float = 1.0,
    rng: RandomState = None,
    return_factor: bool = False,
) -> np.ndarray:
    """Return a finite-rank spiked Wishart covariance or its data factor.

    ``spikes`` contains absolute population covariance eigenvalues; all
    remaining population directions have eigenvalue ``noise_variance``.
    """

    n = _positive_int(n, "n")
    m = _positive_int(m, "m")
    variance = float(noise_variance)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("noise_variance must be finite and positive")
    population_spikes = np.asarray(spikes, dtype=np.float64).ravel()
    dimension, samples = min(n, m), max(n, m)
    if population_spikes.size > dimension:
        raise ValueError("the number of spikes cannot exceed the reduced dimension")
    if not np.all(np.isfinite(population_spikes)) or np.any(population_spikes <= 0.0):
        raise ValueError("spikes must contain finite positive population eigenvalues")
    population = np.full(dimension, variance, dtype=np.float64)
    population[: population_spikes.size] = population_spikes
    oriented = _rng(rng).normal(size=(dimension, samples))
    oriented *= np.sqrt(population)[:, None]
    factor = oriented if n <= m else oriented.T
    if return_factor:
        return factor
    return oriented @ oriented.T / samples


spiked_wishart = spiked_covariance


def poisson_levels(n: int, rng: RandomState = None) -> np.ndarray:
    """Return sorted independent uniform levels, a finite Poisson-spectrum proxy."""

    n = _positive_int(n, "n")
    return np.sort(_rng(rng).uniform(0.0, 1.0, size=n))


def poisson_points_2d(n: int, rng: RandomState = None) -> np.ndarray:
    """Return independent complex points in the unit square."""

    n = _positive_int(n, "n")
    generator = _rng(rng)
    return generator.uniform(size=n) + 1j * generator.uniform(size=n)


def pareto(
    n: int,
    alpha: float,
    xmin: float = 1.0,
    rng: RandomState = None,
) -> np.ndarray:
    """Sample a continuous Pareto density proportional to ``x**(-alpha)``."""

    n = _positive_int(n, "n")
    alpha = float(alpha)
    xmin = float(xmin)
    if not np.isfinite(alpha) or alpha <= 1.0:
        raise ValueError("alpha must exceed one")
    if not np.isfinite(xmin) or xmin <= 0.0:
        raise ValueError("xmin must be finite and positive")
    uniform = _rng(rng).uniform(np.finfo(float).eps, 1.0, size=n)
    return xmin * np.power(1.0 - uniform, -1.0 / (alpha - 1.0))


def wigner_semicircle_samples(n: int, rng: RandomState = None) -> np.ndarray:
    """Return GOE eigenvalues for calibration against the semicircle law."""

    return np.linalg.eigvalsh(goe(n, rng=rng))


__all__ = [
    "ginue",
    "goe",
    "gue",
    "pareto",
    "poisson_levels",
    "poisson_points_2d",
    "spiked_covariance",
    "spiked_wishart",
    "wigner_semicircle_samples",
    "wishart",
    "wishart_factor",
]
