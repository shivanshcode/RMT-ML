"""Spectral unfolding, level statistics, Brody fitting, and rigidity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.optimize import minimize_scalar
from scipy.special import gamma, ndtr
from scipy.stats import kstest


@dataclass(frozen=True)
class BrodyFit:
    beta: float
    negative_log_likelihood: float
    standard_error: float
    success: bool
    n_spacings: int
    method: str = "mle"
    diagnostics: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_levels(levels: np.ndarray, minimum: int = 3) -> np.ndarray:
    array = np.asarray(levels, dtype=np.float64).ravel()
    array = np.unique(array[np.isfinite(array)])
    if array.size < minimum:
        raise ValueError(f"at least {minimum} distinct finite levels are required")
    return np.sort(array)


def unfold_spectrum(
    levels: np.ndarray,
    *,
    method: str = "polynomial",
    degree: int = 7,
    smoothing: float | None = None,
    kernel_window: int = 15,
    edge_mode: str = "replicate",
) -> np.ndarray:
    """Map ordered levels through a smooth cumulative staircase estimate."""

    ordered = _clean_levels(levels, minimum=4)
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    name = str(method).lower()
    aliases = {
        "polynomial": "polynomial",
        "polynomial_chebyshev": "polynomial_chebyshev",
        "chebyshev": "polynomial_chebyshev",
        "spline": "spline_monotone",
        "spline_monotone": "spline_monotone",
        "gaussian_kernel": "gaussian_kernel",
        "gaussian": "gaussian_kernel",
        "raw_rank_order": "raw_rank_order",
    }
    if name not in aliases:
        raise ValueError(
            "method must be polynomial, polynomial_chebyshev, spline_monotone, "
            "gaussian_kernel, or raw_rank_order"
        )
    name = aliases[name]
    if name == "polynomial":
        fitted_degree = min(max(int(degree), 1), ordered.size - 2)
        polynomial = np.polynomial.Polynomial.fit(ordered, ranks, deg=fitted_degree)
        smooth = np.asarray(polynomial(ordered), dtype=np.float64)
    elif name == "polynomial_chebyshev":
        fitted_degree = min(max(int(degree), 1), ordered.size - 2)
        polynomial = np.polynomial.Chebyshev.fit(ordered, ranks, deg=fitted_degree)
        smooth = np.asarray(polynomial(ordered), dtype=np.float64)
    elif name == "spline_monotone":
        smoothness = float(ordered.size if smoothing is None else smoothing)
        if smoothness < 0.0 or not np.isfinite(smoothness):
            raise ValueError("smoothing must be finite and nonnegative")
        spline = UnivariateSpline(
            ordered,
            ranks,
            k=min(3, ordered.size - 1),
            s=smoothness,
        )
        preliminary = np.maximum.accumulate(np.asarray(spline(ordered), dtype=np.float64))
        epsilon = np.finfo(float).eps * max(1.0, float(np.ptp(preliminary)))
        preliminary += epsilon * np.arange(preliminary.size)
        smooth = np.asarray(PchipInterpolator(ordered, preliminary)(ordered), dtype=np.float64)
    elif name == "gaussian_kernel":
        window = int(kernel_window)
        if window < 1 or 2 * window >= ordered.size:
            raise ValueError("kernel_window must be positive and less than half the level count")
        edge = str(edge_mode).lower()
        if edge not in {"replicate", "drop"}:
            raise ValueError("edge_mode must be replicate or drop")
        source = np.pad(ordered, (window, window), mode="edge") if edge == "replicate" else ordered
        means = source[window:-window]
        widths = 0.5 * (source[2 * window :] - source[: -2 * window])
        positive_widths = widths[widths > 0.0]
        fallback = (
            float(np.median(positive_widths))
            if positive_widths.size
            else float(np.ptp(ordered) / max(ordered.size - 1, 1))
        )
        minimum_width = max(
            np.finfo(float).eps * max(1.0, float(np.max(np.abs(ordered)))),
            fallback * 1e-6,
        )
        widths = np.maximum(widths, minimum_width)
        smooth = np.sum(ndtr((ordered[:, None] - means[None, :]) / widths[None, :]), axis=1)
    else:
        smooth = ranks
    if not np.all(np.isfinite(smooth)):
        raise ValueError("unfolding produced non-finite values")
    monotone = np.maximum.accumulate(smooth)
    scale_epsilon = np.finfo(float).eps * max(1.0, float(np.ptp(monotone)))
    monotone = monotone + scale_epsilon * np.arange(monotone.size)
    mean_spacing = float(np.mean(np.diff(monotone)))
    if mean_spacing <= 0.0 or not np.isfinite(mean_spacing):
        raise ValueError("unfolding produced a degenerate staircase")
    return (monotone - monotone[0]) / mean_spacing


def unfold(levels: np.ndarray, deg: int = 7) -> np.ndarray:
    return unfold_spectrum(levels, method="polynomial", degree=deg)


def nearest_neighbor_spacings(
    levels: np.ndarray,
    *,
    unfolded: bool = False,
    method: str = "polynomial",
    degree: int = 7,
    smoothing: float | None = None,
    kernel_window: int = 15,
    edge_mode: str = "replicate",
) -> np.ndarray:
    """Return positive nearest-neighbor spacings normalized to mean one."""

    ordered = _clean_levels(levels, minimum=3)
    transformed = ordered if unfolded else unfold_spectrum(
        ordered,
        method=method,
        degree=degree,
        smoothing=smoothing,
        kernel_window=kernel_window,
        edge_mode=edge_mode,
    )
    spacings = np.diff(transformed)
    spacings = spacings[np.isfinite(spacings) & (spacings > 0.0)]
    if spacings.size == 0:
        raise ValueError("no positive spacings remain")
    return spacings / np.mean(spacings)


def nn_spacing(levels: np.ndarray, deg: int = 7) -> np.ndarray:
    return nearest_neighbor_spacings(levels, degree=deg)


def _brody_b(beta: float) -> float:
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0, 1]")
    return float(gamma((beta + 2.0) / (beta + 1.0)) ** (beta + 1.0))


def brody_pdf(s: np.ndarray | float, beta: float) -> np.ndarray:
    """Evaluate the mean-one Brody interpolation density."""

    values = np.asarray(s, dtype=np.float64)
    b = _brody_b(beta)
    density = np.zeros_like(values)
    nonnegative = values >= 0.0
    selected = values[nonnegative]
    density[nonnegative] = (beta + 1.0) * b * np.power(selected, beta) * np.exp(
        -b * np.power(selected, beta + 1.0)
    )
    return density


def brody_cdf(s: np.ndarray | float, beta: float) -> np.ndarray:
    values = np.asarray(s, dtype=np.float64)
    b = _brody_b(beta)
    positive = np.maximum(values, 0.0)
    return np.where(values <= 0.0, 0.0, 1.0 - np.exp(-b * np.power(positive, beta + 1.0)))


def fit_brody(
    spacings: np.ndarray,
    *,
    method: str = "mle",
    n_bootstrap: int = 0,
    rng: np.random.Generator | int | None = 0,
) -> BrodyFit:
    """Fit ``beta in [0,1]`` by MLE or empirical-CDF least squares."""

    values = np.asarray(spacings, dtype=np.float64).ravel()
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 8:
        raise ValueError("at least eight positive spacings are required")
    values = values / np.mean(values)
    logs = np.log(values)

    def objective(beta: float) -> float:
        b = _brody_b(beta)
        log_density = (
            np.log(beta + 1.0)
            + np.log(b)
            + beta * logs
            - b * np.power(values, beta + 1.0)
        )
        return float(-np.sum(log_density))

    fit_method = str(method).lower()
    if fit_method in {"mle", "maximum_likelihood"}:
        optimized_objective = objective
        canonical_method = "mle"
    elif fit_method in {"cdf_nls", "nonlinear_least_squares", "reference_cdf"}:
        ordered = np.sort(values)
        empirical_cdf = np.arange(ordered.size, dtype=np.float64) / ordered.size

        def cdf_objective(beta: float) -> float:
            return float(np.sum(np.square(brody_cdf(ordered, beta) - empirical_cdf)))

        optimized_objective = cdf_objective
        canonical_method = "cdf_nls"
    else:
        raise ValueError("method must be mle or cdf_nls")
    optimum = minimize_scalar(
        optimized_objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-7, "maxiter": 500},
    )
    beta = float(np.clip(optimum.x, 0.0, 1.0))
    step = 1e-4
    left, right = max(0.0, beta - step), min(1.0, beta + step)
    if left < beta < right:
        curvature = (
            optimized_objective(right)
            - 2.0 * optimized_objective(beta)
            + optimized_objective(left)
        ) / ((right - beta) ** 2)
        standard_error = float(np.sqrt(1.0 / curvature)) if curvature > 0.0 else float("nan")
    else:
        standard_error = float("nan")
    bootstrap_count = int(n_bootstrap)
    if bootstrap_count < 0:
        raise ValueError("n_bootstrap must be nonnegative")
    bootstrap_values: list[float] = []
    if bootstrap_count:
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        for _ in range(bootstrap_count):
            sample = generator.choice(values, size=values.size, replace=True)
            bootstrap_values.append(fit_brody(sample, method=canonical_method).beta)
        if len(bootstrap_values) > 1:
            standard_error = float(np.std(bootstrap_values, ddof=1))
    return BrodyFit(
        beta=beta,
        negative_log_likelihood=float(objective(beta)),
        standard_error=standard_error,
        success=bool(optimum.success),
        n_spacings=int(values.size),
        method=canonical_method,
        diagnostics={
            "fit_objective": float(optimum.fun),
            "n_bootstrap": bootstrap_count,
        },
    )


def fit_brody_cdf_nls(
    spacings: np.ndarray,
    *,
    n_bootstrap: int = 0,
    rng: np.random.Generator | int | None = 0,
) -> BrodyFit:
    """Convenience wrapper for the reference CDF least-squares fit."""

    return fit_brody(
        spacings,
        method="cdf_nls",
        n_bootstrap=n_bootstrap,
        rng=rng,
    )


def r_statistic(levels: np.ndarray) -> float:
    """Return the mean adjacent-gap ratio, which requires no unfolding."""

    ordered = _clean_levels(levels, minimum=4)
    gaps = np.diff(ordered)
    first, second = gaps[:-1], gaps[1:]
    denominator = np.maximum(first, second)
    valid = denominator > 0.0
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.minimum(first[valid], second[valid]) / denominator[valid]))


def wigner_goe_cdf(s: np.ndarray | float) -> np.ndarray:
    values = np.asarray(s, dtype=np.float64)
    return np.where(values <= 0.0, 0.0, 1.0 - np.exp(-np.pi * np.square(values) / 4.0))


def poisson_cdf(s: np.ndarray | float) -> np.ndarray:
    values = np.asarray(s, dtype=np.float64)
    return np.where(values <= 0.0, 0.0, 1.0 - np.exp(-values))


def nn_spacing_ks(levels: np.ndarray, deg: int = 7) -> dict[str, float]:
    spacings = nn_spacing(levels, deg=deg)
    return {
        "nn_KS_GOE": float(kstest(spacings, wigner_goe_cdf).statistic),
        "nn_KS_Poisson": float(kstest(spacings, poisson_cdf).statistic),
    }


def number_variance(
    levels: np.ndarray,
    L: float,
    *,
    unfolded: bool = False,
    degree: int = 7,
    n_windows: int | None = None,
    method: str = "sliding",
    tolerance: float = 1e-4,
    rng: np.random.Generator | int | None = 0,
) -> float:
    """Return variance of level counts in sliding intervals of length ``L``."""

    L = float(L)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("L must be finite and positive")
    ordered = _clean_levels(levels, minimum=5)
    transformed = ordered if unfolded else unfold_spectrum(ordered, degree=degree)
    available = float(transformed[-1] - transformed[0])
    if available <= L:
        return float("nan")
    count = max(32, min(2048, 4 * transformed.size)) if n_windows is None else int(n_windows)
    if count < 2:
        raise ValueError("n_windows must be at least two")
    strategy = str(method).lower()
    if strategy == "sliding":
        starts = np.linspace(transformed[0], transformed[-1] - L, count)
    elif strategy in {"monte_carlo", "reference_monte_carlo"}:
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        running_mean = 0.0
        running_square_mean = 0.0
        estimates: list[float] = []
        minimum_iterations = min(count, max(32, int(np.ceil(np.sqrt(count)))))
        convergence_window = min(24, minimum_iterations)
        for index in range(count):
            start = float(generator.uniform(transformed[0], transformed[-1] - L))
            left = int(np.searchsorted(transformed, start, side="left"))
            right = int(np.searchsorted(transformed, start + L, side="left"))
            level_count = float(right - left)
            running_mean = (index * running_mean + level_count) / (index + 1)
            running_square_mean = (
                index * running_square_mean + level_count**2
            ) / (index + 1)
            estimate = max(0.0, running_square_mean - running_mean**2)
            estimates.append(estimate)
            if index + 1 >= minimum_iterations:
                recent = estimates[-convergence_window:]
                if max(recent) - min(recent) < tolerance:
                    break
        return float(estimates[-1])
    else:
        raise ValueError("method must be sliding or monte_carlo")
    left = np.searchsorted(transformed, starts, side="left")
    right = np.searchsorted(transformed, starts + L, side="left")
    counts = (right - left).astype(np.float64)
    return float(np.mean(np.square(counts - np.mean(counts))))


def sigma2(levels: np.ndarray, L: float, deg: int = 7) -> float:
    return number_variance(levels, L, degree=deg)


def _delta3_window(levels: np.ndarray, start: float, length: float) -> float:
    stop = start + length
    internal = levels[(levels > start) & (levels < stop)]
    boundaries = np.concatenate(([start], internal, [stop]))
    left_edges, right_edges = boundaries[:-1], boundaries[1:]
    midpoints = 0.5 * (left_edges + right_edges)
    baseline = np.searchsorted(levels, start, side="right")
    staircase = np.searchsorted(levels, midpoints, side="right") - baseline
    widths = right_edges - left_edges
    integral_one = float(np.sum(widths))
    integral_x = float(np.sum(0.5 * (right_edges**2 - left_edges**2)))
    integral_x2 = float(np.sum((right_edges**3 - left_edges**3) / 3.0))
    rhs_constant = float(np.sum(staircase * widths))
    rhs_linear = float(np.sum(staircase * 0.5 * (right_edges**2 - left_edges**2)))
    staircase_square = float(np.sum(np.square(staircase) * widths))
    gram = np.asarray([[integral_x2, integral_x], [integral_x, integral_one]])
    rhs = np.asarray([rhs_linear, rhs_constant])
    coefficients = np.linalg.solve(gram, rhs)
    residual = staircase_square - float(rhs @ coefficients)
    return max(0.0, residual / length)


def dyson_mehta_delta3(
    levels: np.ndarray,
    L: float,
    *,
    unfolded: bool = False,
    degree: int = 7,
    n_windows: int | None = None,
) -> float:
    """Return the sliding-window Dyson-Mehta spectral rigidity."""

    L = float(L)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("L must be finite and positive")
    ordered = _clean_levels(levels, minimum=6)
    transformed = ordered if unfolded else unfold_spectrum(ordered, degree=degree)
    if transformed[-1] - transformed[0] <= L:
        return float("nan")
    count = max(16, min(512, transformed.size)) if n_windows is None else int(n_windows)
    if count < 2:
        raise ValueError("n_windows must be at least two")
    starts = np.linspace(transformed[0], transformed[-1] - L, count)
    values = [_delta3_window(transformed, float(start), L) for start in starts]
    return float(np.mean(values))


def delta3(levels: np.ndarray, L: float, deg: int = 7) -> float:
    return dyson_mehta_delta3(levels, L, degree=deg)


def complex_spacing_ratio(matrix: np.ndarray) -> dict[str, float]:
    """Return nearest/next-nearest complex spacing-ratio moments."""

    array = np.asarray(matrix)
    if array.ndim == 1:
        eigenvalues = array.astype(np.complex128)
    elif array.ndim == 2 and array.shape[0] == array.shape[1]:
        eigenvalues = np.linalg.eigvals(array)
    else:
        raise ValueError("matrix must be square or a one-dimensional complex spectrum")
    eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
    if eigenvalues.size < 4:
        raise ValueError("at least four eigenvalues are required")
    ratios: list[complex] = []
    for index, value in enumerate(eigenvalues):
        distances = np.abs(eigenvalues - value)
        distances[index] = np.inf
        candidates = np.argpartition(distances, 2)[:2]
        ordered = candidates[np.argsort(distances[candidates])]
        nearest, next_nearest = eigenvalues[ordered[0]], eigenvalues[ordered[1]]
        denominator = next_nearest - value
        if denominator != 0.0:
            ratios.append((nearest - value) / denominator)
    if not ratios:
        return {"abs_mean": float("nan"), "cos_mean": float("nan")}
    ratio_array = np.asarray(ratios)
    angles = np.angle(ratio_array)
    return {
        "abs_mean": float(np.mean(np.abs(ratio_array))),
        "cos_mean": float(np.mean(np.cos(angles))),
    }


__all__ = [
    "BrodyFit",
    "brody_cdf",
    "brody_pdf",
    "complex_spacing_ratio",
    "delta3",
    "dyson_mehta_delta3",
    "fit_brody",
    "fit_brody_cdf_nls",
    "nearest_neighbor_spacings",
    "nn_spacing",
    "nn_spacing_ks",
    "number_variance",
    "poisson_cdf",
    "r_statistic",
    "sigma2",
    "unfold",
    "unfold_spectrum",
    "wigner_goe_cdf",
]
