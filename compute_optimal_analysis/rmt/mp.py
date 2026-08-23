"""Marchenko-Pastur theory, robust scale fitting, and spectral edges."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares, minimize_scalar


@dataclass(frozen=True)
class MPFitResult:
    """Result of a one-parameter MP variance fit."""

    aspect_ratio: float
    variance: float
    sigma: float
    lambda_minus: float
    lambda_plus: float
    ks_distance: float
    bulk_fraction: float
    n_lower_outliers: int
    n_upper_outliers: int
    method: str = "analytic_mp"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpikeDetectionResult:
    """Thresholded right-edge spike detection summary."""

    method: str
    threshold: float
    bulk_edge: float
    spikes: np.ndarray
    indices: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spikes = np.asarray(self.spikes, dtype=np.float64).ravel()
        indices = np.asarray(self.indices, dtype=np.int64).ravel()
        if spikes.size != indices.size:
            raise ValueError("spikes and indices must have equal length")
        object.__setattr__(self, "spikes", spikes)
        object.__setattr__(self, "indices", indices)

    @property
    def n_spikes(self) -> int:
        return int(self.spikes.size)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["spikes"] = self.spikes.tolist()
        payload["indices"] = self.indices.tolist()
        payload["n_spikes"] = self.n_spikes
        return payload


@dataclass(frozen=True)
class ModifiedMPFitResult:
    """Empirical singular-domain curve fit used by the level-statistics archive."""

    amplitude: float
    nu_min: float
    nu_max: float
    rmse: float
    success: bool
    bandwidth_window: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_aspect_ratio(aspect_ratio: float) -> float:
    q = float(aspect_ratio)
    if not np.isfinite(q) or q <= 0.0 or q > 1.0:
        raise ValueError("aspect_ratio must lie in (0, 1]")
    return q


def _validate_variance(variance: float) -> float:
    value = float(variance)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("variance must be finite and positive")
    return value


def _validate_dimensions(n: int, m: int) -> tuple[int, int]:
    n_int, m_int = int(n), int(m)
    if n_int < 1 or m_int < 1 or n_int != n or m_int != m:
        raise ValueError("matrix dimensions must be positive integers")
    return n_int, m_int


def _positive_finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    return array[np.isfinite(array) & (array > 0.0)]


def marchenko_pastur_bounds(
    aspect_ratio: float,
    variance: float = 1.0,
) -> tuple[float, float]:
    """Return the continuous MP support for a covariance spectrum."""

    q = _validate_aspect_ratio(aspect_ratio)
    variance = _validate_variance(variance)
    root = np.sqrt(q)
    return variance * (1.0 - root) ** 2, variance * (1.0 + root) ** 2


def marchenko_pastur_density(
    x: np.ndarray | float,
    aspect_ratio: float,
    variance: float = 1.0,
) -> np.ndarray:
    """Evaluate the normalized continuous MP density for ``q <= 1``."""

    q = _validate_aspect_ratio(aspect_ratio)
    variance = _validate_variance(variance)
    values = np.asarray(x, dtype=np.float64)
    lower, upper = marchenko_pastur_bounds(q, variance)
    density = np.zeros_like(values, dtype=np.float64)
    mask = (values > lower) & (values < upper) & (values > 0.0)
    if np.any(mask):
        inside = values[mask]
        radicand = np.maximum((upper - inside) * (inside - lower), 0.0)
        density[mask] = np.sqrt(radicand) / (2.0 * np.pi * q * variance * inside)
    return density


def _mp_cdf_scalar(value: float, q: float, variance: float) -> float:
    lower, upper = marchenko_pastur_bounds(q, variance)
    if value <= lower:
        return 0.0
    if value >= upper:
        return 1.0

    def integrand(point: float) -> float:
        return float(marchenko_pastur_density(np.asarray([point]), q, variance)[0])

    result, _ = quad(
        integrand,
        lower,
        value,
        epsabs=2e-10,
        epsrel=2e-9,
        limit=200,
        points=[lower, value],
    )
    return float(np.clip(result, 0.0, 1.0))


def marchenko_pastur_cdf(
    x: np.ndarray | float,
    aspect_ratio: float,
    variance: float = 1.0,
) -> np.ndarray | float:
    """Evaluate the MP CDF by stable quadrature on its compact support."""

    q = _validate_aspect_ratio(aspect_ratio)
    variance = _validate_variance(variance)
    values = np.asarray(x, dtype=np.float64)
    flat = np.asarray(
        [_mp_cdf_scalar(float(value), q, variance) for value in values.ravel()],
        dtype=np.float64,
    ).reshape(values.shape)
    return float(flat) if values.ndim == 0 else flat


def _mp_quantile(probability: float, q: float, variance: float = 1.0) -> float:
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    lower, upper = marchenko_pastur_bounds(q, variance)
    epsilon = np.finfo(float).eps * max(1.0, upper)
    return float(
        brentq(
            lambda value: _mp_cdf_scalar(value, q, variance) - probability,
            lower + epsilon,
            upper - epsilon,
            xtol=1e-11,
            rtol=1e-11,
            maxiter=150,
        )
    )


def mp_eigenvalues(weight: np.ndarray, normalization: float | None = None) -> np.ndarray:
    """Return nonzero covariance eigenvalues in descending order."""

    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("weight must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("weight must be finite")
    norm = float(max(matrix.shape) if normalization is None else normalization)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("normalization must be finite and positive")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return np.square(singular_values) / norm


def fit_marchenko_pastur(
    eigenvalues: np.ndarray,
    aspect_ratio: float,
    *,
    variance: float | None = None,
    trim_upper: float = 0.1,
) -> MPFitResult:
    """Fit MP variance by robust quantile matching and report edge departures."""

    values = np.sort(_positive_finite(eigenvalues))
    if values.size < 4:
        raise ValueError("at least four positive eigenvalues are required")
    q = _validate_aspect_ratio(aspect_ratio)
    trim = float(trim_upper)
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim_upper must lie in [0, 0.5)")
    if variance is None:
        probability = 0.5 * (1.0 - trim)
        empirical = float(np.quantile(values, probability, method="linear"))
        theoretical = _mp_quantile(probability, q, 1.0)
        fitted_variance = empirical / theoretical
    else:
        fitted_variance = _validate_variance(variance)
    lower, upper = marchenko_pastur_bounds(q, fitted_variance)
    model_cdf = np.asarray(marchenko_pastur_cdf(values, q, fitted_variance))
    empirical_cdf = (np.arange(values.size, dtype=np.float64) + 0.5) / values.size
    ks_distance = float(np.max(np.abs(empirical_cdf - model_cdf)))
    lower_count = int(np.count_nonzero(values < lower))
    upper_count = int(np.count_nonzero(values > upper))
    bulk_fraction = float(np.mean((values >= lower) & (values <= upper)))
    return MPFitResult(
        aspect_ratio=q,
        variance=float(fitted_variance),
        sigma=float(np.sqrt(fitted_variance)),
        lambda_minus=float(lower),
        lambda_plus=float(upper),
        ks_distance=ks_distance,
        bulk_fraction=bulk_fraction,
        n_lower_outliers=lower_count,
        n_upper_outliers=upper_count,
    )


def triangular_kde(
    values: np.ndarray,
    grid: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Evaluate the compact linear-kernel density used by the archive fitter."""

    data = _positive_finite(values)
    points = np.asarray(grid, dtype=np.float64).ravel()
    width = float(bandwidth)
    if data.size < 2:
        raise ValueError("at least two positive values are required")
    if not np.all(np.isfinite(points)):
        raise ValueError("grid must be finite")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("bandwidth must be finite and positive")
    density = np.zeros(points.size, dtype=np.float64)
    chunk = max(1, int(2_000_000 // max(data.size, 1)))
    for start in range(0, points.size, chunk):
        selected = points[start : start + chunk]
        distances = np.abs((selected[:, None] - data[None, :]) / width)
        density[start : start + chunk] = np.mean(np.maximum(1.0 - distances, 0.0), axis=1) / width
    return density


def modified_mp_singular_density(
    x: np.ndarray | float,
    amplitude: float,
    nu_min: float,
    nu_max: float,
) -> np.ndarray:
    """Evaluate the archive's unconstrained singular-domain MP-shaped curve."""

    values = np.asarray(x, dtype=np.float64)
    scale = float(amplitude)
    lower, upper = float(nu_min), float(nu_max)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("amplitude must be finite and nonnegative")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0.0 or upper <= lower:
        raise ValueError("singular support must satisfy 0 <= nu_min < nu_max")
    density = np.zeros_like(values, dtype=np.float64)
    mask = (values > max(lower, 0.0)) & (values < upper)
    if np.any(mask):
        selected = values[mask]
        radicand = np.maximum(
            (upper**2 - selected**2) * (selected**2 - lower**2),
            0.0,
        )
        density[mask] = scale * np.sqrt(radicand) / selected
    return density


def adaptive_gaussian_spectral_density(
    singular_values: np.ndarray,
    grid: np.ndarray,
    *,
    window: int = 15,
) -> np.ndarray:
    """Broaden a singular spectrum with rank-adaptive Gaussian widths."""

    values = np.sort(np.asarray(singular_values, dtype=np.float64).ravel())
    points = np.asarray(grid, dtype=np.float64).ravel()
    width_count = int(window)
    if values.size < 2 * width_count + 2 or width_count < 1:
        raise ValueError("window must be positive and leave at least two interior values")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or not np.all(np.isfinite(points)):
        raise ValueError("singular values and grid must be finite and nonnegative")
    padded = np.pad(values, (width_count, width_count), mode="edge")
    means = padded[width_count:-width_count]
    widths = 0.5 * (padded[2 * width_count :] - padded[: -2 * width_count])
    positive = widths[widths > 0.0]
    fallback = float(np.median(positive)) if positive.size else max(float(np.ptp(values)), 1.0)
    widths = np.maximum(widths, max(np.finfo(float).eps, fallback * 1e-8))
    standardized = (points[:, None] - means[None, :]) / widths[None, :]
    density = np.exp(-0.5 * np.square(standardized)) / (
        np.sqrt(2.0 * np.pi) * widths[None, :]
    )
    return np.mean(density, axis=1)


def fit_modified_mp_singular(
    singular_values: np.ndarray,
    *,
    lower_index: int = 0,
    x_min: float = 0.0,
    fit_peak_fraction: float = 0.7,
    kernel_window: int = 15,
    grid_size: int = 512,
) -> ModifiedMPFitResult:
    """Fit free amplitude and upper edge with the observed lower edge fixed."""

    values = np.sort(np.asarray(singular_values, dtype=np.float64).ravel())
    values = values[np.isfinite(values) & (values >= 0.0)]
    index = int(lower_index)
    window = int(kernel_window)
    if values.size < 2 * window + 4:
        raise ValueError("the spectrum is too short for the requested kernel window")
    if index < 0 or index >= values.size:
        raise ValueError("lower_index lies outside the singular spectrum")
    minimum = float(x_min)
    fraction = float(fit_peak_fraction)
    if not np.isfinite(minimum) or minimum < 0.0:
        raise ValueError("x_min must be finite and nonnegative")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fit_peak_fraction must lie in (0, 1]")
    size = int(grid_size)
    if size < 64:
        raise ValueError("grid_size must be at least 64")
    lower = max(float(values[index]), minimum)
    data_range = max(float(values[-1] - values[0]), np.finfo(float).eps)
    grid = np.linspace(max(minimum, float(values[0])), float(values[-1] + 0.1 * data_range), size)
    empirical = adaptive_gaussian_spectral_density(values, grid, window=window)
    peak_index = int(np.argmax(empirical))
    mask = (grid >= minimum) & (
        (grid <= grid[peak_index]) | (empirical >= fraction * empirical[peak_index])
    )
    selected_grid = grid[mask]
    selected_density = empirical[mask]
    initial_upper = max(float(values[-1]), lower + data_range * 0.1)
    initial_amplitude = max(float(np.max(selected_density)), np.finfo(float).eps) / max(
        initial_upper**2 - lower**2,
        np.finfo(float).eps,
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        amplitude = float(np.exp(parameters[0]))
        upper = lower + float(np.exp(parameters[1]))
        return modified_mp_singular_density(selected_grid, amplitude, lower, upper) - selected_density

    optimum = least_squares(
        residual,
        np.log(np.asarray([initial_amplitude, initial_upper - lower])),
        bounds=(np.asarray([-40.0, -40.0]), np.asarray([40.0, 40.0])),
        max_nfev=5000,
    )
    amplitude = float(np.exp(optimum.x[0]))
    upper = lower + float(np.exp(optimum.x[1]))
    rmse = float(np.sqrt(np.mean(np.square(residual(optimum.x)))))
    return ModifiedMPFitResult(
        amplitude=amplitude,
        nu_min=lower,
        nu_max=upper,
        rmse=rmse,
        success=bool(optimum.success),
        bandwidth_window=window,
        diagnostics={
            "fit_peak_fraction": fraction,
            "grid_size": size,
            "fit_points": int(selected_grid.size),
            "optimizer_message": str(optimum.message),
        },
    )


def fit_marchenko_pastur_thamm(
    weight: np.ndarray,
    *,
    lower_index: int = 0,
    x_min: float = 0.0,
    fit_peak_fraction: float = 0.7,
    kernel_window: int = 15,
    grid_size: int = 512,
) -> MPFitResult:
    """Fit the empirical singular-domain MP curve used by Thamm et al.

    The reference fit leaves the amplitude and upper singular edge free while
    fixing the lower edge to an observed order statistic.  This adapter keeps
    that unconstrained fit intact, then converts its singular edges to the
    repository's canonical ``s**2 / max(shape)`` domain.  ``variance`` is only
    an upper-edge-matched compatibility scale; the fitted support itself is
    authoritative and need not obey the one-parameter analytic MP relation.
    """

    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("weight must be a finite matrix with both dimensions at least two")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    curve = fit_modified_mp_singular(
        singular_values,
        lower_index=lower_index,
        x_min=x_min,
        fit_peak_fraction=fit_peak_fraction,
        kernel_window=kernel_window,
        grid_size=grid_size,
    )
    large = max(matrix.shape)
    q = min(matrix.shape) / large
    eigenvalues = np.square(singular_values) / large
    lower = curve.nu_min**2 / large
    upper = curve.nu_max**2 / large
    effective_variance = upper / (1.0 + np.sqrt(q)) ** 2

    cdf_grid = np.linspace(curve.nu_min, curve.nu_max, max(1024, int(grid_size)))
    curve_density = modified_mp_singular_density(
        cdf_grid,
        curve.amplitude,
        curve.nu_min,
        curve.nu_max,
    )
    increments = 0.5 * (curve_density[1:] + curve_density[:-1]) * np.diff(cdf_grid)
    cumulative = np.concatenate((np.asarray([0.0]), np.cumsum(increments)))
    total = float(cumulative[-1])
    if total > 0.0 and np.isfinite(total):
        cumulative /= total
        ordered_singular = np.sort(singular_values)
        model_cdf = np.interp(ordered_singular, cdf_grid, cumulative, left=0.0, right=1.0)
        empirical_cdf = (np.arange(ordered_singular.size, dtype=np.float64) + 0.5) / ordered_singular.size
        ks_distance = float(np.max(np.abs(empirical_cdf - model_cdf)))
    else:
        ks_distance = float("nan")

    return MPFitResult(
        aspect_ratio=float(q),
        variance=float(effective_variance),
        sigma=float(np.sqrt(effective_variance)),
        lambda_minus=float(lower),
        lambda_plus=float(upper),
        ks_distance=ks_distance,
        bulk_fraction=float(np.mean((eigenvalues >= lower) & (eigenvalues <= upper))),
        n_lower_outliers=int(np.count_nonzero(eigenvalues < lower)),
        n_upper_outliers=int(np.count_nonzero(eigenvalues > upper)),
        method="thamm_modified_singular",
        diagnostics={
            "amplitude": curve.amplitude,
            "nu_min_raw": curve.nu_min,
            "nu_max_raw": curve.nu_max,
            "curve_rmse": curve.rmse,
            "optimizer_success": curve.success,
            "bandwidth_window": curve.bandwidth_window,
            "normalization_denominator": int(large),
            "variance_semantics": "analytic upper-edge projection only",
            **curve.diagnostics,
        },
    )


def fit_marchenko_pastur_kde(
    eigenvalues: np.ndarray,
    aspect_ratio: float,
    *,
    bandwidth: float | None = None,
    trim_upper: float = 0.1,
    grid_size: int = 512,
) -> MPFitResult:
    """Fit MP scale to a triangular KDE of the retained empirical bulk.

    This reproduces the scientific structure of the early automated
    WeightWatcher notebook while using a bounded scalar optimizer.
    """

    values = np.sort(_positive_finite(eigenvalues))
    if values.size < 8:
        raise ValueError("at least eight positive eigenvalues are required")
    q = _validate_aspect_ratio(aspect_ratio)
    trim = float(trim_upper)
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim_upper must lie in [0, 0.5)")
    retain_count = max(4, int(np.floor((1.0 - trim) * values.size)))
    retained = values[:retain_count]
    scale = float(np.std(retained, ddof=1))
    automatic = 1.06 * scale * retained.size ** (-0.2)
    width = max(automatic, np.finfo(float).eps * max(1.0, float(np.max(retained)))) if bandwidth is None else float(bandwidth)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("bandwidth must be finite and positive")
    size = int(grid_size)
    if size < 64:
        raise ValueError("grid_size must be at least 64")
    grid_max = float(np.max(retained) + 2.0 * width)
    grid = np.linspace(max(np.finfo(float).eps, float(np.min(retained)) - width), grid_max, size)
    empirical_density = triangular_kde(retained, grid, width)
    initial = fit_marchenko_pastur(retained, q, trim_upper=0.0).variance
    lower_log = np.log(max(initial * 0.05, np.finfo(float).tiny))
    upper_log = np.log(initial * 20.0)

    def objective(log_variance: float) -> float:
        variance = float(np.exp(log_variance))
        model_density = marchenko_pastur_density(grid, q, variance)
        weights = np.maximum(empirical_density, 0.05 * float(np.max(empirical_density)))
        residual = model_density - empirical_density
        return float(np.sum(weights * np.square(residual)) / np.sum(weights))

    optimum = minimize_scalar(
        objective,
        bounds=(lower_log, upper_log),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    variance = float(np.exp(optimum.x))
    fitted = fit_marchenko_pastur(values, q, variance=variance, trim_upper=trim)
    return replace(
        fitted,
        method="kde_bulk_fit",
        diagnostics={
            "bandwidth": width,
            "trim_upper": trim,
            "retained_count": int(retained.size),
            "objective": float(optimum.fun),
            "optimizer_success": bool(optimum.success),
        },
    )


def fit_marchenko_pastur_farms(
    weight: np.ndarray,
    *,
    target_aspect_ratio: float = 1.0,
    window_size: int | None = None,
    row_windows: int = 5,
    column_windows: int = 5,
    sampling: str = "reference_fixed",
    step_size: int = 10,
    orient_tall: bool = False,
    seed: int | None = 0,
    trim_upper: float = 0.1,
) -> MPFitResult:
    """Fit MP theory to a fixed-ratio pooled submatrix ESD."""

    from .farms_aspect_ratio import FARMSConfig, farms_spectrum

    result = farms_spectrum(
        weight,
        FARMSConfig(
            target_aspect_ratio=target_aspect_ratio,
            window_size=window_size,
            row_windows=row_windows,
            column_windows=column_windows,
            sampling=sampling,
            step_size=step_size,
            normalization="canonical",
            orient_tall=orient_tall,
            seed=seed,
        ),
    )
    fitted = fit_marchenko_pastur(
        result.eigenvalues,
        result.canonical_aspect_ratio,
        trim_upper=trim_upper,
    )
    return replace(
        fitted,
        method="farms_unbiased",
        diagnostics={
            **result.as_dict(),
            "spectral_max": float(np.max(result.eigenvalues)),
        },
    )


def tracy_widom_edge_scale(
    n: int,
    m: int,
    variance: float = 1.0,
    *,
    edge: str = "upper",
) -> float:
    """Return Johnstone's finite-size TW scale for a covariance edge."""

    n, m = _validate_dimensions(n, m)
    variance = _validate_variance(variance)
    small, large = min(n, m), max(n, m)
    a = np.sqrt(large - 0.5)
    b = np.sqrt(small - 0.5)
    if edge == "upper":
        scale = (a + b) * (1.0 / a + 1.0 / b) ** (1.0 / 3.0)
    elif edge == "lower":
        if np.isclose(a, b):
            return 0.0
        scale = (a - b) * abs(1.0 / b - 1.0 / a) ** (1.0 / 3.0)
    else:
        raise ValueError("edge must be 'upper' or 'lower'")
    return float(variance * scale / large)


def tracy_widom_quantile(confidence: float = 0.95) -> float:
    """Return an interpolated real-Wishart (TW1) right-tail quantile.

    The tabulated values cover the confidence levels used for diagnostics;
    interpolation avoids introducing a specialized probability package.
    """

    probability = float(confidence)
    probabilities = np.asarray([0.50, 0.80, 0.90, 0.95, 0.975, 0.99, 0.999])
    quantiles = np.asarray([-1.2686, -0.1653, 0.4501, 0.9793, 1.4530, 2.0234, 3.2720])
    if not probabilities[0] <= probability <= probabilities[-1]:
        raise ValueError("confidence must lie in [0.5, 0.999]")
    return float(np.interp(probability, probabilities, quantiles))


def tracy_widom_upper_threshold(
    n: int,
    m: int,
    variance: float = 1.0,
    *,
    confidence: float = 0.95,
) -> float:
    """Return Johnstone's finite-sample upper-edge threshold for ``XX.T/m``."""

    n, m = _validate_dimensions(n, m)
    variance = _validate_variance(variance)
    small, large = min(n, m), max(n, m)
    center = variance * (
        np.sqrt(large - 0.5) + np.sqrt(small - 0.5)
    ) ** 2 / large
    scale = tracy_widom_edge_scale(small, large, variance, edge="upper")
    return float(center + tracy_widom_quantile(confidence) * scale)


def detect_spikes_tracy_widom(
    eigenvalues: np.ndarray,
    n: int,
    m: int,
    variance: float = 1.0,
    *,
    confidence: float = 0.95,
) -> SpikeDetectionResult:
    """Identify sample eigenvalues beyond a finite-size TW1 threshold."""

    values = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("eigenvalues must be a nonempty finite nonnegative array")
    n, m = _validate_dimensions(n, m)
    q = min(n, m) / max(n, m)
    bulk_edge = marchenko_pastur_bounds(q, variance)[1]
    threshold = tracy_widom_upper_threshold(n, m, variance, confidence=confidence)
    indices = np.flatnonzero(values > threshold)
    order = np.argsort(values[indices])[::-1]
    indices = indices[order]
    return SpikeDetectionResult(
        method="tracy_widom_95" if np.isclose(confidence, 0.95) else "tracy_widom",
        threshold=threshold,
        bulk_edge=bulk_edge,
        spikes=values[indices],
        indices=indices,
        diagnostics={"confidence": float(confidence), "aspect_ratio": q},
    )


def bbp_population_threshold(
    aspect_ratio: float,
    variance: float = 1.0,
) -> float:
    """Return the rank-one population-spike BBP threshold."""

    q = _validate_aspect_ratio(aspect_ratio)
    variance = _validate_variance(variance)
    return float(variance * (1.0 + np.sqrt(q)))


def bbp_sample_location(
    population_eigenvalue: float,
    aspect_ratio: float,
    variance: float = 1.0,
) -> float:
    """Map a population covariance spike to its asymptotic sample location."""

    q = _validate_aspect_ratio(aspect_ratio)
    variance = _validate_variance(variance)
    spike = float(population_eigenvalue)
    if not np.isfinite(spike) or spike <= 0.0:
        raise ValueError("population_eigenvalue must be finite and positive")
    relative = spike / variance
    if relative <= 1.0 + np.sqrt(q):
        return float(marchenko_pastur_bounds(q, variance)[1])
    return float(variance * relative * (1.0 + q / (relative - 1.0)))


def detect_spikes_bbp(
    eigenvalues: np.ndarray,
    aspect_ratio: float,
    variance: float = 1.0,
    *,
    margin: float = 0.0,
) -> SpikeDetectionResult:
    """Identify separated sample spikes beyond the null BBP bulk edge."""

    values = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("eigenvalues must be a nonempty finite nonnegative array")
    q = _validate_aspect_ratio(aspect_ratio)
    variance = _validate_variance(variance)
    additive_margin = float(margin)
    if not np.isfinite(additive_margin) or additive_margin < 0.0:
        raise ValueError("margin must be finite and nonnegative")
    bulk_edge = marchenko_pastur_bounds(q, variance)[1]
    threshold = bulk_edge + additive_margin
    indices = np.flatnonzero(values > threshold)
    order = np.argsort(values[indices])[::-1]
    indices = indices[order]
    return SpikeDetectionResult(
        method="bbp_transition",
        threshold=float(threshold),
        bulk_edge=float(bulk_edge),
        spikes=values[indices],
        indices=indices,
        diagnostics={
            "aspect_ratio": q,
            "population_threshold": bbp_population_threshold(q, variance),
            "margin": additive_margin,
        },
    )


def fit_marchenko_pastur_lanczos(
    weight: np.ndarray,
    *,
    steps: int | None = None,
    n_probes: int = 3,
    tail_window: int | None = None,
    threshold_c: float = 1.0,
    threshold_delta: float = 0.25,
    residue_threshold: float = 0.0,
    ridge: float = 0.0,
    adaptive: bool = True,
    convergence_tolerance: float | None = None,
    sequence_length: int | None = None,
    check_interval: int = 2,
    pole_method: str = "reference_ritz",
    rng: np.random.Generator | int | None = 0,
) -> MPFitResult:
    """Fit an effective one-cut MP scale from a Lanczos support estimate."""

    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("weight must be a finite matrix with both dimensions at least two")
    from .lanczos_stieltjes import detect_spikes_from_factor

    detected = detect_spikes_from_factor(
        matrix,
        steps=steps,
        n_probes=n_probes,
        tail_window=tail_window,
        threshold_c=threshold_c,
        threshold_delta=threshold_delta,
        residue_threshold=residue_threshold,
        ridge=ridge,
        adaptive=adaptive,
        convergence_tolerance=convergence_tolerance,
        sequence_length=sequence_length,
        check_interval=check_interval,
        pole_method=pole_method,
        rng=rng,
    )
    q = min(matrix.shape) / max(matrix.shape)
    variance = detected.lambda_plus / (1.0 + np.sqrt(q)) ** 2
    eigenvalues = mp_eigenvalues(matrix)
    model = fit_marchenko_pastur(eigenvalues, q, variance=variance)
    return replace(
        model,
        lambda_minus=float(detected.lambda_minus),
        lambda_plus=float(detected.lambda_plus),
        n_lower_outliers=int(np.count_nonzero(eigenvalues < detected.lambda_minus)),
        n_upper_outliers=int(detected.n_spikes),
        bulk_fraction=float(
            np.mean(
                (eigenvalues >= detected.lambda_minus)
                & (eigenvalues <= detected.lambda_plus)
            )
        ),
        method="lanczos_stieltjes",
        diagnostics=detected.as_dict(),
    )


def mp_soft_rank(eigenvalues: np.ndarray, lambda_plus: float) -> float:
    """Return the MP upper edge divided by the largest observed eigenvalue."""

    values = _positive_finite(eigenvalues)
    edge = float(lambda_plus)
    if values.size == 0 or not np.isfinite(edge) or edge < 0.0:
        return float("nan")
    return float(edge / np.max(values))


def mp_bounds(n: int, m: int, sigma: float) -> tuple[float, float]:
    """Return singular-value MP edges for an ``n x m`` Gaussian factor."""

    n, m = _validate_dimensions(n, m)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    root_large, root_small = np.sqrt(max(n, m)), np.sqrt(min(n, m))
    return float(sigma * (root_large - root_small)), float(sigma * (root_large + root_small))


def mp_pdf(x: np.ndarray | float, n: int, m: int, sigma: float) -> np.ndarray:
    """Evaluate the normalized MP density in the singular-value domain."""

    n, m = _validate_dimensions(n, m)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    values = np.asarray(x, dtype=np.float64)
    large = max(n, m)
    q = min(n, m) / large
    squared = np.square(values) / large
    density = marchenko_pastur_density(squared, q, sigma**2) * (2.0 * np.maximum(values, 0.0) / large)
    lower, upper = mp_bounds(n, m, sigma)
    density = np.where((values >= lower) & (values <= upper), density, 0.0)
    if np.isclose(q, 1.0):
        limit = 2.0 / (np.pi * sigma * np.sqrt(large))
        density = np.where(values == 0.0, limit, density)
    return np.asarray(density, dtype=np.float64)


def mp_cdf(x: np.ndarray | float, n: int, m: int, sigma: float) -> np.ndarray | float:
    """Evaluate the MP CDF in the singular-value domain."""

    n, m = _validate_dimensions(n, m)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    values = np.asarray(x, dtype=np.float64)
    large = max(n, m)
    result = marchenko_pastur_cdf(
        np.square(np.maximum(values, 0.0)) / large,
        min(n, m) / large,
        sigma**2,
    )
    result_array = np.asarray(result)
    result_array = np.where(values < 0.0, 0.0, result_array)
    return float(result_array) if values.ndim == 0 else result_array


def mp_median(n: int, m: int, sigma: float = 1.0) -> float:
    """Return the singular-value median of the corresponding MP law."""

    n, m = _validate_dimensions(n, m)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    large = max(n, m)
    median_eigenvalue = _mp_quantile(0.5, min(n, m) / large, sigma**2)
    return float(np.sqrt(large * median_eigenvalue))


def eigenvalues_of_cov(
    weight: np.ndarray | None = None,
    *,
    s: np.ndarray | None = None,
    n: int | None = None,
    m: int | None = None,
    N: float | None = None,
) -> np.ndarray:
    """Return ``s**2/N`` from a weight or a supplied singular spectrum."""

    if weight is not None:
        matrix = np.asarray(weight, dtype=np.float64)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("weight must be a finite two-dimensional matrix")
        n, m = matrix.shape
        singular_values = np.linalg.svd(matrix, compute_uv=False)
    else:
        if s is None or n is None or m is None:
            raise ValueError("s, n, and m are required when weight is omitted")
        n, m = _validate_dimensions(n, m)
        singular_values = np.asarray(s, dtype=np.float64).ravel()
        if not np.all(np.isfinite(singular_values)) or np.any(singular_values < 0.0):
            raise ValueError("s must contain finite nonnegative singular values")
        if singular_values.size != min(n, m):
            raise ValueError("s must contain min(n, m) singular values")
    denominator = float(m if N is None else N)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("N must be finite and positive")
    return np.square(np.asarray(singular_values, dtype=np.float64)) / denominator


def mp_bounds_eig(n: int, m: int, sigma: float, N: float) -> tuple[float, float]:
    denominator = float(N)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("N must be finite and positive")
    lower, upper = mp_bounds(n, m, sigma)
    return float(lower**2 / denominator), float(upper**2 / denominator)


def mp_pdf_eig(x: np.ndarray | float, n: int, m: int, sigma: float, N: float) -> np.ndarray:
    """Evaluate the exact ``lambda=s**2/N`` image of the singular MP law."""

    values = np.asarray(x, dtype=np.float64)
    denominator = float(N)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("N must be finite and positive")
    singular = np.sqrt(np.maximum(values, 0.0) * denominator)
    jacobian = np.zeros_like(values)
    positive = values > 0.0
    jacobian[positive] = np.sqrt(denominator) / (2.0 * np.sqrt(values[positive]))
    density = mp_pdf(singular, n, m, sigma) * jacobian
    return np.where(values >= 0.0, density, 0.0)


def mp_cdf_eig(x: np.ndarray | float, n: int, m: int, sigma: float, N: float) -> np.ndarray | float:
    values = np.asarray(x, dtype=np.float64)
    denominator = float(N)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("N must be finite and positive")
    result = mp_cdf(np.sqrt(np.maximum(values, 0.0) * denominator), n, m, sigma)
    array = np.where(values < 0.0, 0.0, np.asarray(result))
    return float(array) if values.ndim == 0 else array


def _spectrum_and_shape(
    weight: np.ndarray | None,
    s: np.ndarray | None,
    n: int | None,
    m: int | None,
) -> tuple[np.ndarray, int, int]:
    if weight is not None:
        matrix = np.asarray(weight, dtype=np.float64)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("weight must be a finite two-dimensional matrix")
        return np.linalg.svd(matrix, compute_uv=False), matrix.shape[0], matrix.shape[1]
    if s is None or n is None or m is None:
        raise ValueError("provide either weight or all of s, n, and m")
    n, m = _validate_dimensions(n, m)
    spectrum = np.asarray(s, dtype=np.float64).ravel()
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0.0):
        raise ValueError("s must contain finite nonnegative singular values")
    if spectrum.size != min(n, m):
        raise ValueError("s must contain min(n, m) singular values")
    return np.sort(spectrum)[::-1], n, m


def estimate_sigma_gd_median(
    weight: np.ndarray | None = None,
    *,
    s: np.ndarray | None = None,
    n: int | None = None,
    m: int | None = None,
    discard_largest: float = 0.0,
) -> float:
    """Estimate entry standard deviation by MP quantile matching."""

    spectrum, n, m = _spectrum_and_shape(weight, s, n, m)
    discard = float(discard_largest)
    if not 0.0 <= discard < 0.5:
        raise ValueError("discard_largest must lie in [0, 0.5)")
    remove = int(np.floor(discard * spectrum.size))
    retained = spectrum[remove:]
    empirical = float(np.median(retained))
    probability = 0.5 * (1.0 - remove / spectrum.size)
    q = min(n, m) / max(n, m)
    unit_quantile = _mp_quantile(probability, q, 1.0)
    return float(empirical / np.sqrt(max(n, m) * unit_quantile))


estimate_sigma_med = estimate_sigma_gd_median


def estimate_sigma_med_refined(
    weight: np.ndarray | None = None,
    *,
    s: np.ndarray | None = None,
    n: int | None = None,
    m: int | None = None,
    max_iter: int = 3,
) -> tuple[float, float, int]:
    """Iteratively remove upper-edge spikes and re-estimate the MP scale."""

    spectrum, n, m = _spectrum_and_shape(weight, s, n, m)
    max_iter = int(max_iter)
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative")
    initial = estimate_sigma_gd_median(s=spectrum, n=n, m=m)
    refined = initial
    previous_count = spectrum.size
    iterations = 0
    q = min(n, m) / max(n, m)
    for iteration in range(max_iter):
        upper = mp_bounds(n, m, refined)[1]
        retained = spectrum[spectrum <= upper]
        if retained.size < max(4, spectrum.size // 2):
            break
        probability = 0.5 * retained.size / spectrum.size
        empirical = float(np.median(retained))
        unit_quantile = _mp_quantile(probability, q, 1.0)
        candidate = empirical / np.sqrt(max(n, m) * unit_quantile)
        iterations = iteration + 1
        stable_count = retained.size == previous_count
        stable_scale = abs(candidate - refined) <= 1e-6 * max(refined, 1.0)
        refined = float(candidate)
        previous_count = retained.size
        if stable_count or stable_scale:
            break
    return float(initial), float(refined), iterations


def usvt_hard_threshold(
    n: int,
    m: int,
    sigma: float,
    *,
    square_optimal: bool = True,
) -> float:
    """Return a universal or square-optimal singular-value threshold."""

    n, m = _validate_dimensions(n, m)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if square_optimal and n == m:
        return float((4.0 / np.sqrt(3.0)) * sigma * np.sqrt(n))
    return float(2.01 * sigma * np.sqrt(max(n, m)))


def small_sv_deviation(s: np.ndarray, n: int, m: int, sigma: float) -> dict[str, float | int]:
    """Summarize lower-edge departures from the fitted MP law."""

    n, m = _validate_dimensions(n, m)
    spectrum = np.sort(np.asarray(s, dtype=np.float64).ravel())
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0.0):
        raise ValueError("s must contain finite nonnegative singular values")
    if spectrum.size != min(n, m):
        raise ValueError("s must contain min(n, m) singular values")
    lower, _ = mp_bounds(n, m, sigma)
    median = mp_median(n, m, sigma)
    lower_values = spectrum[spectrum <= median]
    if lower_values.size:
        empirical = (np.arange(lower_values.size) + 0.5) / spectrum.size
        theoretical = np.asarray(mp_cdf(lower_values, n, m, sigma))
        ks_lower = float(np.max(np.abs(empirical - theoretical)))
    else:
        ks_lower = float("nan")
    below = spectrum < lower
    total_energy = float(np.sum(np.square(spectrum)))
    below_energy = float(np.sum(np.square(spectrum[below])))
    q = min(n, m) / max(n, m)
    q10_eigen = _mp_quantile(0.1, q, sigma**2)
    q10_singular = np.sqrt(max(n, m) * q10_eigen)
    expected = int(round(0.1 * spectrum.size))
    return {
        "ks_lower": ks_lower,
        "n_below_minus": int(np.count_nonzero(below)),
        "frac_mass_below_minus": 0.0 if total_energy == 0.0 else below_energy / total_energy,
        "excess_small_sv": int(np.count_nonzero(spectrum <= q10_singular) - expected),
    }


__all__ = [
    "MPFitResult",
    "ModifiedMPFitResult",
    "SpikeDetectionResult",
    "bbp_population_threshold",
    "bbp_sample_location",
    "adaptive_gaussian_spectral_density",
    "detect_spikes_bbp",
    "detect_spikes_tracy_widom",
    "eigenvalues_of_cov",
    "estimate_sigma_gd_median",
    "estimate_sigma_med",
    "estimate_sigma_med_refined",
    "fit_marchenko_pastur",
    "fit_marchenko_pastur_farms",
    "fit_marchenko_pastur_kde",
    "fit_marchenko_pastur_lanczos",
    "fit_marchenko_pastur_thamm",
    "fit_modified_mp_singular",
    "marchenko_pastur_bounds",
    "marchenko_pastur_cdf",
    "marchenko_pastur_density",
    "mp_bounds",
    "mp_bounds_eig",
    "mp_cdf",
    "mp_cdf_eig",
    "mp_eigenvalues",
    "mp_median",
    "mp_pdf",
    "mp_pdf_eig",
    "mp_soft_rank",
    "modified_mp_singular_density",
    "small_sv_deviation",
    "tracy_widom_quantile",
    "tracy_widom_edge_scale",
    "tracy_widom_upper_threshold",
    "triangular_kde",
    "usvt_hard_threshold",
]
