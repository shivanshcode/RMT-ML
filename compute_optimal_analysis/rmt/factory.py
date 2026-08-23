"""Validated multi-method dispatch for the numerical spectral engine."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .farms_aspect_ratio import FARMSConfig, FARMSResult, farms_spectrum, shape_normalize_eigenvalues
from .lanczos_stieltjes import detect_spikes_from_factor
from .mp import (
    MPFitResult,
    SpikeDetectionResult,
    detect_spikes_bbp,
    detect_spikes_tracy_widom,
    fit_marchenko_pastur,
    fit_marchenko_pastur_farms,
    fit_marchenko_pastur_kde,
    fit_marchenko_pastur_lanczos,
    mp_eigenvalues,
)
from .overlap import evaluate_overlap_metric
from .spacing import unfold_spectrum
from .tail import select_tail_estimator


MP_FIT_METHODS = (
    "analytic_mp",
    "kde_bulk_fit",
    "lanczos_stieltjes",
    "farms_unbiased",
)
UNFOLDING_STRATEGIES = (
    "polynomial_chebyshev",
    "spline_monotone",
    "gaussian_kernel",
    "raw_rank_order",
)
TAIL_SOLVERS = (
    "clauset_mle",
    "hill_estimator",
    "fixed_cutoff_mle",
    "rank_ordered_mle",
)
OVERLAP_METRICS = (
    "staats_dual_end",
    "subspace_principal_angles",
    "frobenius_projection",
)
ASPECT_RATIO_MODES = (
    "raw",
    "farms_normalized",
    "farms_unbiased",
    "shape_normalized",
)
SPIKE_DETECTORS = (
    "tracy_widom_95",
    "bbp_transition",
    "lanczos_poles",
)


def _choice(value: str, choices: Sequence[str], name: str) -> str:
    selected = str(value).lower()
    if selected not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")
    return selected


def _finite_weight(weight: np.ndarray) -> np.ndarray:
    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("weight must be a finite matrix with both dimensions at least two")
    return matrix


@dataclass(frozen=True)
class RMTMethodConfig:
    """Complete scientific-method selection shared by CLI and library callers."""

    mp_fit_method: str = "lanczos_stieltjes"
    unfolding_strategy: str = "spline_monotone"
    tail_solver: str = "clauset_mle"
    overlap_metric: str = "staats_dual_end"
    aspect_ratio_mode: str = "farms_normalized"
    spike_detector: str = "lanczos_poles"
    polynomial_degree: int = 15
    spline_smoothing: float | None = None
    gaussian_kernel_window: int = 15
    mp_trim_upper: float = 0.1
    kde_bandwidth: float | None = None
    tail_minimum: int = 50
    tail_fraction: float = 0.1
    farms_target_aspect_ratio: float = 1.0
    farms_window_size: int | None = None
    farms_row_windows: int = 5
    farms_column_windows: int = 5
    farms_sampling: str = "reference_fixed"
    farms_step_size: int = 10
    farms_normalization: str = "canonical"
    farms_orient_tall: bool = False
    lanczos_steps: int | None = 50
    lanczos_probes: int = 3
    lanczos_tail_window: int | None = None
    lanczos_threshold_c: float = 1.0
    lanczos_threshold_delta: float = 0.25
    lanczos_residue_threshold: float = 0.0
    lanczos_ridge: float = 0.0
    lanczos_adaptive: bool = True
    lanczos_convergence_tolerance: float | None = None
    lanczos_sequence_length: int | None = None
    lanczos_check_interval: int = 2
    lanczos_pole_method: str = "reference_ritz"
    seed: int | None = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mp_fit_method", _choice(self.mp_fit_method, MP_FIT_METHODS, "mp_fit_method"))
        object.__setattr__(
            self,
            "unfolding_strategy",
            _choice(self.unfolding_strategy, UNFOLDING_STRATEGIES, "unfolding_strategy"),
        )
        object.__setattr__(self, "tail_solver", _choice(self.tail_solver, TAIL_SOLVERS, "tail_solver"))
        object.__setattr__(
            self,
            "overlap_metric",
            _choice(self.overlap_metric, OVERLAP_METRICS, "overlap_metric"),
        )
        object.__setattr__(
            self,
            "aspect_ratio_mode",
            _choice(self.aspect_ratio_mode, ASPECT_RATIO_MODES, "aspect_ratio_mode"),
        )
        object.__setattr__(
            self,
            "spike_detector",
            _choice(self.spike_detector, SPIKE_DETECTORS, "spike_detector"),
        )
        if int(self.polynomial_degree) < 1 or int(self.polynomial_degree) != self.polynomial_degree:
            raise ValueError("polynomial_degree must be positive")
        if int(self.gaussian_kernel_window) < 1 or int(self.gaussian_kernel_window) != self.gaussian_kernel_window:
            raise ValueError("gaussian_kernel_window must be positive")
        if self.spline_smoothing is not None and (
            not np.isfinite(float(self.spline_smoothing)) or float(self.spline_smoothing) < 0.0
        ):
            raise ValueError("spline_smoothing must be finite and nonnegative")
        if not 0.0 <= float(self.mp_trim_upper) < 0.5:
            raise ValueError("mp_trim_upper must lie in [0, 0.5)")
        if self.kde_bandwidth is not None and (
            not np.isfinite(float(self.kde_bandwidth)) or float(self.kde_bandwidth) <= 0.0
        ):
            raise ValueError("kde_bandwidth must be finite and positive")
        if int(self.tail_minimum) < 2:
            raise ValueError("tail_minimum must be at least two")
        if not 0.0 < float(self.tail_fraction) <= 1.0:
            raise ValueError("tail_fraction must lie in (0, 1]")
        if (
            not np.isfinite(float(self.farms_target_aspect_ratio))
            or float(self.farms_target_aspect_ratio) <= 0.0
        ):
            raise ValueError("farms_target_aspect_ratio must be finite and positive")
        if self.farms_window_size is not None and (
            int(self.farms_window_size) < 2
            or int(self.farms_window_size) != self.farms_window_size
        ):
            raise ValueError("farms_window_size must be an integer of at least two")
        if (
            int(self.farms_row_windows) < 1
            or int(self.farms_row_windows) != self.farms_row_windows
            or int(self.farms_column_windows) < 1
            or int(self.farms_column_windows) != self.farms_column_windows
        ):
            raise ValueError("FARMS window counts must be positive integers")
        if self.farms_sampling not in {
            "reference_fixed",
            "reference_sliding",
            "grid",
            "random",
        }:
            raise ValueError("unsupported FARMS sampling method")
        if int(self.farms_step_size) < 1:
            raise ValueError("farms_step_size must be positive")
        if self.farms_normalization not in {"canonical", "raw", "trace"}:
            raise ValueError("unsupported FARMS normalization")
        if self.lanczos_steps is not None and int(self.lanczos_steps) < 2:
            raise ValueError("lanczos_steps must be at least two")
        if int(self.lanczos_probes) < 1 or int(self.lanczos_probes) != self.lanczos_probes:
            raise ValueError("lanczos_probes must be positive")
        if self.lanczos_tail_window is not None and int(self.lanczos_tail_window) < 2:
            raise ValueError("lanczos_tail_window must be at least two")
        if not np.isfinite(float(self.lanczos_threshold_c)) or float(self.lanczos_threshold_c) < 0.0:
            raise ValueError("lanczos_threshold_c must be finite and nonnegative")
        if not 0.0 < float(self.lanczos_threshold_delta) < 0.5:
            raise ValueError("lanczos_threshold_delta must lie in (0, 0.5)")
        if (
            not np.isfinite(float(self.lanczos_residue_threshold))
            or float(self.lanczos_residue_threshold) < 0.0
        ):
            raise ValueError("lanczos_residue_threshold must be finite and nonnegative")
        if not np.isfinite(float(self.lanczos_ridge)) or float(self.lanczos_ridge) < 0.0:
            raise ValueError("lanczos_ridge must be finite and nonnegative")
        if self.lanczos_convergence_tolerance is not None and (
            not np.isfinite(float(self.lanczos_convergence_tolerance))
            or float(self.lanczos_convergence_tolerance) <= 0.0
        ):
            raise ValueError("lanczos_convergence_tolerance must be finite and positive")
        if self.lanczos_sequence_length is not None and int(self.lanczos_sequence_length) < 1:
            raise ValueError("lanczos_sequence_length must be positive")
        if int(self.lanczos_check_interval) < 1:
            raise ValueError("lanczos_check_interval must be positive")
        if self.lanczos_pole_method not in {"reference_ritz", "constant_tail"}:
            raise ValueError("unsupported Lanczos pole method")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedSpectrum:
    """Spectrum after the configured aspect-ratio preprocessing step."""

    eigenvalues: np.ndarray
    aspect_ratio: float
    mode: str
    farms: FARMSResult | None = field(default=None, repr=False)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.eigenvalues, dtype=np.float64).ravel()
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("eigenvalues must be finite, nonnegative, and nonempty")
        q = float(self.aspect_ratio)
        if not 0.0 < q <= 1.0:
            raise ValueError("aspect_ratio must lie in (0, 1]")
        object.__setattr__(self, "eigenvalues", np.sort(values)[::-1])
        object.__setattr__(self, "aspect_ratio", q)


def _farms_config(config: RMTMethodConfig) -> FARMSConfig:
    return FARMSConfig(
        target_aspect_ratio=config.farms_target_aspect_ratio,
        window_size=config.farms_window_size,
        row_windows=config.farms_row_windows,
        column_windows=config.farms_column_windows,
        sampling=config.farms_sampling,
        step_size=config.farms_step_size,
        normalization=config.farms_normalization,
        orient_tall=config.farms_orient_tall,
        seed=config.seed,
    )


def prepare_spectrum(
    weight: np.ndarray,
    config: RMTMethodConfig = RMTMethodConfig(),
    *,
    variance: float = 1.0,
) -> PreparedSpectrum:
    """Apply raw, fixed-ratio, or analytic edge normalization to a matrix."""

    matrix = _finite_weight(weight)
    q = min(matrix.shape) / max(matrix.shape)
    raw = mp_eigenvalues(matrix)
    if config.aspect_ratio_mode == "raw":
        return PreparedSpectrum(raw, q, "raw")
    if config.aspect_ratio_mode in {"farms_normalized", "farms_unbiased"}:
        farms = farms_spectrum(matrix, _farms_config(config))
        return PreparedSpectrum(
            farms.eigenvalues,
            farms.canonical_aspect_ratio,
            "farms_normalized",
            farms=farms,
            diagnostics=farms.as_dict(),
        )
    normalized = shape_normalize_eigenvalues(raw, q, variance)
    return PreparedSpectrum(
        normalized,
        q,
        "shape_normalized",
        diagnostics={"target_upper_edge": 4.0, "source_variance": float(variance)},
    )


def dispatch_mp_fit(
    weight: np.ndarray,
    config: RMTMethodConfig = RMTMethodConfig(),
    *,
    variance: float | None = None,
) -> MPFitResult:
    """Run the configured MP bulk fitting method."""

    matrix = _finite_weight(weight)
    method = config.mp_fit_method
    if method == "lanczos_stieltjes":
        return fit_marchenko_pastur_lanczos(
            matrix,
            steps=config.lanczos_steps,
            n_probes=config.lanczos_probes,
            tail_window=config.lanczos_tail_window,
            threshold_c=config.lanczos_threshold_c,
            threshold_delta=config.lanczos_threshold_delta,
            residue_threshold=config.lanczos_residue_threshold,
            ridge=config.lanczos_ridge,
            adaptive=config.lanczos_adaptive,
            convergence_tolerance=config.lanczos_convergence_tolerance,
            sequence_length=config.lanczos_sequence_length,
            check_interval=config.lanczos_check_interval,
            pole_method=config.lanczos_pole_method,
            rng=config.seed,
        )
    if method == "farms_unbiased":
        return fit_marchenko_pastur_farms(
            matrix,
            target_aspect_ratio=config.farms_target_aspect_ratio,
            window_size=config.farms_window_size,
            row_windows=config.farms_row_windows,
            column_windows=config.farms_column_windows,
            sampling=config.farms_sampling,
            step_size=config.farms_step_size,
            orient_tall=config.farms_orient_tall,
            seed=config.seed,
            trim_upper=config.mp_trim_upper,
        )
    prepared = prepare_spectrum(matrix, config, variance=1.0 if variance is None else variance)
    if method == "kde_bulk_fit":
        return fit_marchenko_pastur_kde(
            prepared.eigenvalues,
            prepared.aspect_ratio,
            bandwidth=config.kde_bandwidth,
            trim_upper=config.mp_trim_upper,
        )
    return fit_marchenko_pastur(
        prepared.eigenvalues,
        prepared.aspect_ratio,
        variance=variance,
        trim_upper=config.mp_trim_upper,
    )


def dispatch_tail_solver(
    eigenvalues: np.ndarray,
    config: RMTMethodConfig = RMTMethodConfig(),
    **overrides: Any,
) -> dict[str, Any]:
    """Run the configured heavy-tail estimator."""

    parameters: dict[str, Any] = {
        "min_tail": config.tail_minimum,
        "tail_frac": config.tail_fraction,
        "tail_fraction": config.tail_fraction,
    }
    parameters.update(overrides)
    return select_tail_estimator(eigenvalues, config.tail_solver, **parameters)


def dispatch_unfolding(
    levels: np.ndarray,
    config: RMTMethodConfig = RMTMethodConfig(),
) -> np.ndarray:
    """Run the configured staircase smoother."""

    return unfold_spectrum(
        levels,
        method=config.unfolding_strategy,
        degree=config.polynomial_degree,
        smoothing=config.spline_smoothing,
        kernel_window=config.gaussian_kernel_window,
    )


def dispatch_overlap(
    weight_vectors: np.ndarray,
    activation_vectors: np.ndarray,
    config: RMTMethodConfig = RMTMethodConfig(),
) -> dict[str, object]:
    """Run the configured explicit-subspace alignment metric."""

    return evaluate_overlap_metric(
        weight_vectors,
        activation_vectors,
        metric=config.overlap_metric,
    )


def dispatch_spike_detector(
    weight: np.ndarray,
    config: RMTMethodConfig = RMTMethodConfig(),
    *,
    variance: float = 1.0,
) -> SpikeDetectionResult:
    """Run the configured right-spike detector on a rectangular factor."""

    matrix = _finite_weight(weight)
    eigenvalues = mp_eigenvalues(matrix)
    q = min(matrix.shape) / max(matrix.shape)
    if config.spike_detector == "tracy_widom_95":
        return detect_spikes_tracy_widom(
            eigenvalues,
            min(matrix.shape),
            max(matrix.shape),
            variance,
            confidence=0.95,
        )
    if config.spike_detector == "bbp_transition":
        return detect_spikes_bbp(eigenvalues, q, variance)
    detected = detect_spikes_from_factor(
        matrix,
        steps=config.lanczos_steps,
        n_probes=config.lanczos_probes,
        tail_window=config.lanczos_tail_window,
        threshold_c=config.lanczos_threshold_c,
        threshold_delta=config.lanczos_threshold_delta,
        residue_threshold=config.lanczos_residue_threshold,
        ridge=config.lanczos_ridge,
        adaptive=config.lanczos_adaptive,
        convergence_tolerance=config.lanczos_convergence_tolerance,
        sequence_length=config.lanczos_sequence_length,
        check_interval=config.lanczos_check_interval,
        pole_method=config.lanczos_pole_method,
        rng=config.seed,
    )
    return SpikeDetectionResult(
        method="lanczos_poles",
        threshold=detected.threshold,
        bulk_edge=detected.lambda_plus,
        spikes=detected.poles,
        indices=np.arange(detected.n_spikes, dtype=np.int64),
        diagnostics=detected.as_dict(),
    )


__all__ = [
    "ASPECT_RATIO_MODES",
    "MP_FIT_METHODS",
    "OVERLAP_METRICS",
    "PreparedSpectrum",
    "RMTMethodConfig",
    "SPIKE_DETECTORS",
    "TAIL_SOLVERS",
    "UNFOLDING_STRATEGIES",
    "dispatch_mp_fit",
    "dispatch_overlap",
    "dispatch_spike_detector",
    "dispatch_tail_solver",
    "dispatch_unfolding",
    "prepare_spectrum",
]
