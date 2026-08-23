"""Central command-line schema for spectral-method selection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from rmt.factory import (
    ASPECT_RATIO_MODES,
    MP_FIT_METHODS,
    OVERLAP_METRICS,
    SPIKE_DETECTORS,
    TAIL_SOLVERS,
    UNFOLDING_STRATEGIES,
    RMTMethodConfig,
)


BOUNDARY_DETECTORS = (
    "analytic_mp",
    "tracy_widom",
    "weightwatcher_kde",
    "lanczos_stieltjes",
)

EXPERIMENT_MODES = (
    "reproduce_paper1",
    "reproduce_paper2",
    "reproduce_paper3",
    "compute_optimal_rmt",
    "custom",
)
MODEL_TYPES = ("causal_transformer",)
DEVICE_CHOICES = ("cuda", "cpu", "auto")
SVD_BACKENDS = ("auto", "cuda", "cpu")
SVD_DRIVERS = ("gesvdj", "gesvd", "gesvda", "default")


@dataclass(frozen=True)
class SpectralCLIConfig:
    """Serializable CLI-facing counterpart of :class:`RMTMethodConfig`."""

    mp_fit_method: str = "lanczos_stieltjes"
    unfolding_strategy: str = "spline_monotone"
    tail_solver: str = "clauset_mle"
    overlap_metric: str = "staats_dual_end"
    aspect_ratio_mode: str = "farms_normalized"
    spike_detector: str = "lanczos_poles"
    boundary_detector: str | None = None
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

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "SpectralCLIConfig":
        """Extract known spectral fields from an argparse namespace."""

        names = cls.__dataclass_fields__
        values = {
            name: getattr(namespace, name)
            for name in names
            if hasattr(namespace, name)
        }
        return cls(**values)

    def resolved_methods(self) -> tuple[str, str]:
        """Resolve the compatibility boundary flag into fit and spike methods."""

        if self.boundary_detector is None:
            return self.mp_fit_method, self.spike_detector
        mapping = {
            "analytic_mp": ("analytic_mp", self.spike_detector),
            "tracy_widom": ("analytic_mp", "tracy_widom_95"),
            "weightwatcher_kde": ("kde_bulk_fit", self.spike_detector),
            "lanczos_stieltjes": ("lanczos_stieltjes", "lanczos_poles"),
        }
        if self.boundary_detector not in mapping:
            raise ValueError(f"unsupported boundary detector: {self.boundary_detector}")
        return mapping[self.boundary_detector]

    def to_rmt_config(self) -> RMTMethodConfig:
        """Build the validated numerical-engine configuration."""

        mp_method, spike_method = self.resolved_methods()
        return RMTMethodConfig(
            mp_fit_method=mp_method,
            unfolding_strategy=self.unfolding_strategy,
            tail_solver=self.tail_solver,
            overlap_metric=self.overlap_metric,
            aspect_ratio_mode=self.aspect_ratio_mode,
            spike_detector=spike_method,
            polynomial_degree=self.polynomial_degree,
            spline_smoothing=self.spline_smoothing,
            gaussian_kernel_window=self.gaussian_kernel_window,
            mp_trim_upper=self.mp_trim_upper,
            kde_bandwidth=self.kde_bandwidth,
            tail_minimum=self.tail_minimum,
            tail_fraction=self.tail_fraction,
            farms_target_aspect_ratio=self.farms_target_aspect_ratio,
            farms_window_size=self.farms_window_size,
            farms_row_windows=self.farms_row_windows,
            farms_column_windows=self.farms_column_windows,
            farms_sampling=self.farms_sampling,
            farms_step_size=self.farms_step_size,
            farms_normalization=self.farms_normalization,
            farms_orient_tall=self.farms_orient_tall,
            lanczos_steps=self.lanczos_steps,
            lanczos_probes=self.lanczos_probes,
            lanczos_tail_window=self.lanczos_tail_window,
            lanczos_threshold_c=self.lanczos_threshold_c,
            lanczos_threshold_delta=self.lanczos_threshold_delta,
            lanczos_residue_threshold=self.lanczos_residue_threshold,
            lanczos_ridge=self.lanczos_ridge,
            lanczos_adaptive=self.lanczos_adaptive,
            lanczos_convergence_tolerance=self.lanczos_convergence_tolerance,
            lanczos_sequence_length=self.lanczos_sequence_length,
            lanczos_check_interval=self.lanczos_check_interval,
            lanczos_pole_method=self.lanczos_pole_method,
            seed=self.seed,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_rmt_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach every scientific-method switch to an existing parser."""

    group = parser.add_argument_group("spectral method selection")
    group.add_argument("--mp-fit-method", choices=MP_FIT_METHODS, default="lanczos_stieltjes")
    group.add_argument(
        "--unfolding-strategy",
        choices=UNFOLDING_STRATEGIES,
        default="spline_monotone",
    )
    group.add_argument("--tail-solver", choices=TAIL_SOLVERS, default="clauset_mle")
    group.add_argument(
        "--overlap-metric",
        "--overlap-mode",
        dest="overlap_metric",
        choices=OVERLAP_METRICS,
        default="staats_dual_end",
    )
    group.add_argument(
        "--aspect-ratio-mode",
        choices=ASPECT_RATIO_MODES,
        default="farms_normalized",
    )
    group.add_argument("--spike-detector", choices=SPIKE_DETECTORS, default="lanczos_poles")
    group.add_argument(
        "--boundary-detector",
        choices=BOUNDARY_DETECTORS,
        default=None,
        help="compatibility alias that jointly selects the MP fit and edge detector",
    )

    tuning = parser.add_argument_group("spectral method tuning")
    tuning.add_argument(
        "--polynomial-degree",
        "--unfolding-degree",
        dest="polynomial_degree",
        type=int,
        default=15,
    )
    tuning.add_argument("--spline-smoothing", type=float, default=None)
    tuning.add_argument("--gaussian-kernel-window", type=int, default=15)
    tuning.add_argument("--mp-trim-upper", type=float, default=0.1)
    tuning.add_argument("--kde-bandwidth", type=float, default=None)
    tuning.add_argument("--tail-minimum", type=int, default=50)
    tuning.add_argument("--tail-fraction", type=float, default=0.1)
    tuning.add_argument("--farms-target-aspect-ratio", type=float, default=1.0)
    tuning.add_argument("--farms-window-size", type=int, default=None)
    tuning.add_argument("--farms-row-windows", type=int, default=5)
    tuning.add_argument("--farms-column-windows", type=int, default=5)
    tuning.add_argument(
        "--farms-sampling",
        choices=("reference_fixed", "reference_sliding", "grid", "random"),
        default="reference_fixed",
    )
    tuning.add_argument("--farms-step-size", type=int, default=10)
    tuning.add_argument(
        "--farms-normalization",
        choices=("canonical", "raw", "trace"),
        default="canonical",
    )
    tuning.add_argument(
        "--farms-orient-tall",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    tuning.add_argument("--lanczos-steps", type=int, default=50)
    tuning.add_argument("--lanczos-probes", type=int, default=3)
    tuning.add_argument("--lanczos-tail-window", type=int, default=None)
    tuning.add_argument("--lanczos-threshold-c", type=float, default=1.0)
    tuning.add_argument("--lanczos-threshold-delta", type=float, default=0.25)
    tuning.add_argument("--lanczos-residue-threshold", type=float, default=0.0)
    tuning.add_argument("--lanczos-ridge", type=float, default=0.0)
    tuning.add_argument(
        "--lanczos-adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    tuning.add_argument("--lanczos-convergence-tolerance", type=float, default=None)
    tuning.add_argument("--lanczos-sequence-length", type=int, default=None)
    tuning.add_argument("--lanczos-check-interval", type=int, default=2)
    tuning.add_argument(
        "--lanczos-pole-method",
        choices=("reference_ritz", "constant_tail"),
        default="reference_ritz",
    )
    return parser


def rmt_config_from_namespace(namespace: argparse.Namespace) -> RMTMethodConfig:
    """Convert parsed command-line values into a validated engine config."""

    return SpectralCLIConfig.from_namespace(namespace).to_rmt_config()


def parse_rmt_args(arguments: Sequence[str] | None = None) -> RMTMethodConfig:
    """Parse only the spectral method switches for library-level use."""

    parser = add_rmt_cli_arguments(argparse.ArgumentParser(description=__doc__))
    namespace = parser.parse_args(None if arguments is None else list(arguments))
    return rmt_config_from_namespace(namespace)


def add_pipeline_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the complete offline experiment, hardware, and analysis schema."""

    control = parser.add_argument_group("pipeline control")
    control.add_argument(
        "--experiment-mode",
        choices=EXPERIMENT_MODES,
        default="compute_optimal_rmt",
    )
    control.add_argument("--model-type", choices=MODEL_TYPES, default="causal_transformer")
    control.add_argument("--output-dir", default="results")
    control.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="execute training instead of writing a manifest only",
    )
    control.add_argument(
        "--dataset-path",
        default="data/tokenized/wikitext-103-raw-v1_gpt2.npy",
        help="local NPY or NPZ token array; no network fallback is attempted",
    )
    control.add_argument("--cells", default="all")

    scaling = parser.add_argument_group("compute scaling")
    scaling.add_argument("--scaling-budget-flops", type=float, nargs="+", default=[1e16])
    scaling.add_argument(
        "--allocation-ratios",
        type=float,
        nargs="+",
        default=[0.25, 1.0, 4.0],
    )
    scaling.add_argument("--vocab-size", type=int, default=50257)
    scaling.add_argument("--parameter-cap", type=float, default=None)
    scaling.add_argument("--max-train-tokens", type=float, default=None)

    data = parser.add_argument_group("offline data loading")
    data.add_argument("--sequence-length", type=int, default=256)
    data.add_argument("--batch-size", type=int, default=8)
    data.add_argument("--dataloader-workers", type=int, default=4)
    data.add_argument("--prefetch-factor", type=int, default=2)
    data.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    training = parser.add_argument_group("training")
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--warmup-steps", type=int, default=100)
    training.add_argument("--gradient-clip", type=float, default=1.0)
    training.add_argument("--log-every", type=int, default=50)
    training.add_argument("--seed", type=int, default=0)

    hardware = parser.add_argument_group("hardware acceleration")
    hardware.add_argument("--device", choices=DEVICE_CHOICES, default="cuda")
    hardware.add_argument(
        "--amp-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    hardware.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    hardware.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    hardware.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    hardware.add_argument("--svd-backend", choices=SVD_BACKENDS, default="auto")
    hardware.add_argument("--svd-driver", choices=SVD_DRIVERS, default="gesvdj")
    hardware.add_argument(
        "--covariance-device",
        choices=DEVICE_CHOICES,
        default="auto",
    )
    hardware.add_argument(
        "--covariance-dtype",
        choices=("float32", "float64"),
        default="float32",
    )

    metrics = parser.add_argument_group("diagnostic execution")
    metrics.add_argument("--activation-batches", type=int, default=5)
    metrics.add_argument("--validation-batches", type=int, default=20)
    metrics.add_argument(
        "--activation-centered",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    metrics.add_argument(
        "--compute-activation-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    metrics.add_argument(
        "--compute-spacing-distribution",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    metrics.add_argument(
        "--compute-number-variance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    metrics.add_argument(
        "--compute-stable-rank",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    metrics.add_argument(
        "--compute-delta3",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    metrics.add_argument(
        "--compute-porter-thomas",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    metrics.add_argument("--brody-fit-method", choices=("mle", "cdf_nls"), default="mle")
    metrics.add_argument(
        "--number-variance-method",
        choices=("sliding", "monte_carlo"),
        default="sliding",
    )

    lesions = parser.add_argument_group("spectral lesioning")
    lesions.add_argument(
        "--run-spectral-lesioning",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    lesions.add_argument("--lesion-tranches", default="top,bulk,bottom")
    lesions.add_argument("--lesion-fraction", type=float, default=0.05)
    lesions.add_argument("--lesion-mode", choices=("count", "energy"), default="count")
    lesions.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    add_rmt_cli_arguments(parser)
    return parser


def parse_pipeline_args(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the complete production runner schema without external side effects."""

    parser = add_pipeline_cli_arguments(argparse.ArgumentParser(description=__doc__))
    return parser.parse_args(None if arguments is None else list(arguments))


add_spectral_method_arguments = add_rmt_cli_arguments
config_from_args = rmt_config_from_namespace


__all__ = [
    "BOUNDARY_DETECTORS",
    "DEVICE_CHOICES",
    "EXPERIMENT_MODES",
    "MODEL_TYPES",
    "SVD_BACKENDS",
    "SVD_DRIVERS",
    "SpectralCLIConfig",
    "add_pipeline_cli_arguments",
    "add_rmt_cli_arguments",
    "add_spectral_method_arguments",
    "config_from_args",
    "parse_rmt_args",
    "parse_pipeline_args",
    "rmt_config_from_namespace",
]
