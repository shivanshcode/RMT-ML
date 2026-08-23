"""Numerical random-matrix diagnostics for Spectral-Chinchilla."""

from .farms_aspect_ratio import (
    FARMSConfig,
    FARMSResult,
    farms_spectrum,
    shape_normalize_eigenvalues,
)
from .factory import (
    RMTMethodConfig,
    dispatch_mp_fit,
    dispatch_overlap,
    dispatch_spike_detector,
    dispatch_tail_solver,
    dispatch_unfolding,
    prepare_spectrum,
)
from .lanczos_stieltjes import (
    LanczosSpikeResult,
    asymptotic_spectral_density,
    detect_spikes_lanczos,
    finite_vest_poles,
    reference_modified_cholesky,
)
from .svd_result import SVDResult, compute_svd

__all__ = [
    "FARMSConfig",
    "FARMSResult",
    "LanczosSpikeResult",
    "RMTMethodConfig",
    "SVDResult",
    "compute_svd",
    "asymptotic_spectral_density",
    "dispatch_mp_fit",
    "dispatch_overlap",
    "dispatch_spike_detector",
    "dispatch_tail_solver",
    "dispatch_unfolding",
    "detect_spikes_lanczos",
    "finite_vest_poles",
    "farms_spectrum",
    "prepare_spectrum",
    "reference_modified_cholesky",
    "shape_normalize_eigenvalues",
]
__version__ = "0.3.0"
