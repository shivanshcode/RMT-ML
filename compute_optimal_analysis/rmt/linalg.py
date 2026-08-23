"""Compatibility SVD entry point for the numerical engine."""

from __future__ import annotations

import numpy as np

from .svd_result import SVDResult, compute_svd


def cached_svd(
    weight: np.ndarray,
    full_matrices: bool = False,
    *,
    backend: str = "auto",
) -> SVDResult:
    """Compute one SVD.

    The pure engine intentionally supports only the NumPy backend.  GPU model
    surgery lives in ``pipelines.spectral_lesioning`` so importing this module
    can never activate a model-framework dependency.
    """

    if backend not in {"auto", "numpy"}:
        raise ValueError("the pure RMT SVD backend must be 'auto' or 'numpy'")
    return compute_svd(weight, full_matrices=full_matrices)


__all__ = ["SVDResult", "cached_svd", "compute_svd"]
