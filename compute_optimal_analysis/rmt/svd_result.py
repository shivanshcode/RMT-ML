"""Validated NumPy SVD container shared by the pure RMT modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_finite_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if min(array.shape) < 1:
        raise ValueError("matrix dimensions must be positive")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must contain only finite values")
    return np.asarray(array, dtype=np.float64)


@dataclass(frozen=True)
class SVDResult:
    """Reduced or full SVD with an explicit covariance normalization.

    Singular values are required to be nonnegative and descending.  The raw
    Gram eigenvalues are ``s**2`` and the covariance eigenvalues are
    ``s**2 / normalization``.  By default the normalization is the larger
    matrix dimension, which gives an MP aspect ratio in ``(0, 1]``.
    """

    U: np.ndarray
    s: np.ndarray
    Vh: np.ndarray
    n: int
    m: int
    normalization: float | None = None
    lambda_minus: float | None = None
    lambda_plus: float | None = None

    def __post_init__(self) -> None:
        U = np.asarray(self.U, dtype=np.float64)
        s = np.asarray(self.s, dtype=np.float64)
        Vh = np.asarray(self.Vh, dtype=np.float64)
        if self.n < 1 or self.m < 1:
            raise ValueError("n and m must be positive")
        if s.ndim != 1 or s.size != min(self.n, self.m):
            raise ValueError("s must contain min(n, m) singular values")
        if U.ndim != 2 or U.shape[0] != self.n or U.shape[1] < s.size:
            raise ValueError("U has an incompatible shape")
        if Vh.ndim != 2 or Vh.shape[1] != self.m or Vh.shape[0] < s.size:
            raise ValueError("Vh has an incompatible shape")
        if not (np.all(np.isfinite(U)) and np.all(np.isfinite(s)) and np.all(np.isfinite(Vh))):
            raise ValueError("SVD arrays must be finite")
        if np.any(s < 0.0) or np.any(np.diff(s) > 1e-12):
            raise ValueError("singular values must be nonnegative and descending")
        norm = float(max(self.n, self.m) if self.normalization is None else self.normalization)
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("normalization must be finite and positive")
        for bound in (self.lambda_minus, self.lambda_plus):
            if bound is not None and (not np.isfinite(bound) or bound < 0.0):
                raise ValueError("spectral bounds must be finite and nonnegative")
        if self.lambda_minus is not None and self.lambda_plus is not None:
            if self.lambda_minus > self.lambda_plus:
                raise ValueError("lambda_minus cannot exceed lambda_plus")
        object.__setattr__(self, "U", U)
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "Vh", Vh)
        object.__setattr__(self, "normalization", norm)

    @property
    def gamma(self) -> float:
        return self.aspect_ratio

    @property
    def aspect_ratio(self) -> float:
        return float(min(self.n, self.m) / max(self.n, self.m))

    @property
    def Q(self) -> float:
        return self.aspect_ratio

    @property
    def V(self) -> np.ndarray:
        return self.Vh.T

    @property
    def singular_values(self) -> np.ndarray:
        return self.s

    @property
    def eigenvalues(self) -> np.ndarray:
        return np.square(self.s)

    @property
    def covariance_eigenvalues(self) -> np.ndarray:
        return np.square(self.s) / float(self.normalization)

    @property
    def spectral_bounds(self) -> tuple[float | None, float | None]:
        return self.lambda_minus, self.lambda_plus

    def reconstruct(self) -> np.ndarray:
        k = self.s.size
        return (self.U[:, :k] * self.s) @ self.Vh[:k, :]


def compute_svd(
    matrix: np.ndarray,
    *,
    full_matrices: bool = False,
    normalization: float | None = None,
) -> SVDResult:
    """Compute a deterministic NumPy SVD and return the standard container."""

    array = _as_finite_matrix(matrix)
    U, s, Vh = np.linalg.svd(array, full_matrices=full_matrices)
    return SVDResult(
        U=U,
        s=s,
        Vh=Vh,
        n=array.shape[0],
        m=array.shape[1],
        normalization=normalization,
    )
