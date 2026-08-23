"""Sign-invariant weight/activation subspace alignment metrics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.linalg import subspace_angles
from scipy.special import erfc, erfcinv
from scipy.stats import spearmanr

from .mp import estimate_sigma_gd_median, mp_bounds
from .svd_result import SVDResult, compute_svd


def activation_eigensystem(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return covariance eigenvalues and eigenvectors in descending order."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def _normalized_columns(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("vectors must be a finite two-dimensional array")
    norms = np.linalg.norm(array, axis=0)
    if np.any(norms == 0.0):
        raise ValueError("vectors must have nonzero norm")
    return array / norms


def projection_overlap(
    weight_vectors: np.ndarray,
    activation_vectors: np.ndarray,
) -> np.ndarray:
    """Return ``|V_weight^T V_activation|^2`` for column-oriented bases."""

    weight_basis = _normalized_columns(weight_vectors)
    activation_basis = _normalized_columns(activation_vectors)
    if weight_basis.shape[0] != activation_basis.shape[0]:
        raise ValueError("weight and activation vectors must share ambient dimension")
    result = np.abs(weight_basis.T @ activation_basis) ** 2
    return np.clip(result, 0.0, 1.0)


def subspace_alignment(
    weight_vectors: np.ndarray,
    activation_vectors: np.ndarray,
) -> float:
    """Return the contract's mean pairwise squared projection density."""

    overlap = projection_overlap(weight_vectors, activation_vectors)
    return float(np.mean(overlap))


def principal_angle_spectrum(
    weight_vectors: np.ndarray,
    activation_vectors: np.ndarray,
) -> np.ndarray:
    """Return canonical principal angles between two column subspaces."""

    weight_basis = _normalized_columns(weight_vectors)
    activation_basis = _normalized_columns(activation_vectors)
    if weight_basis.shape[0] != activation_basis.shape[0]:
        raise ValueError("weight and activation vectors must share ambient dimension")
    weight_q = np.linalg.qr(weight_basis, mode="reduced")[0]
    activation_q = np.linalg.qr(activation_basis, mode="reduced")[0]
    return np.asarray(subspace_angles(weight_q, activation_q), dtype=np.float64)


def principal_angle_alignment(
    weight_vectors: np.ndarray,
    activation_vectors: np.ndarray,
) -> float:
    """Return mean squared canonical correlation from principal angles."""

    angles = principal_angle_spectrum(weight_vectors, activation_vectors)
    return float(np.mean(np.square(np.cos(angles))))


def frobenius_projection_overlap(
    weight_vectors: np.ndarray,
    activation_vectors: np.ndarray,
) -> float:
    """Return normalized projector trace overlap in ``[0, 1]``."""

    weight_basis = _normalized_columns(weight_vectors)
    activation_basis = _normalized_columns(activation_vectors)
    if weight_basis.shape[0] != activation_basis.shape[0]:
        raise ValueError("weight and activation vectors must share ambient dimension")
    weight_q = np.linalg.qr(weight_basis, mode="reduced")[0]
    activation_q = np.linalg.qr(activation_basis, mode="reduced")[0]
    denominator = min(weight_q.shape[1], activation_q.shape[1])
    if denominator < 1:
        raise ValueError("each subspace must contain at least one vector")
    score = np.linalg.norm(weight_q.T @ activation_q, ord="fro") ** 2 / denominator
    return float(np.clip(score, 0.0, 1.0))


def reference_max_cosine_overlap(
    weight_vectors: np.ndarray,
    activation_vectors: np.ndarray,
    *,
    absolute: bool = False,
    squared: bool = False,
) -> np.ndarray:
    """Return the archive's per-vector maximum cosine overlap convention.

    The defaults reproduce its sign-sensitive row maximum.  ``absolute`` or
    ``squared`` should be preferred for basis-sign-invariant analysis.
    """

    weight_basis = _normalized_columns(weight_vectors)
    activation_basis = _normalized_columns(activation_vectors)
    if weight_basis.shape[0] != activation_basis.shape[0]:
        raise ValueError("weight and activation vectors must share ambient dimension")
    cosine = weight_basis.T @ activation_basis
    if squared:
        cosine = np.square(np.abs(cosine))
    elif absolute:
        cosine = np.abs(cosine)
    return np.max(cosine, axis=1)


def evaluate_overlap_metric(
    weight_vectors: np.ndarray,
    activation_vectors: np.ndarray,
    *,
    metric: str = "staats_dual_end",
) -> dict[str, object]:
    """Dispatch a CLI overlap metric on two explicit column subspaces."""

    name = str(metric).lower()
    if name == "staats_dual_end":
        matrix = projection_overlap(weight_vectors, activation_vectors)
        return {"metric": name, "score": float(np.mean(matrix)), "overlap_matrix": matrix}
    if name == "subspace_principal_angles":
        angles = principal_angle_spectrum(weight_vectors, activation_vectors)
        return {
            "metric": name,
            "score": float(np.mean(np.square(np.cos(angles)))),
            "principal_angles": angles,
        }
    if name == "frobenius_projection":
        return {
            "metric": name,
            "score": frobenius_projection_overlap(weight_vectors, activation_vectors),
        }
    raise ValueError(
        "metric must be staats_dual_end, subspace_principal_angles, or frobenius_projection"
    )


def tranche_indices(
    n_values: int,
    *,
    top_fraction: float = 0.1,
    bottom_fraction: float = 0.1,
) -> dict[str, np.ndarray]:
    """Partition descending singular-value indices into top, bulk, and bottom."""

    n_values = int(n_values)
    top_fraction, bottom_fraction = float(top_fraction), float(bottom_fraction)
    if n_values < 3:
        raise ValueError("at least three singular values are required")
    if top_fraction <= 0.0 or bottom_fraction <= 0.0 or top_fraction + bottom_fraction >= 1.0:
        raise ValueError("fractions must be positive and sum to less than one")
    top_count = min(n_values - 2, max(1, int(np.ceil(top_fraction * n_values))))
    bottom_count = min(n_values - top_count - 1, max(1, int(np.ceil(bottom_fraction * n_values))))
    top = np.arange(0, top_count, dtype=int)
    bulk = np.arange(top_count, n_values - bottom_count, dtype=int)
    bottom = np.arange(n_values - bottom_count, n_values, dtype=int)
    return {"top": top, "bulk": bulk, "bottom": bottom}


def dual_end_alignment(
    svd: SVDResult,
    covariance: np.ndarray,
    *,
    activation_fraction: float = 0.1,
    top_fraction: float = 0.1,
    bottom_fraction: float = 0.1,
    metric: str = "staats_dual_end",
) -> dict[str, object]:
    """Compare each singular tranche with the dominant activation subspace."""

    eigenvalues, activation_vectors = activation_eigensystem(covariance)
    if svd.V.shape[0] != activation_vectors.shape[0]:
        raise ValueError("right singular vectors and covariance dimensions do not match")
    activation_fraction = float(activation_fraction)
    if not 0.0 < activation_fraction <= 1.0:
        raise ValueError("activation_fraction must lie in (0, 1]")
    target_count = max(1, int(np.ceil(activation_fraction * activation_vectors.shape[1])))
    target = activation_vectors[:, :target_count]
    tranches = tranche_indices(
        svd.s.size,
        top_fraction=top_fraction,
        bottom_fraction=bottom_fraction,
    )
    full_overlap = projection_overlap(svd.V[:, : svd.s.size], activation_vectors)
    scores = {
        name: float(
            evaluate_overlap_metric(
                svd.V[:, indices],
                target,
                metric=metric,
            )["score"]
        )
        for name, indices in tranches.items()
    }
    return {
        "activation_eigenvalues": eigenvalues,
        "overlap_matrix": full_overlap,
        "top_alignment": scores["top"],
        "bulk_alignment": scores["bulk"],
        "bottom_alignment": scores["bottom"],
        "tranche_indices": tranches,
        "metric": str(metric),
    }


def overlap_analysis(
    weight: np.ndarray | None,
    feature_matrix: np.ndarray,
    *,
    svd: SVDResult | None = None,
) -> dict[str, object]:
    """Return per-singular-vector activation overlap and fitted MP edges."""

    if svd is None:
        if weight is None:
            raise ValueError("weight is required when svd is omitted")
        svd = compute_svd(weight)
    eigenvalues, activation_vectors = activation_eigensystem(feature_matrix)
    right_vectors = svd.V[:, : svd.s.size]
    matrix = projection_overlap(right_vectors, activation_vectors)
    per_vector = np.max(matrix, axis=1)
    sigma = estimate_sigma_gd_median(s=svd.s, n=svd.n, m=svd.m)
    lower, upper = mp_bounds(svd.n, svd.m, sigma)
    return {
        "svals": svd.s.copy(),
        "overlap": per_vector,
        "overlap_matrix": matrix,
        "mp_min": lower,
        "mp_max": upper,
        "sigma_med": sigma,
        "right_outliers": svd.s > upper,
        "left_outliers": svd.s < lower,
        "evals": eigenvalues,
    }


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or second.size < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return 0.0
    result = float(spearmanr(first, second).statistic)
    return 0.0 if not np.isfinite(result) else result


def eigenvector_eigenvalue_coincidence(
    weight: np.ndarray | None,
    feature_matrix: np.ndarray,
    *,
    svd: SVDResult | None = None,
) -> dict[str, object]:
    """Return cosine coincidence map and rank-correlation summaries."""

    if svd is None:
        if weight is None:
            raise ValueError("weight is required when svd is omitted")
        svd = compute_svd(weight)
    eigenvalues, activation_vectors = activation_eigensystem(feature_matrix)
    right_vectors = _normalized_columns(svd.V[:, : svd.s.size])
    activation_vectors = _normalized_columns(activation_vectors)
    if right_vectors.shape[0] != activation_vectors.shape[0]:
        raise ValueError("right singular vectors and feature matrix dimensions do not match")
    cosine = np.abs(right_vectors.T @ activation_vectors)
    diagonal_size = min(cosine.shape)
    diagonal = np.diag(cosine[:diagonal_size, :diagonal_size])
    row_argmax = np.argmax(cosine, axis=1)
    diagonal_coincidence = float(np.mean(row_argmax[:diagonal_size] == np.arange(diagonal_size)))
    top_column = cosine[:, 0]
    top_row = cosine[0, :]
    return {
        "cos_matrix": cosine,
        "svals_desc": svd.s.copy(),
        "evals_desc": eigenvalues,
        "rho_top_eigenvector_vs_svals": _safe_spearman(top_column, svd.s),
        "rho_top_singular_vs_evals": _safe_spearman(top_row, eigenvalues[: top_row.size]),
        "rho_diag_vs_svals": _safe_spearman(diagonal, svd.s[:diagonal_size]),
        "max_overlap_with_top_eigenvector": float(np.max(top_column)),
        "argmax_singular_for_top_eigenvector": int(np.argmax(top_column)),
        "diagonal_coincidence": diagonal_coincidence,
    }


def three_sigma_band(N: int, sigma_level: float = 3.0) -> tuple[float, float]:
    """Return per-entry and extreme-value null thresholds for absolute overlaps."""

    N = int(N)
    sigma_level = float(sigma_level)
    if N < 2 or not np.isfinite(sigma_level) or sigma_level <= 0.0:
        raise ValueError("N must exceed one and sigma_level must be positive")
    per_entry = sigma_level / np.sqrt(N)
    family_tail = float(erfc(sigma_level / np.sqrt(2.0)))
    per_comparison_tail = 1.0 - (1.0 - family_tail) ** (1.0 / N)
    upper = np.sqrt(2.0) * float(erfcinv(per_comparison_tail)) / np.sqrt(N)
    return float(per_entry), float(min(1.0, upper))


def resolve_fm_key(record_name: str, fm_keys: Iterable[str]) -> str | None:
    """Resolve ordinary and fused-QKV weight names to activation-map keys."""

    base = str(record_name)
    for suffix in ("[Q]", "[K]", "[V]"):
        base = base.replace(suffix, "")
    if base.endswith(".weight"):
        base = base[: -len(".weight")]
    keys = list(fm_keys)
    if base in keys:
        return base
    matches = [key for key in keys if base.endswith(key) or key.endswith(base)]
    return matches[0] if len(matches) == 1 else None


__all__ = [
    "activation_eigensystem",
    "dual_end_alignment",
    "eigenvector_eigenvalue_coincidence",
    "evaluate_overlap_metric",
    "frobenius_projection_overlap",
    "overlap_analysis",
    "projection_overlap",
    "principal_angle_alignment",
    "principal_angle_spectrum",
    "reference_max_cosine_overlap",
    "resolve_fm_key",
    "subspace_alignment",
    "three_sigma_band",
    "tranche_indices",
]
