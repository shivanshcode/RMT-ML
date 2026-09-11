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


def _covariance_tolerance(covariance: np.ndarray, eigenvalues: np.ndarray) -> float:
    source = np.asarray(covariance)
    source_dtype = source.dtype if np.issubdtype(source.dtype, np.floating) else np.dtype("float64")
    # Promotion for eigh does not improve the precision with which covariance
    # entries were accumulated.
    epsilon = np.finfo(source_dtype).eps
    scale = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    return float(epsilon * max(source.shape) * scale)


def _positive_covariance_rank(covariance: np.ndarray, eigenvalues: np.ndarray,
                              tolerance: float) -> int:
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    count = getattr(covariance, "observation_count", None)
    if count is not None:
        centered = bool(getattr(covariance, "centered", True))
        identifiable = max(0, int(count) - (1 if centered else 0))
        rank = min(rank, identifiable)
    return rank


def _weight_vectors_identifiable(svd: SVDResult) -> str | None:
    values = np.asarray(svd.s, dtype=np.float64)
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    epsilon = np.finfo(np.dtype(svd.factorization_dtype)).eps
    tolerance = epsilon * max(svd.n, svd.m) * scale
    if scale == 0.0 or np.any(values <= tolerance):
        return "weight has a nonidentifiable null singular subspace"
    if values.size > 1 and np.any(np.abs(np.diff(values)) <= tolerance):
        return "weight has an unresolved repeated singular-value subspace"
    return None


def weight_basis_status(svd: SVDResult) -> str:
    """Return ``available`` or an explicit singular-basis limitation."""

    reason = _weight_vectors_identifiable(svd)
    return "available" if reason is None else f"unavailable: {reason}"


def _positive_covariance_clusters(eigenvalues: np.ndarray, positive_rank: int,
                                  tolerance: float) -> bool:
    """Whether individually reported positive covariance modes are ambiguous."""

    positive = np.asarray(eigenvalues[:positive_rank], dtype=np.float64)
    return bool(positive.size > 1 and np.any(np.abs(np.diff(positive)) <= tolerance))


def _normalized_columns(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("vectors must be a finite two-dimensional array")
    norms = np.linalg.norm(array, axis=0)
    if np.any(norms == 0.0):
        raise ValueError("vectors must have nonzero norm")
    return array / norms


def _orthonormal_span(vectors: np.ndarray) -> np.ndarray:
    normalized = _normalized_columns(vectors)
    U, singular_values, _ = np.linalg.svd(normalized, full_matrices=False)
    if singular_values.size == 0:
        raise ValueError("subspace contains no vectors")
    tolerance = np.finfo(float).eps * max(normalized.shape) * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < 1:
        raise ValueError("subspace is numerically rank zero")
    return U[:, :rank]


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
    weight_q = _orthonormal_span(weight_basis)
    activation_q = _orthonormal_span(activation_basis)
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
    weight_q = _orthonormal_span(weight_basis)
    activation_q = _orthonormal_span(activation_basis)
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
    scale = float(np.max(np.abs(eigenvalues)))
    tolerance = _covariance_tolerance(covariance, eigenvalues)
    positive_rank = (_positive_covariance_rank(covariance, eigenvalues, tolerance)
                     if scale > 0.0 else 0)
    tranches = tranche_indices(
        svd.s.size,
        top_fraction=top_fraction,
        bottom_fraction=bottom_fraction,
    )
    weight_reason = _weight_vectors_identifiable(svd)
    covariance_reason = (
        "activation covariance is significantly indefinite"
        if np.any(eigenvalues < -tolerance) else None
    )
    unavailable_reason = covariance_reason or weight_reason
    if positive_rank == 0 or unavailable_reason is not None:
        reason = unavailable_reason or "activation covariance has numerical rank zero"
        return {
            "activation_eigenvalues": eigenvalues,
            "overlap_matrix": np.empty((svd.s.size, 0), dtype=np.float64),
            "top_alignment": float("nan"),
            "bulk_alignment": float("nan"),
            "bottom_alignment": float("nan"),
            "tranche_indices": tranches,
            "metric": str(metric),
            "available": False,
            "status": f"unavailable: {reason}",
            "activation_rank": positive_rank,
            "activation_observation_count": getattr(covariance, "observation_count", None),
            "activation_accumulation_dtype": getattr(covariance, "accumulation_dtype", str(np.asarray(covariance).dtype)),
        }
    signal_vectors = activation_vectors[:, :positive_rank]
    target_count = max(1, int(np.ceil(activation_fraction * positive_rank)))
    # Never split an unresolved positive-eigenvalue cluster.  Including the
    # complete cluster makes every subspace score invariant to eigensolver
    # rotations inside that cluster.
    while (
        target_count < positive_rank
        and abs(eigenvalues[target_count - 1] - eigenvalues[target_count])
        <= tolerance
    ):
        target_count += 1
    target = signal_vectors[:, :target_count]
    full_overlap = projection_overlap(svd.V[:, : svd.s.size], signal_vectors)
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
        "available": True,
        "status": "available",
        "activation_rank": positive_rank,
        "activation_target_dimension": target_count,
        "activation_observation_count": getattr(covariance, "observation_count", None),
        "activation_accumulation_dtype": getattr(covariance, "accumulation_dtype", str(np.asarray(covariance).dtype)),
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
    tolerance = _covariance_tolerance(feature_matrix, eigenvalues)
    positive_rank = _positive_covariance_rank(feature_matrix, eigenvalues, tolerance)
    reason = (_weight_vectors_identifiable(svd)
              or ("activation covariance is significantly indefinite"
                  if np.any(eigenvalues < -tolerance) else None)
              or ("activation covariance has numerical rank zero" if positive_rank == 0 else None)
              or ("activation covariance has an unresolved repeated positive eigenspace"
                  if _positive_covariance_clusters(eigenvalues, positive_rank, tolerance)
                  else None))
    if reason is not None:
        return {"svals": svd.s.copy(), "overlap": np.asarray([]),
                "overlap_matrix": np.empty((svd.s.size, 0)), "evals": eigenvalues,
                "available": False, "status": f"unavailable: {reason}",
                "activation_rank": positive_rank}
    activation_vectors = activation_vectors[:, :positive_rank]
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
        "evals": eigenvalues, "available": True, "status": "available",
        "activation_rank": positive_rank,
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
    tolerance = _covariance_tolerance(feature_matrix, eigenvalues)
    positive_rank = _positive_covariance_rank(feature_matrix, eigenvalues, tolerance)
    reason = (_weight_vectors_identifiable(svd)
              or ("activation covariance is significantly indefinite"
                  if np.any(eigenvalues < -tolerance) else None)
              or ("activation covariance has numerical rank zero" if positive_rank == 0 else None)
              or ("activation covariance has an unresolved repeated positive eigenspace"
                  if _positive_covariance_clusters(eigenvalues, positive_rank, tolerance)
                  else None))
    if reason is not None:
        return {"available": False, "status": f"unavailable: {reason}",
                "activation_rank": positive_rank}
    activation_vectors = activation_vectors[:, :positive_rank]
    eigenvalues = eigenvalues[:positive_rank]
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
        "available": True, "status": "available", "activation_rank": positive_rank,
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
