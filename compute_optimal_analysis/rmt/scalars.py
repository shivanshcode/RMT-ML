"""Scalar matrix-complexity, entropy, and eigenvector-localization metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import kstest, norm

from .mp import mp_bounds


def _singular_values(weight: np.ndarray | None, s: np.ndarray | None) -> np.ndarray:
    if s is not None:
        values = np.asarray(s, dtype=np.float64).ravel()
    elif weight is not None:
        matrix = np.asarray(weight, dtype=np.float64)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("weight must be a finite two-dimensional matrix")
        values = np.linalg.svd(matrix, compute_uv=False)
    else:
        raise ValueError("provide either weight or s")
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("singular values must be a nonempty finite nonnegative array")
    return np.sort(values)[::-1]


def _log_base(base: float) -> float:
    base = float(base)
    if not np.isfinite(base) or base <= 0.0 or np.isclose(base, 1.0):
        raise ValueError("base must be positive and different from one")
    return float(np.log(base))


def spectral_norm(weight: np.ndarray | None = None, *, s: np.ndarray | None = None) -> float:
    return float(_singular_values(weight, s)[0])


def frobenius_norm(weight: np.ndarray | None = None, *, s: np.ndarray | None = None) -> float:
    values = _singular_values(weight, s)
    return float(np.linalg.norm(values))


def stable_rank(weight: np.ndarray | None = None, *, s: np.ndarray | None = None) -> float:
    """Return ``||W||_F^2 / ||W||_2^2`` with zero-matrix convention zero."""

    values = _singular_values(weight, s)
    largest = float(values[0])
    if largest == 0.0:
        return 0.0
    return float(np.sum(np.square(values)) / largest**2)


def condition_number(
    weight: np.ndarray | None = None,
    *,
    s: np.ndarray | None = None,
    rcond: float | None = None,
) -> float:
    """Return the reduced 2-norm condition number, or infinity when singular."""

    values = _singular_values(weight, s)
    cutoff = (
        np.finfo(np.float64).eps * max(values.size, 1) * values[0]
        if rcond is None
        else float(rcond) * values[0]
    )
    if cutoff < 0.0 or not np.isfinite(cutoff):
        raise ValueError("rcond must be finite and nonnegative")
    smallest = float(values[-1])
    return float("inf") if smallest <= cutoff else float(values[0] / smallest)


def hard_rank(
    weight: np.ndarray | None = None,
    *,
    s: np.ndarray | None = None,
    tolerance: float | None = None,
) -> int:
    """Return numerical rank from the reduced singular spectrum."""

    values = _singular_values(weight, s)
    threshold = (
        np.finfo(np.float64).eps * values.size * values[0]
        if tolerance is None
        else float(tolerance)
    )
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    return int(np.count_nonzero(values > threshold))


def spectral_entropy(svals: np.ndarray, base: float = np.e) -> float:
    """Return entropy of normalized squared singular values."""

    values = _singular_values(None, np.asarray(svals))
    energies = np.square(values)
    total = float(np.sum(energies))
    if total == 0.0:
        return 0.0
    probabilities = energies / total
    positive = probabilities > 0.0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])) / _log_base(base))


def von_neumann_entropy(
    weight: np.ndarray | None = None,
    *,
    s: np.ndarray | None = None,
    base: float = np.e,
) -> float:
    """Return matrix/von-Neumann entropy of ``WW^T / trace(WW^T)``."""

    return spectral_entropy(_singular_values(weight, s), base=base)


matrix_entropy = von_neumann_entropy


def normalized_matrix_entropy(
    weight: np.ndarray | None = None,
    *,
    s: np.ndarray | None = None,
) -> float:
    """Return singular-energy entropy divided by ``log(numerical rank)``."""

    values = _singular_values(weight, s)
    rank = hard_rank(s=values)
    if rank <= 1:
        return 0.0
    entropy = spectral_entropy(values)
    return float(entropy / np.log(rank))


def row_wise_entropy(weight: np.ndarray, base: float = np.e) -> float:
    """Return mean entropy of each row's normalized squared entries."""

    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("weight must be a finite two-dimensional matrix")
    energies = np.square(matrix)
    totals = np.sum(energies, axis=1, keepdims=True)
    probabilities = np.divide(energies, totals, out=np.zeros_like(energies), where=totals > 0.0)
    logs = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    logs[positive] = np.log(probabilities[positive])
    entropies = -np.sum(probabilities * logs, axis=1) / _log_base(base)
    return float(np.mean(entropies))


def ipr(vectors: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return inverse participation ratio after unit-normalizing each vector."""

    array = np.asarray(vectors)
    if array.ndim < 1 or not np.all(np.isfinite(array)):
        raise ValueError("vectors must be a finite array")
    squared = np.abs(array) ** 2
    norms = np.sum(squared, axis=axis, keepdims=True)
    normalized = np.divide(squared, norms, out=np.zeros_like(squared, dtype=np.float64), where=norms > 0.0)
    return np.sum(np.square(normalized), axis=axis)


def localization_ratio(vector: np.ndarray) -> float:
    """Return the archive's ``L1/L-infinity`` localization ratio."""

    values = np.asarray(vector)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("vector must be nonempty and finite")
    maximum = float(np.linalg.norm(values.ravel(), ord=np.inf))
    return 0.0 if maximum == 0.0 else float(np.linalg.norm(values.ravel(), ord=1) / maximum)


def participation_ratio(vector: np.ndarray) -> float:
    """Return the archive's ``L2/L4`` participation ratio convention."""

    values = np.asarray(vector)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("vector must be nonempty and finite")
    denominator = float(np.linalg.norm(values.ravel(), ord=4))
    return 0.0 if denominator == 0.0 else float(np.linalg.norm(values.ravel(), ord=2) / denominator)


def ipr_summary(Vh: np.ndarray, s: np.ndarray, n: int, m: int, sigma: float) -> dict[str, float]:
    """Summarize right-singular-vector localization at the top and MP bulk."""

    vectors = np.asarray(Vh)
    values = _singular_values(None, s)
    if vectors.ndim != 2 or vectors.shape[0] < values.size:
        raise ValueError("Vh is incompatible with s")
    localization = ipr(vectors[: values.size], axis=1)
    top_count = max(1, int(np.ceil(0.1 * values.size)))
    lower, upper = mp_bounds(n, m, sigma)
    bulk = (values >= lower) & (values <= upper)
    return {
        "ipr_top10_mean": float(np.mean(localization[:top_count])),
        "ipr_bulk_mean": float(np.mean(localization[bulk])) if np.any(bulk) else float("nan"),
    }


def porter_thomas_ks(Vh: np.ndarray, *, n_vectors: int | None = None) -> dict[str, float]:
    """Test normalized singular-vector entries against the Gaussian PT law."""

    vectors = np.asarray(Vh, dtype=np.float64)
    if vectors.ndim != 2 or not np.all(np.isfinite(vectors)):
        raise ValueError("Vh must be a finite two-dimensional array")
    count = vectors.shape[0] if n_vectors is None else min(int(n_vectors), vectors.shape[0])
    if count < 1:
        raise ValueError("n_vectors must select at least one vector")
    statistics: list[float] = []
    pvalues: list[float] = []
    for vector in vectors[:count]:
        normalized = vector / max(np.linalg.norm(vector), np.finfo(float).eps)
        statistic, pvalue = kstest(normalized * np.sqrt(vector.size), norm.cdf)
        statistics.append(float(statistic))
        pvalues.append(float(pvalue))
    return {
        "pt_ks_mean": float(np.mean(statistics)),
        "pt_frac_random": float(np.mean(np.asarray(pvalues) >= 0.05)),
    }


def porter_thomas_monte_carlo(
    vectors: np.ndarray,
    *,
    n_reference: int = 512,
    pooling_window: int = 1,
    rng: np.random.Generator | int | None = 0,
) -> dict[str, object]:
    """Calibrate normalized-vector KS distances by finite-dimensional simulation."""

    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 2 or not np.all(np.isfinite(array)):
        raise ValueError("vectors must be a finite vector or row-oriented matrix")
    references = int(n_reference)
    pool = int(pooling_window)
    if references < 32 or pool < 1:
        raise ValueError("n_reference must be at least 32 and pooling_window positive")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    dimension = array.shape[1]
    reference_vectors = generator.normal(size=(references, dimension))
    reference_vectors /= np.linalg.norm(reference_vectors, axis=1, keepdims=True)
    reference_cdf_x = np.sort(reference_vectors.ravel())
    reference_cdf_y = np.arange(reference_cdf_x.size, dtype=np.float64) / max(
        reference_cdf_x.size - 1, 1
    )

    def ks_distance(sample: np.ndarray) -> float:
        ordered = np.sort(sample.ravel())
        empirical = np.arange(ordered.size, dtype=np.float64) / max(ordered.size - 1, 1)
        expected = np.interp(ordered, reference_cdf_x, reference_cdf_y)
        return float(np.max(np.abs(expected - empirical)))

    calibration = np.empty(references, dtype=np.float64)
    for index in range(references):
        selected = generator.normal(size=(pool, dimension))
        selected /= np.linalg.norm(selected, axis=1, keepdims=True)
        calibration[index] = ks_distance(selected)
    observed_distances = np.empty(array.shape[0], dtype=np.float64)
    pvalues = np.empty(array.shape[0], dtype=np.float64)
    for index, vector in enumerate(array):
        normalized = vector / max(np.linalg.norm(vector), np.finfo(float).eps)
        observed_distances[index] = ks_distance(normalized)
        pvalues[index] = (1.0 + np.count_nonzero(calibration >= observed_distances[index])) / (
            references + 1.0
        )
    return {
        "distances": observed_distances,
        "pvalues": pvalues,
        "pt_ks_mean": float(np.mean(observed_distances)),
        "pt_frac_random": float(np.mean(pvalues >= 0.05)),
        "n_reference": references,
        "pooling_window": pool,
    }


def porter_thomas_monte_carlo_pooled(
    vectors: np.ndarray,
    *,
    n_reference: int = 512,
    pooling_window: int = 5,
    rng: np.random.Generator | int | None = 0,
) -> dict[str, object]:
    """Reproduce the archive's adjacent-vector pooled PT calibration."""

    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2 or not np.all(np.isfinite(array)):
        raise ValueError("vectors must be a finite row-oriented matrix")
    references = int(n_reference)
    window = int(pooling_window)
    if references < 32 or window < 1 or array.shape[0] <= 2 * window:
        raise ValueError(
            "n_reference must be at least 32 and vectors must contain more than twice the pooling window"
        )
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    dimension = array.shape[1]
    pooled_count = 2 * window
    reference_vectors = generator.normal(size=(references * pooled_count, dimension))
    reference_vectors /= np.linalg.norm(reference_vectors, axis=1, keepdims=True)
    reference_x = np.sort(reference_vectors.ravel())
    reference_y = np.arange(reference_x.size, dtype=np.float64) / max(reference_x.size - 1, 1)

    def distance(sample: np.ndarray) -> float:
        ordered = np.sort(sample.ravel())
        empirical = np.arange(ordered.size, dtype=np.float64) / max(ordered.size - 1, 1)
        expected = np.interp(ordered, reference_x, reference_y)
        return float(np.max(np.abs(expected - empirical)))

    calibration = np.empty(references, dtype=np.float64)
    for index in range(references):
        reference_pool = generator.normal(size=(pooled_count, dimension))
        reference_pool /= np.linalg.norm(reference_pool, axis=1, keepdims=True)
        calibration[index] = distance(reference_pool)
    centers = np.arange(window, array.shape[0] - window, dtype=np.int64)
    distances = np.empty(centers.size, dtype=np.float64)
    pvalues = np.empty(centers.size, dtype=np.float64)
    normalized = array / np.maximum(
        np.linalg.norm(array, axis=1, keepdims=True),
        np.finfo(float).eps,
    )
    for output_index, center in enumerate(centers):
        pooled = normalized[center - window : center + window]
        distances[output_index] = distance(pooled)
        pvalues[output_index] = (
            1.0 + np.count_nonzero(calibration >= distances[output_index])
        ) / (references + 1.0)
    return {
        "center_indices": centers,
        "distances": distances,
        "pvalues": pvalues,
        "pt_ks_mean": float(np.mean(distances)),
        "pt_frac_random": float(np.mean(pvalues >= 0.05)),
        "n_reference": references,
        "pooling_window": window,
    }


def mp_softrank(s: np.ndarray, nu_plus: float) -> float:
    values = _singular_values(None, s)
    edge = float(nu_plus)
    if not np.isfinite(edge) or edge < 0.0:
        return float("nan")
    return float("nan") if values[0] == 0.0 else float(edge / values[0])


def bulk_mass_frac(s: np.ndarray, nu_plus: float) -> float:
    values = _singular_values(None, s)
    edge = float(nu_plus)
    if not np.isfinite(edge) or edge < 0.0:
        return float("nan")
    energies = np.square(values)
    total = float(np.sum(energies))
    return 0.0 if total == 0.0 else float(np.sum(energies[values <= edge]) / total)


def decile_index_ranges(
    k: int,
    n_deciles: int = 10,
    ascending: bool = True,
) -> list[tuple[int, int]]:
    """Partition ``[0,k)`` into balanced half-open ranges."""

    k, n_deciles = int(k), int(n_deciles)
    if k < 0 or n_deciles < 1:
        raise ValueError("k must be nonnegative and n_deciles positive")
    boundaries = np.linspace(0, k, n_deciles + 1, dtype=int)
    ranges = [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(n_deciles)]
    if ascending:
        return ranges
    return [(k - end, k - start) for start, end in ranges]


def per_decile(s: np.ndarray, n_deciles: int = 10) -> dict[str, float]:
    """Return globally additive entropy and stable-rank contributions by tranche."""

    values = np.sort(_singular_values(None, s))
    energies = np.square(values)
    total = float(np.sum(energies))
    maximum = float(np.max(energies)) if energies.size else 0.0
    result: dict[str, float] = {}
    for index, (start, end) in enumerate(decile_index_ranges(values.size, n_deciles), start=1):
        tranche = energies[start:end]
        if total == 0.0 or tranche.size == 0:
            entropy_contribution = 0.0
        else:
            probabilities = tranche / total
            positive = probabilities > 0.0
            entropy_contribution = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        rank_contribution = 0.0 if maximum == 0.0 else float(np.sum(tranche) / maximum)
        result[f"entropy_decile_{index}"] = entropy_contribution
        result[f"srk_decile_{index}"] = rank_contribution
    return result


__all__ = [
    "bulk_mass_frac",
    "condition_number",
    "decile_index_ranges",
    "frobenius_norm",
    "hard_rank",
    "ipr",
    "ipr_summary",
    "localization_ratio",
    "matrix_entropy",
    "mp_softrank",
    "normalized_matrix_entropy",
    "per_decile",
    "participation_ratio",
    "porter_thomas_ks",
    "porter_thomas_monte_carlo",
    "porter_thomas_monte_carlo_pooled",
    "row_wise_entropy",
    "spectral_entropy",
    "spectral_norm",
    "stable_rank",
    "von_neumann_entropy",
]
