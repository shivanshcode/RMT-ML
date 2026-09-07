"""Lanczos-Cholesky asymptotic support and Stieltjes pole estimation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, TypeAlias

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import LinearOperator, aslinearoperator


RandomState: TypeAlias = np.random.Generator | int | None
Reorthogonalization: TypeAlias = Literal["none", "partial", "full"]
PoleMethod: TypeAlias = Literal["reference_ritz", "constant_tail"]


def _generator(rng: RandomState) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _unit_probe(dimension: int, probe: np.ndarray | None, rng: RandomState) -> np.ndarray:
    if probe is None:
        vector = _generator(rng).normal(size=dimension)
    else:
        vector = np.asarray(probe, dtype=np.float64).reshape(-1)
        if vector.size != dimension or not np.all(np.isfinite(vector)):
            raise ValueError("probe must be a finite vector matching the operator dimension")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("probe must have finite nonzero norm")
    return vector / norm


def _symmetric_operator(
    matrix: np.ndarray | LinearOperator | Callable[[np.ndarray], np.ndarray],
    dimension: int | None,
) -> tuple[LinearOperator, int]:
    if isinstance(matrix, np.ndarray):
        array = np.asarray(matrix, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 2:
            raise ValueError("matrix must be square with dimension at least two")
        if not np.all(np.isfinite(array)):
            raise ValueError("matrix must contain only finite values")
        tolerance = 1e-10 * max(1.0, float(np.linalg.norm(array, ord=np.inf)))
        if not np.allclose(array, array.T, atol=tolerance, rtol=1e-10):
            raise ValueError("matrix must be symmetric")
        return aslinearoperator(0.5 * (array + array.T)), int(array.shape[0])
    if isinstance(matrix, LinearOperator):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("operator must be square")
        return matrix, int(matrix.shape[0])
    if not callable(matrix) or dimension is None or int(dimension) < 2:
        raise ValueError("a callable operator requires dimension at least two")
    size = int(dimension)

    def matvec(vector: np.ndarray) -> np.ndarray:
        result = np.asarray(matrix(vector), dtype=np.float64).reshape(-1)
        if result.size != size or not np.all(np.isfinite(result)):
            raise ValueError("operator returned an incompatible or non-finite vector")
        return result

    return LinearOperator((size, size), matvec=matvec, rmatvec=matvec, dtype=np.float64), size


def covariance_linear_operator(
    factor: np.ndarray,
    *,
    normalization: float | None = None,
) -> LinearOperator:
    """Return the reduced covariance operator of a rectangular factor."""

    matrix = np.asarray(factor, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("factor must be a finite matrix with both dimensions at least two")
    oriented = matrix if matrix.shape[0] <= matrix.shape[1] else matrix.T
    denominator = float(max(matrix.shape) if normalization is None else normalization)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("normalization must be finite and positive")
    dimension = int(oriented.shape[0])

    def matvec(vector: np.ndarray) -> np.ndarray:
        return oriented @ (oriented.T @ vector) / denominator

    return LinearOperator(
        (dimension, dimension),
        matvec=matvec,
        rmatvec=matvec,
        dtype=np.float64,
    )


@dataclass(frozen=True)
class LanczosResult:
    """Finite Jacobi matrix generated from one vector spectral measure."""

    diagonal: np.ndarray
    off_diagonal: np.ndarray
    probe: np.ndarray
    iterations: int
    breakdown: bool
    orthogonality_error: float
    converged: bool = False
    convergence_iteration: int | None = None
    convergence_tolerance: float | None = None
    sequence_length: int | None = None
    check_interval: int | None = None
    basis: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        diagonal = np.asarray(self.diagonal, dtype=np.float64).ravel()
        off_diagonal = np.asarray(self.off_diagonal, dtype=np.float64).ravel()
        probe = np.asarray(self.probe, dtype=np.float64).ravel()
        if diagonal.size < 1 or off_diagonal.size != diagonal.size - 1:
            raise ValueError("off_diagonal must have one fewer entry than diagonal")
        if int(self.iterations) != diagonal.size:
            raise ValueError("iterations must equal the realized diagonal length")
        if not np.all(np.isfinite(diagonal)) or not np.all(np.isfinite(off_diagonal)):
            raise ValueError("Jacobi entries must be finite")
        if np.any(off_diagonal < 0.0):
            raise ValueError("Jacobi off-diagonal entries must be nonnegative")
        object.__setattr__(self, "diagonal", diagonal)
        object.__setattr__(self, "off_diagonal", off_diagonal)
        object.__setattr__(self, "probe", probe)
        if self.convergence_iteration is not None and not 1 <= int(self.convergence_iteration) <= diagonal.size:
            raise ValueError("convergence_iteration must index the realized recurrence")
        if self.convergence_tolerance is not None and (
            not np.isfinite(float(self.convergence_tolerance))
            or float(self.convergence_tolerance) <= 0.0
        ):
            raise ValueError("convergence_tolerance must be finite and positive")
        if self.sequence_length is not None and int(self.sequence_length) < 1:
            raise ValueError("sequence_length must be positive")
        if self.check_interval is not None and int(self.check_interval) < 1:
            raise ValueError("check_interval must be positive")

    def matrix(self) -> np.ndarray:
        result = np.diag(self.diagonal)
        if self.off_diagonal.size:
            result += np.diag(self.off_diagonal, 1) + np.diag(self.off_diagonal, -1)
        return result


@dataclass(frozen=True)
class JacobiCholesky:
    """Lower-bidiagonal Cholesky entries of a positive Jacobi matrix."""

    diagonal: np.ndarray
    sub_diagonal: np.ndarray
    ridge: float

    def __post_init__(self) -> None:
        diagonal = np.asarray(self.diagonal, dtype=np.float64).ravel()
        sub = np.asarray(self.sub_diagonal, dtype=np.float64).ravel()
        if diagonal.size < 1 or sub.size != diagonal.size - 1:
            raise ValueError("sub_diagonal must have one fewer entry than diagonal")
        if not np.all(np.isfinite(diagonal)) or not np.all(np.isfinite(sub)):
            raise ValueError("Cholesky entries must be finite")
        if np.any(diagonal <= 0.0) or np.any(sub < 0.0):
            raise ValueError("Cholesky entries must have positive diagonal and nonnegative sub-diagonal")
        object.__setattr__(self, "diagonal", diagonal)
        object.__setattr__(self, "sub_diagonal", sub)


@dataclass(frozen=True)
class LanczosSpikeResult:
    """Support, poles, residues, and convergence diagnostics."""

    lambda_minus: float
    lambda_plus: float
    poles: np.ndarray
    residues: np.ndarray
    n_spikes: int
    threshold: float
    tail_alpha: float
    tail_beta: float
    probe_counts: np.ndarray
    probe_edges: np.ndarray
    iterations: np.ndarray
    converged: bool
    probe_converged: np.ndarray
    pole_method: str
    representative_probe: int
    stieltjes_diagonal: np.ndarray = field(repr=False)
    stieltjes_sub_diagonal: np.ndarray = field(repr=False)
    probe_stieltjes_diagonals: Sequence[np.ndarray] = field(default_factory=tuple, repr=False)
    probe_stieltjes_sub_diagonals: Sequence[np.ndarray] = field(default_factory=tuple, repr=False)
    probe_tail_metadata: Sequence[dict[str, object]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        poles = np.asarray(self.poles, dtype=np.float64).ravel()
        residues = np.asarray(self.residues, dtype=np.float64).ravel()
        if poles.size != residues.size:
            raise ValueError("poles and residues must have equal length")
        if int(self.n_spikes) != poles.size:
            raise ValueError("n_spikes must equal the number of retained poles")
        if not np.all(np.isfinite(poles)) or not np.all(np.isfinite(residues)):
            raise ValueError("poles and residues must be finite")
        if np.any(residues < 0.0):
            raise ValueError("residues must be nonnegative")
        if (
            float(self.lambda_minus) < 0.0
            or float(self.lambda_plus) < float(self.lambda_minus)
            or float(self.lambda_plus) <= 0.0
        ):
            raise ValueError("support endpoints must be nonnegative and ordered")
        probe_counts = np.asarray(self.probe_counts, dtype=np.int64).ravel()
        probe_edges = np.asarray(self.probe_edges, dtype=np.float64)
        iterations = np.asarray(self.iterations, dtype=np.int64).ravel()
        probe_converged = np.asarray(self.probe_converged, dtype=bool).ravel()
        if probe_counts.size < 1 or np.any(probe_counts < 0):
            raise ValueError("probe_counts must contain nonnegative counts")
        if probe_edges.shape != (probe_counts.size, 2) or not np.all(np.isfinite(probe_edges)):
            raise ValueError("probe_edges must contain one finite endpoint pair per probe")
        if iterations.size != probe_counts.size or np.any(iterations < 1):
            raise ValueError("iterations must contain one positive value per probe")
        if probe_converged.size != probe_counts.size:
            raise ValueError("probe_converged must contain one flag per probe")
        object.__setattr__(self, "poles", poles)
        object.__setattr__(self, "residues", residues)
        object.__setattr__(self, "probe_counts", probe_counts)
        object.__setattr__(self, "probe_edges", probe_edges)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "probe_converged", probe_converged)
        if not 0 <= int(self.representative_probe) < probe_counts.size:
            raise ValueError("representative_probe must index the supplied probes")
        object.__setattr__(self, "stieltjes_diagonal", np.asarray(self.stieltjes_diagonal, dtype=np.float64))
        object.__setattr__(self, "stieltjes_sub_diagonal", np.asarray(self.stieltjes_sub_diagonal, dtype=np.float64))
        diagonals = tuple(
            np.asarray(values, dtype=np.float64)
            for values in self.probe_stieltjes_diagonals
        )
        sub_diagonals = tuple(
            np.asarray(values, dtype=np.float64)
            for values in self.probe_stieltjes_sub_diagonals
        )
        if len(diagonals) != len(sub_diagonals):
            raise ValueError("probe Stieltjes recurrence collections must have equal length")
        if diagonals and len(diagonals) != probe_counts.size:
            raise ValueError("probe Stieltjes recurrences must align with probe diagnostics")
        if self.probe_tail_metadata and len(self.probe_tail_metadata) != len(diagonals):
            raise ValueError("probe tail metadata must align with the recurrence collection")
        object.__setattr__(self, "probe_stieltjes_diagonals", diagonals)
        object.__setattr__(self, "probe_stieltjes_sub_diagonals", sub_diagonals)

    def stieltjes(self, z: np.ndarray | complex) -> np.ndarray | complex:
        if self.probe_stieltjes_diagonals:
            estimates = [
                extended_stieltjes_transform(
                    z,
                    diagonal,
                    sub_diagonal,
                    tail_alpha=self.tail_alpha,
                    tail_beta=self.tail_beta,
                )
                for diagonal, sub_diagonal in zip(
                    self.probe_stieltjes_diagonals,
                    self.probe_stieltjes_sub_diagonals,
                )
            ]
            averaged = np.mean(np.asarray(estimates), axis=0)
            return complex(averaged) if np.asarray(z).ndim == 0 else averaged
        return extended_stieltjes_transform(
            z,
            self.stieltjes_diagonal,
            self.stieltjes_sub_diagonal,
            tail_alpha=self.tail_alpha,
            tail_beta=self.tail_beta,
        )

    def density(self, energies: np.ndarray, *, eta: float = 1e-3) -> np.ndarray:
        return asymptotic_spectral_density(self, energies, eta=eta)

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("stieltjes_diagonal")
        payload.pop("stieltjes_sub_diagonal")
        payload.pop("probe_stieltjes_diagonals")
        payload.pop("probe_stieltjes_sub_diagonals")
        for name in (
            "poles",
            "residues",
            "probe_counts",
            "probe_edges",
            "iterations",
            "probe_converged",
        ):
            payload[name] = np.asarray(payload[name]).tolist()
        return payload


def default_lanczos_steps(dimension: int) -> int:
    """Return the released logarithmic, quarter-size, and root-size ceiling."""

    size = int(dimension)
    if size < 2:
        raise ValueError("dimension must be at least two")
    proposed = int(
        np.ceil(max(6.0 * np.log(size) + 24.0, size / 4.0, np.sqrt(size)))
    )
    return max(2, min(size - 1, proposed))


def lanczos_tridiagonalize(
    matrix: np.ndarray | LinearOperator | Callable[[np.ndarray], np.ndarray],
    *,
    dimension: int | None = None,
    steps: int | None = None,
    probe: np.ndarray | None = None,
    reorthogonalization: Reorthogonalization = "full",
    tolerance: float | None = None,
    adaptive: bool = False,
    convergence_tolerance: float | None = None,
    sequence_length: int | None = None,
    check_interval: int = 2,
    rng: RandomState = 0,
    return_basis: bool = False,
) -> LanczosResult:
    """Generate a symmetric tridiagonal projection with optional adaptive stopping.

    Adaptive stopping mirrors the released reference: two consecutive windows
    of Jacobi diagonal/off-diagonal means and sample deviations must agree
    within an absolute tolerance. Full reorthogonalization performs two
    projection passes, matching the reference implementation.
    """

    operator, size = _symmetric_operator(matrix, dimension)
    iterations = default_lanczos_steps(size) if steps is None else int(steps)
    if not 2 <= iterations <= size:
        raise ValueError("steps must lie between two and the operator dimension")
    if reorthogonalization not in {"none", "partial", "full"}:
        raise ValueError("reorthogonalization must be none, partial, or full")
    threshold = (
        np.finfo(np.float64).eps * max(1.0, float(size))
        if tolerance is None
        else float(tolerance)
    )
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    window = (
        max(1, int(np.floor(np.log(size) / 2.0)))
        if sequence_length is None
        else int(sequence_length)
    )
    interval = int(check_interval)
    if window < 1 or window >= iterations:
        if adaptive:
            raise ValueError("sequence_length must lie between one and steps minus one")
        window = min(max(1, iterations - 1), window)
    if interval < 1:
        raise ValueError("check_interval must be positive")
    convergence_threshold = (
        None if convergence_tolerance is None else float(convergence_tolerance)
    )
    if convergence_threshold is None and isinstance(matrix, np.ndarray):
        dense = np.asarray(matrix, dtype=np.float64)
        convergence_threshold = max(
            3.0 * float(np.sum(np.abs(np.diag(dense)))) / (size * np.sqrt(size)),
            np.finfo(float).eps,
        )
    if convergence_threshold is not None and (
        not np.isfinite(convergence_threshold) or convergence_threshold <= 0.0
    ):
        raise ValueError("convergence_tolerance must be finite and positive")
    initial = _unit_probe(size, probe, rng)
    basis = np.zeros((size, iterations), dtype=np.float64)
    basis[:, 0] = initial
    diagonal: list[float] = []
    off_diagonal: list[float] = []
    previous = np.zeros(size, dtype=np.float64)
    current = initial.copy()
    previous_beta = 0.0
    breakdown = False
    converged = False
    convergence_iteration: int | None = None
    previous_statistics: tuple[float, float, float, float] | None = (0.0, 0.0, 0.0, 0.0)
    next_check = window
    stop_after_diagonal = False
    used = 0
    for index in range(iterations):
        residual = np.asarray(operator.matvec(current), dtype=np.float64).reshape(-1)
        if residual.size != size or not np.all(np.isfinite(residual)):
            raise ValueError("operator returned an incompatible or non-finite vector")
        if index:
            residual -= previous_beta * previous
        alpha = float(current @ residual)
        residual -= alpha * current
        active_basis = basis[:, : index + 1]
        if reorthogonalization == "full":
            for _ in range(2):
                residual -= active_basis @ (active_basis.T @ residual)
        elif reorthogonalization == "partial":
            projections = active_basis.T @ residual
            if np.max(np.abs(projections), initial=0.0) > np.sqrt(np.finfo(float).eps) * max(
                1.0, float(np.linalg.norm(residual))
            ):
                residual -= active_basis @ projections
        beta = float(np.linalg.norm(residual))
        diagonal.append(alpha)
        used = index + 1
        if stop_after_diagonal:
            break
        if index == iterations - 1:
            break
        if not np.isfinite(beta) or beta <= threshold:
            breakdown = True
            break
        off_diagonal.append(beta)
        if adaptive and used >= next_check:
            diagonal_segment = np.asarray(diagonal[-window:], dtype=np.float64)
            off_segment = np.asarray(off_diagonal[-window:], dtype=np.float64)
            diagonal_mean = float(np.mean(diagonal_segment))
            off_mean = float(np.mean(off_segment))
            diagonal_deviation = (
                float(np.std(diagonal_segment, ddof=1))
                if diagonal_segment.size > 1
                else 0.0
            )
            off_deviation = (
                float(np.std(off_segment, ddof=1))
                if off_segment.size > 1
                else 0.0
            )
            active_tolerance = (
                3.0 * max(abs(diagonal_mean), np.finfo(float).eps) / np.sqrt(size)
                if convergence_threshold is None
                else convergence_threshold
            )
            statistics = (
                diagonal_mean,
                off_mean,
                diagonal_deviation,
                off_deviation,
            )
            if previous_statistics is not None:
                previous_mean, previous_off, previous_dstd, previous_ostd = previous_statistics
                converged = bool(
                    diagonal_deviation < active_tolerance
                    and off_deviation < active_tolerance
                    and previous_dstd < active_tolerance
                    and previous_ostd < active_tolerance
                    and abs(diagonal_mean - previous_mean) < active_tolerance
                    and abs(off_mean - previous_off) < active_tolerance
                )
            previous_statistics = statistics
            next_check += interval
            if converged:
                convergence_iteration = used
                convergence_threshold = active_tolerance
                stop_after_diagonal = True
        previous, current = current, residual / beta
        previous_beta = beta
        basis[:, index + 1] = current
    used_basis = basis[:, :used]
    gram = used_basis.T @ used_basis
    orthogonality_error = float(np.linalg.norm(gram - np.eye(used), ord=np.inf))
    return LanczosResult(
        diagonal=np.asarray(diagonal, dtype=np.float64),
        off_diagonal=np.asarray(off_diagonal[: max(0, used - 1)], dtype=np.float64),
        probe=initial,
        iterations=used,
        breakdown=breakdown,
        orthogonality_error=orthogonality_error,
        converged=converged,
        convergence_iteration=convergence_iteration,
        convergence_tolerance=convergence_threshold,
        sequence_length=window,
        check_interval=interval,
        basis=used_basis.copy() if return_basis else None,
    )


def jacobi_cholesky(
    lanczos: LanczosResult,
    *,
    ridge: float = 0.0,
    pivot_tolerance: float = 1e-14,
) -> JacobiCholesky:
    """Cholesky-factorize a positive Jacobi matrix in bidiagonal form."""

    ridge = float(ridge)
    pivot_tolerance = float(pivot_tolerance)
    if ridge < 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative")
    if pivot_tolerance < 0.0 or not np.isfinite(pivot_tolerance):
        raise ValueError("pivot_tolerance must be finite and nonnegative")
    source_diagonal = lanczos.diagonal + ridge
    diagonal = np.empty_like(source_diagonal)
    sub = np.empty_like(lanczos.off_diagonal)
    first = float(source_diagonal[0])
    if first <= pivot_tolerance:
        raise ValueError("Jacobi matrix is not numerically positive definite")
    diagonal[0] = np.sqrt(first)
    for index, off_value in enumerate(lanczos.off_diagonal):
        sub[index] = off_value / diagonal[index]
        pivot = float(source_diagonal[index + 1] - sub[index] ** 2)
        if pivot <= pivot_tolerance:
            raise ValueError("Jacobi matrix is not numerically positive definite")
        diagonal[index + 1] = np.sqrt(pivot)
    return JacobiCholesky(diagonal=diagonal, sub_diagonal=sub, ridge=ridge)


def estimate_constant_tail(
    cholesky: JacobiCholesky,
    *,
    tail_window: int | None = None,
) -> tuple[float, float, float, float]:
    """Estimate constant Cholesky entries and their relative scatter."""

    available = min(cholesky.diagonal.size, cholesky.sub_diagonal.size)
    if available < 2:
        raise ValueError("at least three Lanczos iterations are required for tail estimation")
    width = (
        max(4, int(np.ceil(np.log(cholesky.diagonal.size))))
        if tail_window is None
        else int(tail_window)
    )
    width = min(width, available)
    if width < 2:
        raise ValueError("tail_window must select at least two entries")
    diagonal_tail = cholesky.diagonal[-width:]
    sub_tail = cholesky.sub_diagonal[-width:]
    alpha = float(np.mean(diagonal_tail))
    beta = float(np.mean(sub_tail))
    alpha_scatter = float(np.std(diagonal_tail, ddof=1) / max(alpha, np.finfo(float).eps))
    beta_scatter = float(np.std(sub_tail, ddof=1) / max(beta, np.finfo(float).eps))
    return alpha, beta, alpha_scatter, beta_scatter


def reference_modified_cholesky(
    lanczos: LanczosResult,
    *,
    tail_window: int | None = None,
    ridge: float = 0.0,
) -> tuple[JacobiCholesky, dict[str, float | int | str]]:
    """Construct the released reference's two-entry constant Jacobi tail.

    The stable suffix is averaged in Jacobi coordinates, two terminal
    diagonal entries and one terminal off-diagonal entry are appended, and the
    shortened matrix is then Cholesky-factorized. This ordering matters at
    finite iteration count.
    """

    used = lanczos.diagonal.size
    available = lanczos.off_diagonal.size
    if available < 2 or used < 3:
        raise ValueError("at least three Lanczos iterations are required")
    tolerance = lanczos.convergence_tolerance
    if tolerance is None:
        tolerance = 3.0 * float(np.sum(np.abs(lanczos.diagonal))) / (
            used * np.sqrt(max(lanczos.probe.size, 1))
        )
    tolerance = max(float(tolerance), np.finfo(float).eps)
    if tail_window is not None:
        width = min(int(tail_window), available)
        if width < 2:
            raise ValueError("tail_window must select at least two entries")
        cut = max(0, used - width)
        diagonal_tail = lanczos.diagonal[cut:]
        off_tail = lanczos.off_diagonal[cut:]
        search_backwards = False
        tail_mode = "fixed_window"
    elif lanczos.converged:
        sequence = int(lanczos.sequence_length or max(1, np.floor(np.log(lanczos.probe.size) / 2.0)))
        jump = int(lanczos.check_interval or 2)
        cut = max(1, used - jump - sequence - 1)
        diagonal_tail = lanczos.diagonal[cut:]
        off_tail = lanczos.off_diagonal[cut:]
        search_backwards = True
        tail_mode = "reference_converged"
    else:
        cut = max(0, used - 2)
        diagonal_tail = lanczos.diagonal[used - 2 : used - 1]
        off_tail = lanczos.off_diagonal[used - 2 : used - 1]
        search_backwards = True
        tail_mode = "reference_ceiling"
    if diagonal_tail.size == 0 or off_tail.size == 0:
        raise ValueError("the realized recurrence has no estimable asymptotic tail")
    comparison_diagonal = float(np.mean(diagonal_tail))
    comparison_off = float(np.mean(off_tail))
    diagonal_values = list(np.asarray(diagonal_tail, dtype=np.float64))
    off_values = list(np.asarray(off_tail, dtype=np.float64))
    minimum_cut = 1 if lanczos.converged else 0
    while search_backwards and cut > minimum_cut:
        candidate_diagonal = float(lanczos.diagonal[cut - 1])
        candidate_off = float(lanczos.off_diagonal[cut - 1])
        if (
            abs(candidate_diagonal - comparison_diagonal) >= tolerance
            or abs(candidate_off - comparison_off) >= tolerance
        ):
            break
        diagonal_values.insert(0, candidate_diagonal)
        off_values.insert(0, candidate_off)
        cut -= 1
    diagonal_mean = float(np.mean(diagonal_values))
    off_mean = float(np.mean(off_values))
    modified_diagonal = np.concatenate(
        (lanczos.diagonal[:cut], np.asarray([diagonal_mean, diagonal_mean]))
    )
    modified_off = np.concatenate(
        (lanczos.off_diagonal[:cut], np.asarray([off_mean]))
    )
    modified = LanczosResult(
        diagonal=modified_diagonal,
        off_diagonal=modified_off,
        probe=lanczos.probe,
        iterations=int(modified_diagonal.size),
        breakdown=lanczos.breakdown,
        orthogonality_error=lanczos.orthogonality_error,
        converged=lanczos.converged,
        convergence_iteration=None,
        convergence_tolerance=lanczos.convergence_tolerance,
        sequence_length=lanczos.sequence_length,
        check_interval=lanczos.check_interval,
    )
    cholesky = jacobi_cholesky(modified, ridge=ridge)
    return cholesky, {
        "tail_start": cut,
        "tail_window": len(diagonal_values),
        "tail_mode": tail_mode,
        "tail_tolerance": tolerance,
        "jacobi_diagonal_mean": diagonal_mean,
        "jacobi_off_diagonal_mean": off_mean,
    }


def support_from_cholesky_tail(alpha: float, beta: float) -> tuple[float, float]:
    """Return one-cut support endpoints from limiting bidiagonal entries."""

    alpha, beta = float(alpha), float(beta)
    if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0.0 or beta < 0.0:
        raise ValueError("alpha must be positive and beta nonnegative")
    return float((alpha - beta) ** 2), float((alpha + beta) ** 2)


def vector_empirical_stieltjes(
    lanczos: LanczosResult,
    z: np.ndarray | complex,
) -> np.ndarray | complex:
    """Evaluate the finite vector empirical Stieltjes transform."""

    spectral_parameter = np.asarray(z, dtype=np.complex128)
    eigenvalues, eigenvectors = eigh_tridiagonal(lanczos.diagonal, lanczos.off_diagonal)
    weights = np.square(eigenvectors[0, :])
    expanded_parameter = np.expand_dims(spectral_parameter, axis=-1)
    result = np.sum(weights / (eigenvalues - expanded_parameter), axis=-1)
    return complex(result) if spectral_parameter.ndim == 0 else result


def _constant_tail_transform(
    z: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    lower, upper = support_from_cholesky_tail(alpha, beta)
    if beta == 0.0:
        return 1.0 / (alpha**2 - z)
    root = np.lib.scimath.sqrt(z - upper) * np.lib.scimath.sqrt(z - lower)
    return (alpha**2 - z - beta**2 + root) / (2.0 * z * beta**2)


def extended_stieltjes_transform(
    z: np.ndarray | complex,
    diagonal: np.ndarray,
    sub_diagonal: np.ndarray,
    *,
    tail_alpha: float,
    tail_beta: float,
) -> np.ndarray | complex:
    """Evaluate the paper's constant-tail continued-fraction transform."""

    spectral_parameter = np.asarray(z, dtype=np.complex128)
    alpha_entries = np.asarray(diagonal, dtype=np.float64).ravel()
    beta_entries = np.asarray(sub_diagonal, dtype=np.float64).ravel()
    if alpha_entries.size < 1 or beta_entries.size != alpha_entries.size - 1:
        raise ValueError("bidiagonal entries have incompatible sizes")
    if not np.all(np.isfinite(alpha_entries)) or not np.all(np.isfinite(beta_entries)):
        raise ValueError("bidiagonal entries must be finite")
    transform = _constant_tail_transform(spectral_parameter, float(tail_alpha), float(tail_beta))
    for index in range(alpha_entries.size - 2, -1, -1):
        alpha = alpha_entries[index]
        beta = beta_entries[index]
        correction = alpha**2 * beta**2 * transform / (1.0 + beta**2 * transform)
        transform = 1.0 / (alpha**2 - spectral_parameter - correction)
    return complex(transform) if spectral_parameter.ndim == 0 else transform


def asymptotic_spectral_density(
    result: LanczosSpikeResult,
    energies: np.ndarray,
    *,
    eta: float = 1e-3,
) -> np.ndarray:
    """Evaluate the smoothed constant-tail density as ``Im s(E+i eta)/pi``."""

    grid = np.asarray(energies, dtype=np.float64)
    broadening = float(eta)
    if not np.all(np.isfinite(grid)):
        raise ValueError("energies must be finite")
    if not np.isfinite(broadening) or broadening <= 0.0:
        raise ValueError("eta must be finite and positive")
    transform = np.asarray(result.stieltjes(grid + 1j * broadening), dtype=np.complex128)
    return np.maximum(np.imag(transform) / np.pi, 0.0)


def _extended_jacobi(
    cholesky: JacobiCholesky,
    tail_alpha: float,
    tail_beta: float,
    *,
    tail_window: int,
    extension_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_size = cholesky.diagonal.size
    length = max(int(extension_size), base_size + 8)
    cut = max(1, base_size - int(tail_window))
    diagonal_l = np.full(length, float(tail_alpha), dtype=np.float64)
    sub_l = np.full(length - 1, float(tail_beta), dtype=np.float64)
    diagonal_l[:cut] = cholesky.diagonal[:cut]
    sub_l[:cut] = cholesky.sub_diagonal[:cut]
    jacobi_diagonal = np.square(diagonal_l)
    jacobi_diagonal[1:] += np.square(sub_l)
    jacobi_off = diagonal_l[:-1] * sub_l
    return jacobi_diagonal, jacobi_off, diagonal_l[:cut], sub_l[:cut]


def finite_section_poles(
    cholesky: JacobiCholesky,
    *,
    tail_alpha: float,
    tail_beta: float,
    threshold: float,
    residue_threshold: float = 0.0,
    tail_window: int | None = None,
    extension_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate right poles from a finite section of the constant-tail operator."""

    residue_threshold = float(residue_threshold)
    if not np.isfinite(residue_threshold) or residue_threshold < 0.0:
        raise ValueError("residue_threshold must be finite and nonnegative")
    available = min(cholesky.diagonal.size, cholesky.sub_diagonal.size)
    width = (
        max(4, int(np.ceil(np.log(cholesky.diagonal.size))))
        if tail_window is None
        else int(tail_window)
    )
    width = min(width, available)
    if width < 2:
        raise ValueError("tail_window must select at least two entries")
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    extension = max(4 * cholesky.diagonal.size, cholesky.diagonal.size + 128) if extension_size is None else int(extension_size)
    diagonal, off, _, _ = _extended_jacobi(
        cholesky,
        tail_alpha,
        tail_beta,
        tail_window=width,
        extension_size=extension,
    )
    eigenvalues, eigenvectors = eigh_tridiagonal(diagonal, off)
    residues = np.square(eigenvectors[0, :])
    mask = (eigenvalues > threshold) & (residues > residue_threshold)
    poles = eigenvalues[mask]
    selected_residues = residues[mask]
    order = np.argsort(poles)[::-1]
    return poles[order], selected_residues[order]


def finite_vest_poles(
    lanczos: LanczosResult,
    *,
    threshold: float,
    residue_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract finite VEST Ritz poles and first-component residues.

    This is the released detector's finite-Jacobi spike rule augmented with an
    optional residue floor. Setting the floor to zero reproduces its counting
    behavior while retaining the VEST weights for diagnostics.
    """

    edge = float(threshold)
    residue_floor = float(residue_threshold)
    if not np.isfinite(edge):
        raise ValueError("threshold must be finite")
    if not np.isfinite(residue_floor) or residue_floor < 0.0:
        raise ValueError("residue_threshold must be finite and nonnegative")
    eigenvalues, eigenvectors = eigh_tridiagonal(
        lanczos.diagonal,
        lanczos.off_diagonal,
    )
    residues = np.square(eigenvectors[0, :])
    mask = (eigenvalues > edge) & (residues > residue_floor)
    poles = eigenvalues[mask]
    selected_residues = residues[mask]
    order = np.argsort(poles)[::-1]
    return poles[order], selected_residues[order]


def _mode_count(counts: np.ndarray) -> int:
    values, frequencies = np.unique(np.asarray(counts, dtype=np.int64), return_counts=True)
    winners = values[frequencies == np.max(frequencies)]
    if winners.size == 1:
        return int(winners[0])
    mean = float(np.mean(counts))
    return int(winners[np.argmin(np.abs(winners - mean))])


def detect_spikes_lanczos(
    matrix: np.ndarray | LinearOperator | Callable[[np.ndarray], np.ndarray],
    *,
    dimension: int | None = None,
    steps: int | None = None,
    n_probes: int = 1,
    reorthogonalization: Reorthogonalization = "full",
    tail_window: int | None = None,
    threshold_c: float = 1.0,
    threshold_delta: float = 0.25,
    residue_threshold: float = 0.0,
    ridge: float = 0.0,
    extension_size: int | None = None,
    adaptive: bool = True,
    convergence_tolerance: float | None = None,
    sequence_length: int | None = None,
    check_interval: int = 2,
    pole_method: PoleMethod = "reference_ritz",
    rng: RandomState = 0,
) -> LanczosSpikeResult:
    """Estimate one-cut support and separated right poles from random probes."""

    operator, size = _symmetric_operator(matrix, dimension)
    if steps is not None and (int(steps) < 3 or int(steps) > size):
        raise ValueError(f"steps must satisfy 3 <= steps <= operator dimension ({size})")
    active_convergence_tolerance = convergence_tolerance
    if active_convergence_tolerance is None and isinstance(matrix, np.ndarray):
        dense = np.asarray(matrix, dtype=np.float64)
        active_convergence_tolerance = max(
            3.0 * float(np.sum(np.abs(np.diag(dense)))) / (size * np.sqrt(size)),
            np.finfo(float).eps,
        )
    probes = int(n_probes)
    if probes < 1:
        raise ValueError("n_probes must be positive")
    threshold_c = float(threshold_c)
    threshold_delta = float(threshold_delta)
    if not np.isfinite(threshold_c) or threshold_c < 0.0:
        raise ValueError("threshold_c must be finite and nonnegative")
    if not np.isfinite(threshold_delta) or not 0.0 < threshold_delta < 0.5:
        raise ValueError("threshold_delta must lie in (0, 0.5)")
    if pole_method not in {"reference_ritz", "constant_tail"}:
        raise ValueError("pole_method must be reference_ritz or constant_tail")
    generator = _generator(rng)
    lanczos_results: list[LanczosResult] = []
    cholesky_results: list[JacobiCholesky] = []
    modified_cholesky_results: list[JacobiCholesky] = []
    tail_estimates: list[tuple[float, float, float, float]] = []
    tail_metadata: list[dict[str, object]] = []
    for _ in range(probes):
        initial = _unit_probe(size, None, generator)
        lanczos = lanczos_tridiagonalize(
            operator,
            steps=steps,
            probe=initial,
            reorthogonalization=reorthogonalization,
            adaptive=adaptive,
            convergence_tolerance=active_convergence_tolerance,
            sequence_length=sequence_length,
            check_interval=check_interval,
            rng=generator,
        )
        cholesky = jacobi_cholesky(lanczos, ridge=ridge)
        modified_cholesky, metadata = reference_modified_cholesky(
            lanczos,
            tail_window=tail_window,
            ridge=ridge,
        )
        alpha_values = modified_cholesky.diagonal[-2:]
        beta_value = float(modified_cholesky.sub_diagonal[-1])
        tail = (
            float(np.mean(alpha_values)),
            beta_value,
            float(np.std(alpha_values, ddof=1) / max(np.mean(alpha_values), np.finfo(float).eps)),
            0.0,
        )
        lanczos_results.append(lanczos)
        cholesky_results.append(cholesky)
        modified_cholesky_results.append(modified_cholesky)
        tail_estimates.append(tail)
        tail_metadata.append(metadata)
    tail_array = np.asarray(tail_estimates, dtype=np.float64)
    tail_alpha = float(np.mean(tail_array[:, 0]))
    tail_beta = float(np.mean(tail_array[:, 1]))
    consensus_modified: list[JacobiCholesky] = []
    for modified in modified_cholesky_results:
        diagonal = modified.diagonal.copy()
        sub_diagonal = modified.sub_diagonal.copy()
        diagonal[-2:] = tail_alpha
        sub_diagonal[-1] = tail_beta
        consensus_modified.append(
            JacobiCholesky(
                diagonal=diagonal,
                sub_diagonal=sub_diagonal,
                ridge=modified.ridge,
            )
        )
    lower, upper = support_from_cholesky_tail(tail_alpha, tail_beta)
    # Ridge stabilizes Cholesky only; report support in the original operator
    # domain so it is comparable with unshifted Ritz values/eigenvalues.
    lower, upper = max(0.0, lower - ridge), max(0.0, upper - ridge)
    gap = threshold_c * size ** (-threshold_delta)
    threshold = float(upper + gap)
    per_probe_poles: list[np.ndarray] = []
    per_probe_residues: list[np.ndarray] = []
    probe_edges: list[tuple[float, float]] = []
    for lanczos, cholesky, modified in zip(
        lanczos_results,
        cholesky_results,
        modified_cholesky_results,
    ):
        probe_alpha = float(np.mean(modified.diagonal[-2:]))
        probe_beta = float(modified.sub_diagonal[-1])
        probe_lower, probe_upper = support_from_cholesky_tail(probe_alpha, probe_beta)
        probe_edges.append((max(0.0, probe_lower - ridge),
                            max(0.0, probe_upper - ridge)))
        if pole_method == "reference_ritz":
            poles, residues = finite_vest_poles(
                lanczos,
                threshold=threshold,
                residue_threshold=residue_threshold,
            )
        else:
            poles, residues = finite_section_poles(
                cholesky,
                tail_alpha=tail_alpha,
                tail_beta=tail_beta,
                threshold=threshold + ridge,
                residue_threshold=residue_threshold,
                tail_window=tail_window,
                extension_size=extension_size,
            )
            poles = poles - ridge
        per_probe_poles.append(poles)
        per_probe_residues.append(residues)
    counts = np.asarray([values.size for values in per_probe_poles], dtype=np.int64)
    spike_count = _mode_count(counts)
    eligible = [index for index, count in enumerate(counts) if count == spike_count]
    longest = int(np.argmax([result.iterations for result in lanczos_results]))
    if pole_method == "reference_ritz":
        eligible_by_length = sorted(
            eligible,
            key=lambda index: lanczos_results[index].iterations,
            reverse=True,
        )
        representative = eligible_by_length[0] if eligible_by_length else longest
        if spike_count:
            poles = per_probe_poles[representative][:spike_count]
            residues = per_probe_residues[representative][:spike_count]
        else:
            poles = np.asarray([], dtype=np.float64)
            residues = np.asarray([], dtype=np.float64)
    elif spike_count and len(eligible) > 1:
        representative = eligible[0]
        pole_stack = np.vstack([per_probe_poles[index][:spike_count] for index in eligible])
        residue_stack = np.vstack([per_probe_residues[index][:spike_count] for index in eligible])
        poles = np.median(pole_stack, axis=0)
        residues = np.median(residue_stack, axis=0)
    else:
        representative = eligible[0] if eligible else int(np.argmin(np.abs(counts - spike_count)))
        poles = per_probe_poles[representative][:spike_count]
        residues = per_probe_residues[representative][:spike_count]
    probe_stieltjes_diagonals: list[np.ndarray] = []
    probe_stieltjes_sub_diagonals: list[np.ndarray] = []
    for modified in consensus_modified:
        diagonal = modified.diagonal[:-1].copy()
        sub_diagonal = modified.sub_diagonal[:-1].copy()
        probe_stieltjes_diagonals.append(diagonal)
        probe_stieltjes_sub_diagonals.append(sub_diagonal)
    stieltjes_diagonal = probe_stieltjes_diagonals[representative]
    stieltjes_sub = probe_stieltjes_sub_diagonals[representative]
    relative_scatter = float(np.max(tail_array[:, 2:]))
    edge_array = np.asarray(probe_edges, dtype=np.float64)
    edge_spread = float(np.ptp(edge_array[:, 1]) / max(upper, np.finfo(float).eps)) if probes > 1 else 0.0
    adaptive_flags = np.asarray([result.converged for result in lanczos_results], dtype=bool)
    recurrence_converged = bool(np.all(adaptive_flags)) if adaptive else True
    converged = bool(recurrence_converged and relative_scatter <= 0.2 and edge_spread <= 0.1)
    return LanczosSpikeResult(
        lambda_minus=lower,
        lambda_plus=upper,
        poles=np.asarray(poles, dtype=np.float64),
        residues=np.asarray(residues, dtype=np.float64),
        n_spikes=spike_count,
        threshold=threshold,
        tail_alpha=tail_alpha,
        tail_beta=tail_beta,
        probe_counts=counts,
        probe_edges=edge_array,
        iterations=np.asarray([result.iterations for result in lanczos_results], dtype=np.int64),
        converged=converged,
        probe_converged=adaptive_flags,
        pole_method=str(pole_method),
        representative_probe=representative,
        stieltjes_diagonal=stieltjes_diagonal,
        stieltjes_sub_diagonal=stieltjes_sub,
        probe_stieltjes_diagonals=tuple(probe_stieltjes_diagonals),
        probe_stieltjes_sub_diagonals=tuple(probe_stieltjes_sub_diagonals),
        probe_tail_metadata=tuple(tail_metadata),
    )


def detect_spikes_from_factor(
    factor: np.ndarray,
    **kwargs: Any,
) -> LanczosSpikeResult:
    """Build a reduced covariance operator and run the Lanczos detector."""

    matrix = np.asarray(factor, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("factor must be a finite matrix with both dimensions at least two")
    dimension = min(matrix.shape)
    normalization = max(matrix.shape)
    if "convergence_tolerance" not in kwargs or kwargs["convergence_tolerance"] is None:
        trace = float(np.sum(np.square(matrix)) / normalization)
        kwargs["convergence_tolerance"] = max(
            3.0 * trace / (dimension * np.sqrt(dimension)),
            np.finfo(float).eps,
        )
    operator = covariance_linear_operator(matrix, normalization=normalization)
    result = detect_spikes_lanczos(operator, **kwargs)
    # Debias finite-recurrence support scale with a cheap trace identity after
    # excluding separated poles (no dense decomposition is introduced).
    bulk_count = dimension - result.n_spikes
    if bulk_count > 0:
        trace = float(np.sum(np.square(matrix)) / normalization)
        bulk_variance = max(np.finfo(float).eps,
                            (trace - float(np.sum(result.poles))) / bulk_count)
        q = dimension / max(matrix.shape)
        moment_lower = bulk_variance * (1.0 - np.sqrt(q)) ** 2
        moment_upper = bulk_variance * (1.0 + np.sqrt(q)) ** 2
        gap = float(kwargs.get("threshold_c", 1.0)) * dimension ** (
            -float(kwargs.get("threshold_delta", 0.25))
        )
        corrected_threshold = float(moment_upper + gap)
        retained = result.poles > corrected_threshold
        result = replace(
            result,
            lambda_minus=float(moment_lower),
            lambda_plus=float(moment_upper),
            threshold=corrected_threshold,
            poles=result.poles[retained],
            residues=result.residues[retained],
            n_spikes=int(np.count_nonzero(retained)),
        )
    return result


lanczos_tridiagonalization = lanczos_tridiagonalize
vector_empirical_stieltjes_transform = vector_empirical_stieltjes
lanczos_stieltjes_detector = detect_spikes_lanczos


__all__ = [
    "asymptotic_spectral_density",
    "JacobiCholesky",
    "LanczosResult",
    "LanczosSpikeResult",
    "covariance_linear_operator",
    "default_lanczos_steps",
    "detect_spikes_from_factor",
    "detect_spikes_lanczos",
    "estimate_constant_tail",
    "extended_stieltjes_transform",
    "finite_section_poles",
    "finite_vest_poles",
    "jacobi_cholesky",
    "reference_modified_cholesky",
    "lanczos_tridiagonalize",
    "lanczos_tridiagonalization",
    "lanczos_stieltjes_detector",
    "support_from_cholesky_tail",
    "vector_empirical_stieltjes",
    "vector_empirical_stieltjes_transform",
]
