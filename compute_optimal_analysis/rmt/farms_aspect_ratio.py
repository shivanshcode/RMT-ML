"""Fixed-aspect-ratio matrix subsampling and pooled spectral estimates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator, Literal, TypeAlias

import numpy as np


RandomState: TypeAlias = np.random.Generator | int | None
SamplingMode: TypeAlias = Literal[
    "reference_fixed",
    "reference_sliding",
    "grid",
    "random",
]
SpectrumNormalization: TypeAlias = Literal["canonical", "raw", "trace"]


def _generator(rng: RandomState) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _finite_matrix(weight: np.ndarray) -> np.ndarray:
    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("weight must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("weight must contain only finite values")
    return matrix


@dataclass(frozen=True)
class FARMSConfig:
    """Configuration for fixed-ratio sliding or random windows.

    ``target_aspect_ratio`` follows the reference convention: sampled columns
    divided by sampled rows. ``window_size`` is the sampled row count. When it
    is omitted, the largest fitting row count is used. ``reference_fixed``
    reproduces the released fixed-operation sampler, while
    ``reference_sliding`` reproduces its fixed-step sampler.
    """

    target_aspect_ratio: float = 1.0
    window_size: int | None = None
    row_windows: int = 5
    column_windows: int = 5
    sampling: SamplingMode = "reference_fixed"
    n_submatrices: int | None = None
    step_size: int = 10
    normalization: SpectrumNormalization = "canonical"
    orient_tall: bool = False
    seed: int | None = 0

    def __post_init__(self) -> None:
        ratio = float(self.target_aspect_ratio)
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("target_aspect_ratio must be finite and positive")
        if self.window_size is not None and (
            int(self.window_size) < 2 or int(self.window_size) != self.window_size
        ):
            raise ValueError("window_size must be an integer of at least two")
        if (
            int(self.row_windows) < 1
            or int(self.row_windows) != self.row_windows
            or int(self.column_windows) < 1
            or int(self.column_windows) != self.column_windows
        ):
            raise ValueError("row_windows and column_windows must be positive integers")
        if self.sampling not in {"reference_fixed", "reference_sliding", "grid", "random"}:
            raise ValueError(
                "sampling must be reference_fixed, reference_sliding, grid, or random"
            )
        if self.n_submatrices is not None and (
            int(self.n_submatrices) < 1
            or int(self.n_submatrices) != self.n_submatrices
        ):
            raise ValueError("n_submatrices must be a positive integer")
        if int(self.step_size) < 1 or int(self.step_size) != self.step_size:
            raise ValueError("step_size must be a positive integer")
        if self.normalization not in {"canonical", "raw", "trace"}:
            raise ValueError("normalization must be canonical, raw, or trace")


@dataclass(frozen=True)
class FARMSResult:
    """Pooled ESD and coverage metadata from equal-shape submatrices."""

    eigenvalues: np.ndarray
    source_shape: tuple[int, int]
    oriented_shape: tuple[int, int]
    window_shape: tuple[int, int]
    starts: np.ndarray
    target_aspect_ratio: float
    canonical_aspect_ratio: float
    normalization: str
    transposed: bool
    coverage_fraction: float
    reference_aspect_ratio: float

    def __post_init__(self) -> None:
        values = np.asarray(self.eigenvalues, dtype=np.float64).ravel()
        starts = np.asarray(self.starts, dtype=np.int64)
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("eigenvalues must be finite, nonnegative, and nonempty")
        if starts.ndim != 2 or starts.shape[1] != 2 or starts.shape[0] < 1:
            raise ValueError("starts must have shape (n_submatrices, 2)")
        if not np.isfinite(float(self.reference_aspect_ratio)) or float(self.reference_aspect_ratio) <= 0.0:
            raise ValueError("reference_aspect_ratio must be finite and positive")
        object.__setattr__(self, "eigenvalues", np.sort(values)[::-1])
        object.__setattr__(self, "starts", starts)

    @property
    def n_submatrices(self) -> int:
        return int(self.starts.shape[0])

    @property
    def eigenvalues_per_submatrix(self) -> int:
        return int(min(self.window_shape))

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("eigenvalues")
        payload["starts"] = self.starts.tolist()
        payload["n_submatrices"] = self.n_submatrices
        payload["n_eigenvalues"] = int(self.eigenvalues.size)
        return payload


def fixed_ratio_window_shape(
    source_shape: tuple[int, int],
    target_aspect_ratio: float = 1.0,
    window_size: int | None = None,
) -> tuple[int, int]:
    """Return a fitting ``(rows, columns)`` window with reference ratio ``columns/rows``."""

    rows, columns = int(source_shape[0]), int(source_shape[1])
    ratio = float(target_aspect_ratio)
    if rows < 1 or columns < 1:
        raise ValueError("source dimensions must be positive")
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("target_aspect_ratio must be finite and positive")
    window_rows = (
        min(rows, int(np.floor(columns / ratio)))
        if window_size is None
        else int(window_size)
    )
    if window_rows < 2:
        raise ValueError("the source cannot fit a window with both dimensions at least two")
    window_columns = int(np.floor(window_rows * ratio))
    if window_columns < 2:
        raise ValueError("the requested ratio cannot produce two sampled columns")
    if window_rows > rows or window_columns > columns:
        raise ValueError("requested fixed-ratio window does not fit the oriented matrix")
    return window_rows, window_columns


def _grid_positions(limit: int, width: int, count: int) -> np.ndarray:
    maximum = int(limit - width)
    if maximum < 0:
        raise ValueError("window exceeds source dimension")
    if maximum == 0 or count == 1:
        return np.asarray([0], dtype=np.int64)
    positions = np.rint(np.linspace(0.0, float(maximum), int(count))).astype(np.int64)
    return np.unique(positions)


def _reference_fixed_positions(limit: int, width: int, count: int) -> np.ndarray:
    """Match the released fixed-operation sampler's floor-step schedule."""

    maximum = int(limit - width)
    if maximum < 0:
        raise ValueError("window exceeds source dimension")
    requested = int(count)
    if requested < 1:
        raise ValueError("window count must be positive")
    if maximum == 0 or requested == 1:
        return np.asarray([0], dtype=np.int64)
    step = max(1, maximum // (requested - 1))
    realized = min(requested, max(1, maximum // step + 1))
    return np.asarray(
        [min(index * step, maximum) for index in range(realized)],
        dtype=np.int64,
    )


def _reference_sliding_positions(limit: int, width: int, step_size: int) -> np.ndarray:
    """Match the released fixed-step sliding sampler without forcing the endpoint."""

    maximum = int(limit - width)
    step = int(step_size)
    if maximum < 0:
        raise ValueError("window exceeds source dimension")
    if step < 1:
        raise ValueError("step_size must be positive")
    count = max(1, maximum // step + 1)
    return np.arange(count, dtype=np.int64) * step


def farms_window_starts(
    source_shape: tuple[int, int],
    window_shape: tuple[int, int],
    *,
    row_windows: int = 5,
    column_windows: int = 5,
    sampling: SamplingMode = "reference_fixed",
    n_submatrices: int | None = None,
    step_size: int = 10,
    rng: RandomState = 0,
) -> np.ndarray:
    """Return starts for a reference, uniform-grid, or seeded-random schedule."""

    rows, columns = map(int, source_shape)
    window_rows, window_columns = map(int, window_shape)
    if min(rows, columns, window_rows, window_columns) < 1:
        raise ValueError("all dimensions must be positive")
    max_row, max_column = rows - window_rows, columns - window_columns
    if max_row < 0 or max_column < 0:
        raise ValueError("window_shape must fit source_shape")
    if sampling == "reference_fixed":
        row_positions = _reference_fixed_positions(rows, window_rows, int(row_windows))
        column_positions = _reference_fixed_positions(
            columns,
            window_columns,
            int(column_windows),
        )
        return np.asarray(
            [(row, column) for row in row_positions for column in column_positions],
            dtype=np.int64,
        )
    if sampling == "reference_sliding":
        row_positions = _reference_sliding_positions(rows, window_rows, int(step_size))
        column_positions = _reference_sliding_positions(columns, window_columns, int(step_size))
        return np.asarray(
            [(row, column) for row in row_positions for column in column_positions],
            dtype=np.int64,
        )
    if sampling == "grid":
        row_positions = _grid_positions(rows, window_rows, int(row_windows))
        column_positions = _grid_positions(columns, window_columns, int(column_windows))
        return np.asarray(
            [(row, column) for row in row_positions for column in column_positions],
            dtype=np.int64,
        )
    if sampling != "random":
        raise ValueError(
            "sampling must be reference_fixed, reference_sliding, grid, or random"
        )
    count = int(n_submatrices if n_submatrices is not None else row_windows * column_windows)
    if count < 1:
        raise ValueError("n_submatrices must be positive")
    available = (max_row + 1) * (max_column + 1)
    if count > available:
        raise ValueError("n_submatrices exceeds the number of unique window starts")
    generator = _generator(rng)
    flat = np.sort(generator.choice(available, size=count, replace=False))
    starts = np.column_stack((flat // (max_column + 1), flat % (max_column + 1)))
    return starts.astype(np.int64)


def iter_farms_submatrices(
    weight: np.ndarray,
    window_shape: tuple[int, int],
    starts: np.ndarray,
) -> Iterator[np.ndarray]:
    """Yield views into a matrix for the supplied window starts."""

    matrix = _finite_matrix(weight)
    window_rows, window_columns = map(int, window_shape)
    positions = np.asarray(starts, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("starts must have shape (n_submatrices, 2)")
    for row, column in positions:
        row_stop, column_stop = int(row + window_rows), int(column + window_columns)
        if row < 0 or column < 0 or row_stop > matrix.shape[0] or column_stop > matrix.shape[1]:
            raise ValueError("a window start lies outside the source matrix")
        yield matrix[int(row) : row_stop, int(column) : column_stop]


def _coverage_fraction(
    source_shape: tuple[int, int],
    window_shape: tuple[int, int],
    starts: np.ndarray,
) -> float:
    rows, columns = source_shape
    window_rows, window_columns = window_shape
    sample_rows = np.unique(np.rint(np.linspace(0, rows - 1, min(rows, 128))).astype(int))
    sample_columns = np.unique(np.rint(np.linspace(0, columns - 1, min(columns, 128))).astype(int))
    covered = np.zeros((sample_rows.size, sample_columns.size), dtype=bool)
    for row, column in np.asarray(starts, dtype=np.int64):
        row_mask = (sample_rows >= row) & (sample_rows < row + window_rows)
        column_mask = (sample_columns >= column) & (sample_columns < column + window_columns)
        covered |= row_mask[:, None] & column_mask[None, :]
    return float(np.mean(covered))


def farms_spectrum(
    weight: np.ndarray,
    config: FARMSConfig = FARMSConfig(),
) -> FARMSResult:
    """Compute the pooled ESD of fixed-ratio submatrices.

    Windows are processed one at a time. Concatenating equal-length
    eigenvalue series gives the arithmetic mean of their empirical densities.
    """

    source = _finite_matrix(weight)
    transposed = bool(config.orient_tall and source.shape[0] < source.shape[1])
    matrix = source.T if transposed else source
    window_shape = fixed_ratio_window_shape(
        matrix.shape,
        config.target_aspect_ratio,
        config.window_size,
    )
    starts = farms_window_starts(
        matrix.shape,
        window_shape,
        row_windows=config.row_windows,
        column_windows=config.column_windows,
        sampling=config.sampling,
        n_submatrices=config.n_submatrices,
        step_size=config.step_size,
        rng=config.seed,
    )
    spectra: list[np.ndarray] = []
    for submatrix in iter_farms_submatrices(matrix, window_shape, starts):
        singular_values = np.linalg.svd(submatrix, compute_uv=False)
        eigenvalues = np.square(singular_values)
        if config.normalization == "canonical":
            eigenvalues = eigenvalues / max(window_shape)
        elif config.normalization == "trace":
            total = float(np.sum(eigenvalues))
            eigenvalues = eigenvalues if total == 0.0 else eigenvalues / total
        spectra.append(np.asarray(eigenvalues, dtype=np.float64))
    pooled = np.concatenate(spectra)
    canonical = min(window_shape) / max(window_shape)
    return FARMSResult(
        eigenvalues=pooled,
        source_shape=tuple(map(int, source.shape)),
        oriented_shape=tuple(map(int, matrix.shape)),
        window_shape=tuple(map(int, window_shape)),
        starts=starts,
        target_aspect_ratio=float(config.target_aspect_ratio),
        canonical_aspect_ratio=float(canonical),
        normalization=str(config.normalization),
        transposed=transposed,
        coverage_fraction=_coverage_fraction(matrix.shape, window_shape, starts),
        reference_aspect_ratio=float(window_shape[1] / window_shape[0]),
    )


def shape_normalize_eigenvalues(
    eigenvalues: np.ndarray,
    aspect_ratio: float,
    variance: float = 1.0,
    *,
    target_upper_edge: float = 4.0,
) -> np.ndarray:
    """Rescale an MP spectrum so its analytic upper edge is fixed.

    This analytic baseline is intentionally separate from fixed-ratio
    subsampling.
    """

    values = np.asarray(eigenvalues, dtype=np.float64)
    q = float(aspect_ratio)
    scale = float(variance)
    target = float(target_upper_edge)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("eigenvalues must be finite and nonnegative")
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("aspect_ratio must be finite and positive")
    q = min(q, 1.0 / q)
    if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(target) or target <= 0.0:
        raise ValueError("variance and target_upper_edge must be finite and positive")
    upper = scale * (1.0 + np.sqrt(q)) ** 2
    return values * (target / upper)


farms_unbiased_spectrum = farms_spectrum


__all__ = [
    "FARMSConfig",
    "FARMSResult",
    "farms_spectrum",
    "farms_unbiased_spectrum",
    "farms_window_starts",
    "fixed_ratio_window_shape",
    "iter_farms_submatrices",
    "shape_normalize_eigenvalues",
]
