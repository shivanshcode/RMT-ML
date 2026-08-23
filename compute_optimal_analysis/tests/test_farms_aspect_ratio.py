import numpy as np
import pytest

from rmt.ensembles import wishart_factor
from rmt.farms_aspect_ratio import (
    FARMSConfig,
    farms_spectrum,
    farms_window_starts,
    fixed_ratio_window_shape,
    shape_normalize_eigenvalues,
)
from rmt.mp import fit_marchenko_pastur, marchenko_pastur_bounds


def _shape_for_ratio(short_dimension: int, ratio: float) -> tuple[int, int]:
    if ratio <= 1.0:
        return short_dimension, int(round(short_dimension / ratio))
    return int(round(short_dimension * ratio)), short_dimension


def test_fixed_ratio_windows_are_shape_identical_and_reproducible() -> None:
    matrix = wishart_factor(96, 384, rng=5)
    config = FARMSConfig(
        target_aspect_ratio=1.0,
        window_size=80,
        row_windows=3,
        column_windows=4,
        seed=11,
    )
    first = farms_spectrum(matrix, config)
    second = farms_spectrum(matrix, config)
    assert first.window_shape == (80, 80)
    assert first.canonical_aspect_ratio == 1.0
    assert np.array_equal(first.starts, second.starts)
    assert np.array_equal(first.eigenvalues, second.eigenvalues)
    assert first.eigenvalues.size == first.n_submatrices * 80
    assert 0.0 < first.coverage_fraction <= 1.0


def test_farms_removes_source_aspect_ratio_bias_from_fitted_edge() -> None:
    fitted_edges: list[float] = []
    for index, ratio in enumerate((0.1, 0.25, 0.5, 1.0, 2.0, 4.0)):
        n, m = _shape_for_ratio(256, ratio)
        matrix = wishart_factor(n, m, rng=100 + index)
        result = farms_spectrum(
            matrix,
            FARMSConfig(
                target_aspect_ratio=1.0,
                window_size=256,
                row_windows=3,
                column_windows=3,
            ),
        )
        fit = fit_marchenko_pastur(result.eigenvalues, result.canonical_aspect_ratio)
        fitted_edges.append(fit.lambda_plus)
    edges = np.asarray(fitted_edges)
    assert np.max(np.abs(edges / 4.0 - 1.0)) < 0.02


def test_shape_normalization_is_an_explicit_analytic_baseline() -> None:
    q = 0.25
    raw_upper = marchenko_pastur_bounds(q, 1.0)[1]
    mapped = shape_normalize_eigenvalues(np.asarray([0.0, raw_upper]), q)
    assert np.allclose(mapped, np.asarray([0.0, 4.0]))
    assert fixed_ratio_window_shape((400, 100), 1.0) == (100, 100)
    with pytest.raises(ValueError, match="two sampled columns"):
        fixed_ratio_window_shape((10, 10), 0.1, window_size=10)


def test_reference_ratio_and_floor_stride_match_released_sampler() -> None:
    shape = fixed_ratio_window_shape((10, 20), 2.0, window_size=3)
    assert shape == (3, 6)
    starts = farms_window_starts(
        (10, 20),
        shape,
        row_windows=3,
        column_windows=3,
        sampling="reference_fixed",
    )
    expected = np.asarray(
        [
            (0, 0),
            (0, 7),
            (0, 14),
            (3, 0),
            (3, 7),
            (3, 14),
            (6, 0),
            (6, 7),
            (6, 14),
        ]
    )
    assert np.array_equal(starts, expected)


def test_reference_fixed_step_sampler_does_not_force_last_start() -> None:
    starts = farms_window_starts(
        (11, 17),
        (4, 6),
        sampling="reference_sliding",
        step_size=5,
    )
    expected = np.asarray(
        [
            (0, 0),
            (0, 5),
            (0, 10),
            (5, 0),
            (5, 5),
            (5, 10),
        ]
    )
    assert np.array_equal(starts, expected)


def test_reference_raw_and_canonical_pooling_differ_only_by_window_scale() -> None:
    matrix = np.random.default_rng(29).normal(size=(8, 16))
    shared = dict(
        target_aspect_ratio=2.0,
        window_size=4,
        row_windows=2,
        column_windows=2,
        sampling="reference_fixed",
    )
    raw = farms_spectrum(matrix, FARMSConfig(**shared, normalization="raw"))
    canonical = farms_spectrum(
        matrix,
        FARMSConfig(**shared, normalization="canonical"),
    )
    assert raw.window_shape == (4, 8)
    assert raw.reference_aspect_ratio == 2.0
    assert np.allclose(raw.eigenvalues / 8.0, canonical.eigenvalues)
