from pathlib import Path

import numpy as np
from scipy.integrate import quad

from rmt.ensembles import wishart_factor
from rmt.mp import (
    bbp_population_threshold,
    bbp_sample_location,
    detect_spikes_bbp,
    detect_spikes_tracy_widom,
    eigenvalues_of_cov,
    estimate_sigma_gd_median,
    fit_marchenko_pastur,
    fit_marchenko_pastur_kde,
    fit_marchenko_pastur_thamm,
    fit_modified_mp_singular,
    marchenko_pastur_bounds,
    marchenko_pastur_density,
    mp_bounds,
    mp_bounds_eig,
    tracy_widom_upper_threshold,
)


def test_pure_rmt_tree_has_no_torch_import() -> None:
    root = Path(__file__).resolve().parents[1] / "rmt"
    for source in root.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "import torch" not in text
        assert "from torch" not in text


def test_mp_density_integrates_to_one() -> None:
    for q in (0.25, 0.5, 1.0):
        lower, upper = marchenko_pastur_bounds(q, 1.7)
        integral = quad(
            lambda value: float(marchenko_pastur_density(value, q, 1.7)),
            lower,
            upper,
            epsabs=1e-9,
        )[0]
        assert abs(integral - 1.0) < 1e-3


def test_mp_fit_recovers_wishart_scale_and_bulk() -> None:
    matrix = wishart_factor(400, 800, sigma=1.3, rng=1234)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    sigma = estimate_sigma_gd_median(s=singular_values, n=400, m=800)
    eigenvalues = singular_values**2 / 800
    fit = fit_marchenko_pastur(eigenvalues, 0.5)
    assert abs(sigma / 1.3 - 1.0) < 0.02
    assert abs(fit.sigma / 1.3 - 1.0) < 0.03
    assert fit.bulk_fraction > 0.94


def test_eigenvalue_and_singular_edge_views_agree() -> None:
    matrix = wishart_factor(32, 64, sigma=0.7, rng=7)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    eigenvalues = eigenvalues_of_cov(matrix, N=64)
    assert np.allclose(eigenvalues, singular_values**2 / 64)
    singular_bounds = mp_bounds(32, 64, 0.7)
    eigen_bounds = mp_bounds_eig(32, 64, 0.7, 64)
    assert np.allclose(eigen_bounds, np.square(singular_bounds) / 64)


def test_kde_bulk_fit_recovers_wishart_scale() -> None:
    matrix = wishart_factor(240, 480, sigma=1.2, rng=61)
    eigenvalues = np.linalg.svd(matrix, compute_uv=False) ** 2 / 480.0
    fit = fit_marchenko_pastur_kde(eigenvalues, 0.5, trim_upper=0.05)
    assert fit.method == "kde_bulk_fit"
    assert abs(fit.sigma / 1.2 - 1.0) < 0.08
    assert fit.lambda_minus < fit.lambda_plus


def test_twidom_and_bbp_thresholds_are_distinct_and_ordered() -> None:
    q = 0.25
    population_threshold = bbp_population_threshold(q)
    assert np.isclose(population_threshold, 1.5)
    assert bbp_sample_location(population_threshold * 0.99, q) == marchenko_pastur_bounds(q)[1]
    assert bbp_sample_location(3.0, q) > marchenko_pastur_bounds(q)[1]
    tw_threshold = tracy_widom_upper_threshold(200, 800, confidence=0.95)
    assert tw_threshold > marchenko_pastur_bounds(q)[1]
    values = np.asarray([1.0, 2.0, 2.3, 4.0])
    tw = detect_spikes_tracy_widom(values, 200, 800)
    bbp = detect_spikes_bbp(values, q)
    assert tw.n_spikes <= bbp.n_spikes
    assert np.array_equal(bbp.spikes, np.asarray([4.0, 2.3]))


def test_modified_singular_mp_curve_returns_ordered_empirical_edges() -> None:
    matrix = wishart_factor(96, 192, rng=73)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    fit = fit_modified_mp_singular(
        singular_values,
        kernel_window=8,
        grid_size=256,
    )
    assert fit.success
    assert fit.amplitude > 0.0
    assert 0.0 <= fit.nu_min < fit.nu_max
    assert np.isfinite(fit.rmse)


def test_thamm_mp_adapter_preserves_empirical_singular_edges() -> None:
    matrix = wishart_factor(96, 192, rng=79)
    fit = fit_marchenko_pastur_thamm(
        matrix,
        kernel_window=8,
        grid_size=256,
    )
    denominator = max(matrix.shape)
    assert fit.method == "thamm_modified_singular"
    assert np.isclose(fit.lambda_minus, fit.diagnostics["nu_min_raw"] ** 2 / denominator)
    assert np.isclose(fit.lambda_plus, fit.diagnostics["nu_max_raw"] ** 2 / denominator)
    assert 0.0 <= fit.lambda_minus < fit.lambda_plus
    assert np.isfinite(fit.variance)
    assert np.isfinite(fit.ks_distance)
