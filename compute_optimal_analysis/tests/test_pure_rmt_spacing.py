import numpy as np

from rmt.ensembles import goe, poisson_levels, wishart_factor
from rmt.spacing import (
    fit_brody,
    fit_brody_cdf_nls,
    nearest_neighbor_spacings,
    number_variance,
    r_statistic,
    unfold,
    unfold_spectrum,
)


def test_unfolding_normalizes_mean_spacing() -> None:
    levels = np.linalg.eigvalsh(goe(240, rng=1234))
    unfolded = unfold(levels, deg=7)
    assert abs(np.mean(np.diff(unfolded)) - 1.0) < 1e-10


def test_brody_and_gap_ratio_discriminate_goe_from_poisson() -> None:
    goe_levels = np.linalg.eigvalsh(goe(420, rng=13))
    goe_levels = goe_levels[50:-50]
    poisson = poisson_levels(3000, rng=13)
    goe_beta = fit_brody(nearest_neighbor_spacings(goe_levels)).beta
    poisson_beta = fit_brody(nearest_neighbor_spacings(poisson)).beta
    assert 0.0 <= poisson_beta <= 1.0
    assert 0.0 <= goe_beta <= 1.0
    assert goe_beta > poisson_beta
    assert poisson_beta < 0.25
    assert r_statistic(goe_levels) > r_statistic(poisson)


def test_number_variance_is_finite_and_nonnegative() -> None:
    levels = poisson_levels(2500, rng=77)
    value = number_variance(levels, 10.0)
    assert np.isfinite(value)
    assert value >= 0.0


def test_wishart_bulk_has_goe_like_brody_repulsion() -> None:
    factor = wishart_factor(320, 640, rng=2025)
    levels = np.linalg.eigvalsh(factor @ factor.T / 640.0)[30:-30]
    beta = fit_brody(nearest_neighbor_spacings(levels)).beta
    assert 0.55 <= beta <= 1.0


def test_chebyshev_and_monotone_spline_agree_on_goe_calibration() -> None:
    levels = np.linalg.eigvalsh(goe(700, rng=2026))[100:-100]
    for strategy in ("polynomial_chebyshev", "spline_monotone"):
        unfolded = unfold_spectrum(levels, method=strategy, degree=7)
        spacings = np.diff(unfolded)
        spacings = spacings[spacings > 0.0]
        spacings /= np.mean(spacings)
        assert abs(np.mean(spacings) - 1.0) < 0.01
        beta = fit_brody(spacings).beta
        assert 0.90 <= beta <= 1.00


def test_reference_brody_and_number_variance_paths_are_seeded() -> None:
    levels = poisson_levels(1200, rng=35)
    spacings = nearest_neighbor_spacings(levels, method="spline_monotone")
    cdf_fit = fit_brody_cdf_nls(spacings, n_bootstrap=8, rng=4)
    assert 0.0 <= cdf_fit.beta <= 1.0
    assert cdf_fit.standard_error >= 0.0
    first = number_variance(
        levels,
        8.0,
        method="monte_carlo",
        n_windows=128,
        rng=22,
    )
    second = number_variance(
        levels,
        8.0,
        method="monte_carlo",
        n_windows=128,
        rng=22,
    )
    assert first == second
    assert first >= 0.0
