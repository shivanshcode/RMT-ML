import numpy as np

from rmt.ensembles import pareto
from rmt.tail import (
    csn_goodness_of_fit,
    fit_powerlaw_csn,
    fixed_cutoff_mle,
    hill_alpha_at,
    hill_estimator,
    rank_ordered_mle,
)


def test_csn_recovers_pareto_density_exponent() -> None:
    values = pareto(100_000, alpha=3.0, rng=1234)
    fit = fit_powerlaw_csn(values, min_tail=500, max_xmin_candidates=120)
    assert abs(fit["alpha"] / 3.0 - 1.0) < 0.02
    assert fit["ks_D"] < 0.03


def test_hill_is_survival_exponent_and_squaring_halves_it() -> None:
    values = pareto(50_000, alpha=3.0, rng=91)
    k = 1500
    alpha_values = hill_alpha_at(values, k)
    alpha_squared = hill_alpha_at(values**2, k)
    assert abs(alpha_values - 2.0) < 0.2
    assert abs(alpha_squared / alpha_values - 0.5) < 0.05


def test_short_tail_returns_structured_nonfinite_fit() -> None:
    result = fit_powerlaw_csn(np.arange(1.0, 20.0), min_tail=50)
    assert np.isnan(result["alpha"])
    assert result["n_tail"] == 0
    ks, estimates = hill_estimator(np.asarray([1.0, 2.0, 3.0]), k_min=5)
    assert ks.size == 0
    assert estimates.size == 0


def test_fixed_cutoff_and_rank_ordered_density_exponents() -> None:
    values = pareto(100_000, alpha=3.0, rng=2026)
    fixed = fixed_cutoff_mle(values, xmin=1.0)
    ranked = rank_ordered_mle(values, xmin=1.0)
    assert abs(fixed["alpha"] / 3.0 - 1.0) < 0.02
    assert abs(ranked["alpha"] / 3.0 - 1.0) < 0.05
    assert ranked["r_squared"] > 0.98


def test_csn_semiparametric_bootstrap_is_seeded_and_bounded() -> None:
    values = pareto(4000, alpha=2.7, rng=41)
    first = csn_goodness_of_fit(
        values,
        n_bootstrap=20,
        min_tail=100,
        max_xmin_candidates=40,
        rng=12,
    )
    second = csn_goodness_of_fit(
        values,
        n_bootstrap=20,
        min_tail=100,
        max_xmin_candidates=40,
        rng=12,
    )
    assert 0.0 <= first["p_value"] <= 1.0
    assert np.array_equal(first["bootstrap_distances"], second["bootstrap_distances"])
