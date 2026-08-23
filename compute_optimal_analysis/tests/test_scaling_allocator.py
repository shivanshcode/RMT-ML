import math

from models.chinchilla_scaling import (
    ScalingLaw,
    allocation_for_regime,
    compute_optimal_allocation,
    isoflop_grid,
)


def test_target_ratio_allocation_conserves_compute() -> None:
    law = ScalingLaw()
    allocation = compute_optimal_allocation(1e16, law, target_tokens_per_parameter=20.0)
    assert math.isclose(allocation.tokens_per_parameter, 20.0, rel_tol=1e-12)
    assert math.isclose(allocation.realized_compute, 1e16, rel_tol=1e-12)


def test_regimes_move_inversely_on_same_isoflop_surface() -> None:
    optimum = compute_optimal_allocation(1e15, target_tokens_per_parameter=20.0)
    undertrained = allocation_for_regime(optimum, 0.25)
    overtrained = allocation_for_regime(optimum, 4.0)
    assert math.isclose(undertrained.parameters, optimum.parameters / 0.25)
    assert math.isclose(undertrained.tokens, optimum.tokens * 0.25)
    assert math.isclose(overtrained.parameters, optimum.parameters / 4.0)
    assert math.isclose(overtrained.tokens, optimum.tokens * 4.0)
    assert math.isclose(undertrained.realized_compute, optimum.compute_budget, rel_tol=1e-12)
    assert math.isclose(overtrained.realized_compute, optimum.compute_budget, rel_tol=1e-12)


def test_default_grid_has_nine_cells() -> None:
    grid = isoflop_grid((1e15, 1e16, 1e17))
    assert len(grid) == 9
    assert {allocation.regime for allocation in grid} == {
        "undertrained",
        "compute_optimal",
        "overtrained",
    }

