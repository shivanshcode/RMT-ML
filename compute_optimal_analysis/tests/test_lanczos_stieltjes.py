import numpy as np

from rmt.ensembles import spiked_covariance
from rmt.lanczos_stieltjes import (
    LanczosResult,
    detect_spikes_from_factor,
    default_lanczos_steps,
    finite_vest_poles,
    lanczos_tridiagonalize,
    reference_modified_cholesky,
    support_from_cholesky_tail,
    vector_empirical_stieltjes,
)


def test_vector_empirical_stieltjes_matches_direct_resolvent() -> None:
    generator = np.random.default_rng(17)
    raw = generator.normal(size=(24, 24))
    matrix = raw @ raw.T / 24.0 + 0.25 * np.eye(24)
    probe = generator.normal(size=24)
    normalized = probe / np.linalg.norm(probe)
    lanczos = lanczos_tridiagonalize(
        matrix,
        steps=24,
        probe=probe,
        reorthogonalization="full",
    )
    z = 1.3 + 0.2j
    expected = normalized @ np.linalg.solve(matrix - z * np.eye(24), normalized)
    observed = vector_empirical_stieltjes(lanczos, z)
    assert abs(observed - expected) < 1e-9


def test_constant_cholesky_tail_has_declared_support() -> None:
    lower, upper = support_from_cholesky_tail(1.0, 0.5)
    assert np.isclose(lower, 0.25)
    assert np.isclose(upper, 2.25)


def test_single_entry_reference_stability_window_is_finite() -> None:
    result = lanczos_tridiagonalize(
        np.diag(np.linspace(0.5, 2.0, 8)),
        steps=6,
        adaptive=True,
        sequence_length=1,
        convergence_tolerance=1e-12,
        rng=4,
    )
    assert result.sequence_length == 1
    assert np.all(np.isfinite(result.diagonal))
    assert np.all(np.isfinite(result.off_diagonal))


def test_reference_finite_vest_poles_use_ritz_residues() -> None:
    matrix = np.diag(np.asarray([0.4, 0.8, 1.2, 3.0, 4.0]))
    lanczos = lanczos_tridiagonalize(
        matrix,
        steps=5,
        probe=np.ones(5),
        reorthogonalization="full",
    )
    poles, residues = finite_vest_poles(lanczos, threshold=2.0)
    assert np.allclose(poles, np.asarray([4.0, 3.0]))
    assert np.all(residues > 0.0)
    assert default_lanczos_steps(400) >= 100


def test_reference_modified_tail_uses_convergence_window_before_cholesky() -> None:
    recurrence = LanczosResult(
        diagonal=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        off_diagonal=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
        probe=np.ones(10) / np.sqrt(10.0),
        iterations=6,
        breakdown=False,
        orthogonality_error=0.0,
        converged=True,
        convergence_iteration=5,
        convergence_tolerance=1e-6,
        sequence_length=2,
        check_interval=2,
    )
    cholesky, metadata = reference_modified_cholesky(recurrence)
    lower = np.diag(cholesky.diagonal) + np.diag(cholesky.sub_diagonal, -1)
    reconstructed = lower @ lower.T
    expected = np.diag(np.asarray([1.0, 4.0, 4.0]))
    expected += np.diag(np.asarray([0.1, 0.35]), 1)
    expected += np.diag(np.asarray([0.1, 0.35]), -1)
    assert np.allclose(reconstructed, expected)
    assert metadata["tail_start"] == 1
    assert metadata["tail_mode"] == "reference_converged"


def test_lanczos_detects_three_separated_spikes_and_mp_edge() -> None:
    n, m = 240, 960
    q = n / m
    population_spikes = np.asarray([4.5, 3.6, 2.8])
    factor = spiked_covariance(
        n,
        m,
        population_spikes,
        rng=2026,
        return_factor=True,
    )
    result = detect_spikes_from_factor(
        factor,
        steps=96,
        n_probes=3,
        threshold_c=0.5,
        threshold_delta=0.25,
        residue_threshold=0.0,
        rng=91,
    )
    theoretical_edge = (1.0 + np.sqrt(q)) ** 2
    assert result.n_spikes == 3
    assert result.poles.size == 3
    assert np.all(result.poles > result.threshold)
    assert np.all(result.residues > 0.0)
    assert result.pole_method == "reference_ritz"
    assert result.probe_converged.shape == (3,)
    assert result.representative_probe == int(np.argmax(result.iterations))
    assert len(result.probe_tail_metadata) == 3
    # The factor adapter preserves the finite Lanczos estimate rather than
    # replacing it with an analytic MP moment projection.
    assert abs(result.lambda_plus / theoretical_edge - 1.0) < 0.04
    grid = np.linspace(result.lambda_minus, result.lambda_plus, 64)
    density = result.density(grid, eta=0.02)
    assert np.all(np.isfinite(density))
    assert np.all(density >= 0.0)
