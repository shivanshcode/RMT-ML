import numpy as np

from rmt.scalars import (
    condition_number,
    frobenius_norm,
    localization_ratio,
    normalized_matrix_entropy,
    participation_ratio,
    porter_thomas_monte_carlo,
    porter_thomas_monte_carlo_pooled,
    spectral_norm,
    stable_rank,
    von_neumann_entropy,
)


def test_identity_scalar_invariants() -> None:
    identity = np.eye(12)
    assert np.isclose(spectral_norm(identity), 1.0)
    assert np.isclose(frobenius_norm(identity), np.sqrt(12.0))
    assert np.isclose(stable_rank(identity), 12.0)
    assert np.isclose(condition_number(identity), 1.0)
    assert np.isclose(von_neumann_entropy(identity), np.log(12.0))
    assert np.isclose(normalized_matrix_entropy(identity), 1.0)


def test_rank_one_and_singular_conventions() -> None:
    matrix = np.outer(np.arange(1.0, 9.0), np.arange(1.0, 6.0))
    assert np.isclose(stable_rank(matrix), 1.0)
    assert von_neumann_entropy(matrix) < 1e-25
    assert normalized_matrix_entropy(matrix) == 0.0
    assert np.isinf(condition_number(np.diag([2.0, 1.0, 0.0])))


def test_archive_localization_and_participation_conventions() -> None:
    localized = np.asarray([1.0, 0.0, 0.0, 0.0])
    delocalized = np.ones(4) / 2.0
    assert np.isclose(localization_ratio(localized), 1.0)
    assert np.isclose(localization_ratio(delocalized), 4.0)
    assert np.isclose(participation_ratio(localized), 1.0)
    assert participation_ratio(delocalized) > participation_ratio(localized)


def test_porter_thomas_monte_carlo_is_seeded_and_bounded() -> None:
    generator = np.random.default_rng(51)
    vectors = generator.normal(size=(5, 32))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    first = porter_thomas_monte_carlo(vectors, n_reference=64, rng=3)
    second = porter_thomas_monte_carlo(vectors, n_reference=64, rng=3)
    assert np.array_equal(first["distances"], second["distances"])
    assert np.array_equal(first["pvalues"], second["pvalues"])
    assert np.all((first["pvalues"] >= 0.0) & (first["pvalues"] <= 1.0))
    pooled = porter_thomas_monte_carlo_pooled(
        np.vstack((vectors, vectors, vectors)),
        n_reference=64,
        pooling_window=2,
        rng=5,
    )
    assert pooled["center_indices"].size == 11
    assert np.all((pooled["pvalues"] >= 0.0) & (pooled["pvalues"] <= 1.0))
