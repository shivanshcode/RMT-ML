import numpy as np

from rmt.overlap import (
    frobenius_projection_overlap,
    principal_angle_alignment,
    principal_angle_spectrum,
    projection_overlap,
    reference_max_cosine_overlap,
    subspace_alignment,
    tranche_indices,
)


def test_aligned_projection_overlap_is_identity() -> None:
    basis = np.linalg.qr(np.random.default_rng(1234).normal(size=(32, 32)))[0]
    overlap = projection_overlap(basis, basis)
    assert np.allclose(overlap, np.eye(32), atol=1e-12)
    assert np.isclose(subspace_alignment(basis[:, :4], basis[:, :4]), 0.25)


def test_projection_bounds_and_tranche_partition() -> None:
    generator = np.random.default_rng(8)
    first = np.linalg.qr(generator.normal(size=(48, 12)))[0]
    second = np.linalg.qr(generator.normal(size=(48, 9)))[0]
    overlap = projection_overlap(first, second)
    assert np.all((overlap >= 0.0) & (overlap <= 1.0))
    tranches = tranche_indices(53)
    combined = np.concatenate([tranches["top"], tranches["bulk"], tranches["bottom"]])
    assert np.array_equal(np.sort(combined), np.arange(53))
    assert np.unique(combined).size == 53


def test_principal_angle_and_frobenius_metrics_are_basis_invariant() -> None:
    generator = np.random.default_rng(19)
    basis = np.linalg.qr(generator.normal(size=(24, 6)))[0]
    rotation = np.linalg.qr(generator.normal(size=(6, 6)))[0]
    rotated = basis @ rotation
    assert np.allclose(principal_angle_spectrum(basis, rotated), 0.0, atol=1e-7)
    assert np.isclose(principal_angle_alignment(basis, rotated), 1.0)
    assert np.isclose(frobenius_projection_overlap(basis, rotated), 1.0)
    signed = reference_max_cosine_overlap(-basis, basis)
    invariant = reference_max_cosine_overlap(-basis, basis, absolute=True)
    assert np.all(signed <= invariant)
    assert np.allclose(invariant, 1.0)
