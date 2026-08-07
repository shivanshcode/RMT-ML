import numpy as np
import pytest

from rmt import overlap as OV


def _orthonormal(n, rng):
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    return Q


def test_overlap_aligned_basis_is_one(rng):
    n = 64
    Q = _orthonormal(n, rng)               # eigenvectors of C (columns)
    evals = np.sort(rng.uniform(1, 5, n))[::-1]
    C = Q @ np.diag(evals) @ Q.T
    # build W so its right singular vectors Vh^T == Q  (square, U=I)
    svals = np.sort(rng.uniform(1, 3, n))[::-1]
    W = (Q * svals) @ Q.T                  # symmetric PSD; right sing vecs = Q
    res = OV.overlap_analysis(W, C)
    assert np.allclose(res["overlap"], 1.0, atol=1e-6)


def test_overlap_random_is_small(rng):
    n = 512
    W = rng.standard_normal((n, n))
    C = rng.standard_normal((n, n)); C = C @ C.T
    res = OV.overlap_analysis(W, C)
    _, upper = OV.three_sigma_band(n)
    assert np.mean(res["overlap"]) < 0.2
    band = 5 * np.sqrt(2 * np.log(n) / n)
    assert np.mean(res["overlap"] < band) > 0.95


def test_three_sigma_band_depends_on_sigma_level():
    b3 = OV.three_sigma_band(1000, 3.0)[1]
    b1 = OV.three_sigma_band(1000, 1.0)[1]
    assert b3 != b1
    assert b3 >= b1


def test_three_sigma_band_shrinks_with_N():
    uppers = [OV.three_sigma_band(N)[1] for N in (256, 1024, 4096)]
    assert uppers[0] > uppers[1] > uppers[2]


def test_coincidence_diagonal_when_aligned(rng):
    n = 48
    Q = _orthonormal(n, rng)
    evals = np.sort(rng.uniform(1, 5, n))[::-1]
    C = Q @ np.diag(evals) @ Q.T
    svals = np.sort(rng.uniform(1, 3, n))[::-1]
    W = (Q * svals) @ Q.T
    res = OV.eigenvector_eigenvalue_coincidence(W, C)
    assert abs(res["diagonal_coincidence"] - 1.0) < 1e-6
    assert res["argmax_singular_for_top_eigenvector"] == 0


def test_coincidence_keys_present(rng):
    n = 32
    W = rng.standard_normal((n, n))
    C = rng.standard_normal((n, n)); C = C @ C.T
    res = OV.eigenvector_eigenvalue_coincidence(W, C)
    for k in ("cos_matrix", "svals_desc", "evals_desc",
              "rho_top_eigenvector_vs_svals", "rho_top_singular_vs_evals",
              "rho_diag_vs_svals", "max_overlap_with_top_eigenvector",
              "argmax_singular_for_top_eigenvector", "diagonal_coincidence"):
        assert k in res
        v = res[k]
        if np.isscalar(v):
            assert np.isfinite(v)


def test_resolve_fm_key():
    keys = ["model.layers.4.self_attn.q_proj", "model.layers.4.self_attn.k_proj",
            "gpt_neox.layers.0.attention.query_key_value"]
    assert OV.resolve_fm_key("model.layers.4.self_attn.q_proj.weight", keys) == \
        "model.layers.4.self_attn.q_proj"
    assert OV.resolve_fm_key("gpt_neox.layers.0.attention.query_key_value.weight[Q]",
                             keys) == "gpt_neox.layers.0.attention.query_key_value"
    assert OV.resolve_fm_key("model.layers.9.mlp.gate_proj.weight", keys) is None
