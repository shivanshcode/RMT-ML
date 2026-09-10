import numpy as np
import pytest

from rmt import svd_cache as SC


def test_save_load_roundtrip(tmp_path):
    U = np.random.default_rng(0).standard_normal((10, 4))
    s = np.array([4.0, 3.0, 2.0, 1.0])
    Vh = np.random.default_rng(1).standard_normal((4, 6))
    SC.save_svd(str(tmp_path), "model.layers.0.self_attn.q_proj.weight", U, s, Vh)
    out = SC.load_svd(str(tmp_path), "model.layers.0.self_attn.q_proj.weight")
    assert out is not None
    U2, s2, Vh2 = out
    assert np.allclose(U, U2) and np.allclose(s, s2) and np.allclose(Vh, Vh2)


def test_missing_returns_none(tmp_path):
    assert SC.load_svd(str(tmp_path), "does.not.exist.weight") is None


def test_fused_tag_filename_sanitized(tmp_path):
    s = np.array([2.0, 1.0])
    p = SC.save_svd(str(tmp_path), "gpt_neox.layers.0.attention.query_key_value.weight[Q]",
                    np.zeros((2, 2)), s, np.zeros((2, 2)))
    assert "[" not in p and "]" not in p and "'" not in p and " " not in p
    out = SC.load_svd(str(tmp_path),
                      "gpt_neox.layers.0.attention.query_key_value.weight[Q]")
    assert out is not None and np.allclose(out[1], s)
