import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _synthetic_models import tiny_llama, tiny_pythia          # noqa: E402

from rmt import discovery as D                                  # noqa: E402

pytestmark = pytest.mark.torch


def test_get_model_spec_known_archs():
    assert D.get_model_spec("meta-llama/Llama-3.1-8B").name == "llama"
    assert D.get_model_spec(tiny_llama()).name == "llama"
    assert D.get_model_spec(tiny_pythia()).name == "gpt_neox"


def test_get_model_spec_generic_fallback():
    spec = D.get_model_spec("some-unknown-model-xyz")
    assert spec.name == "generic"
    assert spec.patterns is D.MATRIX_PATTERNS


def test_classify_llama():
    spec = D.get_model_spec("llama")
    assert D.classify("model.layers.0.self_attn.q_proj", spec) == "Q"
    assert D.classify("model.layers.0.mlp.down_proj", spec) == "D"
    assert D.classify("model.layers.0.input_layernorm", spec) is None


def test_classify_pythia_fused():
    spec = D.get_model_spec("gpt_neox")
    assert D.classify("gpt_neox.layers.0.attention.query_key_value", spec) == "QKV"
    assert D.classify("gpt_neox.layers.0.mlp.dense_h_to_4h", spec) == "U"


def test_extract_layer_index():
    spec = D.get_model_spec("llama")
    assert D.extract_layer_index("model.layers.7.self_attn.q_proj", spec) == 7
    assert D.extract_layer_index("model.norm", spec) == -1


def test_fused_qkv_split_contiguous_thirds():
    spec = D.get_model_spec("gpt_neox")
    d = 16
    W = np.zeros((3 * d, d), dtype=np.float32)
    W[:d] = 1.0; W[d:2 * d] = 2.0; W[2 * d:] = 3.0
    recs = D.split_fused_qkv(W, "blk.query_key_value.weight", 0, spec)
    assert [r.short for r in recs] == ["Q", "K", "V"]
    assert np.allclose(recs[0].weight, 1.0)
    assert np.allclose(recs[1].weight, 2.0)
    assert np.allclose(recs[2].weight, 3.0)
    assert recs[0].name.endswith("[Q]")


def test_fused_qkv_divisibility_assert():
    spec = D.get_model_spec("gpt_neox")
    W = np.zeros((10, 4), dtype=np.float32)        # 10 % 3 != 0
    with pytest.raises(AssertionError):
        D.split_fused_qkv(W, "x.query_key_value.weight", 0, spec)


def test_discover_skips_embeddings_and_head():
    recs = D.discover_weight_matrices(tiny_llama(n_layers=2, d=32))
    names = [r.name for r in recs]
    assert not any("embed" in n for n in names)
    assert not any("lm_head" in n for n in names)
    shorts = {r.short for r in recs}
    assert {"Q", "K", "V", "O", "G", "U", "D"} <= shorts


def test_discover_respects_layer_filter():
    recs = D.discover_weight_matrices(tiny_llama(n_layers=4, d=32),
                                      layer_indices=[0, 2])
    assert {r.layer_idx for r in recs} == {0, 2}


def test_discover_pythia_yields_qkv_records():
    recs = D.discover_weight_matrices(tiny_pythia(n_layers=2, d=48))
    shorts = [r.short for r in recs]
    assert "Q" in shorts and "K" in shorts and "V" in shorts
    # each fused split has n == d (=48), m == d
    q = [r for r in recs if r.short == "Q"][0]
    assert q.n == 48 and q.m == 48
    assert q.name.endswith("[Q]")


def test_conv1d_transposed():
    # emulate HF Conv1D: weight stored as (in, out); discovery must transpose.
    class Conv1D(torch.nn.Module):
        def __init__(self, nf, nx):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(nx, nf))   # (in, out)
            self.nf = nf

    m = torch.nn.Module()
    m.config = type("C", (), {"model_type": "gpt2", "architectures": ["gpt2"]})()
    m.h = torch.nn.ModuleList()
    blk = torch.nn.Module()
    attn = torch.nn.Module()
    attn.c_attn = Conv1D(3 * 12, 12)        # fused QKV, (in=12, out=36)
    attn.c_proj = Conv1D(12, 12)
    blk.attn = attn
    m.h.append(blk)
    # patch module path so named_modules yields '.h.0.attn.c_attn'
    recs = D.discover_weight_matrices(m)
    cproj = [r for r in recs if r.short == "O"][0]
    assert (cproj.n, cproj.m) == (12, 12)
    qkv = [r for r in recs if r.short in ("Q", "K", "V")]
    assert len(qkv) == 3
    for r in qkv:
        assert (r.n, r.m) == (12, 12)       # 36 rows split into 3×12 after transpose
