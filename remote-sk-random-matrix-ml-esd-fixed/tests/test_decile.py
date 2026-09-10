import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from tests.synthetic_models import tiny_llama, tiny_pythia, tiny_causal_lm   # noqa: E402

from rmt import decile as DEC                                   # noqa: E402
from rmt import discovery as D                                   # noqa: E402

pytestmark = pytest.mark.torch


def test_zeroing_decile_reduces_rank():
    m = tiny_llama(n_layers=1, d=32)
    recs = [r for r in D.discover_weight_matrices(m) if r.short == "Q"]
    q = m.get_submodule("model.layers.0.self_attn.q_proj")
    before = np.linalg.matrix_rank(q.weight.detach().numpy())
    DEC.set_layer_svd_decile(m, recs, decile=5, n_deciles=10)
    after = np.linalg.matrix_rank(q.weight.detach().numpy())
    assert after < before


def test_decile_out_of_range_raises():
    m = tiny_llama(n_layers=1, d=16)
    recs = D.discover_weight_matrices(m)
    with pytest.raises(ValueError):
        DEC.set_layer_svd_decile(m, recs, decile=0)
    with pytest.raises(ValueError):
        DEC.set_layer_svd_decile(m, recs, decile=11)


def test_smallest_decile_changes_small_svs():
    m = tiny_llama(n_layers=1, d=32)
    recs = [r for r in D.discover_weight_matrices(m) if r.short == "O"]
    o = m.get_submodule("model.layers.0.self_attn.o_proj")
    s_before = np.sort(np.linalg.svd(o.weight.detach().numpy(), compute_uv=False))
    DEC.set_layer_svd_decile(m, recs, decile=1, n_deciles=10)   # smallest 10%
    s_after = np.sort(np.linalg.svd(o.weight.detach().numpy(), compute_uv=False))
    k = len(s_after) // 10
    # weights are float32, so reconstruction leaves ~1e-7 noise, not exact 0
    assert np.sum(s_after < 1e-5) >= max(1, k)
    assert np.allclose(s_before[-k:], s_after[-k:], atol=1e-4)


def test_fused_qkv_writes_only_its_block():
    m = tiny_pythia(n_layers=1, d=48)
    recs = D.discover_weight_matrices(m)
    qrec = [r for r in recs if r.short == "Q"]
    mod = m.get_submodule("gpt_neox.layers.0.attention.query_key_value")
    W0 = mod.weight.detach().numpy().copy()         # head-interleaved (3d, d)
    DEC.set_layer_svd_decile(m, qrec, decile=3, n_deciles=10)
    W1 = mod.weight.detach().numpy()
    for index, tag in enumerate(("Q", "K", "V")):
        before = D.extract_qkv_block(W0, index, num_heads=4, interleaved=True)
        after = D.extract_qkv_block(W1, index, num_heads=4, interleaved=True)
        assert (not np.allclose(before, after)) if tag == "Q" else np.allclose(before, after)


def test_top_decile_preserves_small_svs():
    m = tiny_llama(n_layers=1, d=32)
    recs = [r for r in D.discover_weight_matrices(m) if r.short == "V"]
    v = m.get_submodule("model.layers.0.self_attn.v_proj")
    s_before = np.sort(np.linalg.svd(v.weight.detach().numpy(), compute_uv=False))
    DEC.set_layer_svd_decile(m, recs, decile=10, n_deciles=10)  # largest 10%
    s_after = np.sort(np.linalg.svd(v.weight.detach().numpy(), compute_uv=False))
    k = len(s_after) // 10
    # zeroing the largest decile removes the top SVs (max drops) and the
    # original small/mid values survive as a set (they reappear in s_after).
    assert s_after.max() < s_before.max() - 1e-4
    assert np.sum(s_after < 1e-5) >= max(1, k)
    surviving = s_before[: len(s_before) - k]                  # all but the top k
    for val in surviving[:k]:                                  # smallest few survive
        assert np.min(np.abs(s_after - val)) < 1e-4


def test_decile_index_consistency_with_scalars():
    from rmt.scalars import decile_index_ranges
    rngs = decile_index_ranges(50, 10, ascending=True)
    assert rngs[0] == (0, 5)
    assert rngs[-1] == (45, 50)


def test_decile_scope_all_vs_analyzed(monkeypatch):
    # spy on set_layer_svd_decile to count how many records get ablated per scope
    seen = {"all": [], "analyzed": []}
    real = DEC.set_layer_svd_decile

    def make_factory():
        return lambda: tiny_causal_lm(n_layers=2, d=16)

    def fake_ppl(model, tokenizer, device, **kw):
        return ({"perplexity": 1.0, "scored_tokens": 7}
                if kw.get("return_details") else 1.0)

    monkeypatch.setattr("rmt.perplexity.perplexity_wikitext", fake_ppl, raising=False)

    # count records touched in scope='all'
    m_all = tiny_causal_lm(n_layers=2, d=16)
    recs_all = D.discover_weight_matrices(m_all)
    analyzed = [r for r in recs_all if r.layer_idx == 0 and r.short == "Q"]

    captured = {}

    def spy(model, records, decile, **kw):
        captured.setdefault(decile, len(records))
        return real(model, records, decile, **kw)

    monkeypatch.setattr(DEC, "set_layer_svd_decile", spy)

    DEC.perplexity_vs_decile(make_factory(), None, analyzed, "cpu",
                             n_tokens=8, decile_scope="analyzed", n_deciles=2)
    n_analyzed = captured[1]
    captured.clear()
    DEC.perplexity_vs_decile(make_factory(), None, analyzed, "cpu",
                             n_tokens=8, decile_scope="all", n_deciles=2)
    n_all = captured[1]
    assert n_all > n_analyzed                        # 'all' touches more matrices
