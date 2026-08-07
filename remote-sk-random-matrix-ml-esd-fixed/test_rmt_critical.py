"""Regression tests for the CRITICAL rmt bugs (verified report).

Every test here is written to FAIL on the current code and PASS once the
corresponding fix in REPORT_consolidated.md is applied.  They target the bugs
that change scientific output or only bite on the real A100 / fp16 run (which
the existing 106-test suite never exercises).

Run:  pytest test_rmt_critical.py -q
The pure-numpy tests need no torch; the rest are marked `torch`.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))  # for _synthetic_models (torch tests)


# ---------------------------------------------------------------------------- #
# PRECISION: the small-singular-value pipeline (Paper 3's whole thesis)         #
# ---------------------------------------------------------------------------- #
def _ill_conditioned(n=256, lo_exp=-8, rng=None):
    """Orthogonal U,V with a planted SV tail spanning 1 .. 10**lo_exp."""
    rng = rng or np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    true_s = np.logspace(0, lo_exp, n)
    return (U * true_s) @ V.T, true_s


@pytest.mark.torch
def test_torch_svd_preserves_small_singular_values():
    """Flaw 4.1: the torch SVD path must not cast to float32.

    A float32 SVD floors the small singular values at ~1e-7*smax, destroying
    exactly the regime Paper 3 studies. The torch backend must match the
    float64 numpy backend on the smallest singular values.
    """
    pytest.importorskip("torch")
    from rmt.linalg import cached_svd

    W, true_s = _ill_conditioned()
    s_np = np.sort(cached_svd(W, backend="numpy").s)[::-1]
    s_t = np.sort(cached_svd(W, backend="torch").s)[::-1]
    # relative error on the smallest 20 singular values
    tail = slice(-20, None)
    rel = np.abs(s_t[tail] - s_np[tail]) / np.maximum(s_np[tail], 1e-300)
    assert np.max(rel) < 1e-4, (
        f"torch path loses small-SV precision (max rel-err {np.max(rel):.2e}); "
        "use float64 in torch.linalg.svd")


@pytest.mark.torch
def test_weight_materialized_at_full_precision():
    """Flaw 4.3/4.2: discover_weight_matrices must not downcast through float32.

    A weight value that differs from its float32 rounding must survive
    materialization, or the smallest singular values become float32 noise.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from rmt.discovery import discover_weight_matrices, get_model_spec

    # value NOT representable in float32 (needs >24 bits of mantissa)
    val = 1.0 + 2.0 ** -40
    lin = nn.Linear(4, 4, bias=False).to(torch.float64)
    with torch.no_grad():
        lin.weight.fill_(val)
    m = nn.Module()
    m.config = type("C", (), {"model_type": "generic", "architectures": ["generic"]})()
    # NOTE (corrected): the module must carry a name the spec's classifier
    # recognises ("lin" matches nothing, so discovery would return 0 records and
    # the precision path would never be exercised). "q_proj" classifies as Q.
    m.q_proj = lin
    recs = discover_weight_matrices(m, spec=get_model_spec(m))
    assert recs, "no records discovered"
    w = np.asarray(recs[0].weight, dtype=np.float64)
    assert abs(float(w.flat[0]) - val) < 1e-15, (
        "weight was downcast to float32 during discovery (lost sub-float32 bits)")


# ---------------------------------------------------------------------------- #
# DEAD FLAGS that the SLURM passes (mislead about what actually ran)            #
# ---------------------------------------------------------------------------- #
class _Rec:
    short = "Q"
    layer_idx = 0

    def __init__(self, W):
        self.name = "blk.0.lin.weight"
        self.weight = W
        self.n, self.m = W.shape


def _pareto_matrix(n=200, alpha=3.0, rng=None):
    rng = rng or np.random.default_rng(1)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = (rng.pareto(alpha, n) + 1.0)
    return (U * np.sort(s)[::-1]) @ V.T


def test_do_powerlaw_false_skips_tail_fit():
    """Flaw 2.2: with do_powerlaw=False the CSN/Hill tail fits must not run."""
    from rmt.per_matrix import per_matrix_analysis
    from rmt.config import RunConfig

    W = _pareto_matrix()
    cfg = RunConfig(do_powerlaw=False, do_spacing=False, do_ipr=False,
                    do_overlap=False)
    row = per_matrix_analysis(_Rec(W), fm_dict=None, cfg=cfg)
    assert np.isnan(row["alpha"]) and np.isnan(row["alpha_hill_lambda"]), (
        "do_powerlaw=False but alpha/Hill were still computed (dead flag)")


def test_do_ipr_false_skips_ipr():
    """Flaw 2.3: with do_ipr=False the IPR summary must not run."""
    from rmt.per_matrix import per_matrix_analysis
    from rmt.config import RunConfig

    W = _pareto_matrix()
    cfg = RunConfig(do_ipr=False, do_powerlaw=False, do_spacing=False,
                    do_overlap=False)
    row = per_matrix_analysis(_Rec(W), fm_dict=None, cfg=cfg)
    assert np.isnan(row["ipr_top10_mean"]) and np.isnan(row["ipr_bulk_mean"]), (
        "do_ipr=False but IPR was still computed (dead flag)")


# ---------------------------------------------------------------------------- #
# MODEL-MUTATION: activation wrappers are never removed (NEW, verified)         #
# ---------------------------------------------------------------------------- #
@pytest.mark.torch
def test_feature_layers_inert_after_capture():
    """N1: after capturing the activation covariance, the model must be left so
    that ordinary forward passes (== perplexity) do NOT keep running the
    per-token O(d^2) covariance loop. Either the wrappers are removed, or they
    are switched off.
    """
    torch = pytest.importorskip("torch")
    from _synthetic_models import tiny_causal_lm
    from rmt.activations import compute_activation_covariance, FeatureLayer
    from rmt.discovery import get_model_spec

    m = tiny_causal_lm(n_layers=2, d=32, vocab=64)
    compute_activation_covariance(m, tokenizer=None, layer_indices=[0, 1],
                                  device="cpu", n_text_batches=2, max_length=16,
                                  stride=8, spec=get_model_spec(m))
    fls = [mod for _, mod in m.named_modules() if isinstance(mod, FeatureLayer)]
    counters_before = [f.fm_computation for f in fls]
    with torch.no_grad():
        m(input_ids=torch.zeros((1, 16), dtype=torch.long))
    counters_after = [f.fm_computation for f in fls]
    assert counters_before == counters_after, (
        "a plain forward kept accumulating the activation covariance — the "
        "FeatureLayer wrappers were never removed/disabled after capture")


@pytest.mark.torch
def test_perplexity_snapshot_is_weight_only():
    """N2: perplexity_vs_decile must not deepcopy giant float64 _cov buffers.

    After overlap capture the model carries (d_in x d_in) covariance buffers;
    snapshotting the full state_dict copies GBs on the real model. The snapshot
    used to restore weights between deciles must contain only weights.
    """
    torch = pytest.importorskip("torch")
    from _synthetic_models import tiny_causal_lm
    from rmt.activations import compute_activation_covariance
    from rmt.discovery import get_model_spec, discover_weight_matrices
    from rmt.decile import perplexity_vs_decile

    m = tiny_causal_lm(n_layers=2, d=32, vocab=64)
    spec = get_model_spec(m)
    compute_activation_covariance(m, tokenizer=None, layer_indices=[0, 1],
                                  device="cpu", n_text_batches=2, max_length=16,
                                  stride=8, spec=spec)
    recs = discover_weight_matrices(m, spec=spec)

    import copy
    seen = {}

    real_deepcopy = copy.deepcopy

    def spy(obj, *a, **k):
        if isinstance(obj, dict):
            seen["keys"] = list(obj.keys())
        return real_deepcopy(obj, *a, **k)

    copy.deepcopy = spy
    try:
        perplexity_vs_decile(lambda: m, None, recs, "cpu", n_tokens=16,
                             decile_scope="analyzed", n_deciles=2, spec=spec,
                             text_path="/nonexistent")
    finally:
        copy.deepcopy = real_deepcopy

    cov_keys = [k for k in seen.get("keys", []) if k.endswith("_cov")]
    assert not cov_keys, (
        f"perplexity snapshot deepcopied {len(cov_keys)} covariance buffers "
        "(should snapshot weights only)")


# ---------------------------------------------------------------------------- #
# DECILE ablation precision (fp16 store-back)                                   #
# ---------------------------------------------------------------------------- #
@pytest.mark.torch
def test_decile_ablation_not_floored_by_fp16():
    """Flaw 4.4: zeroing a decile and storing back must not quantise to fp16.

    The ablation must reconstruct in high precision and store back at >= float32.
    With an fp16 store-back the result is re-quantised to ~1e-3 rel-err and the
    small-SV signal Paper 3 ablates is destroyed.

    NOTE (corrected reference): the model here is *built* in fp16 only to stress
    the store-back path, so the fp16 representation of the input matrix is itself
    ~2e-4 in Frobenius norm. A float64 reference computed from the original
    full-precision ``W`` is therefore unreachable from an fp16 input regardless of
    the store-back dtype (input quantisation alone exceeds 1e-4). To isolate the
    bug under test — the store-back quantisation — the reference is reconstructed
    from the SAME fp16 input the code reads. A correct >= float32 store yields
    ~1e-8; the buggy fp16 store yields ~2e-4, so 1e-4 cleanly separates them.
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from rmt.decile import _reconstruct_zeroed, set_layer_svd_decile
    from rmt.discovery import get_model_spec, discover_weight_matrices
    from rmt.scalars import decile_index_ranges

    rng = np.random.default_rng(3)
    W = rng.standard_normal((48, 32)).astype(np.float64)
    k = min(W.shape)
    lo, hi = decile_index_ranges(k, 10, ascending=True)[0]

    # build a half-precision model holding W (discoverable name so the ablation
    # actually runs) and ablate decile 1 in place
    lin = nn.Linear(32, 48, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.as_tensor(W))
    lin = lin.half()
    m = nn.Module()
    m.config = type("C", (), {"model_type": "generic", "architectures": ["generic"]})()
    m.q_proj = lin
    spec = get_model_spec(m)
    recs = discover_weight_matrices(m, spec=spec)
    assert recs, "no records discovered (the ablation never ran)"

    # reference: reconstruct from the SAME fp16 input the code will read, so the
    # comparison isolates the store-back precision (not the input's fp16 error).
    w_in = lin.weight.detach().cpu().double().numpy()
    ref = _reconstruct_zeroed(w_in, lo, hi)

    set_layer_svd_decile(m, recs, 1, n_deciles=10, spec=spec)
    got = lin.weight.detach().cpu().double().numpy()

    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 1e-4, (
        f"decile ablation floored at fp16 (rel-err {rel:.2e}); reconstruct and "
        "store at >= float32")


# ---------------------------------------------------------------------------- #
# PLOT WIRING: only summary.png is produced today                              #
# ---------------------------------------------------------------------------- #
@pytest.mark.torch
def test_per_matrix_plots_are_emitted(tmp_path):
    """Flaws 1.2/1.3/1.6: enabling the analyses must emit their plots."""
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from _synthetic_models import tiny_causal_lm
    from rmt.pipeline import analyze_one_model

    m = tiny_causal_lm(n_layers=2, d=64, vocab=64)
    out = str(tmp_path)
    analyze_one_model(m, "tiny", out, tokenizer=None,
                      do_powerlaw=True, do_spacing=True, do_overlap=False,
                      do_perplexity=False)
    esd = os.path.isdir(os.path.join(out, "esd"))
    hill = os.path.isdir(os.path.join(out, "hill"))
    assert esd and hill, (
        "per-matrix ESD/Hill plots were not emitted (only summary.png is wired)")
