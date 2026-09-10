from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from rmt import activations, decile, pipeline, tail
from rmt.config import RunConfig, effective_context_length
from rmt.linalg import SVDResult
from rmt.plots.esd import _bin_edges


class _PositionModel(nn.Module):
    def __init__(self, padding_idx):
        super().__init__()
        self.position_embeddings = nn.Embedding(16, 4, padding_idx=padding_idx)
        self.config = SimpleNamespace(max_position_embeddings=64)


def test_histogram_edges_are_bounded_for_nearly_constant_spectrum():
    values = np.linspace(1.0 - 1e-10, 1.0 + 1e-10, 64)
    edges = _bin_edges(values, 0.0, 2.475414472)
    assert 2 < edges.size <= 401
    assert np.all(np.isfinite(edges))
    assert np.all(np.diff(edges) > 0.0)
    assert edges[0] <= 0.0 and edges[-1] >= 2.475414472


def test_decile_rejects_degraded_factorization_before_mutation(monkeypatch):
    from _synthetic_models import tiny_llama
    from rmt.discovery import discover_weight_matrices

    model = tiny_llama(n_layers=1, d=16)
    records = [r for r in discover_weight_matrices(model) if r.short == "Q"]
    parameter = model.get_submodule("model.layers.0.self_attn.q_proj").weight
    pristine = parameter.detach().clone()

    def degraded(weight, **kwargs):
        del kwargs
        matrix = np.asarray(weight, dtype=np.float32)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        return SVDResult(u, s, vh, *matrix.shape, backend="injected",
                         factorization_dtype="float32", degraded=True)

    monkeypatch.setattr("rmt.linalg.cached_svd", degraded)
    with pytest.raises(RuntimeError, match="non-degraded float64"):
        decile.set_layer_svd_decile(model, records, 1)
    assert torch.equal(parameter, pristine)


def test_effective_context_accounts_for_positive_position_offset():
    assert effective_context_length(_PositionModel(None), 2048) == 16
    assert effective_context_length(_PositionModel(1), 2048) == 14


def test_activation_capture_restores_mixed_module_modes(tmp_path):
    from _synthetic_models import tiny_causal_lm

    model = tiny_causal_lm(n_layers=1, d=16).train()
    dropout = next((m for m in model.modules() if isinstance(m, nn.Dropout)), None)
    if dropout is None:
        model.mode_probe = nn.Dropout()
        dropout = model.mode_probe
    dropout.eval()
    expected = [module.training for module in model.modules()]
    activations.compute_activation_covariance(
        model, None, [0], "cpu", n_text_batches=1, max_length=4, stride=4,
        text_path=str(tmp_path / "missing.txt"), allow_fallback=True,
        allow_tokenizer_fallback=True, target_names=["model.layers.0.self_attn.q_proj"],
    )
    assert [module.training for module in model.modules()] == expected


def test_pipeline_omits_numeric_filter_for_unindexed_exact_target(monkeypatch):
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return {"q_proj": {"FM": np.eye(2)}}

    monkeypatch.setattr(activations, "compute_activation_covariance", capture)
    model = nn.Linear(2, 2)
    record = SimpleNamespace(layer_idx=-1, name="q_proj.weight")
    result = pipeline._maybe_activation_cov(
        model, [record], RunConfig(strict=True), object())
    assert result
    assert seen["layer_indices"] is None
    assert seen["target_names"] == ["q_proj"]


def test_hill_observation_support_includes_boundary_sample():
    values = np.arange(1.0, 201.0) ** -0.5
    result = tail.hill_plateau(values, window=20)
    assert result["hill_plateau_end_rank"] == (
        result["hill_plateau_start_rank"]
        + result["hill_support_observations"] - 1)
    # Every selected spacing window needs its final cutoff observation.
    assert result["hill_support_observations"] >= result["hill_plateau_width"] + 20
