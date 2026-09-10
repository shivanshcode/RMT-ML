"""Regressions for the resolved findings in bug_report.md."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from rmt import activations, cli, decile, overlap, per_matrix, pipeline, svd_cache, tail
from rmt.config import RunConfig, SEED
from rmt.discovery import discover_weight_matrices, get_model_spec


class _AliasedProjections(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.k_proj.weight = self.q_proj.weight
        self.config = SimpleNamespace(model_type="generic", architectures=["generic"])


def test_tied_parameter_is_lesioned_once_and_oversized_partition_fails_first():
    model = _AliasedProjections()
    with torch.no_grad():
        model.q_proj.weight.copy_(torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0])))
    records = discover_weight_matrices(model)
    decile.set_layer_svd_decile(model, records, 4, n_deciles=4, factor_cache={})
    assert torch.allclose(torch.linalg.svdvals(model.q_proj.weight),
                          torch.tensor([3.0, 2.0, 1.0, 0.0]), atol=1e-5)
    pristine = model.q_proj.weight.detach().clone()
    with pytest.raises(ValueError, match="exceeds singular count"):
        decile.set_layer_svd_decile(model, records, 1, n_deciles=5)
    assert torch.equal(model.q_proj.weight, pristine)


def test_explicit_multihead_projection_is_not_blacklisted():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.multihead_attention = nn.Module()
            self.multihead_attention.q_proj = nn.Linear(4, 4)
            self.lm_head = nn.Linear(4, 4)
            self.config = SimpleNamespace(model_type="generic", architectures=["generic"])

    model = Model()
    wrapped = activations.replace_with_feature_layers(
        model, None, "cpu", spec=get_model_spec(model),
        target_names=["multihead_attention.q_proj", "lm_head"],
    )
    assert wrapped == ["multihead_attention.q_proj"]
    activations.restore_linears(model)


def test_cli_selftest_seed_is_fixed(monkeypatch):
    seen = []
    monkeypatch.setattr("rmt.selftest.run", lambda seed, verbose=True: seen.append(seed) or True)
    assert cli.main(["--selftest", "--seed", "7"]) == 0
    assert seen == [SEED]


def test_degenerate_activation_overlap_is_unavailable():
    result = overlap.overlap_analysis(np.diag([4.0, 3.0, 2.0, 1.0]), np.zeros((4, 4)))
    assert result["available"] is False
    assert result["overlap"].size == 0


def test_bounded_tail_recovers_conditional_exponent():
    u = (np.arange(10000) + 0.5) / 10000
    sample = (1.0 - u * (1.0 - 2.0 ** -2)) ** -0.5
    fit = tail.fit_powerlaw_csn(sample, min_tail=1000, xmax=2)
    assert np.isclose(fit["alpha"], 3.0, atol=0.02)


def test_corrupt_cache_is_a_miss(tmp_path):
    weight = np.eye(3)
    digest = svd_cache.weight_digest(weight)
    path = svd_cache.save_svd(tmp_path, "q_proj.weight", weight, np.ones(3), weight,
                              digest=digest, factorization_dtype="float64")
    with open(path, "wb") as handle:
        handle.write(b"not an npz")
    assert svd_cache.load_svd(tmp_path, "q_proj.weight", digest=digest) is None


def test_random_control_matches_selected_hill_convention():
    record = SimpleNamespace(
        name="q_proj.weight", short="Q", layer_idx=-1,
        weight=np.random.default_rng(4).normal(size=(64, 64)), n=64, m=64,
    )
    row = per_matrix.per_matrix_analysis(record, cfg=RunConfig(
        alpha_estimator="hill", do_randomize=True, do_overlap=False,
        do_spacing=False, use_svd_cache=False,
    ))
    assert row["alpha_rand_estimator"] == "hill"
    assert row["alpha_rand_kind"] == row["alpha_kind"] == "survival"
    assert np.isfinite(row["alpha_rand_xmin"])


def test_requested_weightwatcher_unavailability_is_partial(tmp_path, monkeypatch):
    from _synthetic_models import tiny_llama
    import rmt.baselines

    monkeypatch.setattr(rmt.baselines, "run_weightwatcher", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_maybe_plots", lambda *args, **kwargs: None)
    output = tmp_path / "baseline"
    pipeline.analyze_one_model(
        tiny_llama(n_layers=1, d=16), "tiny", str(output),
        do_overlap=False, do_perplexity=False, do_ww=True, do_spacing=False,
        do_powerlaw=False, do_ipr=False, use_svd_cache=False,
    )
    import json
    status = json.loads((output / "tiny_run_status.json").read_text())
    baseline = json.loads((output / "tiny_weightwatcher_status.json").read_text())
    assert status["status"] == "partial"
    assert baseline["status"] == "unavailable"


def test_checkpoint_tracking_uses_configured_dispatcher(tmp_path, monkeypatch):
    from _synthetic_models import tiny_causal_lm
    import rmt.linalg

    real = rmt.linalg.cached_svd
    seen = []

    def spy(weight, **kwargs):
        seen.append(dict(kwargs))
        return real(weight, **kwargs)

    monkeypatch.setattr(rmt.linalg, "cached_svd", spy)
    path = pipeline.analyze_checkpoints(
        lambda fraction: tiny_causal_lm(n_layers=1, d=16), [0.0], [0],
        str(tmp_path), "checkpoint", backend="numpy", gpu_svd_min_dim=7,
    )
    assert seen and all(call["backend"] == "numpy" for call in seen)
    header = Path(path).read_text(encoding="utf-8").splitlines()[0]
    assert "svd_backend" in header and "svd_dtype" in header


def test_output_directory_has_exclusive_owner(tmp_path):
    output = tmp_path / "run"
    pipeline._acquire_output_directory(output)
    with pytest.raises(FileExistsError):
        pipeline._acquire_output_directory(output)
