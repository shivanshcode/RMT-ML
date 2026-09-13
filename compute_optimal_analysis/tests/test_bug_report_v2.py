import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from models.chinchilla_scaling import suggest_architecture
from models.modules import CausalSelfAttention
from pipelines.activation_extractor import ActivationExtractor
from pipelines.spectral_lesioning import lesion_matrix
from rmt.farms_aspect_ratio import farms_spectrum
from rmt.lanczos_stieltjes import (
    covariance_linear_operator,
    detect_spikes_from_factor,
    extended_stieltjes_transform,
)
from rmt.mp import marchenko_pastur_cdf, marchenko_pastur_density, small_sv_deviation
from rmt.tail import rank_ordered_mle
from scripts import download_assets


def test_spacing_fit_is_skipped_when_disabled_or_rank_is_insufficient(monkeypatch) -> None:
    import run_experiments
    from rmt.factory import RMTMethodConfig

    called = False

    def unexpected_fit(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError((args, kwargs))

    monkeypatch.setattr(run_experiments, "fit_marchenko_pastur", unexpected_fit)
    weight = np.diag([3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    eigenvalues = np.asarray([9.0, 4.0, 1.0, 0.0, 0.0, 0.0]) / 6.0
    svd = SimpleNamespace(aspect_ratio=1.0, factorization_dtype="float64")
    headline = SimpleNamespace(diagnostics={"available": False})
    for requested in (False, True):
        values, fitted = run_experiments._prepare_spacing_fit(
            weight, eigenvalues, svd, RMTMethodConfig(), headline,
            headline_is_raw=False, requested=requested,
        )
        assert np.count_nonzero(values) == 3
        assert fitted is None
    assert called is False


def test_analyze_model_records_an_unavailable_spacing_fit(monkeypatch) -> None:
    import run_experiments
    from rmt.factory import RMTMethodConfig

    class OneMatrixModel:
        def __init__(self):
            self.parameter = nn.Parameter(torch.randn(32, 32, dtype=torch.float64))

        def iter_spectral_weights(self):
            yield "layers.0.q_proj.weight", self.parameter

    def unavailable_spacing(weight, raw_eigenvalues, *args, **kwargs):
        del weight, args, kwargs
        return raw_eigenvalues, None

    monkeypatch.setattr(run_experiments, "_prepare_spacing_fit", unavailable_spacing)
    cell = {
        "cell_index": 0,
        "compute_budget": 1.0,
        "regime": "calibration",
        "kappa": 1.0,
        "requested_parameters": 1024.0,
        "realized_parameters": 1024.0,
        "requested_tokens": 100.0,
        "realized_tokens": 100.0,
        "realized_training_compute": 1.0,
    }
    rows, _ = run_experiments.analyze_model(
        OneMatrixModel(),
        {},
        cell,
        RMTMethodConfig(
            mp_fit_method="analytic_mp",
            aspect_ratio_mode="raw",
            spike_detector="tracy_widom_95",
            tail_minimum=10,
        ),
        svd_backend="cpu",
        compute_activation_overlap=False,
    )
    assert len(rows) == 1
    assert rows[0]["spacing_bulk_fit_available"] is False
    assert np.isnan(rows[0]["spacing_bulk_lambda_minus"])
    assert np.isnan(rows[0]["spacing_bulk_lambda_plus"])
    assert rows[0]["unfolding_status"] == "unavailable: raw-domain MP fit unavailable"


def test_fp16_attention_scores_do_not_overflow_before_scaling() -> None:
    torch.manual_seed(4)
    reference = CausalSelfAttention(64, 1, 1, rope_theta=None).eval()
    half = CausalSelfAttention(64, 1, 1, rope_theta=None).eval().half()
    half.load_state_dict(reference.state_dict())
    hidden32 = torch.full((1, 3, 64), 40.0, requires_grad=True)
    hidden16 = hidden32.detach().half().requires_grad_(True)
    expected = reference(hidden32)
    observed = half(hidden16).float()
    assert torch.all(torch.isfinite(observed))
    assert torch.allclose(observed, expected, atol=0.25, rtol=0.02)
    observed.square().mean().backward()
    assert torch.all(torch.isfinite(hidden16.grad))


def test_energy_match_status_is_invariant_to_weight_units() -> None:
    weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64))
    _, base = lesion_matrix(weight, "top", fraction=0.05, mode="energy")
    _, scaled = lesion_matrix(weight * 1e-10, "top", fraction=0.05, mode="energy")
    assert base.indices == scaled.indices
    assert base.energy_match_status == scaled.energy_match_status == "nearest_discrete"
    assert scaled.relative_energy_error == pytest.approx(base.relative_energy_error)


def test_rank_ordered_ks_uses_both_sides_of_empirical_jumps() -> None:
    result = rank_ordered_mle(np.asarray([1.0, 2.0, 4.0]), tail_fraction=1.0)
    assert result["ks_D"] == pytest.approx(1.0 / 3.0)


def test_lanczos_automatic_diagnostics_are_scale_relative() -> None:
    factor = np.random.default_rng(12).normal(size=(32, 64))
    options = dict(steps=24, adaptive=True, sequence_length=3, check_interval=2, rng=7)
    base = detect_spikes_from_factor(factor, **options)
    scaled = detect_spikes_from_factor(factor * 1e-10, **options)
    assert np.array_equal(base.iterations, scaled.iterations)
    assert np.array_equal(base.probe_converged, scaled.probe_converged)
    assert scaled.lambda_plus / 1e-20 == pytest.approx(base.lambda_plus, rel=1e-7)


def test_manifest_verification_rejects_added_and_duplicate_files(tmp_path: Path) -> None:
    asset = tmp_path / "data" / "tokenized" / "tokens.npy"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"tokens")
    manifest = {"files": download_assets._file_manifest(tmp_path)}
    download_assets._write_json(tmp_path / "data" / "asset_manifest.json", manifest)
    assert download_assets._verify_manifest(tmp_path) == 1
    (tmp_path / "data" / "tokenized" / "added.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="membership mismatch"):
        download_assets._verify_manifest(tmp_path)
    (tmp_path / "data" / "tokenized" / "added.json").unlink()
    manifest["files"].append(dict(manifest["files"][0]))
    download_assets._write_json(tmp_path / "data" / "asset_manifest.json", manifest)
    with pytest.raises(ValueError, match="duplicate path"):
        download_assets._verify_manifest(tmp_path)


def test_failed_restage_preserves_previous_asset_release(tmp_path: Path, monkeypatch) -> None:
    asset = tmp_path / "data" / "tokenized" / "synthetic_zipf.npy"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"old-release")
    manifest_path = tmp_path / "data" / "asset_manifest.json"
    download_assets._write_json(
        manifest_path,
        {"assets": {}, "source_revisions": {}, "files": download_assets._file_manifest(tmp_path)},
    )
    old_manifest = manifest_path.read_bytes()
    args = argparse.Namespace(
        root=tmp_path, assets="synthetic", allow_network=False, verify_only=False,
        max_wikitext_tokens=0, synthetic_tokens=20, synthetic_vocab_size=8, seed=3,
    )
    monkeypatch.setattr(download_assets, "parse_args", lambda: args)

    def fail_after_stage(paths, **kwargs):
        del kwargs
        (paths["tokenized"] / "synthetic_zipf.npy").write_bytes(b"new-release")
        raise RuntimeError("injected staging failure")

    monkeypatch.setattr(download_assets, "_generate_synthetic_assets", fail_after_stage)
    with pytest.raises(RuntimeError, match="injected"):
        download_assets.main()
    assert asset.read_bytes() == b"old-release"
    assert manifest_path.read_bytes() == old_manifest


def test_failed_hook_registration_removes_earlier_hooks() -> None:
    model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))

    def selection(name, module):
        del module
        if name == "1":
            raise RuntimeError("injected filter failure")
        return name == "0"

    extractor = ActivationExtractor(model, selection)
    with pytest.raises(RuntimeError, match="injected"):
        extractor.__enter__()
    assert extractor.handles == []
    model(torch.ones(1, 3))
    assert extractor.accumulators == {}


def test_architecture_search_rejects_an_invalid_rope_head_dimension() -> None:
    with pytest.raises(ValueError, match="no valid architecture"):
        suggest_architecture(
            8484, vocab_size=512, width_multiple=12, max_layers=1, max_parameters=9000
        )
    result = suggest_architecture(50_000, vocab_size=512, width_multiple=12, max_layers=2)
    assert (result["d_model"] // result["n_heads"]) % 2 == 0


def test_real_only_alternate_entry_points_reject_complex_values() -> None:
    complex_numpy = 1j * np.eye(4)
    complex_torch = torch.as_tensor(complex_numpy)
    with pytest.raises(TypeError):
        lesion_matrix(complex_torch, "top")
    with pytest.raises(TypeError):
        farms_spectrum(complex_numpy)
    with pytest.raises(TypeError):
        covariance_linear_operator(complex_numpy)
    with pytest.raises(TypeError):
        detect_spikes_from_factor(complex_numpy)


def test_mp_density_cdf_and_energy_fraction_survive_extreme_units() -> None:
    base_density = float(marchenko_pastur_density(1.0, 1.0, 1.0))
    for scale in (1e-200, 1e200):
        density = float(marchenko_pastur_density(scale, 1.0, scale))
        assert np.isfinite(density)
        assert density * scale == pytest.approx(base_density)
        assert marchenko_pastur_cdf(scale, 1.0, scale) == pytest.approx(
            marchenko_pastur_cdf(1.0, 1.0, 1.0)
        )
    spectrum = np.asarray([1.0, 2.0, 3.0, 4.0])
    base = small_sv_deviation(spectrum, 4, 16, 1.0)
    for scale in (1e-200, 1e200):
        scaled = small_sv_deviation(spectrum * scale, 4, 16, scale)
        assert scaled["frac_mass_below_minus"] == pytest.approx(
            base["frac_mass_below_minus"]
        )


def test_constant_tail_transform_has_finite_zero_limit() -> None:
    observed = extended_stieltjes_transform(
        0j, np.asarray([2.0]), np.asarray([]), tail_alpha=2.0, tail_beta=1.0
    )
    assert observed == pytest.approx(1.0 / 3.0)
