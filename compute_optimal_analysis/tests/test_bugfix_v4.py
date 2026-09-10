import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import run_experiments as runner
from pipelines.activation_extractor import CovarianceAccumulator, compute_activation_covariances
from pipelines.spectral_lesioning import independent_decile_benchmark, independent_lesion_benchmark
from pipelines.trainer import LanguageModelTrainer, TrainConfig, evaluate_language_model
from rmt.factory import RMTMethodConfig, dispatch_mp_fit, dispatch_spike_detector
from rmt.lanczos_stieltjes import detect_spikes_from_factor
from rmt.mp import tracy_widom_upper_threshold
from rmt.tail import hill_plateau


class _OneMatrixModel:
    def __init__(self, weight):
        self.weight = nn.Parameter(torch.as_tensor(weight, dtype=torch.float32))

    def iter_spectral_weights(self):
        yield "layers.0.q_proj.weight", self.weight


def _cell():
    return {
        "cell_index": 0, "compute_budget": 1.0, "regime": "test", "kappa": 1.0,
        "requested_parameters": 1.0, "realized_parameters": 1.0,
        "requested_tokens": 1.0, "realized_tokens": 1.0,
        "realized_training_compute": 1.0,
    }


def test_farms_esd_is_separate_from_single_operator_spacing():
    a = np.random.default_rng(22).normal(size=(32, 32))
    weight = np.vstack((a, a))
    config = RMTMethodConfig(
        mp_fit_method="analytic_mp", aspect_ratio_mode="farms_normalized",
        spike_detector="bbp_transition", farms_window_size=32,
        farms_row_windows=2, farms_column_windows=1, tail_minimum=4,
    )
    rows, artifacts = runner.analyze_model(
        _OneMatrixModel(weight), {}, _cell(), config,
        compute_activation_overlap=False, compute_number_variance=False,
    )
    row = rows[0]
    assert row["spectrum_observation_count"] == 64
    assert row["spacing_observation_count"] == 32
    assert artifacts[0]["spacings"].size <= 31
    assert row["r_statistic"] > 0.0


def test_golden_lanczos_keeps_raw_operator_and_executes_farms_esd():
    weight = np.random.default_rng(2).normal(size=(32, 64))
    config = RMTMethodConfig(
        mp_fit_method="lanczos_stieltjes", aspect_ratio_mode="farms_normalized",
        spike_detector="lanczos_poles", farms_window_size=16,
        farms_row_windows=2, farms_column_windows=2, tail_minimum=4,
        lanczos_steps=16, lanczos_probes=1,
    )
    rows, _ = runner.analyze_model(
        _OneMatrixModel(weight), {}, _cell(), config,
        compute_activation_overlap=False, compute_number_variance=False,
    )
    row = rows[0]
    assert row["spectrum_mode"] == "farms_canonical"
    assert row["mp_spectrum_domain"] == "raw"
    assert row["spike_detector_domain"] == "raw"
    assert row["spectrum_observation_count"] != row["mp_observation_count"]


def test_incompatible_farms_fit_is_rejected_and_tw_uses_operator_geometry():
    weight = np.random.default_rng(3).normal(size=(32, 64))
    with pytest.raises(ValueError, match="requires aspect_ratio_mode"):
        dispatch_mp_fit(weight, RMTMethodConfig(
            mp_fit_method="farms_unbiased", aspect_ratio_mode="raw"))

    values = np.linspace(0.1, 4.0, 32)
    config = RMTMethodConfig(spike_detector="tracy_widom_95")
    first = dispatch_spike_detector(
        weight, config, eigenvalues=values, aspect_ratio=1.0,
        operator_shape=(32, 32))
    repeated = dispatch_spike_detector(
        weight, config, eigenvalues=np.tile(values, 10), aspect_ratio=1.0,
        operator_shape=(32, 32))
    assert first.threshold == repeated.threshold
    assert np.isclose(first.threshold, tracy_widom_upper_threshold(32, 32, 1.0))


def test_aliases_survive_runtime_presets_and_abbreviations_are_disabled():
    args = runner.parse_args([
        "--experiment-mode", "reproduce_paper2", "--unfolding-degree", "3"])
    runner._resolve_runtime_configuration(args)
    assert args.polynomial_degree == 3
    args = runner.parse_args(["--overlap-mode=frobenius_projection"])
    runner._resolve_runtime_configuration(args)
    assert args.overlap_metric == "frobenius_projection"
    with pytest.raises(SystemExit):
        runner.parse_args(["--overlap-m", "frobenius_projection"])


def test_gap_ratio_survives_unfolding_failure(monkeypatch):
    weight = np.random.default_rng(18).normal(size=(64, 128))
    monkeypatch.setattr(runner, "dispatch_unfolding", lambda *_: (_ for _ in ()).throw(
        ValueError("forced fold")))
    rows, _ = runner.analyze_model(
        _OneMatrixModel(weight), {}, _cell(),
        RMTMethodConfig(mp_fit_method="analytic_mp", aspect_ratio_mode="raw",
                        spike_detector="bbp_transition", tail_minimum=4),
        compute_activation_overlap=False, compute_number_variance=False,
    )
    row = rows[0]
    assert math.isfinite(row["r_statistic"])
    assert row["unfolding_status"].startswith("unavailable")


def test_float32_covariance_is_stable_for_large_mean_small_variance():
    x = (1.0 + 1e-4 * np.random.default_rng(7).normal(size=(4096, 4))).astype(np.float32)
    accumulator = CovarianceAccumulator(4, dtype=torch.float32, device="cpu")
    accumulator.update(torch.from_numpy(x[:2000]))
    accumulator.update(torch.from_numpy(x[2000:]))
    observed = accumulator.covariance().numpy().astype(np.float64)
    reference = np.cov(x.astype(np.float64), rowvar=False, bias=True)
    assert np.linalg.norm(observed - reference) / np.linalg.norm(reference) < 0.01
    assert np.linalg.eigvalsh(observed).min() > -1e-11


class _MixedModeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.proj = nn.Linear(4, 16)
        self.dropout = nn.Dropout()

    def forward(self, input_ids, labels=None, attention_mask=None):
        del attention_mask
        logits = self.proj(self.dropout(self.embedding(input_ids)))
        loss = (None if labels is None else torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1)))
        return SimpleNamespace(loss=loss, logits=logits)


def test_measurement_helpers_restore_every_module_mode():
    model = _MixedModeLM().train()
    model.dropout.eval()
    expected = [module.training for module in model.modules()]
    batch = {"input_ids": torch.tensor([[1, 2, 3]]),
             "labels": torch.tensor([[1, 2, 3]])}
    evaluate_language_model(model, [batch], "cpu")
    assert [module.training for module in model.modules()] == expected
    compute_activation_covariances(
        model, [batch], device="cpu", module_filter=("proj",), max_batches=1)
    assert [module.training for module in model.modules()] == expected


def test_nonfinite_gradients_fail_before_optimizer_mutation():
    class ScalarLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, **kwargs):
            del kwargs
            return SimpleNamespace(loss=self.weight.square())

    model = ScalarLoss()
    model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    trainer = LanguageModelTrainer(model, TrainConfig(
        device="cpu", amp_dtype="float32", max_train_tokens=1, warmup_steps=0))
    batch = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[1, 2]])}
    with pytest.raises(FloatingPointError, match="gradient norm"):
        trainer.fit([batch])
    assert model.weight.item() == 1.0
    assert trainer.global_step == trainer.processed_train_tokens == 0


def test_epoch_only_no_target_loader_terminates_and_evaluation_is_honest():
    model = _MixedModeLM()
    ignored = {"input_ids": torch.tensor([[1, 2]]),
               "labels": torch.tensor([[-100, -100]])}
    trainer = LanguageModelTrainer(model, TrainConfig(
        device="cpu", amp_dtype="float32", epochs=1, warmup_steps=0))
    with pytest.raises(RuntimeError, match="no optimizer updates"):
        trainer.fit([ignored])

    class ConstantLoss(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.0))
            self.value = value

        def forward(self, **kwargs):
            del kwargs
            return SimpleNamespace(loss=self.anchor * 0 + self.value)

    batch = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[1, 2]])}
    p81 = evaluate_language_model(ConstantLoss(81.0), [batch], "cpu")
    p100 = evaluate_language_model(ConstantLoss(100.0), [batch], "cpu")
    assert p81["perplexity"] != p100["perplexity"]
    with pytest.raises(FloatingPointError):
        evaluate_language_model(ConstantLoss(float("inf")), [batch], "cpu")


def test_lesion_benchmarks_reject_nonfinite_interventions_and_restore_weights():
    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    name = "0.weight"
    pristine = model[0].weight.detach().clone()
    calls = iter((1.0, float("nan")))
    with pytest.raises(ValueError, match="lesion evaluation"):
        independent_lesion_benchmark(model, [name], lambda: next(calls), tranches=("top",))
    assert torch.equal(model[0].weight, pristine)
    calls = iter((1.0, float("inf")))
    with pytest.raises(ValueError, match="decile 0"):
        independent_decile_benchmark(model, [name], lambda: next(calls), n_deciles=1)
    assert torch.equal(model[0].weight, pristine)


def test_lanczos_margin_and_hill_support_are_scale_correct():
    factor = np.random.default_rng(0).normal(size=(128, 256))
    factor[0] *= 6.0
    unit = detect_spikes_from_factor(factor, steps=50, n_probes=1, rng=0)
    scaled = detect_spikes_from_factor(0.02 * factor, steps=50, n_probes=1, rng=0)
    assert unit.n_spikes == scaled.n_spikes == 1
    assert np.isclose(scaled.threshold / unit.threshold, 0.02 ** 2, rtol=1e-6)
    plateau = hill_plateau(np.arange(1.0, 201.0) ** -0.5, window=20)
    assert plateau["hill_plateau_end_rank"] == (
        plateau["hill_plateau_start_rank"]
        + plateau["hill_support_observations"] - 1)
