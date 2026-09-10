"""Regressions for the resolved findings in bug_report.md."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import run_experiments as runner
from pipelines.trainer import LanguageModelTrainer, TrainConfig
from rmt.factory import RMTMethodConfig, dispatch_spike_detector
from rmt.mp import fit_marchenko_pastur_kde
from rmt.overlap import dual_end_alignment
from rmt.spacing import r_statistic
from rmt.svd_result import compute_svd
from rmt.tail import fit_powerlaw_csn, fixed_cutoff_mle, rank_ordered_mle


def test_square_kde_fit_is_hard_edge_safe_and_scale_equivariant():
    weight = np.random.default_rng(3).normal(size=(256, 256))
    values = np.linalg.svd(weight, compute_uv=False) ** 2 / 256
    fit = fit_marchenko_pastur_kde(values, 1.0)
    scaled = fit_marchenko_pastur_kde(9.0 * values, 1.0)
    assert 0.7 < fit.variance < 1.3
    assert not fit.diagnostics["boundary_solution"]
    assert np.isclose(scaled.variance, 9.0 * fit.variance, rtol=1e-5)


def test_scaler_overflow_backs_off_and_retries_without_advancing_budget():
    class LargeGradient(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, **kwargs):
            del kwargs
            return SimpleNamespace(loss=self.weight * 1e34)

    model = LargeGradient()
    trainer = LanguageModelTrainer(model, TrainConfig(
        device="cpu", amp_dtype="float32", max_steps=1, max_train_tokens=1,
        warmup_steps=0, max_consecutive_skipped_updates=10,
    ))
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=65536)
    batch = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[1, 2]])}
    trainer.fit([batch])
    assert trainer.skipped_steps == 1
    assert trainer.attempted_steps == 2
    assert trainer.global_step == trainer.processed_train_tokens == 1


def test_token_budget_scheduler_crosses_loader_passes():
    class Quadratic(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, **kwargs):
            del kwargs
            return SimpleNamespace(loss=self.weight.square())

    trainer = LanguageModelTrainer(Quadratic(), TrainConfig(
        device="cpu", amp_dtype="float32", epochs=1, max_steps=5,
        max_train_tokens=5, warmup_steps=0,
    ))
    batch = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[1, 2]])}
    learning_rates = []
    trainer.optimizer.register_step_pre_hook(
        lambda optimizer, *_: learning_rates.append(optimizer.param_groups[0]["lr"])
    )
    trainer.fit([batch])
    assert len(learning_rates) == 5
    assert learning_rates[1] > trainer.config.learning_rate * trainer.config.min_learning_rate_ratio


def test_three_level_ratio_and_lanczos_detector_unavailability():
    assert np.isclose(r_statistic(np.array([0.0, 1.0, 3.0])), 0.5)
    detected = dispatch_spike_detector(
        np.eye(64), RMTMethodConfig(mp_fit_method="analytic_mp",
                                    spike_detector="lanczos_poles")
    )
    assert detected.diagnostics["available"] is False


def test_bounded_tail_and_rank_tail_are_scale_correct():
    u = (np.arange(10000) + 0.5) / 10000
    sample = (1.0 - u * (1.0 - 2.0 ** -2)) ** -0.5
    assert np.isclose(fixed_cutoff_mle(sample, xmin=1, xmax=2)["alpha"], 3, atol=0.01)
    assert np.isclose(fit_powerlaw_csn(sample, min_tail=1000, xmax=2)["alpha"], 3, atol=0.02)
    ranked = np.arange(1.0, 1001.0) ** -0.5
    first = rank_ordered_mle(ranked, tail_fraction=1)["alpha"]
    second = rank_ordered_mle(1e-10 * ranked, tail_fraction=1)["alpha"]
    assert np.isclose(first, second)


def test_zero_covariance_overlap_is_unavailable():
    result = dual_end_alignment(compute_svd(np.diag([4.0, 3.0, 2.0, 1.0])), np.zeros((4, 4)))
    assert result["available"] is False
    assert np.isnan(result["bottom_alignment"])


def test_unselected_collapsed_cells_do_not_block_selected_cell(tmp_path):
    with pytest.raises(FileNotFoundError, match="offline token array"):
        runner.main([
            "--execute", "--device", "cpu", "--parameter-cap", "3500000",
            "--max-train-tokens", "10", "--cells", "0",
            "--dataset-path", str(tmp_path / "missing.npy"),
            "--output-dir", str(tmp_path / "selected"),
        ])


def test_output_directory_has_exclusive_owner(tmp_path):
    output = tmp_path / "run"
    runner._acquire_output_directory(output)
    with pytest.raises(FileExistsError):
        runner._acquire_output_directory(output)
