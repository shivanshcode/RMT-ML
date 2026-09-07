from dataclasses import replace
from pathlib import Path
import re

import numpy as np
import pytest

from pipelines.cli_config import parse_pipeline_args, parse_rmt_args
from rmt.factory import (
    ASPECT_RATIO_MODES,
    MP_FIT_METHODS,
    OVERLAP_METRICS,
    SPIKE_DETECTORS,
    TAIL_SOLVERS,
    UNFOLDING_STRATEGIES,
    RMTMethodConfig,
    dispatch_mp_fit,
    dispatch_overlap,
    dispatch_spike_detector,
    dispatch_tail_solver,
    dispatch_unfolding,
)


def test_pure_source_tree_contains_no_framework_name() -> None:
    root = Path(__file__).resolve().parents[1] / "rmt"
    for source in root.glob("*.py"):
        assert "torch" not in source.read_text(encoding="utf-8").lower()


def test_cli_defaults_and_boundary_compatibility_alias() -> None:
    defaults = parse_rmt_args([])
    assert defaults == RMTMethodConfig()
    compatibility = parse_rmt_args(["--boundary-detector", "lanczos_stieltjes"])
    assert compatibility.mp_fit_method == "lanczos_stieltjes"
    assert compatibility.spike_detector == "lanczos_poles"
    overlap_alias = parse_rmt_args(["--overlap-mode", "frobenius_projection"])
    assert overlap_alias.overlap_metric == "frobenius_projection"
    degree_alias = parse_rmt_args(["--unfolding-degree", "15"])
    assert degree_alias.polynomial_degree == 15


def test_phase_three_golden_defaults_are_resolved() -> None:
    parsed = parse_pipeline_args([])
    methods = parse_rmt_args([])
    assert parsed.experiment_mode == "compute_optimal_rmt"
    assert parsed.device == "cuda"
    assert parsed.amp_dtype == "bfloat16"
    assert parsed.compile_model
    assert methods.mp_fit_method == "lanczos_stieltjes"
    assert methods.aspect_ratio_mode == "farms_normalized"
    assert methods.spike_detector == "lanczos_poles"
    assert methods.unfolding_strategy == "spline_monotone"


def test_all_four_production_track_argument_sets_parse() -> None:
    tracks = (
        [
            "--experiment-mode", "reproduce_paper1",
            "--aspect-ratio-mode", "raw",
            "--mp-fit-method", "analytic_mp",
            "--spike-detector", "tracy_widom_95",
            "--overlap-metric", "staats_dual_end",
            "--unfolding-strategy", "gaussian_kernel",
            "--tail-solver", "rank_ordered_mle",
            "--run-spectral-lesioning",
            "--lesion-tranches", "top,bulk,bottom",
        ],
        [
            "--experiment-mode", "reproduce_paper2",
            "--aspect-ratio-mode", "raw",
            "--mp-fit-method", "thamm_modified_singular",
            "--spike-detector", "tracy_widom_95",
            "--unfolding-strategy", "polynomial_chebyshev",
            "--unfolding-degree", "15",
            "--compute-spacing-distribution",
            "--compute-number-variance",
            "--compute-delta3",
        ],
        [
            "--experiment-mode", "reproduce_paper3",
            "--aspect-ratio-mode", "raw",
            "--mp-fit-method", "kde_bulk_fit",
            "--spike-detector", "tracy_widom_95",
            "--tail-solver", "clauset_mle",
            "--compute-stable-rank",
        ],
        [
            "--experiment-mode", "compute_optimal_rmt",
            "--scaling-budget-flops", "1e16",
            "--allocation-ratios", "0.25", "1.0", "4.0",
            "--aspect-ratio-mode", "farms_normalized",
            "--farms-sampling", "reference_fixed",
            "--mp-fit-method", "lanczos_stieltjes",
            "--spike-detector", "lanczos_poles",
            "--lanczos-steps", "50",
            "--lanczos-pole-method", "reference_ritz",
            "--unfolding-strategy", "spline_monotone",
            "--tail-solver", "clauset_mle",
            "--overlap-metric", "staats_dual_end",
        ],
    )
    for arguments in tracks:
        namespace = parse_pipeline_args(arguments)
        assert namespace.experiment_mode in {
            "reproduce_paper1",
            "reproduce_paper2",
            "reproduce_paper3",
            "compute_optimal_rmt",
        }


def test_slurm_harness_contains_every_track_and_strict_offline_exports() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "run_hpc.slurm").read_text(encoding="utf-8")
    for track in ("run_paper1", "run_paper2", "run_paper3", "run_golden"):
        assert track in source
    for variable in ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_DATASETS_OFFLINE=1"):
        assert variable in source


def test_offline_staging_contract_is_fail_closed_and_directly_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    downloader = (root / "scripts" / "download_assets.py").read_text(encoding="utf-8")
    assert 'DATASET_ID = "Salesforce/wikitext"' in downloader
    assert 'DATASET_CONFIG = "wikitext-103-raw-v1"' in downloader
    assert 'TOKENIZER_ID = "openai-community/gpt2"' in downloader
    assert "--allow-network" in downloader
    assert "--verify-only" in downloader
    assert "sha256" in downloader.lower()
    assert "dataset_info" in downloader and "model_info" in downloader
    assert '"source_revisions"' in downloader
    slurm = (root / "run_hpc.slurm").read_text(encoding="utf-8")
    assert "--verify-only" in slurm
    requirement_lines = [
        line.strip()
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirement_lines
    assert all("==" in line and ">=" not in line for line in requirement_lines)


def test_maintained_tree_contains_no_placeholder_stubs() -> None:
    root = Path(__file__).resolve().parents[1]
    maintained = [
        *(root / "rmt").glob("*.py"),
        *(root / "models").glob("*.py"),
        *(root / "pipelines").glob("*.py"),
        *(root / "scripts").glob("*.py"),
        *(root / "tests").glob("*.py"),
        *root.glob("*.py"),
        *root.glob("*.md"),
        *root.glob("*.slurm"),
    ]
    body_stub = re.compile(r"(?m)^\s*(?:pass|\.\.\.)\s*(?:#.*)?$")
    forbidden_exception = "NotImplemented" + "Error"
    forbidden_marker = "TO" + "DO"
    for path in maintained:
        source = path.read_text(encoding="utf-8")
        assert forbidden_exception not in source, path
        assert not re.search(rf"\b{forbidden_marker}\b", source), path
        assert body_stub.search(source) is None, path


@pytest.mark.parametrize("method", MP_FIT_METHODS)
def test_every_mp_fit_method_dispatches(method: str) -> None:
    matrix = np.random.default_rng(7).normal(size=(64, 128))
    config = RMTMethodConfig(
        mp_fit_method=method,
        lanczos_steps=28,
        lanczos_probes=1,
        lanczos_tail_window=5,
        farms_window_size=48,
        farms_row_windows=2,
        farms_column_windows=2,
    )
    result = dispatch_mp_fit(matrix, config)
    assert result.method == method
    assert 0.0 <= result.lambda_minus < result.lambda_plus
    assert np.isfinite(result.variance)


@pytest.mark.parametrize("strategy", UNFOLDING_STRATEGIES)
def test_every_unfolding_strategy_dispatches(strategy: str) -> None:
    raw = np.random.default_rng(9).normal(size=(96, 96))
    levels = np.linalg.eigvalsh(raw + raw.T)
    config = RMTMethodConfig(unfolding_strategy=strategy, gaussian_kernel_window=8)
    unfolded = dispatch_unfolding(levels, config)
    assert unfolded.shape == levels.shape
    assert np.all(np.diff(unfolded) > 0.0)
    assert abs(np.mean(np.diff(unfolded)) - 1.0) < 1e-10


@pytest.mark.parametrize("solver", TAIL_SOLVERS)
def test_every_tail_solver_dispatches(solver: str) -> None:
    values = np.random.default_rng(11).pareto(2.0, size=5000) + 1.0
    result = dispatch_tail_solver(
        values,
        RMTMethodConfig(tail_solver=solver, tail_minimum=100, tail_fraction=0.2),
    )
    assert result["selected_estimator"] == solver
    assert np.isfinite(result["selected_alpha"])


@pytest.mark.parametrize("metric", OVERLAP_METRICS)
def test_every_overlap_metric_dispatches(metric: str) -> None:
    identity = np.eye(8)
    result = dispatch_overlap(
        identity[:, :3],
        identity[:, :3],
        RMTMethodConfig(overlap_metric=metric),
    )
    assert 0.0 <= float(result["score"]) <= 1.0
    expected = 1.0 / 3.0 if metric == "staats_dual_end" else 1.0
    assert np.isclose(float(result["score"]), expected)


@pytest.mark.parametrize("detector", SPIKE_DETECTORS)
def test_every_spike_detector_dispatches(detector: str) -> None:
    matrix = np.random.default_rng(13).normal(size=(48, 96))
    config = RMTMethodConfig(
        spike_detector=detector,
        lanczos_steps=24,
        lanczos_probes=1,
        lanczos_tail_window=5,
    )
    result = dispatch_spike_detector(matrix, config)
    assert result.method == detector
    assert result.n_spikes >= 0
    assert result.threshold >= result.bulk_edge


def test_all_declared_choice_values_construct_together() -> None:
    base = RMTMethodConfig()
    for mp_method in MP_FIT_METHODS:
        for unfolding in UNFOLDING_STRATEGIES:
            for tail in TAIL_SOLVERS:
                for overlap in OVERLAP_METRICS:
                    for aspect in ASPECT_RATIO_MODES:
                        for spike in SPIKE_DETECTORS:
                            configured = replace(
                                base,
                                mp_fit_method=mp_method,
                                unfolding_strategy=unfolding,
                                tail_solver=tail,
                                overlap_metric=overlap,
                                aspect_ratio_mode=aspect,
                                spike_detector=spike,
                            )
                            assert configured.mp_fit_method == mp_method
