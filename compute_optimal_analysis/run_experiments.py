"""Spectral-Chinchilla allocation, training, diagnostics, lesions, and plots."""

from __future__ import annotations

import argparse
import csv
from importlib import metadata
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.chinchilla_scaling import (
    ScalingLaw,
    isoflop_grid,
    suggest_architecture,
)
from models.transformer import CausalTransformer, TransformerConfig
from pipelines.activation_extractor import compute_activation_covariances, compute_tensor_svd
from pipelines.cli_config import add_pipeline_cli_arguments, rmt_config_from_namespace
from pipelines.dataset import TokenSequenceDataset, load_token_array, split_tokens
from pipelines.spectral_lesioning import independent_lesion_benchmark, spectral_parameter_names
from pipelines.trainer import LanguageModelTrainer, TrainConfig, evaluate_language_model
from rmt.factory import (
    RMTMethodConfig,
    dispatch_mp_fit,
    dispatch_spike_detector,
    dispatch_tail_solver,
    dispatch_unfolding,
    prepare_spectrum,
)
from rmt.mp import fit_marchenko_pastur, mp_soft_rank
from rmt.overlap import dual_end_alignment
from rmt.scalars import (
    condition_number,
    porter_thomas_monte_carlo_pooled,
    stable_rank,
    von_neumann_entropy,
)
from rmt.spacing import (
    dyson_mehta_delta3,
    fit_brody,
    fit_brody_cdf_nls,
    number_variance,
    r_statistic,
)
from rmt.tail import hill_alpha_at, hill_plateau


COMPUTE_BUDGETS = (1e15, 1e16, 1e17)
REGIME_MULTIPLIERS = (0.25, 1.0, 4.0)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sanitize_json(value: Any) -> Any:
    converted = _json_value(value)
    if isinstance(converted, dict):
        return {str(key): _sanitize_json(item) for key, item in converted.items()}
    if isinstance(converted, (list, tuple)):
        return [_sanitize_json(item) for item in converted]
    if isinstance(converted, float) and not math.isfinite(converted):
        return None
    return converted


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize_json(payload), handle, indent=2, allow_nan=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(row.get(key)) for key in fields})


def _runtime_environment(args: argparse.Namespace) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in (
        "numpy",
        "scipy",
        "torch",
        "matplotlib",
        "datasets",
        "transformers",
        "tokenizers",
        "huggingface-hub",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    accelerator: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_build": torch.version.cuda,
        "requested_device": args.device,
        "amp_dtype": args.amp_dtype,
        "allow_tf32": bool(args.allow_tf32),
        "svd_backend": args.svd_backend,
        "svd_driver": args.svd_driver,
    }
    if args.execute and args.device == "cuda" and torch.cuda.is_available():
        accelerator.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": list(torch.cuda.get_device_capability(0)),
                "device_count_visible": torch.cuda.device_count(),
            }
        )
    dataset: dict[str, Any] = {"path": str(Path(args.dataset_path).resolve())}
    asset_manifest = Path("data/asset_manifest.json")
    if asset_manifest.is_file():
        try:
            with asset_manifest.open("r", encoding="utf-8") as handle:
                staged = json.load(handle)
            dataset["asset_manifest"] = str(asset_manifest.resolve())
            dataset["source_revisions"] = staged.get("source_revisions", {})
            dataset_target = Path(args.dataset_path).resolve()
            for record in staged.get("files", []):
                candidate = (Path.cwd() / str(record.get("path", ""))).resolve()
                if candidate == dataset_target:
                    dataset["bytes"] = record.get("bytes")
                    dataset["sha256"] = record.get("sha256")
                    break
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as error:
            dataset["manifest_error"] = str(error)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "accelerator": accelerator,
        "dataset": dataset,
        "scheduler": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
    }


def build_manifest(
    *,
    vocab_size: int,
    compute_budgets: Iterable[float] = COMPUTE_BUDGETS,
    allocation_ratios: Iterable[float] = REGIME_MULTIPLIERS,
    parameter_cap: float | None = None,
    token_cap: float | None = None,
    law: ScalingLaw = ScalingLaw(),
) -> list[dict[str, Any]]:
    """Build the exact three-by-three IsoFLOP manifest with realized caps labeled."""

    allocations = isoflop_grid(
        tuple(float(value) for value in compute_budgets),
        tuple(float(value) for value in allocation_ratios),
        law,
        target_tokens_per_parameter=20.0,
    )
    manifest: list[dict[str, Any]] = []
    for cell_index, allocation in enumerate(allocations):
        architecture_target = allocation.parameters
        if parameter_cap is not None:
            architecture_target = min(architecture_target, float(parameter_cap))
        architecture = suggest_architecture(architecture_target, vocab_size=vocab_size)
        realized_tokens = allocation.tokens if token_cap is None else min(allocation.tokens, float(token_cap))
        realized_parameters = float(architecture["estimated_parameters"])
        manifest.append(
            {
                "cell_index": cell_index,
                **allocation.as_dict(),
                "requested_parameters": allocation.parameters,
                "requested_tokens": allocation.tokens,
                "architecture_target_parameters": architecture_target,
                "realized_parameters": realized_parameters,
                "realized_tokens": float(realized_tokens),
                "realized_training_compute": float(
                    law.flops_per_parameter_token * realized_parameters * realized_tokens
                ),
                "parameter_cap_applied": bool(parameter_cap is not None and allocation.parameters > parameter_cap),
                "token_cap_applied": bool(token_cap is not None and allocation.tokens > token_cap),
                "architecture": architecture,
            }
        )
    return manifest


def _config_from_manifest(cell: dict[str, Any], sequence_length: int) -> TransformerConfig:
    architecture = dict(cell["architecture"])
    return TransformerConfig(
        vocab_size=int(architecture["vocab_size"]),
        d_model=int(architecture["d_model"]),
        n_layers=int(architecture["n_layers"]),
        n_heads=int(architecture["n_heads"]),
        n_kv_heads=int(architecture["n_kv_heads"]),
        max_seq_len=int(sequence_length),
        mlp_ratio=float(architecture["mlp_ratio"]),
        mlp_type=str(architecture["mlp_type"]),
        position_embedding=str(architecture["position_embedding"]),
        tie_embeddings=bool(architecture["tie_embeddings"]),
        bias=bool(architecture["bias"]),
        dropout=0.0,
    )


def _role_and_layer(parameter_name: str) -> tuple[str, int]:
    role = parameter_name.rsplit(".", 2)[-2].replace("_proj", "")
    match = re.search(r"layers\.(\d+)", parameter_name)
    return role, -1 if match is None else int(match.group(1))


def analyze_model(
    model: CausalTransformer,
    covariances: dict[str, np.ndarray],
    cell: dict[str, Any],
    method_config: RMTMethodConfig = RMTMethodConfig(),
    *,
    svd_backend: str = "auto",
    svd_driver: str = "gesvdj",
    compute_activation_overlap: bool = True,
    compute_spacing_distribution: bool = True,
    compute_number_variance: bool = True,
    compute_stable_rank: bool = True,
    compute_delta3: bool = False,
    compute_porter_thomas: bool = False,
    brody_fit_method: str = "mle",
    number_variance_method: str = "sliding",
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Analyze every attention/MLP matrix from one SVD per matrix."""

    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for parameter_name, parameter in model.iter_spectral_weights():
        svd = compute_tensor_svd(
            parameter,
            backend=svd_backend,
            driver=svd_driver,
        )
        weight = parameter.detach().float().cpu().numpy()
        prepared = prepare_spectrum(weight, method_config)
        eigenvalues = prepared.eigenvalues
        mp_fit = dispatch_mp_fit(weight, method_config)
        spike_fit = dispatch_spike_detector(
            weight,
            method_config,
            variance=mp_fit.variance,
        )
        tail_minimum = max(8, min(method_config.tail_minimum, eigenvalues.size // 3))
        tail = dispatch_tail_solver(
            eigenvalues,
            method_config,
            min_tail=tail_minimum,
        )
        hill_k = max(2, min(eigenvalues.size - 1, int(round(np.sqrt(eigenvalues.size)))))
        hill = hill_alpha_at(eigenvalues, hill_k)
        plateau = hill_plateau(eigenvalues, window=max(3, min(20, eigenvalues.size // 4)))
        spacing_eigenvalues = svd.covariance_eigenvalues
        spacing_mp_fit = fit_marchenko_pastur(spacing_eigenvalues, svd.aspect_ratio)
        if method_config.mp_fit_method == "farms_unbiased":
            mp_soft_spectrum = np.asarray([float(mp_fit.diagnostics["spectral_max"])])
        elif method_config.mp_fit_method == "lanczos_stieltjes":
            mp_soft_spectrum = spacing_eigenvalues
        else:
            mp_soft_spectrum = eigenvalues
        bulk_mask = (
            (spacing_eigenvalues >= spacing_mp_fit.lambda_minus)
            & (spacing_eigenvalues <= spacing_mp_fit.lambda_plus)
        )
        bulk_levels = np.sort(spacing_eigenvalues[bulk_mask])
        beta = float("nan")
        ratio = float("nan")
        variance_10 = float("nan")
        rigidity_10 = float("nan")
        spacings = np.asarray([], dtype=np.float64)
        minimum_bulk = (
            max(10, 2 * method_config.gaussian_kernel_window + 1)
            if method_config.unfolding_strategy == "gaussian_kernel"
            else 10
        )
        needs_spacing = bool(
            compute_spacing_distribution or compute_number_variance or compute_delta3
        )
        if needs_spacing and bulk_levels.size >= minimum_bulk:
            unfolded = dispatch_unfolding(bulk_levels, method_config)
            spacings = np.diff(unfolded)
            spacings = spacings[np.isfinite(spacings) & (spacings > 0.0)]
            if spacings.size:
                spacings = spacings / np.mean(spacings)
            if spacings.size >= 8:
                beta = (
                    fit_brody_cdf_nls(spacings).beta
                    if brody_fit_method == "cdf_nls"
                    else fit_brody(spacings).beta
                )
            ratio = r_statistic(bulk_levels)
            if compute_number_variance:
                variance_10 = number_variance(
                    unfolded,
                    10.0,
                    unfolded=True,
                    method=number_variance_method,
                    rng=seed,
                )
            if compute_delta3:
                rigidity_10 = dyson_mehta_delta3(
                    unfolded,
                    10.0,
                    unfolded=True,
                )
        module_name = parameter_name[: -len(".weight")]
        covariance = covariances.get(f"{module_name}:pre")
        top_alignment = bulk_alignment = bottom_alignment = float("nan")
        overlap_matrix: np.ndarray | None = None
        activation_eigenvalues: np.ndarray | None = None
        if (
            compute_activation_overlap
            and covariance is not None
            and covariance.shape[0] == weight.shape[1]
            and svd.s.size >= 3
        ):
            alignment = dual_end_alignment(
                svd,
                covariance,
                metric=method_config.overlap_metric,
            )
            top_alignment = float(alignment["top_alignment"])
            bulk_alignment = float(alignment["bulk_alignment"])
            bottom_alignment = float(alignment["bottom_alignment"])
            overlap_matrix = np.asarray(alignment["overlap_matrix"])
            activation_eigenvalues = np.asarray(alignment["activation_eigenvalues"])
        role, layer = _role_and_layer(parameter_name)
        porter_thomas_ks = float("nan")
        porter_thomas_fraction = float("nan")
        if compute_porter_thomas and svd.Vh.shape[0] > 10:
            porter = porter_thomas_monte_carlo_pooled(
                svd.Vh,
                n_reference=256,
                pooling_window=5,
                rng=seed,
            )
            porter_thomas_ks = float(porter["pt_ks_mean"])
            porter_thomas_fraction = float(porter["pt_frac_random"])
        csn_selected = method_config.tail_solver == "clauset_mle"
        row = {
            "cell_index": cell["cell_index"],
            "compute_budget": cell["compute_budget"],
            "regime": cell["regime"],
            "kappa": cell["kappa"],
            "requested_parameters": cell["requested_parameters"],
            "realized_parameters": cell["realized_parameters"],
            "requested_tokens": cell["requested_tokens"],
            "realized_tokens": cell["realized_tokens"],
            "parameter_name": parameter_name,
            "role": role,
            "layer": layer,
            "n": svd.n,
            "m": svd.m,
            "normalization": svd.normalization,
            "aspect_ratio": prepared.aspect_ratio,
            "aspect_ratio_mode": method_config.aspect_ratio_mode,
            "spectrum_mode": prepared.mode,
            "mp_fit_method": method_config.mp_fit_method,
            "mp_variance": mp_fit.variance,
            "mp_lambda_minus": mp_fit.lambda_minus,
            "mp_lambda_plus": mp_fit.lambda_plus,
            "mp_ks": mp_fit.ks_distance,
            "mp_bulk_fraction": mp_fit.bulk_fraction,
            "lower_outliers": mp_fit.n_lower_outliers,
            "upper_outliers": mp_fit.n_upper_outliers,
            "spike_detector": method_config.spike_detector,
            "spike_threshold": spike_fit.threshold,
            "detected_spikes": spike_fit.n_spikes,
            "mp_soft_rank": mp_soft_rank(
                mp_soft_spectrum,
                mp_fit.lambda_plus,
            ),
            "spacing_bulk_lambda_minus": spacing_mp_fit.lambda_minus,
            "spacing_bulk_lambda_plus": spacing_mp_fit.lambda_plus,
            "tail_solver": method_config.tail_solver,
            "tail_alpha_lambda": tail["selected_alpha"],
            "tail_alpha_kind": tail["selected_kind"],
            "tail_xmin_lambda": tail["xmin"],
            "tail_ks": tail["ks_D"],
            "tail_count": tail["n_tail"],
            "csn_alpha_density_lambda": tail["selected_alpha"] if csn_selected else float("nan"),
            "csn_xmin_lambda": tail["xmin"] if csn_selected else float("nan"),
            "csn_ks": tail["ks_D"] if csn_selected else float("nan"),
            "csn_tail_count": tail["n_tail"] if csn_selected else 0,
            "hill_alpha_survival_lambda": hill,
            "hill_plateau_alpha": plateau["hill_plateau_alpha"],
            "hill_plateau_width": plateau["hill_plateau_width"],
            "hill_is_powerlaw": plateau["hill_is_powerlaw"],
            "spectral_norm": float(svd.s[0]),
            "frobenius_norm": float(np.linalg.norm(svd.s)),
            "stable_rank": stable_rank(s=svd.s) if compute_stable_rank else float("nan"),
            "condition_number": condition_number(s=svd.s),
            "von_neumann_entropy": von_neumann_entropy(s=svd.s),
            "brody_beta": beta,
            "brody_fit_method": brody_fit_method,
            "unfolding_strategy": method_config.unfolding_strategy,
            "r_statistic": ratio,
            "number_variance_L10": variance_10,
            "number_variance_method": number_variance_method,
            "dyson_mehta_delta3_L10": rigidity_10,
            "porter_thomas_ks_mean": porter_thomas_ks,
            "porter_thomas_fraction_random": porter_thomas_fraction,
            "top_activation_alignment": top_alignment,
            "bulk_activation_alignment": bulk_alignment,
            "bottom_activation_alignment": bottom_alignment,
            "overlap_metric": method_config.overlap_metric,
        }
        rows.append(row)
        artifacts.append(
            {
                "row": row,
                "eigenvalues": eigenvalues,
                "spacings": spacings,
                "overlap_matrix": overlap_matrix,
                "activation_eigenvalues": activation_eigenvalues,
            }
        )
    return rows, artifacts


def _plot_esd(artifacts: list[dict[str, Any]], output: Path) -> None:
    selected = artifacts[: min(9, len(artifacts))]
    figure, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)
    for axis in axes.ravel():
        axis.set_visible(False)
    for axis, artifact in zip(axes.ravel(), selected):
        axis.set_visible(True)
        row = artifact["row"]
        values = np.asarray(artifact["eigenvalues"])
        positive = values[values > 0.0]
        bins = np.geomspace(positive.min(), positive.max(), min(60, max(12, positive.size)))
        density, edges = np.histogram(positive, bins=bins, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
        axis.loglog(centers, np.maximum(density, np.finfo(float).tiny), marker="o", linestyle="none", ms=3)
        alpha, xmin = row["tail_alpha_lambda"], row["tail_xmin_lambda"]
        if isinstance(alpha, float) and math.isfinite(alpha) and isinstance(xmin, float) and math.isfinite(xmin):
            tail_x = centers[centers >= xmin]
            if tail_x.size:
                density_alpha = alpha + 1.0 if row["tail_alpha_kind"] == "survival" else alpha
                anchor_index = int(np.argmin(np.abs(centers - xmin)))
                anchor = max(float(density[anchor_index]), np.finfo(float).tiny)
                axis.loglog(tail_x, anchor * (tail_x / xmin) ** (-density_alpha), linewidth=1.5)
        axis.axvline(row["mp_lambda_minus"], color="tab:green", linestyle="--", linewidth=1)
        axis.axvline(row["mp_lambda_plus"], color="tab:red", linestyle="--", linewidth=1)
        axis.set_title(f"{row['regime']} L{row['layer']} {row['role']}")
        axis.set_xlabel("lambda")
        axis.set_ylabel("density")
    figure.tight_layout()
    figure.savefig(output / "esd_powerlaw_fits.png", dpi=180)
    plt.close(figure)


def _plot_spacing(artifacts: list[dict[str, Any]], output: Path) -> None:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for artifact in artifacts:
        spacings = np.asarray(artifact["spacings"])
        if spacings.size:
            grouped[str(artifact["row"]["regime"])].append(spacings)
    figure, axis = plt.subplots(figsize=(9, 6))
    grid = np.linspace(0.0, 3.5, 300)
    axis.plot(grid, np.exp(-grid), color="black", linestyle=":", label="Poisson")
    axis.plot(grid, 0.5 * np.pi * grid * np.exp(-np.pi * grid**2 / 4.0), color="black", linestyle="--", label="GOE")
    for regime, arrays in sorted(grouped.items()):
        combined = np.concatenate(arrays)
        beta = fit_brody(combined).beta if combined.size >= 8 else float("nan")
        label = f"{regime}, beta={beta:.3f}" if math.isfinite(beta) else regime
        axis.hist(combined, bins=40, range=(0.0, 3.5), density=True, histtype="step", linewidth=1.5, label=label)
    axis.set_xlabel("unfolded nearest-neighbor spacing")
    axis.set_ylabel("density")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "brody_spacing_distribution.png", dpi=180)
    plt.close(figure)


def _plot_overlap(artifacts: list[dict[str, Any]], output: Path) -> None:
    selected: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        regime = str(artifact["row"]["regime"])
        if artifact["overlap_matrix"] is not None and regime not in selected:
            selected[regime] = artifact
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    for axis, regime in zip(axes[0], ("undertrained", "compute_optimal", "overtrained")):
        artifact = selected.get(regime)
        if artifact is None:
            axis.text(0.5, 0.5, "No compatible activation covariance", ha="center", va="center")
            axis.set_axis_off()
            continue
        matrix = np.asarray(artifact["overlap_matrix"])
        row_step = max(1, int(math.ceil(matrix.shape[0] / 256)))
        column_step = max(1, int(math.ceil(matrix.shape[1] / 256)))
        image = axis.imshow(matrix[::row_step, ::column_step], aspect="auto", origin="upper", cmap="magma")
        axis.set_title(regime)
        axis.set_xlabel("activation eigenvector rank")
        axis.set_ylabel("singular-vector rank")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(output / "dual_end_activation_overlap.png", dpi=180)
    plt.close(figure)


def _plot_lesions(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    cells = sorted(set(int(row["cell_index"]) for row in rows))
    tranches = ("top", "bulk", "bottom")
    width = 0.24
    x = np.arange(len(cells), dtype=np.float64)
    for offset, tranche in enumerate(tranches):
        values = [
            next(float(row["delta"]) for row in rows if int(row["cell_index"]) == cell and row["tranche"] == tranche)
            for cell in cells
        ]
        axis.bar(x + (offset - 1) * width, values, width=width, label=tranche)
    axis.set_xticks(x, [str(cell) for cell in cells])
    axis.set_xlabel("manifest cell index")
    axis.set_ylabel("perplexity change")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "lesioning_perplexity_impact.png", dpi=180)
    plt.close(figure)


def _plot_scaling(rows: list[dict[str, Any]], output: Path) -> None:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["compute_budget"]), float(row["kappa"]))].append(row)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for kappa in sorted({float(row["kappa"]) for row in rows}):
        points = sorted((key, values) for key, values in grouped.items() if math.isclose(key[1], kappa))
        budgets = [key[0] for key, _ in points]
        alpha = [
            float(np.nanmean([float(row["tail_alpha_lambda"]) for row in values]))
            for _, values in points
        ]
        beta = [
            float(np.nanmean([float(row["brody_beta"]) for row in values]))
            for _, values in points
        ]
        axes[0].plot(budgets, alpha, marker="o", label=f"kappa={kappa:g}")
        axes[1].plot(budgets, beta, marker="o", label=f"kappa={kappa:g}")
    for axis, ylabel in zip(axes, ("mean selected tail exponent", "mean Brody beta")):
        axis.set_xscale("log")
        axis.set_xlabel("requested training FLOPs")
        axis.set_ylabel(ylabel)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "scaling_collapse_spectral_trajectory.png", dpi=180)
    plt.close(figure)


def execute_cell(
    cell: dict[str, Any],
    train_tokens: np.ndarray,
    validation_tokens: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Train, capture, analyze, and lesion one manifest cell."""

    seed = int(args.seed + cell["cell_index"])
    config = _config_from_manifest(cell, args.sequence_length)
    model = CausalTransformer(config)
    train_dataset = TokenSequenceDataset(train_tokens, args.sequence_length, stride=args.sequence_length)
    validation_dataset = TokenSequenceDataset(
        validation_tokens,
        args.sequence_length,
        stride=args.sequence_length,
    )
    loader_generator = torch.Generator().manual_seed(seed)
    loader_options: dict[str, Any] = {
        "num_workers": int(args.dataloader_workers),
        "pin_memory": bool(args.pin_memory and args.device == "cuda"),
    }
    if args.dataloader_workers > 0:
        loader_options["prefetch_factor"] = int(args.prefetch_factor)
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )
    tokens_per_step = args.batch_size * (args.sequence_length - 1)
    max_steps = max(1, int(math.ceil(float(cell["realized_tokens"]) / tokens_per_step)))
    epochs = max(1, int(math.ceil(max_steps / len(train_loader))))
    train_config = TrainConfig(
        epochs=epochs,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=min(args.warmup_steps, max_steps // 5),
        gradient_clip=args.gradient_clip,
        device=args.device,
        amp_dtype=args.amp_dtype,
        compile_model=bool(args.compile_model and args.device == "cuda"),
        compile_mode=args.compile_mode,
        allow_tf32=args.allow_tf32,
        log_every=max(1, args.log_every),
        seed=seed,
    )
    trainer = LanguageModelTrainer(model, train_config)
    history = trainer.fit(train_loader, validation_loader)
    for record in history:
        record.update({"cell_index": cell["cell_index"], "regime": cell["regime"], "kappa": cell["kappa"]})
    covariances = compute_activation_covariances(
        model,
        validation_loader,
        device=args.device,
        module_filter=lambda name, module: isinstance(module, torch.nn.Linear) and name != "lm_head",
        max_batches=args.activation_batches,
        centered=args.activation_centered,
        accumulation_device=(
            args.device if args.covariance_device == "auto" else args.covariance_device
        ),
        accumulation_dtype=args.covariance_dtype,
        amp_dtype=args.amp_dtype,
    )
    spectral_rows, artifacts = analyze_model(
        model,
        covariances,
        cell,
        rmt_config_from_namespace(args),
        svd_backend=args.svd_backend,
        svd_driver=args.svd_driver,
        compute_activation_overlap=args.compute_activation_overlap,
        compute_spacing_distribution=args.compute_spacing_distribution,
        compute_number_variance=args.compute_number_variance,
        compute_stable_rank=args.compute_stable_rank,
        compute_delta3=args.compute_delta3,
        compute_porter_thomas=args.compute_porter_thomas,
        brody_fit_method=args.brody_fit_method,
        number_variance_method=args.number_variance_method,
        seed=seed,
    )

    def evaluate_perplexity() -> float:
        return float(
            evaluate_language_model(
                model,
                validation_loader,
                args.device,
                max_batches=args.validation_batches,
                amp_dtype=args.amp_dtype,
            )["perplexity"]
        )

    lesions: list[dict[str, object]] = []
    if args.run_spectral_lesioning:
        tranches = tuple(
            name.strip().lower()
            for name in str(args.lesion_tranches).split(",")
            if name.strip()
        )
        if not tranches or any(name not in {"top", "bulk", "bottom"} for name in tranches):
            raise ValueError("lesion-tranches must be a comma-separated subset of top,bulk,bottom")
        lesions = independent_lesion_benchmark(
            model,
            spectral_parameter_names(model),
            evaluate_perplexity,
            tranches=tranches,
            fraction=args.lesion_fraction,
            mode=args.lesion_mode,
            seed=seed,
            svd_backend=args.svd_backend,
            svd_driver=args.svd_driver,
        )
    lesion_rows: list[dict[str, Any]] = []
    for result in lesions:
        lesion_rows.append(
            {
                "cell_index": cell["cell_index"],
                "compute_budget": cell["compute_budget"],
                "regime": cell["regime"],
                "kappa": cell["kappa"],
                "tranche": result["tranche"],
                "baseline": result["baseline"],
                "lesioned": result["lesioned"],
                "delta": result["delta"],
                "matrix_count": len(result["matrices"]),
                "mean_removed_energy_fraction": float(
                    np.mean([matrix["removed_energy_fraction"] for matrix in result["matrices"]])
                ),
            }
        )
    if args.save_checkpoints:
        trainer.save_checkpoint(
            Path(args.output_dir) / "checkpoints" / f"cell_{cell['cell_index']}.pt",
            metadata={"manifest_cell": cell},
        )
    return spectral_rows, lesion_rows, history, artifacts


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_pipeline_cli_arguments(parser)
    return parser.parse_args(None if argv is None else list(argv))


def _resolve_runtime_configuration(args: argparse.Namespace) -> None:
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.execute and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA execution was requested but no CUDA device is available")
    if args.execute and args.svd_backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA SVD was requested but no CUDA device is available")
    if args.execute and args.covariance_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA covariance accumulation was requested but CUDA is unavailable")
    if not args.scaling_budget_flops or any(value <= 0.0 for value in args.scaling_budget_flops):
        raise ValueError("scaling-budget-flops must contain positive values")
    if not args.allocation_ratios or any(value <= 0.0 for value in args.allocation_ratios):
        raise ValueError("allocation-ratios must contain positive values")
    if args.parameter_cap is not None and args.parameter_cap <= 0.0:
        raise ValueError("parameter-cap must be positive when supplied")
    if args.max_train_tokens is not None and args.max_train_tokens <= 0.0:
        raise ValueError("max-train-tokens must be positive when supplied")
    if args.dataloader_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("dataloader-workers must be nonnegative and prefetch-factor positive")
    if args.sequence_length < 2 or args.batch_size < 1 or args.vocab_size < 2:
        raise ValueError("sequence length, batch size, and vocabulary size are invalid")
    if args.activation_batches < 1 or args.validation_batches < 1:
        raise ValueError("activation and validation batch limits must be positive")
    if not 0.0 < args.lesion_fraction <= 1.0:
        raise ValueError("lesion-fraction must lie in (0, 1]")
    if args.experiment_mode in {"reproduce_paper1", "compute_optimal_rmt"}:
        args.run_spectral_lesioning = True
        args.compute_activation_overlap = True
    if args.experiment_mode == "reproduce_paper2":
        args.compute_spacing_distribution = True
        args.compute_number_variance = True
    if args.experiment_mode == "reproduce_paper3":
        args.compute_stable_rank = True


def main(argv: Iterable[str] | None = None) -> int:
    started_timer = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    args = parse_args(argv)
    _resolve_runtime_configuration(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        vocab_size=args.vocab_size,
        compute_budgets=args.scaling_budget_flops,
        allocation_ratios=args.allocation_ratios,
        parameter_cap=args.parameter_cap,
        token_cap=args.max_train_tokens,
    )
    _write_json(output / "allocation_manifest.json", manifest)
    _write_json(output / "spectral_method_config.json", rmt_config_from_namespace(args).as_dict())
    _write_json(output / "run_config.json", vars(args))
    environment_record = _runtime_environment(args)
    environment_record["started_at_utc"] = started_at
    _write_json(output / "runtime_environment.json", environment_record)
    if not args.execute:
        environment_record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        environment_record["elapsed_seconds"] = time.perf_counter() - started_timer
        _write_json(output / "runtime_environment.json", environment_record)
        return 0
    if not Path(args.dataset_path).is_file():
        raise FileNotFoundError(
            f"offline token array not found: {args.dataset_path}; run the asset prefetch step first"
        )
    tokens = load_token_array(args.dataset_path)
    if int(np.max(tokens)) >= args.vocab_size:
        raise ValueError("vocab-size must exceed every token id in the local array")
    train_tokens, validation_tokens = split_tokens(tokens)
    if min(train_tokens.size, validation_tokens.size) < args.sequence_length:
        raise ValueError("both token splits must contain at least one full sequence")
    if args.cells == "all":
        selected_indices = set(range(len(manifest)))
    else:
        selected_indices = {int(value.strip()) for value in args.cells.split(",") if value.strip()}
    if not selected_indices or min(selected_indices) < 0 or max(selected_indices) >= len(manifest):
        raise ValueError("--cells contains an invalid manifest index")
    spectral_rows: list[dict[str, Any]] = []
    lesion_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for cell in manifest:
        if int(cell["cell_index"]) not in selected_indices:
            continue
        cell_spectral, cell_lesions, cell_history, cell_artifacts = execute_cell(
            cell,
            train_tokens,
            validation_tokens,
            args,
        )
        spectral_rows.extend(cell_spectral)
        lesion_rows.extend(cell_lesions)
        training_rows.extend(cell_history)
        artifacts.extend(cell_artifacts)
    _write_csv(output / "spectral_metrics.csv", spectral_rows)
    _write_csv(output / "lesion_metrics.csv", lesion_rows)
    with (output / "training_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in training_rows:
            handle.write(json.dumps(_sanitize_json(row), allow_nan=False) + "\n")
    if artifacts:
        _plot_esd(artifacts, output)
        _plot_spacing(artifacts, output)
        _plot_overlap(artifacts, output)
    if lesion_rows:
        _plot_lesions(lesion_rows, output)
    if spectral_rows:
        _plot_scaling(spectral_rows, output)
    environment_record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    environment_record["elapsed_seconds"] = time.perf_counter() - started_timer
    _write_json(output / "runtime_environment.json", environment_record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
