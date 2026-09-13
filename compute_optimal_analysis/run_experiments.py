"""Spectral-Chinchilla allocation, training, diagnostics, lesions, and plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile
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
from pipelines.trainer import (LanguageModelTrainer, TrainConfig,
                               evaluate_language_model, seed_everything)
from rmt.factory import (
    RMTMethodConfig,
    dispatch_mp_fit,
    dispatch_spike_detector,
    dispatch_tail_solver,
    dispatch_unfolding,
    prepare_spectrum,
    qualified_raw_eigenvalues,
)
from rmt.mp import SpikeDetectionResult, fit_marchenko_pastur, mp_soft_rank
from rmt.overlap import dual_end_alignment, weight_basis_status
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


def _prepare_spacing_fit(
    weight: np.ndarray,
    raw_eigenvalues: np.ndarray,
    svd: Any,
    method_config: RMTMethodConfig,
    headline_fit: Any,
    *,
    headline_is_raw: bool,
    requested: bool,
) -> tuple[np.ndarray, Any | None]:
    """Return the qualified raw spectrum and its optional spacing fit."""

    values = qualified_raw_eigenvalues(weight, raw_eigenvalues, svd)
    if not requested:
        return values, None
    if headline_is_raw:
        available = bool(headline_fit.diagnostics.get("available", True))
        return values, headline_fit if available else None
    if np.count_nonzero(values > 0.0) < 4:
        return values, None
    try:
        fitted = fit_marchenko_pastur(
            values, svd.aspect_ratio, trim_upper=method_config.mp_trim_upper
        )
    except (ValueError, np.linalg.LinAlgError):
        fitted = None
    return values, fitted


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
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_sanitize_json(payload), handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _json_value(row.get(key)) for key in fields})
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _acquire_output_directory(output: Path) -> None:
    """Atomically claim an absent or intentionally pre-created empty directory."""

    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output}; use a fresh run directory"
        )
    owner = output / ".run-owner.json"
    try:
        descriptor = os.open(owner, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"output directory is already owned: {output}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "acquired_at_utc": datetime.now(timezone.utc).isoformat()}, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _selected_cell_indices(cells: str, manifest_size: int) -> set[int]:
    if cells == "all":
        selected = set(range(manifest_size))
    else:
        try:
            selected = {int(value.strip()) for value in cells.split(",") if value.strip()}
        except ValueError as error:
            raise ValueError("--cells contains an invalid manifest index") from error
    if not selected or min(selected) < 0 or max(selected) >= manifest_size:
        raise ValueError("--cells contains an invalid manifest index")
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_dataset_asset(dataset_path: str | Path) -> dict[str, Any]:
    target = Path(dataset_path).resolve()
    manifest_path = Path("data/asset_manifest.json").resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("data/asset_manifest.json is required for execution")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for record in manifest.get("files", []):
        portable = str(record.get("path", "")).replace("\\", "/")
        candidate = (Path.cwd() / portable).resolve()
        if candidate != target:
            continue
        actual_size = target.stat().st_size
        actual_digest = _sha256(target)
        if actual_size != int(record.get("bytes", -1)) or actual_digest != record.get("sha256"):
            raise ValueError("selected dataset does not match its staged manifest record")
        return {"verified": True, "bytes": actual_size, "sha256": actual_digest,
                "manifest": str(manifest_path)}
    raise ValueError("selected dataset is not covered by the staged asset manifest")


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
                portable = str(record.get("path", "")).replace("\\", "/")
                candidate = (Path.cwd() / portable).resolve()
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

    budget_values = tuple(float(value) for value in compute_budgets)
    ratio_values = tuple(float(value) for value in allocation_ratios)
    if not budget_values or any(not math.isfinite(value) or value <= 0.0 for value in budget_values):
        raise ValueError("compute_budgets must contain finite positive values")
    if not ratio_values or any(not math.isfinite(value) or value <= 0.0 for value in ratio_values):
        raise ValueError("allocation_ratios must contain finite positive values")
    for name, value in (("parameter_cap", parameter_cap), ("token_cap", token_cap)):
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.0):
            raise ValueError(f"{name} must be finite and positive when supplied")
    allocations = isoflop_grid(
        budget_values,
        ratio_values,
        law,
        target_tokens_per_parameter=20.0,
    )
    manifest: list[dict[str, Any]] = []
    for cell_index, allocation in enumerate(allocations):
        architecture_target = allocation.parameters
        if parameter_cap is not None:
            architecture_target = min(architecture_target, float(parameter_cap))
        architecture = suggest_architecture(
            architecture_target,
            vocab_size=vocab_size,
            max_parameters=parameter_cap,
        )
        realized_parameters = float(architecture["estimated_parameters"])
        # Conserve the requested IsoFLOP budget after discretizing architecture.
        isoflop_tokens = allocation.compute_budget / (
            law.flops_per_parameter_token * realized_parameters
        )
        continuous_tokens = isoflop_tokens if token_cap is None else min(isoflop_tokens, float(token_cap))
        target_tokens = int(math.floor(continuous_tokens))
        if target_tokens < 1:
            raise ValueError(
                "requested compute/caps cannot fund one next-token training target"
            )
        realized_tokens = float(target_tokens)
        manifest.append(
            {
                "cell_index": cell_index,
                **allocation.as_dict(),
                "requested_parameters": allocation.parameters,
                "requested_tokens": allocation.tokens,
                "architecture_target_parameters": architecture_target,
                "realized_parameters": realized_parameters,
                "realized_tokens": realized_tokens,
                "training_target_tokens": target_tokens,
                "continuous_affordable_tokens": float(continuous_tokens),
                "requested_tokens_per_parameter": float(allocation.tokens / allocation.parameters),
                "realized_tokens_per_parameter": float(realized_tokens / realized_parameters),
                "realized_allocation_ratio": float(
                    (realized_tokens / realized_parameters) / 20.0
                ),
                "realized_training_compute": float(
                    law.flops_per_parameter_token * realized_parameters * realized_tokens
                ),
                "parameter_cap_applied": bool(parameter_cap is not None and allocation.parameters > parameter_cap),
                "token_cap_applied": bool(token_cap is not None and isoflop_tokens > token_cap),
                "isoflop_tokens_for_realized_architecture": float(isoflop_tokens),
                "architecture": architecture,
            }
        )
    # Flag designs where caps/discretization erase the requested intervention.
    signatures: dict[tuple[float, int], list[int]] = defaultdict(list)
    for index, cell in enumerate(manifest):
        # Requested budget is intervention metadata, not executable design
        # identity.  Compare the discretized architecture and integer target.
        signature = (float(cell["realized_parameters"]),
                     int(cell["training_target_tokens"]))
        signatures[signature].append(index)
    for indices in signatures.values():
        collapsed = len(indices) > 1
        for index in indices:
            manifest[index]["allocation_collapsed"] = collapsed
            manifest[index]["collapsed_with_cells"] = [
                manifest[other]["cell_index"] for other in indices if other != index
            ]
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
    analysis_dtype: str = "float64",
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
            analysis_dtype=analysis_dtype,
        )
        selected_dtype = torch.float64 if analysis_dtype == "float64" else torch.float32
        weight = parameter.detach().to(dtype=selected_dtype).cpu().numpy()
        raw_methods = {"lanczos_stieltjes", "thamm_modified_singular"}
        raw_eigenvalues = np.asarray(svd.covariance_eigenvalues, dtype=np.float64)
        # ESD/tail preprocessing is independent of operator-domain MP methods.
        # In particular, the Golden Lanczos path still performs its requested
        # FARMS density analysis while Lanczos sees the original operator.
        prepared = prepare_spectrum(weight, method_config, svd=svd)
        eigenvalues = prepared.eigenvalues
        spectrum_mode = prepared.mode
        spectrum_aspect_ratio = prepared.aspect_ratio
        mp_fit = dispatch_mp_fit(weight, method_config, svd=svd, prepared=prepared)
        mp_is_raw = method_config.mp_fit_method in raw_methods
        mp_eigenvalues = raw_eigenvalues if mp_is_raw else eigenvalues
        mp_spectrum_mode = "raw" if mp_is_raw else spectrum_mode
        if prepared.farms is not None:
            farms_geometry = tuple(map(int, prepared.farms.window_shape))
            normalization = prepared.farms.normalization
            spectrum_denominator: object = (
                max(farms_geometry) if normalization == "canonical"
                else None if normalization == "raw"
                else "per_window_trace"
            )
            spectrum_geometry = farms_geometry
            spectrum_normalization_denominators = json.dumps(
                prepared.farms.normalization_denominators.tolist())
            spectrum_observation_count = int(prepared.farms.eigenvalues_per_submatrix)
        else:
            spectrum_geometry = tuple(map(int, weight.shape))
            spectrum_denominator = (
                svd.normalization if spectrum_mode == "raw"
                else "source_covariance_then_shape_normalization"
            )
            spectrum_normalization_denominators = json.dumps([svd.normalization])
            spectrum_observation_count = int(eigenvalues.size)
        mp_geometry = tuple(map(int, weight.shape)) if mp_is_raw else spectrum_geometry
        fit_available = bool(mp_fit.diagnostics.get("available", True))
        if not fit_available:
            spike_fit = SpikeDetectionResult(
                method=method_config.spike_detector, threshold=float("nan"),
                bulk_edge=float("nan"), spikes=np.asarray([]),
                indices=np.asarray([], dtype=np.int64),
                diagnostics={"available": False, "status": "unavailable",
                             "reason": "MP fit unavailable"},
            )
        elif (method_config.mp_fit_method == "lanczos_stieltjes"
                and method_config.spike_detector == "lanczos_poles"):
            diagnostics = mp_fit.diagnostics
            poles = np.asarray(diagnostics.get("poles", []), dtype=np.float64)
            spike_fit = SpikeDetectionResult(
                method="lanczos_poles",
                threshold=float(diagnostics["threshold"]),
                bulk_edge=float(diagnostics["lambda_plus"]),
                spikes=poles,
                indices=np.arange(poles.size, dtype=np.int64),
                diagnostics=diagnostics,
            )
        else:
            spike_fit = dispatch_spike_detector(
                weight,
                method_config,
                variance=mp_fit.variance,
                eigenvalues=mp_eigenvalues,
                aspect_ratio=mp_fit.aspect_ratio,
                operator_shape=mp_geometry,
            )
        tail = dispatch_tail_solver(
            eigenvalues,
            method_config,
            min_tail=method_config.tail_minimum,
        )
        hill_k = max(2, min(eigenvalues.size - 1, int(round(np.sqrt(eigenvalues.size)))))
        try:
            hill = hill_alpha_at(eigenvalues, hill_k)
            plateau = hill_plateau(eigenvalues, window=max(3, min(20, eigenvalues.size // 4)))
        except ValueError:
            hill = float("nan")
            plateau = {"hill_plateau_alpha": float("nan"),
                       "hill_plateau_width": 0, "hill_is_powerlaw": False}
        # Level statistics always describe the one original covariance
        # operator.  Pooled FARMS observations are valid for an ESD, not for
        # nearest-neighbour correlations.  Fit a separate raw-domain bulk when
        # the selected headline MP fit lives in another domain.
        needs_spacing = bool(
            compute_spacing_distribution or compute_number_variance or compute_delta3
        )
        spacing_eigenvalues, spacing_mp_fit = _prepare_spacing_fit(
            weight,
            raw_eigenvalues,
            svd,
            method_config,
            mp_fit,
            headline_is_raw=mp_is_raw,
            requested=needs_spacing,
        )
        if (method_config.mp_fit_method == "farms_unbiased"
                and "spectral_max" in mp_fit.diagnostics):
            mp_soft_spectrum = np.asarray([float(mp_fit.diagnostics["spectral_max"])])
        else:
            mp_soft_spectrum = mp_eigenvalues
        bulk_mask = (
            np.zeros(spacing_eigenvalues.shape, dtype=bool)
            if spacing_mp_fit is None else
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
        unfolding_status = "not_requested_or_insufficient"
        # Adjacent-gap ratios need no unfolding and remain available when a
        # polynomial/spline/kernel smoother is unavailable.
        if compute_spacing_distribution and bulk_levels.size >= 3 and float(np.ptp(bulk_levels)) > 0.0:
            ratio = r_statistic(bulk_levels)
        if (needs_spacing and bulk_levels.size >= minimum_bulk
                and float(np.ptp(bulk_levels)) > 0.0):
            try:
                unfolded = dispatch_unfolding(bulk_levels, method_config)
                unfolding_status = "available"
            except ValueError as error:
                unfolded = None
                unfolding_status = f"unavailable: {error}"
            if unfolded is not None:
                if compute_spacing_distribution:
                    spacings = np.diff(unfolded)
                    spacings = spacings[np.isfinite(spacings) & (spacings >= 0.0)]
                    if spacings.size and np.mean(spacings) > 0.0:
                        spacings = spacings / np.mean(spacings)
                    if np.count_nonzero(spacings > 0.0) >= 8:
                        beta = (
                            fit_brody_cdf_nls(spacings).beta
                            if brody_fit_method == "cdf_nls"
                            else fit_brody(spacings).beta
                        )
                if compute_number_variance:
                    variance_10 = number_variance(
                        unfolded, 10.0, unfolded=True,
                        method=number_variance_method, rng=seed,
                    )
                if compute_delta3:
                    rigidity_10 = dyson_mehta_delta3(
                        unfolded, 10.0, unfolded=True,
                    )
        module_name = parameter_name[: -len(".weight")]
        covariance = covariances.get(f"{module_name}:pre")
        top_alignment = bulk_alignment = bottom_alignment = float("nan")
        overlap_status = "not_requested_or_unavailable"
        activation_rank = 0
        activation_observation_count = 0
        activation_accumulation_dtype = "unavailable"
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
            overlap_status = str(alignment.get("status", "available"))
            activation_rank = int(alignment.get("activation_rank", 0))
            activation_observation_count = int(alignment.get("activation_observation_count") or 0)
            activation_accumulation_dtype = str(
                alignment.get("activation_accumulation_dtype", "unknown")
            )
        role, layer = _role_and_layer(parameter_name)
        porter_thomas_ks = float("nan")
        porter_thomas_fraction = float("nan")
        porter_thomas_status = (
            weight_basis_status(svd) if compute_porter_thomas else "not_requested"
        )
        if compute_porter_thomas and svd.Vh.shape[0] <= 10:
            porter_thomas_status = "unavailable: pooled calibration requires more than 10 vectors"
        if (compute_porter_thomas and porter_thomas_status == "available"
                and svd.Vh.shape[0] > 10):
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
            "realized_tokens_per_parameter": cell.get("realized_tokens_per_parameter"),
            "realized_allocation_ratio": cell.get("realized_allocation_ratio"),
            "allocation_collapsed": cell.get("allocation_collapsed", False),
            "realized_training_compute": cell["realized_training_compute"],
            "optimizer_updates": cell.get("optimizer_updates"),
            "attempted_steps": cell.get("attempted_steps"),
            "parameter_name": parameter_name,
            "role": role,
            "layer": layer,
            "n": svd.n,
            "m": svd.m,
            "normalization": svd.normalization,
            "analysis_dtype": analysis_dtype,
            "aspect_ratio": spectrum_aspect_ratio,
            "mp_aspect_ratio": mp_fit.aspect_ratio,
            "mp_spectrum_domain": mp_spectrum_mode,
            "mp_spectrum_geometry": json.dumps(mp_geometry),
            "mp_spectrum_denominator": (
                svd.normalization if mp_is_raw else spectrum_denominator
            ),
            "mp_observation_count": int(mp_eigenvalues.size),
            "aspect_ratio_mode": method_config.aspect_ratio_mode,
            "spectrum_mode": spectrum_mode,
            "spectrum_units": "eigenvalue",
            "spectrum_geometry": json.dumps(spectrum_geometry),
            "spectrum_denominator": spectrum_denominator,
            "spectrum_normalization_denominators": spectrum_normalization_denominators,
            "spectrum_provenance": json.dumps(
                _sanitize_json(prepared.diagnostics), sort_keys=True),
            "spectrum_observations_per_operator": spectrum_observation_count,
            "spectrum_observation_count": int(eigenvalues.size),
            "mp_fit_method": method_config.mp_fit_method,
            "mp_variance": mp_fit.variance,
            "mp_lambda_minus": mp_fit.lambda_minus,
            "mp_lambda_plus": mp_fit.lambda_plus,
            "mp_ks": mp_fit.ks_distance,
            "mp_bulk_fraction": mp_fit.bulk_fraction,
            "mp_fit_available": fit_available,
            "mp_fit_converged": mp_fit.diagnostics.get(
                "converged", mp_fit.diagnostics.get("optimizer_success", fit_available)
            ),
            "mp_fit_method_status": mp_fit.diagnostics.get(
                "status", mp_fit.diagnostics.get("optimizer_message", "available")
            ),
            "mp_fit_diagnostics": json.dumps(_sanitize_json(mp_fit.diagnostics), sort_keys=True),
            "lower_outliers": mp_fit.n_lower_outliers,
            "upper_outliers": mp_fit.n_upper_outliers,
            "above_edge_observations": int(np.count_nonzero(mp_eigenvalues > mp_fit.lambda_plus))
                if fit_available else 0,
            "spike_detector": method_config.spike_detector,
            "spike_detector_domain": (
                "raw" if method_config.spike_detector == "lanczos_poles"
                else mp_spectrum_mode
            ),
            "spike_detector_geometry": json.dumps(
                tuple(map(int, weight.shape))
                if method_config.spike_detector == "lanczos_poles" else mp_geometry
            ),
            "spike_detector_denominator": (
                svd.normalization if method_config.spike_detector == "lanczos_poles"
                else (svd.normalization if mp_is_raw else spectrum_denominator)
            ),
            "spike_count_semantics": (
                "raw_operator_poles"
                if method_config.spike_detector == "lanczos_poles"
                else "pooled_observations_at_per_operator_threshold"
                if prepared.farms is not None and not mp_is_raw
                else "single_operator_observations"
            ),
            "spike_threshold": spike_fit.threshold,
            "detected_spikes": spike_fit.n_spikes,
            "spike_detector_available": spike_fit.diagnostics.get("available", True),
            "spike_detector_converged": spike_fit.diagnostics.get("converged", fit_available),
            "spike_diagnostics": json.dumps(_sanitize_json(spike_fit.diagnostics), sort_keys=True),
            "mp_soft_rank": mp_soft_rank(
                mp_soft_spectrum,
                mp_fit.lambda_plus,
            ),
            "spacing_spectrum_domain": "raw_single_operator",
            "spacing_spectrum_geometry": json.dumps(tuple(map(int, weight.shape))),
            "spacing_spectrum_denominator": svd.normalization,
            "spacing_observation_count": int(spacing_eigenvalues.size),
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
            "brody_sample_conditioning": "positive_spacings" if np.any(spacings == 0.0) else "complete",
            "zero_spacing_count": int(np.count_nonzero(spacings == 0.0)),
            "unfolding_strategy": method_config.unfolding_strategy,
            "unfolding_status": unfolding_status,
            "r_statistic": ratio,
            "number_variance_L10": variance_10,
            "number_variance_method": number_variance_method,
            "dyson_mehta_delta3_L10": rigidity_10,
            "porter_thomas_ks_mean": porter_thomas_ks,
            "porter_thomas_fraction_random": porter_thomas_fraction,
            "porter_thomas_status": porter_thomas_status,
            "top_activation_alignment": top_alignment,
            "bulk_activation_alignment": bulk_alignment,
            "bottom_activation_alignment": bottom_alignment,
            "overlap_metric": method_config.overlap_metric,
            "overlap_status": overlap_status,
            "activation_covariance_rank": activation_rank,
            "activation_observation_count": activation_observation_count,
            "activation_accumulation_dtype": activation_accumulation_dtype,
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
        positive = values[np.isfinite(values) & (values > 0.0)]
        if positive.size == 0:
            axis.text(0.5, 0.5, "No positive eigenvalues", ha="center", va="center")
            axis.set_title(f"{row['regime']} L{row['layer']} {row['role']}")
            continue
        lower, upper = float(positive.min()), float(positive.max())
        if np.isclose(lower, upper):
            lower, upper = lower * 0.9, upper * 1.1
            if lower <= 0.0:
                lower = max(np.finfo(float).tiny, upper * 0.5)
        bins = np.geomspace(lower, upper, min(60, max(12, positive.size)) + 1)
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
        if row.get("mp_spectrum_domain") == row.get("spectrum_mode"):
            axis.axvline(row["mp_lambda_minus"], color="tab:green", linestyle="--", linewidth=1)
            axis.axvline(row["mp_lambda_plus"], color="tab:red", linestyle="--", linewidth=1)
        axis.set_title(f"{row['regime']} L{row['layer']} {row['role']}")
        axis.set_xlabel("lambda")
        axis.set_ylabel("density")
    figure.tight_layout()
    figure.savefig(output / "esd_powerlaw_fits.png", dpi=180)
    plt.close(figure)


def _plot_spacing(artifacts: list[dict[str, Any]], output: Path, *,
                  brody_fit_method: str = "mle") -> None:
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
        beta = (fit_brody(combined, method=brody_fit_method).beta
                if np.count_nonzero(combined > 0.0) >= 8 else float("nan"))
        label = (f"{regime}, pooled {brody_fit_method} beta={beta:.3f}"
                 if math.isfinite(beta) else f"{regime}, pooled {brody_fit_method}")
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
    canonical = ("top", "bulk", "bottom")
    tranches = [name for name in canonical if any(row.get("tranche") == name for row in rows)]
    width = min(0.8 / max(len(tranches), 1), 0.24)
    x = np.arange(len(cells), dtype=np.float64)
    center = 0.5 * (len(tranches) - 1)
    for offset, tranche in enumerate(tranches):
        values = []
        for cell in cells:
            matches = [float(row["delta"]) for row in rows
                       if int(row["cell_index"]) == cell and row["tranche"] == tranche]
            values.append(matches[0] if matches else float("nan"))
        axis.bar(x + (offset - center) * width, values, width=width, label=tranche)
    axis.set_xticks(x, [str(cell) for cell in cells])
    axis.set_xlabel("manifest cell index")
    axis.set_ylabel("perplexity change")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "lesioning_perplexity_impact.png", dpi=180)
    plt.close(figure)


def _plot_scaling(rows: list[dict[str, Any]], output: Path) -> None:
    # Requested kappa is the stable intervention identity.  Realized ratios are
    # point metadata and legitimately drift as architectures/token counts round.
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row.get("realized_training_compute", row["compute_budget"])),
                 float(row["kappa"]))].append(row)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for kappa in sorted({float(row["kappa"]) for row in rows}):
        points = sorted((key, values) for key, values in grouped.items() if key[1] == kappa)
        budgets = [key[0] for key, _ in points]
        alpha = [
            float(np.nanmean([float(row["tail_alpha_lambda"]) for row in values]))
            for _, values in points
        ]
        beta = [
            float(np.nanmean([float(row["brody_beta"]) for row in values]))
            for _, values in points
        ]
        realized = [float(np.nanmean([
            float(row.get("realized_allocation_ratio", row["kappa"])) for row in values
        ])) for _, values in points]
        collapsed = [any(bool(row.get("allocation_collapsed", False)) for row in values)
                     for _, values in points]
        label = f"requested kappa={kappa:g}"
        axes[0].plot(budgets, alpha, marker="o", label=label)
        axes[1].plot(budgets, beta, marker="o", label=label)
        for axis, values in zip(axes, (alpha, beta)):
            for budget, value, ratio, is_collapsed in zip(
                    budgets, values, realized, collapsed):
                if math.isfinite(value):
                    note = f"r={ratio:.3g}" + ("\ncollapsed" if is_collapsed else "")
                    axis.annotate(note, (budget, value), fontsize=7)
                    if is_collapsed:
                        axis.scatter([budget], [value], marker="x", color="red", zorder=4)
    for axis, ylabel in zip(axes, ("mean selected tail exponent", "mean Brody beta")):
        axis.set_xscale("log")
        axis.set_xlabel("measured training FLOPs")
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
    seed_everything(seed)
    cell["resolved_seed"] = seed
    config = _config_from_manifest(cell, args.sequence_length)
    model = CausalTransformer(config)
    actual_parameters = float(model.num_parameters())
    if args.parameter_cap is not None and actual_parameters > float(args.parameter_cap):
        raise ValueError("realized architecture exceeds the hard parameter cap")
    if actual_parameters != float(cell["realized_parameters"]):
        cell["realized_parameters"] = actual_parameters
        isoflop_tokens = float(cell["compute_budget"]) / (6.0 * actual_parameters)
        cell["realized_tokens"] = (
            isoflop_tokens if args.max_train_tokens is None
            else min(isoflop_tokens, float(args.max_train_tokens))
        )
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
        validation_max_batches=args.validation_batches,
        max_train_tokens=int(cell["training_target_tokens"]),
        seed=seed,
    )
    trainer = LanguageModelTrainer(model, train_config)
    training_path = Path(args.output_dir) / "training_metrics.jsonl"

    def stream_record(record: dict[str, float | int]) -> None:
        enriched = {**record, "cell_index": cell["cell_index"],
                    "regime": cell["regime"], "kappa": cell["kappa"]}
        with training_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_sanitize_json(enriched), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if (args.save_checkpoints and int(record.get("optimizer_update", 0))
                and int(record.get("step", 0)) % max(1, args.log_every) == 0):
            trainer.save_checkpoint(
                Path(args.output_dir) / "checkpoints" / f"cell_{cell['cell_index']}_latest.pt",
                metadata={"manifest_cell": cell},
            )

    history = trainer.fit(train_loader, validation_loader, callback=stream_record)
    for record in history:
        record.update({"cell_index": cell["cell_index"], "regime": cell["regime"], "kappa": cell["kappa"]})
    cell["planned_realized_tokens"] = cell["realized_tokens"]
    cell["realized_tokens"] = float(trainer.processed_train_tokens)
    cell["optimizer_updates"] = trainer.global_step
    cell["attempted_steps"] = trainer.attempted_steps
    cell["skipped_steps"] = trainer.skipped_steps
    cell["attempted_target_tokens"] = trainer.attempted_target_tokens
    cell["forwarded_input_positions"] = trainer.forwarded_input_positions
    cell["successful_update_target_tokens"] = trainer.processed_train_tokens
    cell["realized_tokens_per_parameter"] = float(
        trainer.processed_train_tokens / cell["realized_parameters"]
    )
    cell["realized_allocation_ratio"] = cell["realized_tokens_per_parameter"] / 20.0
    cell["training_budget_complete"] = bool(
        trainer.processed_train_tokens == train_config.max_train_tokens
    )
    cell["realized_training_compute"] = float(
        6.0 * cell["realized_parameters"] * trainer.processed_train_tokens
    )
    cell["realized_training_compute_semantics"] = "6ND approximation; D=successful update targets"
    cell["forwarded_compute_proxy"] = float(
        6.0 * cell["realized_parameters"] * trainer.forwarded_input_positions
    )
    if args.save_checkpoints:
        trainer.save_checkpoint(
            Path(args.output_dir) / "checkpoints" / f"cell_{cell['cell_index']}.pt",
            metadata={"manifest_cell": cell},
        )
    covariances = {}
    if args.compute_activation_overlap:
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
        analysis_dtype=args.analysis_dtype,
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
            analysis_dtype=args.analysis_dtype,
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
                "all_energy_matches_reached": bool(all(
                    matrix["target_reached"] for matrix in result["matrices"]
                )),
                "matrix_interventions": json.dumps(
                    _sanitize_json(result["matrices"]), sort_keys=True
                ),
            }
        )
    return spectral_rows, lesion_rows, history, artifacts


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_pipeline_cli_arguments(parser)
    raw = None if argv is None else list(argv)
    namespace = parser.parse_args(raw)
    import sys
    tokens = sys.argv[1:] if raw is None else raw
    explicit_options = {
        token.split("=", 1)[0] for token in tokens if token.startswith("--")
    }
    option_destinations = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    namespace._explicit_options = sorted(explicit_options)
    namespace._explicit_destinations = sorted({
        option_destinations[option]
        for option in explicit_options
        if option in option_destinations
    })
    return namespace


def _resolve_runtime_configuration(args: argparse.Namespace) -> None:
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.scaling_budget_flops or any(
        not math.isfinite(value) or value <= 0.0 for value in args.scaling_budget_flops
    ):
        raise ValueError("scaling-budget-flops must contain positive values")
    if not args.allocation_ratios or any(
        not math.isfinite(value) or value <= 0.0 for value in args.allocation_ratios
    ):
        raise ValueError("allocation-ratios must contain positive values")
    if (args.parameter_cap is not None
            and (not math.isfinite(args.parameter_cap) or args.parameter_cap <= 0.0)):
        raise ValueError("parameter-cap must be finite and positive when supplied")
    if (args.max_train_tokens is not None
            and (not math.isfinite(args.max_train_tokens) or args.max_train_tokens <= 0.0)):
        raise ValueError("max-train-tokens must be finite and positive when supplied")
    if (not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0
            or not math.isfinite(args.gradient_clip) or args.gradient_clip <= 0.0):
        raise ValueError("learning-rate and gradient-clip must be finite and positive")
    if args.warmup_steps < 0 or args.log_every < 1:
        raise ValueError("warmup-steps must be nonnegative and log-every positive")
    if args.dataloader_workers < 0 or args.prefetch_factor < 1:
        raise ValueError("dataloader-workers must be nonnegative and prefetch-factor positive")
    if args.sequence_length < 2 or args.batch_size < 1 or args.vocab_size < 2:
        raise ValueError("sequence length, batch size, and vocabulary size are invalid")
    if args.activation_batches < 1 or args.validation_batches < 1:
        raise ValueError("activation and validation batch limits must be positive")
    if not math.isfinite(args.lesion_fraction) or not 0.0 < args.lesion_fraction <= 1.0:
        raise ValueError("lesion-fraction must be finite and lie in (0, 1]")
    explicit_destinations = set(getattr(args, "_explicit_destinations", ()))
    def preset(attribute: str, value: Any) -> None:
        if attribute not in explicit_destinations:
            setattr(args, attribute, value)

    method_presets = {
        "reproduce_paper1": {
            "aspect_ratio_mode": "raw", "mp_fit_method": "analytic_mp",
            "spike_detector": "tracy_widom_95", "overlap_metric": "staats_dual_end",
            "unfolding_strategy": "gaussian_kernel", "tail_solver": "rank_ordered_mle",
            "run_spectral_lesioning": True, "compute_activation_overlap": True,
        },
        "reproduce_paper2": {
            "aspect_ratio_mode": "raw", "mp_fit_method": "thamm_modified_singular",
            "spike_detector": "tracy_widom_95",
            "unfolding_strategy": "polynomial_chebyshev", "polynomial_degree": 15,
            "compute_spacing_distribution": True, "compute_number_variance": True,
            "compute_delta3": True,
        },
        "reproduce_paper3": {
            "aspect_ratio_mode": "raw", "mp_fit_method": "kde_bulk_fit",
            "spike_detector": "tracy_widom_95", "tail_solver": "clauset_mle",
            "compute_stable_rank": True,
        },
        "compute_optimal_rmt": {
            "aspect_ratio_mode": "farms_normalized",
            "mp_fit_method": "lanczos_stieltjes", "spike_detector": "lanczos_poles",
            "unfolding_strategy": "spline_monotone", "tail_solver": "clauset_mle",
            "overlap_metric": "staats_dual_end", "run_spectral_lesioning": True,
            "compute_activation_overlap": True,
        },
    }
    for attribute, value in method_presets.get(args.experiment_mode, {}).items():
        preset(attribute, value)

    # Validate dependent options only after presets and explicit --no-* flags
    # have resolved the effective lesion setting.
    tranches = tuple(
        name.strip().lower() for name in args.lesion_tranches.split(",") if name.strip()
    )
    if (args.run_spectral_lesioning
            and (not tranches or any(name not in {"top", "bulk", "bottom"} for name in tranches))):
        raise ValueError("lesion-tranches must be a comma-separated subset of top,bulk,bottom")

    # Constructing the shared config performs selector, numeric, and cross-method
    # compatibility validation before data/model/output acquisition.
    rmt_config_from_namespace(args)
    if args.execute and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA execution was requested but no CUDA device is available")
    if args.execute and args.svd_backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA SVD was requested but no CUDA device is available")
    if args.execute and args.covariance_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA covariance accumulation was requested but CUDA is unavailable")


def main(argv: Iterable[str] | None = None) -> int:
    started_timer = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    args = parse_args(argv)
    _resolve_runtime_configuration(args)
    output = Path(args.output_dir)
    manifest = build_manifest(
        vocab_size=args.vocab_size,
        compute_budgets=args.scaling_budget_flops,
        allocation_ratios=args.allocation_ratios,
        parameter_cap=args.parameter_cap,
        token_cap=args.max_train_tokens,
    )
    selected_indices = _selected_cell_indices(args.cells, len(manifest))
    selected_collapsed_groups = {
        tuple(sorted(
            {int(cell["cell_index"]), *map(int, cell.get("collapsed_with_cells", []))}
            & selected_indices
        ))
        for cell in manifest
        if int(cell["cell_index"]) in selected_indices and cell.get("allocation_collapsed")
    }
    duplicate_groups = sorted(group for group in selected_collapsed_groups if len(group) > 1)
    if args.execute and duplicate_groups and not args.allow_collapsed_allocations:
        collapsed = sorted({index for group in duplicate_groups for index in group})
        raise ValueError(
            f"selected allocation cells {collapsed} collapse to identical realized N/D designs; "
            "change caps or explicitly allow capped calibration runs")
    _acquire_output_directory(output)
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
    verification = _verify_dataset_asset(args.dataset_path)
    environment_record["dataset"].update(verification)
    _write_json(output / "runtime_environment.json", environment_record)
    tokens = load_token_array(args.dataset_path)
    if int(np.max(tokens)) >= args.vocab_size:
        raise ValueError("vocab-size must exceed every token id in the local array")
    train_tokens, validation_tokens = split_tokens(tokens)
    if min(train_tokens.size, validation_tokens.size) < args.sequence_length:
        raise ValueError("both token splits must contain at least one full sequence")
    spectral_rows: list[dict[str, Any]] = []
    lesion_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    # Records are appended and flushed by the trainer callback.
    (output / "training_metrics.jsonl").write_text("", encoding="utf-8")
    cell_status: list[dict[str, Any]] = []
    for cell in manifest:
        if int(cell["cell_index"]) not in selected_indices:
            continue
        try:
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
            cell_status.append({"cell_index": cell["cell_index"], "status": "complete"})
            # Persist every completed cell atomically before starting the next.
            _write_csv(output / "spectral_metrics.csv", spectral_rows)
            _write_csv(output / "lesion_metrics.csv", lesion_rows)
            _write_json(output / "allocation_manifest.json", manifest)
            _write_json(output / "cell_status.json", cell_status)
        except Exception as exc:
            cell_status.append({"cell_index": cell["cell_index"], "status": "failed",
                                "error": repr(exc)})
            _write_json(output / "cell_status.json", cell_status)
            environment_record["status"] = "failed"
            environment_record["error"] = repr(exc)
            _write_json(output / "runtime_environment.json", environment_record)
            raise
    if artifacts:
        _plot_esd(artifacts, output)
        if args.compute_spacing_distribution:
            _plot_spacing(artifacts, output, brody_fit_method=args.brody_fit_method)
        if args.compute_activation_overlap:
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
