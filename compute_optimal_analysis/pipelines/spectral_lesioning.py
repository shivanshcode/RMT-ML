"""Reversible singular-value tranche surgery and independent benchmarks."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from collections.abc import Callable, Iterable, Iterator, Sequence
import math

import numpy as np
import torch
from torch import Tensor, nn

from rmt.mp import estimate_sigma_gd_median, mp_bounds


@dataclass(frozen=True)
class LesionInfo:
    tranche: str
    fraction: float
    mode: str
    indices: Sequence[int]
    removed_frobenius_energy: float
    total_frobenius_energy: float
    rank: int
    target_frobenius_energy: float | None = None
    target_reached: bool = True

    @property
    def removed_energy_fraction(self) -> float:
        if self.total_frobenius_energy == 0.0:
            return 0.0
        return self.removed_frobenius_energy / self.total_frobenius_energy

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["removed_energy_fraction"] = self.removed_energy_fraction
        return result


def _svd_components(
    weight: Tensor,
    *,
    backend: str,
    driver: str,
) -> tuple[Tensor, Tensor, Tensor]:
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("svd backend must be auto, cpu, or cuda")
    if driver not in {"default", "gesvdj", "gesvd", "gesvda"}:
        raise ValueError("unsupported SVD driver")
    if backend == "auto":
        target = weight.device
    else:
        target = torch.device(backend)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA SVD was requested but is unavailable")
    analysis_dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    analysis = weight.detach().to(device=target, dtype=analysis_dtype)
    keyword_arguments: dict[str, object] = {"full_matrices": False}
    if target.type == "cuda" and driver != "default":
        keyword_arguments["driver"] = driver
    return torch.linalg.svd(analysis, **keyword_arguments)


def _canonical_tranche(tranche: str) -> str:
    name = str(tranche).lower()
    aliases = {
        "top": "top",
        "largest": "top",
        "outlier": "top",
        "bulk": "bulk",
        "random_bulk": "bulk",
        "bottom": "bottom",
        "smallest": "bottom",
    }
    if name not in aliases:
        raise ValueError("tranche must identify top, bulk, or bottom singular values")
    return aliases[name]


def _mp_bulk_candidates(singular_values: Tensor, n: int, m: int) -> Tensor:
    values = singular_values.detach().cpu().double().numpy()
    try:
        sigma = estimate_sigma_gd_median(s=values, n=n, m=m)
        lower, upper = mp_bounds(n, m, sigma)
        candidates = np.flatnonzero((values >= lower) & (values <= upper))
    except ValueError:
        candidates = np.asarray([], dtype=int)
    if candidates.size == 0:
        margin = max(1, int(math.ceil(0.1 * values.size)))
        candidates = np.arange(margin, values.size - margin, dtype=int)
    if candidates.size == 0:
        candidates = np.arange(values.size, dtype=int)
    return torch.as_tensor(candidates, dtype=torch.long)


def _count_selection(
    singular_values: Tensor,
    tranche: str,
    fraction: float,
    n: int,
    m: int,
    generator: torch.Generator | None,
) -> Tensor:
    rank = singular_values.numel()
    count = min(rank, max(1, int(math.ceil(fraction * rank))))
    if tranche == "top":
        return torch.arange(count, dtype=torch.long)
    if tranche == "bottom":
        return torch.arange(rank - count, rank, dtype=torch.long)
    candidates = _mp_bulk_candidates(singular_values, n, m)
    if candidates.numel() < count:
        candidates = torch.arange(rank, dtype=torch.long)
    order = torch.randperm(candidates.numel(), generator=generator)
    return candidates[order[:count]]


def _energy_selection(
    singular_values: Tensor,
    tranche: str,
    fraction: float,
    reference_energy: float | None,
    n: int,
    m: int,
    generator: torch.Generator | None,
) -> Tensor:
    energies = singular_values.detach().cpu().double().square()
    total = float(energies.sum())
    target = fraction * total if reference_energy is None else float(reference_energy)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("reference_energy must be finite and positive")
    rank = energies.numel()
    if tranche == "top":
        order = torch.arange(rank, dtype=torch.long)
    elif tranche == "bottom":
        order = torch.arange(rank - 1, -1, -1, dtype=torch.long)
    else:
        candidates = _mp_bulk_candidates(singular_values, n, m)
        permutation = torch.randperm(candidates.numel(), generator=generator)
        order = candidates[permutation]
    cumulative = torch.cumsum(energies[order], dim=0)
    reached = torch.nonzero(cumulative >= target, as_tuple=False)
    count = order.numel() if reached.numel() == 0 else int(reached[0, 0]) + 1
    return order[:count]


def lesion_matrix(
    weight: Tensor,
    tranche: str,
    *,
    fraction: float = 0.05,
    mode: str = "count",
    reference_energy: float | None = None,
    generator: torch.Generator | None = None,
    svd_backend: str = "auto",
    svd_driver: str = "gesvdj",
    svd_factors: tuple[Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, LesionInfo]:
    """Return a reconstructed matrix with one singular tranche zeroed."""

    if weight.ndim != 2 or min(weight.shape) < 1:
        raise ValueError("weight must be a nonempty matrix")
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    name = _canonical_tranche(tranche)
    mode = str(mode).lower()
    if mode not in {"count", "energy"}:
        raise ValueError("mode must be count or energy")
    U, singular_values, Vh = (
        _svd_components(weight, backend=svd_backend, driver=svd_driver)
        if svd_factors is None else svd_factors
    )
    target_energy: float | None = None
    if mode == "count":
        indices = _count_selection(
            singular_values,
            name,
            fraction,
            weight.shape[0],
            weight.shape[1],
            generator,
        )
    else:
        total_before = float(torch.sum(singular_values.detach().cpu().double().square()))
        target_energy = fraction * total_before if reference_energy is None else float(reference_energy)
        indices = _energy_selection(
            singular_values,
            name,
            fraction,
            reference_energy,
            weight.shape[0],
            weight.shape[1],
            generator,
        )
    device_indices = indices.to(device=singular_values.device)
    modified = singular_values.clone()
    removed_energy = float(torch.sum(modified[device_indices].double().square()).cpu())
    total_energy = float(torch.sum(modified.double().square()).cpu())
    modified[device_indices] = 0.0
    reconstructed = (U * modified.unsqueeze(0)) @ Vh
    info = LesionInfo(
        tranche=name,
        fraction=fraction,
        mode=mode,
        indices=tuple(int(index) for index in torch.sort(indices).values.tolist()),
        removed_frobenius_energy=removed_energy,
        total_frobenius_energy=total_energy,
        rank=int(singular_values.numel()),
        target_frobenius_energy=target_energy,
        target_reached=(target_energy is None or removed_energy >= target_energy),
    )
    return reconstructed.to(device=weight.device, dtype=weight.dtype), info


def lesion_matrix_decile(
    weight: Tensor,
    decile: int,
    *,
    n_deciles: int = 10,
    svd_backend: str = "auto",
    svd_driver: str = "gesvdj",
) -> tuple[Tensor, LesionInfo]:
    """Reproduce the reference index-decile singular-value lesion."""

    if weight.ndim != 2 or min(weight.shape) < 1:
        raise ValueError("weight must be a nonempty matrix")
    groups = int(n_deciles)
    selected_group = int(decile)
    if groups < 1 or not 0 <= selected_group < groups:
        raise ValueError("decile must index one of n_deciles groups")
    U, singular_values, Vh = _svd_components(
        weight,
        backend=svd_backend,
        driver=svd_driver,
    )
    if groups > singular_values.numel():
        raise ValueError("n_deciles cannot exceed the reduced matrix rank")
    boundaries = torch.linspace(0, singular_values.numel(), groups + 1, dtype=torch.float64)
    boundaries = torch.floor(boundaries).to(dtype=torch.long)
    start = int(boundaries[selected_group])
    stop = int(boundaries[selected_group + 1])
    indices = torch.arange(start, stop, dtype=torch.long)
    modified = singular_values.clone()
    device_indices = indices.to(device=modified.device)
    removed_energy = float(torch.sum(modified[device_indices].double().square()).cpu())
    total_energy = float(torch.sum(modified.double().square()).cpu())
    modified[device_indices] = 0.0
    reconstructed = (U * modified.unsqueeze(0)) @ Vh
    info = LesionInfo(
        tranche=f"decile_{selected_group}",
        fraction=float(indices.numel() / max(singular_values.numel(), 1)),
        mode="reference_decile",
        indices=tuple(int(index) for index in indices.tolist()),
        removed_frobenius_energy=removed_energy,
        total_frobenius_energy=total_energy,
        rank=int(singular_values.numel()),
    )
    return reconstructed.to(device=weight.device, dtype=weight.dtype), info


@contextmanager
def spectral_decile_lesion(
    model: nn.Module,
    parameter_names: Iterable[str],
    decile: int,
    *,
    n_deciles: int = 10,
    svd_backend: str = "auto",
    svd_driver: str = "gesvdj",
) -> Iterator[list[LesionInfo]]:
    """Temporarily apply one descending index decile to named matrices."""

    names = list(dict.fromkeys(str(name) for name in parameter_names))
    if not names:
        raise ValueError("parameter_names must be nonempty")
    parameters = dict(model.named_parameters())
    missing = [name for name in names if name not in parameters]
    if missing:
        raise KeyError(f"unknown model parameters: {missing}")
    snapshots = {name: parameters[name].detach().clone() for name in names}
    information: list[LesionInfo] = []
    try:
        with torch.no_grad():
            for name in names:
                modified, info = lesion_matrix_decile(
                    parameters[name],
                    decile,
                    n_deciles=n_deciles,
                    svd_backend=svd_backend,
                    svd_driver=svd_driver,
                )
                parameters[name].copy_(modified)
                information.append(info)
        yield information
    finally:
        with torch.no_grad():
            for name, snapshot in snapshots.items():
                parameters[name].copy_(snapshot)


@contextmanager
def spectral_lesion(
    model: nn.Module,
    parameter_names: Iterable[str],
    tranche: str,
    *,
    fraction: float = 0.05,
    mode: str = "count",
    reference_energy: float | None = None,
    seed: int = 0,
    svd_backend: str = "auto",
    svd_driver: str = "gesvdj",
    factor_cache: dict[str, tuple[Tensor, Tensor, Tensor]] | None = None,
) -> Iterator[list[LesionInfo]]:
    """Temporarily lesion named matrices and restore exact bytes on exit."""

    names = list(dict.fromkeys(str(name) for name in parameter_names))
    if not names:
        raise ValueError("parameter_names must be nonempty")
    parameters = dict(model.named_parameters())
    missing = [name for name in names if name not in parameters]
    if missing:
        raise KeyError(f"unknown model parameters: {missing}")
    if any(parameters[name].ndim != 2 for name in names):
        raise ValueError("all lesioned parameters must be matrices")
    snapshots = {name: parameters[name].detach().clone() for name in names}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    information: list[LesionInfo] = []
    try:
        with torch.no_grad():
            for name in names:
                factors = None if factor_cache is None else factor_cache.get(name)
                if factors is None and factor_cache is not None:
                    computed = _svd_components(
                        parameters[name], backend=svd_backend, driver=svd_driver
                    )
                    factors = tuple(value.detach().cpu() for value in computed)
                    factor_cache[name] = factors
                modified, info = lesion_matrix(
                    parameters[name],
                    tranche,
                    fraction=fraction,
                    mode=mode,
                    reference_energy=reference_energy,
                    generator=generator,
                    svd_backend=svd_backend,
                    svd_driver=svd_driver,
                    svd_factors=factors,
                )
                parameters[name].copy_(modified)
                information.append(info)
        yield information
    finally:
        with torch.no_grad():
            for name, snapshot in snapshots.items():
                parameters[name].copy_(snapshot)


def independent_lesion_benchmark(
    model: nn.Module,
    parameter_names: Iterable[str],
    evaluate: Callable[[], float],
    *,
    tranches: Iterable[str] = ("top", "bulk", "bottom"),
    fraction: float = 0.05,
    mode: str = "count",
    reference_energy: float | None = None,
    seed: int = 0,
    svd_backend: str = "auto",
    svd_driver: str = "gesvdj",
) -> list[dict[str, object]]:
    """Evaluate each lesion from the same pristine model state."""

    names = list(parameter_names)
    baseline = float(evaluate())
    if not math.isfinite(baseline):
        raise ValueError("baseline evaluation must be finite")
    results: list[dict[str, object]] = []
    factor_cache: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
    for offset, tranche in enumerate(tranches):
        with spectral_lesion(
            model,
            names,
            tranche,
            fraction=fraction,
            mode=mode,
            reference_energy=reference_energy,
            seed=seed + offset,
            svd_backend=svd_backend,
            svd_driver=svd_driver,
            factor_cache=factor_cache,
        ) as information:
            value = float(evaluate())
        results.append(
            {
                "tranche": _canonical_tranche(tranche),
                "baseline": baseline,
                "lesioned": value,
                "delta": value - baseline,
                "matrices": [info.as_dict() for info in information],
            }
        )
    return results


def independent_decile_benchmark(
    model: nn.Module,
    parameter_names: Iterable[str],
    evaluate: Callable[[], float],
    *,
    n_deciles: int = 10,
    svd_backend: str = "auto",
    svd_driver: str = "gesvdj",
) -> list[dict[str, object]]:
    """Evaluate every descending reference decile from one pristine state."""

    names = list(parameter_names)
    baseline = float(evaluate())
    if not math.isfinite(baseline):
        raise ValueError("baseline evaluation must be finite")
    groups = int(n_deciles)
    if groups < 1:
        raise ValueError("n_deciles must be positive")
    results: list[dict[str, object]] = []
    for decile in range(groups):
        with spectral_decile_lesion(
            model,
            names,
            decile,
            n_deciles=groups,
            svd_backend=svd_backend,
            svd_driver=svd_driver,
        ) as information:
            value = float(evaluate())
        results.append(
            {
                "decile": decile,
                "baseline": baseline,
                "lesioned": value,
                "delta": value - baseline,
                "matrices": [info.as_dict() for info in information],
            }
        )
    return results


def spectral_parameter_names(model: nn.Module) -> list[str]:
    """Return attention and MLP projection matrices in deterministic order."""

    suffixes = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    )
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.ndim == 2 and name.endswith(suffixes)
    ]


__all__ = [
    "LesionInfo",
    "independent_lesion_benchmark",
    "independent_decile_benchmark",
    "lesion_matrix",
    "lesion_matrix_decile",
    "spectral_decile_lesion",
    "spectral_lesion",
    "spectral_parameter_names",
]
