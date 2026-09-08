"""Forward-hook activation moments and empirical covariance extraction."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from rmt.svd_result import SVDResult


@dataclass
class CovarianceAccumulator:
    """Accumulate first and second moments on one device without retaining batches."""

    dimension: int
    count: int = 0
    sum_vector: Tensor | None = None
    gram_matrix: Tensor | None = None
    device: str | torch.device | None = None
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        self.dimension = int(self.dimension)
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        target = torch.device("cpu" if self.device is None else self.device)
        if self.dtype not in {torch.float32, torch.float64}:
            raise ValueError("covariance accumulation dtype must be float32 or float64")
        if self.sum_vector is None:
            self.sum_vector = torch.zeros(self.dimension, dtype=self.dtype, device=target)
        if self.gram_matrix is None:
            self.gram_matrix = torch.zeros(
                self.dimension,
                self.dimension,
                dtype=self.dtype,
                device=target,
            )
        if self.sum_vector.device != self.gram_matrix.device:
            raise ValueError("moment buffers must share a device")
        self.device = self.sum_vector.device

    def update(self, activations: Tensor, *, valid_mask: Tensor | None = None,
               max_samples: int | None = None) -> None:
        if activations.ndim < 2 or activations.shape[-1] != self.dimension:
            raise ValueError("activation trailing dimension does not match accumulator")
        if self.sum_vector is None or self.gram_matrix is None:
            raise RuntimeError("moment buffers are unavailable")
        flattened = activations.detach().reshape(-1, self.dimension)
        if valid_mask is not None:
            mask = valid_mask.detach().reshape(-1).to(device=flattened.device, dtype=torch.bool)
            if mask.numel() != flattened.shape[0]:
                raise ValueError("valid-position mask does not match activation positions")
            flattened = flattened[mask]
        flattened = flattened.to(
            device=self.sum_vector.device,
            dtype=self.sum_vector.dtype,
            non_blocking=self.sum_vector.device.type == "cuda",
        )
        if max_samples is not None:
            limit = int(max_samples)
            if limit < 1:
                raise ValueError("max_samples must be positive")
            flattened = flattened[:limit]
        if flattened.shape[0] == 0:
            return
        self.count += int(flattened.shape[0])
        self.sum_vector.add_(flattened.sum(dim=0))
        self.gram_matrix.addmm_(flattened.T, flattened)

    def second_moment(self) -> Tensor:
        if self.count < 1:
            raise ValueError("no activations have been accumulated")
        if self.gram_matrix is None:
            raise RuntimeError("gram matrix is unavailable")
        return self.gram_matrix / self.count

    def covariance(self, *, centered: bool = True, unbiased: bool = False) -> Tensor:
        if self.count < 1:
            raise ValueError("no activations have been accumulated")
        if unbiased and self.count < 2:
            raise ValueError("unbiased covariance requires at least two observations")
        if self.gram_matrix is None or self.sum_vector is None:
            raise RuntimeError("moment buffers are unavailable")
        second = self.gram_matrix / self.count
        if centered:
            mean = self.sum_vector / self.count
            result = second - torch.outer(mean, mean)
            if unbiased:
                result = result * (self.count / (self.count - 1))
        else:
            result = second
        return 0.5 * (result + result.T)

    def clear(self) -> None:
        self.count = 0
        if self.sum_vector is None or self.gram_matrix is None:
            raise RuntimeError("moment buffers are unavailable")
        self.sum_vector.zero_()
        self.gram_matrix.zero_()


class ActivationExtractor:
    """Context-managed hooks for pre/post module activation covariances."""

    def __init__(
        self,
        model: nn.Module,
        module_filter: Callable[[str, nn.Module], bool] | Iterable[str] | None = None,
        *,
        capture: Sequence[str] = ("pre", "post"),
        max_samples_per_hook: int | None = None,
        accumulation_device: str | torch.device = "auto",
        accumulation_dtype: str = "auto",
    ) -> None:
        invalid = set(capture) - {"pre", "post"}
        if invalid or not capture:
            raise ValueError("capture must contain pre, post, or both")
        self.model = model
        self.module_filter = (
            module_filter
            if module_filter is None or callable(module_filter)
            else frozenset(str(name) for name in module_filter)
        )
        self.capture = tuple(capture)
        self.max_samples_per_hook = max_samples_per_hook
        if str(accumulation_device) not in {"auto", "cpu", "cuda"} and not isinstance(
            accumulation_device,
            torch.device,
        ):
            raise ValueError("accumulation_device must be auto, cpu, cuda, or a device")
        if accumulation_dtype not in {"auto", "float32", "float64"}:
            raise ValueError("accumulation_dtype must be auto, float32, or float64")
        self.accumulation_device = accumulation_device
        self.accumulation_dtype = accumulation_dtype
        self.accumulators: dict[str, CovarianceAccumulator] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.valid_position_mask: Tensor | None = None

    def _selected(self, name: str, module: nn.Module) -> bool:
        if self.module_filter is None:
            return isinstance(module, nn.Linear)
        if callable(self.module_filter):
            return bool(self.module_filter(name, module))
        return name in self.module_filter

    def _update(self, key: str, tensor: Tensor) -> None:
        dimension = int(tensor.shape[-1])
        accumulator = self.accumulators.get(key)
        if accumulator is None:
            target = (
                tensor.device
                if str(self.accumulation_device) == "auto"
                else torch.device(self.accumulation_device)
            )
            dtype = {
                "float32": torch.float32,
                "float64": torch.float64,
                "auto": torch.float32 if target.type == "cuda" else torch.float64,
            }[self.accumulation_dtype]
            accumulator = CovarianceAccumulator(
                dimension,
                device=target,
                dtype=dtype,
            )
            self.accumulators[key] = accumulator
        mask = self.valid_position_mask if tensor.ndim >= 3 else None
        accumulator.update(tensor, valid_mask=mask,
                           max_samples=self.max_samples_per_hook)

    def _make_hook(self, name: str) -> Callable[[nn.Module, Sequence[object], object], None]:
        def hook(module: nn.Module, inputs: Sequence[object], output: object) -> None:
            del module
            if "pre" in self.capture and inputs and isinstance(inputs[0], Tensor):
                self._update(f"{name}:pre", inputs[0])
            if "post" in self.capture:
                candidate = output[0] if isinstance(output, tuple) and output else output
                if isinstance(candidate, Tensor):
                    self._update(f"{name}:post", candidate)

        return hook

    def __enter__(self) -> "ActivationExtractor":
        if self.handles:
            raise RuntimeError("ActivationExtractor is already active")
        for name, module in self.model.named_modules():
            if name and self._selected(name, module):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
        if not self.handles:
            raise ValueError("module_filter selected no modules")
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        del exc_type, exc_value, traceback
        self.remove()
        return False

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self) -> None:
        for accumulator in self.accumulators.values():
            accumulator.clear()

    def covariances(
        self,
        *,
        centered: bool = True,
        unbiased: bool = False,
    ) -> dict[str, np.ndarray]:
        return {
            name: accumulator.covariance(
                centered=centered,
                unbiased=unbiased,
            ).detach().cpu().numpy()
            for name, accumulator in self.accumulators.items()
        }

    def second_moments(self) -> dict[str, np.ndarray]:
        return {
            name: accumulator.second_moment().detach().cpu().numpy()
            for name, accumulator in self.accumulators.items()
        }


def compute_tensor_svd(
    matrix: Tensor,
    *,
    backend: str = "auto",
    driver: str = "gesvdj",
    normalization: float | None = None,
    analysis_dtype: str = "float64",
) -> SVDResult:
    """Compute an accelerated reduced SVD and transfer only factors to the pure container."""

    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("matrix must be a nonempty two-dimensional tensor")
    if backend not in {"auto", "cpu", "cuda"}:
        raise ValueError("backend must be auto, cpu, or cuda")
    if driver not in {"default", "gesvdj", "gesvd", "gesvda"}:
        raise ValueError("unsupported SVD driver")
    if backend == "auto":
        target = matrix.device if matrix.device.type == "cuda" else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        target = torch.device(backend)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA SVD was requested but is unavailable")
    if analysis_dtype not in {"float32", "float64"}:
        raise ValueError("analysis_dtype must be float32 or float64")
    selected_dtype = torch.float64 if analysis_dtype == "float64" else torch.float32
    analysis = matrix.detach().to(device=target, dtype=selected_dtype)
    keyword_arguments: dict[str, object] = {"full_matrices": False}
    if target.type == "cuda" and driver != "default":
        keyword_arguments["driver"] = driver
    with torch.no_grad():
        U, singular_values, Vh = torch.linalg.svd(analysis, **keyword_arguments)
    return SVDResult(
        U=U.cpu().numpy(),
        s=singular_values.cpu().numpy(),
        Vh=Vh.cpu().numpy(),
        n=int(matrix.shape[0]),
        m=int(matrix.shape[1]),
        normalization=normalization,
    )


def compute_activation_covariances(
    model: nn.Module,
    dataloader: Iterable[dict[str, Tensor]],
    *,
    device: str | torch.device,
    module_filter: Callable[[str, nn.Module], bool] | Iterable[str] | None = None,
    max_batches: int | None = None,
    centered: bool = True,
    accumulation_device: str | torch.device = "auto",
    accumulation_dtype: str = "auto",
    amp_dtype: str = "float32",
) -> dict[str, np.ndarray]:
    """Run local batches through a model and return captured pre-activation covariances."""

    target = torch.device(device)
    if amp_dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("amp_dtype must be float32, float16, or bfloat16")
    selected_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[amp_dtype]
    use_amp = target.type == "cuda" and selected_dtype != torch.float32
    was_training = model.training
    model.eval()
    try:
        with ActivationExtractor(
            model,
            module_filter,
            capture=("pre",),
            accumulation_device=accumulation_device,
            accumulation_dtype=accumulation_dtype,
        ) as extractor:
            with torch.no_grad():
                for batch_index, batch in enumerate(dataloader):
                    if max_batches is not None and batch_index >= int(max_batches):
                        break
                    inputs = {
                        key: value.to(target, non_blocking=target.type == "cuda")
                        for key, value in batch.items()
                        if key in {"input_ids", "attention_mask"}
                    }
                    context = (
                        torch.autocast(device_type="cuda", dtype=selected_dtype)
                        if use_amp
                        else nullcontext()
                    )
                    extractor.valid_position_mask = inputs.get("attention_mask")
                    with context:
                        model(**inputs)
                    extractor.valid_position_mask = None
    finally:
        model.train(was_training)
    return extractor.covariances(centered=centered)


__all__ = [
    "ActivationExtractor",
    "CovarianceAccumulator",
    "compute_activation_covariances",
    "compute_tensor_svd",
]
