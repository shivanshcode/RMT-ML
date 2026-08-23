import numpy as np
import torch
from torch import nn

from pipelines.activation_extractor import ActivationExtractor, compute_tensor_svd


def test_activation_covariance_shape_symmetry_and_psd() -> None:
    torch.manual_seed(1234)
    model = nn.Sequential(nn.Linear(7, 11, bias=False), nn.GELU(), nn.Linear(11, 5, bias=False))
    inputs = torch.randn(4, 6, 7)
    with ActivationExtractor(model, module_filter=("0",), capture=("pre",)) as extractor:
        model(inputs)
    covariance = extractor.covariances(centered=True)["0:pre"]
    assert covariance.shape == (7, 7)
    assert np.allclose(covariance, covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10


def test_uncentered_second_moment_matches_exact_xtx() -> None:
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    inputs = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]]])
    with ActivationExtractor(model, module_filter=("0",), capture=("pre",)) as extractor:
        model(inputs)
    observed = extractor.second_moments()["0:pre"]
    flattened = inputs.numpy().reshape(-1, 3)
    expected = flattened.T @ flattened / flattened.shape[0]
    assert np.allclose(observed, expected)


def test_accumulator_uses_requested_device_and_float32_buffers() -> None:
    model = nn.Sequential(nn.Linear(4, 3, bias=False))
    inputs = torch.randn(2, 5, 4)
    with ActivationExtractor(
        model,
        module_filter=("0",),
        capture=("pre",),
        accumulation_device="cpu",
        accumulation_dtype="float32",
    ) as extractor:
        model(inputs)
    accumulator = extractor.accumulators["0:pre"]
    assert accumulator.gram_matrix is not None
    assert accumulator.gram_matrix.device.type == "cpu"
    assert accumulator.gram_matrix.dtype == torch.float32


def test_tensor_svd_bridge_returns_pure_container() -> None:
    matrix = torch.randn(9, 6)
    result = compute_tensor_svd(matrix, backend="cpu", driver="gesvdj")
    assert result.s.shape == (6,)
    assert np.all(np.diff(result.s) <= 0.0)
    assert np.allclose(result.reconstruct(), matrix.numpy(), atol=2e-5)
