import pytest
import torch

from models.transformer import CausalTransformer, TransformerConfig
from pipelines.spectral_lesioning import (
    lesion_matrix,
    lesion_matrix_decile,
    spectral_decile_lesion,
    spectral_lesion,
    spectral_parameter_names,
)


def test_matrix_lesion_zeroes_requested_singular_count() -> None:
    torch.manual_seed(5)
    matrix = torch.randn(20, 16)
    modified, information = lesion_matrix(matrix, "bottom", fraction=0.25)
    singular_values = torch.linalg.svdvals(modified)
    assert information.indices == tuple(range(12, 16))
    assert int(torch.count_nonzero(singular_values < 1e-5)) >= 4


def test_lesion_context_restores_parameters_after_success_and_error() -> None:
    torch.manual_seed(8)
    model = CausalTransformer(
        TransformerConfig(vocab_size=31, d_model=16, n_layers=1, n_heads=4, max_seq_len=8)
    )
    name = spectral_parameter_names(model)[0]
    parameter = dict(model.named_parameters())[name]
    original = parameter.detach().clone()
    with spectral_lesion(model, (name,), "top", fraction=0.2):
        assert not torch.equal(parameter, original)
    assert torch.equal(parameter, original)
    with pytest.raises(RuntimeError):
        with spectral_lesion(model, (name,), "bottom", fraction=0.2):
            raise RuntimeError("evaluation failure")
    assert torch.equal(parameter, original)


def test_reference_decile_lesion_uses_descending_index_groups() -> None:
    matrix = torch.diag(torch.arange(10.0, 0.0, -1.0))
    largest_removed, largest_info = lesion_matrix_decile(matrix, 0)
    smallest_removed, smallest_info = lesion_matrix_decile(matrix, 9)
    assert largest_info.indices == (0,)
    assert smallest_info.indices == (9,)
    assert largest_info.removed_frobenius_energy > smallest_info.removed_frobenius_energy
    assert torch.linalg.matrix_rank(largest_removed) == 9
    assert torch.linalg.matrix_rank(smallest_removed) == 9


def test_reference_decile_context_restores_parameters() -> None:
    model = CausalTransformer(
        TransformerConfig(vocab_size=31, d_model=16, n_layers=1, n_heads=4, max_seq_len=8)
    )
    name = spectral_parameter_names(model)[0]
    parameter = dict(model.named_parameters())[name]
    original = parameter.detach().clone()
    with spectral_decile_lesion(model, (name,), 0):
        assert not torch.equal(parameter, original)
    assert torch.equal(parameter, original)
