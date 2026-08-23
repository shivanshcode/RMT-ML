import torch

from models.transformer import CausalTransformer, TransformerConfig


def tiny_model() -> CausalTransformer:
    torch.manual_seed(1234)
    return CausalTransformer(
        TransformerConfig(
            vocab_size=41,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=16,
            mlp_ratio=2.0,
            dropout=0.0,
            position_embedding="rope",
        )
    )


def test_transformer_forward_shape_and_loss() -> None:
    model = tiny_model()
    tokens = torch.randint(0, 41, (3, 10))
    output = model(tokens, labels=tokens)
    assert output.logits.shape == (3, 10, 41)
    assert output.loss is not None
    assert torch.isfinite(output.loss)


def test_causal_mask_blocks_future_token_changes() -> None:
    model = tiny_model().eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 4, 17, 19]])
    with torch.no_grad():
        first_logits = model(first).logits
        second_logits = model(second).logits
    assert torch.allclose(first_logits[:, :4], second_logits[:, :4], atol=1e-6, rtol=1e-6)


def test_learned_positions_layernorm_gelu_mha_variant() -> None:
    config = TransformerConfig(
        vocab_size=29,
        d_model=24,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        max_seq_len=12,
        norm_type="layernorm",
        mlp_type="gelu",
        position_embedding="learned",
    )
    output = CausalTransformer(config)(torch.randint(0, 29, (2, 7)))
    assert output.logits.shape == (2, 7, 29)

