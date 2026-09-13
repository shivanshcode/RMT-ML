"""Decoder block components: normalization, RoPE, GQA/MHA, and MLPs."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        if dimension < 1 or epsilon <= 0.0:
            raise ValueError("dimension and epsilon must be positive")
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = float(epsilon)

    def forward(self, inputs: Tensor) -> Tensor:
        variance = inputs.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs.float() * torch.rsqrt(variance + self.epsilon)
        return (normalized * self.weight.float()).to(dtype=inputs.dtype)


def rotate_half(inputs: Tensor) -> Tensor:
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim < 2 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if not math.isfinite(theta) or theta <= 1.0:
            raise ValueError("theta must exceed one")
        frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", frequencies, persistent=False)

    def cos_sin(self, position_ids: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if position_ids.ndim == 1:
            frequencies = torch.einsum("t,d->td", position_ids.float(), self.inv_freq.float())
            embedding = torch.cat((frequencies, frequencies), dim=-1)[None, None, :, :]
        elif position_ids.ndim == 2:
            frequencies = torch.einsum("bt,d->btd", position_ids.float(), self.inv_freq.float())
            embedding = torch.cat((frequencies, frequencies), dim=-1)[:, None, :, :]
        else:
            raise ValueError("position_ids must be one- or two-dimensional")
        return embedding.cos().to(dtype=dtype), embedding.sin().to(dtype=dtype)

    def forward(self, query: Tensor, key: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        cosine, sine = self.cos_sin(position_ids.to(device=query.device), query.dtype)
        return query * cosine + rotate_half(query) * sine, key * cosine + rotate_half(key) * sine


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = False,
        rope_theta: float | None = 10000.0,
    ) -> None:
        super().__init__()
        if d_model < 1 or n_heads < 1 or n_kv_heads < 1:
            raise ValueError("attention dimensions must be positive")
        if d_model % n_heads != 0 or n_heads % n_kv_heads != 0:
            raise ValueError("d_model must divide by heads and query heads by KV heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = d_model // n_heads
        self.kv_width = self.n_kv_heads * self.head_dim
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, self.kv_width, bias=bias)
        self.v_proj = nn.Linear(d_model, self.kv_width, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)
        self.rope = None if rope_theta is None else RotaryEmbedding(self.head_dim, rope_theta)

    def _shape(self, tensor: Tensor, heads: int) -> Tensor:
        batch, sequence, _ = tensor.shape
        return tensor.view(batch, sequence, heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        batch, sequence, _ = hidden_states.shape
        query = self._shape(self.q_proj(hidden_states), self.n_heads)
        key = self._shape(self.k_proj(hidden_states), self.n_kv_heads)
        value = self._shape(self.v_proj(hidden_states), self.n_kv_heads)
        if self.rope is not None:
            if position_ids is None:
                position_ids = torch.arange(sequence, device=hidden_states.device)
            query, key = self.rope(query, key, position_ids)
        repeat = self.n_heads // self.n_kv_heads
        if repeat > 1:
            key = key.repeat_interleave(repeat, dim=1)
            value = value.repeat_interleave(repeat, dim=1)
        # Promote reduced-precision inputs before the dot product can overflow.
        score_dtype = (
            torch.float32 if query.dtype in {torch.float16, torch.bfloat16} else query.dtype
        )
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = torch.matmul(
                query.to(dtype=score_dtype), key.to(dtype=score_dtype).transpose(-2, -1)
            )
            scores = scores * (1.0 / math.sqrt(self.head_dim))
        allowed = ~torch.ones(sequence, sequence, dtype=torch.bool,
                              device=hidden_states.device).triu(1)
        allowed = allowed[None, None, :, :].expand(batch, 1, sequence, sequence)
        valid_queries = None
        if attention_mask is not None:
            if attention_mask.shape != (batch, sequence):
                raise ValueError("attention_mask must have shape (batch, sequence)")
            valid_keys = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
            valid_queries = valid_keys
            allowed = allowed & valid_keys[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1)
        # A fully masked row must be exactly zero, not softmax(uniform min).
        probabilities = probabilities * allowed.to(probabilities.dtype)
        denominator = probabilities.sum(dim=-1, keepdim=True)
        probabilities = torch.where(
            denominator > 0.0,
            probabilities / denominator.clamp_min(torch.finfo(probabilities.dtype).tiny),
            torch.zeros_like(probabilities),
        )
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        # Use the projection dtype at the output boundary.  Under autocast the
        # residual stream can stay FP32 while the following projection is BF16;
        # an FP32 context would make Torch 2.4 Inductor emit a mixed-dtype GEMM.
        with torch.autocast(device_type=query.device.type, enabled=False):
            context = torch.matmul(
                probabilities, value.to(dtype=score_dtype)
            ).to(dtype=query.dtype)
        context = context.transpose(1, 2).contiguous().view(batch, sequence, self.d_model)
        output = self.o_proj(context)
        if valid_queries is not None:
            output = output * valid_queries.unsqueeze(-1).to(output.dtype)
        return output


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, *, bias: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_hidden, bias=bias)
        self.up_proj = nn.Linear(d_model, d_hidden, bias=bias)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=bias)
        self.dropout = float(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(F.dropout(gated, p=self.dropout, training=self.training))


class GeLUMLP(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, *, bias: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_hidden, bias=bias)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=bias)
        self.dropout = float(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        activated = F.gelu(self.up_proj(hidden_states), approximate="tanh")
        return self.down_proj(F.dropout(activated, p=self.dropout, training=self.training))


def make_norm(kind: str, dimension: int, epsilon: float) -> nn.Module:
    name = str(kind).lower()
    if name == "rmsnorm":
        return RMSNorm(dimension, epsilon)
    if name == "layernorm":
        return nn.LayerNorm(dimension, eps=epsilon)
    raise ValueError("norm_type must be 'rmsnorm' or 'layernorm'")


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_hidden: int,
        *,
        norm_type: str,
        norm_epsilon: float,
        mlp_type: str,
        dropout: float,
        bias: bool,
        rope_theta: float | None,
    ) -> None:
        super().__init__()
        self.attention_norm = make_norm(norm_type, d_model, norm_epsilon)
        self.self_attn = CausalSelfAttention(
            d_model,
            n_heads,
            n_kv_heads,
            dropout=dropout,
            bias=bias,
            rope_theta=rope_theta,
        )
        self.mlp_norm = make_norm(norm_type, d_model, norm_epsilon)
        if str(mlp_type).lower() == "swiglu":
            self.mlp = SwiGLUMLP(d_model, d_hidden, bias=bias, dropout=dropout)
        elif str(mlp_type).lower() == "gelu":
            self.mlp = GeLUMLP(d_model, d_hidden, bias=bias, dropout=dropout)
        else:
            raise ValueError("mlp_type must be 'swiglu' or 'gelu'")
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        attention_output = self.self_attn(
            self.attention_norm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + self.residual_dropout(attention_output)
        return hidden_states + self.residual_dropout(self.mlp(self.mlp_norm(hidden_states)))


__all__ = [
    "CausalSelfAttention",
    "DecoderBlock",
    "GeLUMLP",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLUMLP",
    "make_norm",
    "rotate_half",
]

