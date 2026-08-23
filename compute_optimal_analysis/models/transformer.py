"""Configurable decoder-only causal language model."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .modules import DecoderBlock, RMSNorm, make_norm


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 512
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int | None = None
    max_seq_len: int = 512
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    bias: bool = False
    norm_type: str = "rmsnorm"
    norm_epsilon: float = 1e-6
    mlp_type: str = "swiglu"
    position_embedding: str = "rope"
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    initializer_std: float = 0.02

    def __post_init__(self) -> None:
        integer_values = (self.vocab_size, self.d_model, self.n_layers, self.n_heads, self.max_seq_len)
        if any(int(value) != value or value < 1 for value in integer_values):
            raise ValueError("integer configuration fields must be positive")
        kv_heads = self.n_heads if self.n_kv_heads is None else int(self.n_kv_heads)
        if kv_heads < 1:
            raise ValueError("n_kv_heads must be positive")
        if self.d_model % self.n_heads != 0 or self.n_heads % kv_heads != 0:
            raise ValueError("d_model must divide by n_heads and n_heads by n_kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0 and self.position_embedding.lower() == "rope":
            raise ValueError("RoPE requires an even head dimension")
        if self.mlp_ratio <= 0.0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("mlp_ratio must be positive and dropout must lie in [0, 1)")
        if self.norm_type.lower() not in {"rmsnorm", "layernorm"}:
            raise ValueError("norm_type must be rmsnorm or layernorm")
        if self.mlp_type.lower() not in {"swiglu", "gelu"}:
            raise ValueError("mlp_type must be swiglu or gelu")
        if self.position_embedding.lower() not in {"rope", "learned"}:
            raise ValueError("position_embedding must be rope or learned")
        if self.norm_epsilon <= 0.0 or self.rope_theta <= 1.0 or self.initializer_std <= 0.0:
            raise ValueError("epsilon, RoPE theta, and initializer scale are invalid")

    @property
    def resolved_kv_heads(self) -> int:
        return self.n_heads if self.n_kv_heads is None else int(self.n_kv_heads)

    @property
    def d_hidden(self) -> int:
        return max(1, int(round(self.mlp_ratio * self.d_model)))


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None


class CausalTransformer(nn.Module):
    """Pre-norm decoder-only transformer with causal next-token loss."""

    spectral_suffixes = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    )

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        if config.position_embedding.lower() == "learned":
            self.position_embedding: nn.Embedding | None = nn.Embedding(config.max_seq_len, config.d_model)
            rope_theta: float | None = None
        else:
            self.position_embedding = None
            rope_theta = config.rope_theta
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    config.d_model,
                    config.n_heads,
                    config.resolved_kv_heads,
                    config.d_hidden,
                    norm_type=config.norm_type,
                    norm_epsilon=config.norm_epsilon,
                    mlp_type=config.mlp_type,
                    dropout=config.dropout,
                    bias=config.bias,
                    rope_theta=rope_theta,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = make_norm(config.norm_type, config.d_model, config.norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_module)
        residual_scale = config.initializer_std / math.sqrt(2.0 * config.n_layers)
        for name, parameter in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_scale)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_std)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            if isinstance(module, nn.LayerNorm) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        *,
        return_dict: bool = True,
    ) -> CausalLMOutput | tuple[Tensor] | tuple[Tensor, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        batch, sequence = input_ids.shape
        if sequence < 1 or sequence > self.config.max_seq_len:
            raise ValueError("sequence length is outside the configured range")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must use an integer tensor dtype")
        positions = torch.arange(sequence, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            hidden_states = hidden_states + self.position_embedding(positions)[None, :, :]
        hidden_states = self.embedding_dropout(hidden_states)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=positions,
            )
        logits = self.lm_head(self.final_norm(hidden_states))
        loss: Tensor | None = None
        if labels is not None:
            if sequence < 2:
                raise ValueError("labeled causal-LM batches require at least two tokens")
            if labels.shape != (batch, sequence):
                raise ValueError("labels must have the same shape as input_ids")
            shifted_labels = labels[:, 1:].contiguous()
            if attention_mask is not None:
                shifted_labels = shifted_labels.clone()
                shifted_labels[~attention_mask[:, 1:].to(dtype=torch.bool)] = -100
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                shifted_labels.view(-1),
                ignore_index=-100,
            )
        if return_dict:
            return CausalLMOutput(logits=logits, loss=loss)
        return (logits,) if loss is None else (loss, logits)

    def num_parameters(self, *, exclude_embeddings: bool = False) -> int:
        if not exclude_embeddings:
            return sum(parameter.numel() for parameter in self.parameters())
        embedding_ids = {id(self.token_embedding.weight)}
        if self.position_embedding is not None:
            embedding_ids.add(id(self.position_embedding.weight))
        return sum(parameter.numel() for parameter in self.parameters() if id(parameter) not in embedding_ids)

    def iter_spectral_weights(self) -> Iterator[tuple[str, Tensor]]:
        for name, parameter in self.named_parameters():
            if parameter.ndim == 2 and name.endswith(self.spectral_suffixes):
                yield name, parameter


__all__ = ["CausalLMOutput", "CausalTransformer", "TransformerConfig"]
