"""Transformer and compute-allocation components for Spectral-Chinchilla."""

from .chinchilla_scaling import (
    Allocation,
    ScalingLaw,
    allocation_for_regime,
    compute_optimal_allocation,
    isoflop_grid,
)
from .transformer import CausalLMOutput, CausalTransformer, TransformerConfig

__all__ = [
    "Allocation",
    "CausalLMOutput",
    "CausalTransformer",
    "ScalingLaw",
    "TransformerConfig",
    "allocation_for_regime",
    "compute_optimal_allocation",
    "isoflop_grid",
]

