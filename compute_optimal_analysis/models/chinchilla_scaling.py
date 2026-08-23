"""Compute-optimal allocation and fixed-compute regime construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections.abc import Iterable, Mapping
import math
from typing import Any


@dataclass(frozen=True)
class ScalingLaw:
    """Parameters of ``L=E+A/N^alpha+B/D^beta`` and ``C=c_f*N*D``."""

    irreducible_loss: float = 1.69
    model_coefficient: float = 406.4
    data_coefficient: float = 410.7
    model_exponent: float = 0.34
    data_exponent: float = 0.28
    flops_per_parameter_token: float = 6.0

    def __post_init__(self) -> None:
        values = (
            self.model_coefficient,
            self.data_coefficient,
            self.model_exponent,
            self.data_exponent,
            self.flops_per_parameter_token,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("scaling coefficients, exponents, and FLOP factor must be positive")
        if not math.isfinite(self.irreducible_loss):
            raise ValueError("irreducible_loss must be finite")

    def loss(self, parameters: float, tokens: float) -> float:
        if parameters <= 0.0 or tokens <= 0.0:
            raise ValueError("parameters and tokens must be positive")
        return float(
            self.irreducible_loss
            + self.model_coefficient / parameters**self.model_exponent
            + self.data_coefficient / tokens**self.data_exponent
        )


@dataclass(frozen=True)
class Allocation:
    compute_budget: float
    parameters: float
    tokens: float
    kappa: float
    regime: str
    predicted_loss: float
    flops_per_parameter_token: float = 6.0
    target_tokens_per_parameter: float | None = None

    def __post_init__(self) -> None:
        numeric = (
            self.compute_budget,
            self.parameters,
            self.tokens,
            self.kappa,
            self.flops_per_parameter_token,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("allocation values must be finite and positive")
        if not math.isfinite(self.predicted_loss):
            raise ValueError("predicted_loss must be finite")

    @property
    def realized_compute(self) -> float:
        return float(self.flops_per_parameter_token * self.parameters * self.tokens)

    @property
    def tokens_per_parameter(self) -> float:
        return float(self.tokens / self.parameters)

    def rounded(self) -> "Allocation":
        """Round counts while recomputing the represented compute value."""

        parameters = max(1, int(round(self.parameters)))
        tokens = max(1, int(round(self.tokens)))
        return replace(
            self,
            compute_budget=float(self.flops_per_parameter_token * parameters * tokens),
            parameters=float(parameters),
            tokens=float(tokens),
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["realized_compute"] = self.realized_compute
        result["tokens_per_parameter"] = self.tokens_per_parameter
        return result


def compute_optimal_allocation(
    compute_budget: float,
    law: ScalingLaw = ScalingLaw(),
    *,
    target_tokens_per_parameter: float | None = None,
) -> Allocation:
    """Return the analytic optimum or an exact empirical token-ratio allocation."""

    compute_budget = float(compute_budget)
    if not math.isfinite(compute_budget) or compute_budget <= 0.0:
        raise ValueError("compute_budget must be finite and positive")
    product = compute_budget / law.flops_per_parameter_token
    if target_tokens_per_parameter is not None:
        ratio = float(target_tokens_per_parameter)
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("target_tokens_per_parameter must be finite and positive")
        parameters = math.sqrt(product / ratio)
        tokens = ratio * parameters
    else:
        alpha, beta = law.model_exponent, law.data_exponent
        prefactor = (
            alpha * law.model_coefficient / (beta * law.data_coefficient)
        ) ** (1.0 / (alpha + beta))
        parameters = prefactor * product ** (beta / (alpha + beta))
        tokens = product / parameters
        ratio = None
    return Allocation(
        compute_budget=compute_budget,
        parameters=float(parameters),
        tokens=float(tokens),
        kappa=1.0,
        regime="compute_optimal",
        predicted_loss=law.loss(parameters, tokens),
        flops_per_parameter_token=law.flops_per_parameter_token,
        target_tokens_per_parameter=ratio,
    )


def allocation_for_regime(
    optimal: Allocation,
    kappa: float,
    *,
    name: str | None = None,
    law: ScalingLaw = ScalingLaw(),
) -> Allocation:
    """Move along a fixed-compute surface by scaling data and model inversely."""

    kappa = float(kappa)
    if not math.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be finite and positive")
    parameters = optimal.parameters / kappa
    tokens = optimal.tokens * kappa
    if name is None:
        if math.isclose(kappa, 1.0, rel_tol=0.0, abs_tol=1e-12):
            name = "compute_optimal"
        elif kappa < 1.0:
            name = "undertrained"
        else:
            name = "overtrained"
    return Allocation(
        compute_budget=optimal.compute_budget,
        parameters=parameters,
        tokens=tokens,
        kappa=kappa,
        regime=str(name),
        predicted_loss=law.loss(parameters, tokens),
        flops_per_parameter_token=optimal.flops_per_parameter_token,
        target_tokens_per_parameter=optimal.target_tokens_per_parameter,
    )


def isoflop_grid(
    compute_budgets: Iterable[float],
    kappas: Iterable[float] = (0.25, 1.0, 4.0),
    law: ScalingLaw = ScalingLaw(),
    *,
    target_tokens_per_parameter: float | None = 20.0,
) -> list[Allocation]:
    """Return a deterministic budget-major grid of fixed-compute allocations."""

    budgets = [float(value) for value in compute_budgets]
    multipliers = [float(value) for value in kappas]
    if not budgets or not multipliers:
        raise ValueError("compute_budgets and kappas must be nonempty")
    grid: list[Allocation] = []
    for budget in budgets:
        optimum = compute_optimal_allocation(
            budget,
            law,
            target_tokens_per_parameter=target_tokens_per_parameter,
        )
        grid.extend(allocation_for_regime(optimum, kappa, law=law) for kappa in multipliers)
    return grid


def estimate_transformer_parameters(config: Mapping[str, Any] | object) -> int:
    """Estimate trainable parameters from a transformer config without importing torch."""

    def get(name: str, default: Any = None) -> Any:
        if isinstance(config, Mapping):
            return config.get(name, default)
        return getattr(config, name, default)

    vocab = int(get("vocab_size"))
    width = int(get("d_model"))
    layers = int(get("n_layers"))
    heads = int(get("n_heads"))
    kv_heads = int(get("n_kv_heads", heads) or heads)
    ratio = float(get("mlp_ratio", 4.0))
    mlp_type = str(get("mlp_type", "swiglu")).lower()
    learned_positions = str(get("position_embedding", "rope")).lower() == "learned"
    max_seq_len = int(get("max_seq_len", 0))
    tied = bool(get("tie_embeddings", True))
    bias = bool(get("bias", False))
    if min(vocab, width, layers, heads, kv_heads) < 1 or width % heads != 0:
        raise ValueError("config dimensions are inconsistent")
    head_dim = width // heads
    kv_width = kv_heads * head_dim
    attention = width * width * 2 + width * kv_width * 2
    hidden = int(round(ratio * width))
    mlp = width * hidden * (3 if mlp_type == "swiglu" else 2)
    norm = 2 * width
    per_layer = attention + mlp + norm
    if bias:
        per_layer += width + 2 * kv_width + width
        per_layer += (2 * hidden + width) if mlp_type == "swiglu" else (hidden + width)
    embeddings = vocab * width
    output = 0 if tied else vocab * width
    positions = max_seq_len * width if learned_positions else 0
    final_norm = width
    return int(embeddings + output + positions + layers * per_layer + final_norm)


def suggest_architecture(
    target_parameters: float,
    *,
    vocab_size: int = 512,
    max_layers: int = 24,
    width_multiple: int = 64,
) -> dict[str, Any]:
    """Search a compact decoder configuration nearest a target parameter count."""

    target = float(target_parameters)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("target_parameters must be finite and positive")
    if vocab_size < 2 or max_layers < 1 or width_multiple < 8:
        raise ValueError("architecture search bounds are invalid")
    best: tuple[float, dict[str, Any]] | None = None
    for layers in range(2, max_layers + 1, 2):
        approximate_width = math.sqrt(target / max(1.0, 16.0 * layers))
        center = max(width_multiple, int(round(approximate_width / width_multiple)) * width_multiple)
        for width in range(max(width_multiple, center - 4 * width_multiple), center + 5 * width_multiple, width_multiple):
            head_options = [value for value in (4, 8, 12, 16, 24, 32) if value <= width and width % value == 0]
            if not head_options:
                continue
            heads = min(head_options, key=lambda value: abs(width // value - 64))
            candidate = {
                "vocab_size": int(vocab_size),
                "d_model": int(width),
                "n_layers": int(layers),
                "n_heads": int(heads),
                "n_kv_heads": int(heads),
                "mlp_ratio": 4.0,
                "mlp_type": "swiglu",
                "position_embedding": "rope",
                "max_seq_len": 2048,
                "tie_embeddings": True,
                "bias": False,
            }
            realized = estimate_transformer_parameters(candidate)
            error = abs(realized - target) / target
            if best is None or error < best[0]:
                best = (error, {**candidate, "estimated_parameters": realized, "relative_error": error})
    if best is None:
        raise ValueError("no valid architecture found for the requested bounds")
    return best[1]


__all__ = [
    "Allocation",
    "ScalingLaw",
    "allocation_for_regime",
    "compute_optimal_allocation",
    "estimate_transformer_parameters",
    "isoflop_grid",
    "suggest_architecture",
]
