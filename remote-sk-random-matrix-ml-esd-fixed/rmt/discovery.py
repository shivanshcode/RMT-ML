"""rmt.discovery — model-agnostic discovery of analyzable 2-D weight matrices.

Walks ``model.named_modules()``, classifies Linear / Conv1D weights into short
roles (Q/K/V/O/G/U/D…), skips embeddings and the LM head, and splits fused QKV
projections.  Registry covers llama, gpt_neox/pythia, qwen2/qwen3, bert, gpt2,
plus a never-None generic fallback.

QKV LAYOUT (important).  Two conventions exist for a fused QKV weight of shape
(3·d, d_in):

  * **contiguous**  rows = [all_Q | all_K | all_V].  HF GPT-2 ``attn.c_attn``.
  * **head-interleaved**  rows = [h0_Q, h0_K, h0_V, h1_Q, h1_K, h1_V, …].
    HF GPT-NeoX / Pythia ``attention.query_key_value``.  Its forward path does
    ``view(..., num_heads, 3*head_size)`` then ``chunk(3, dim=-1)``, i.e. the
    q/k/v split happens *inside* each head.

Taking contiguous thirds of a head-interleaved matrix yields the full q+k+v of
the first num_heads/3 heads, not Q — which corrupts every downstream Q/K/V
metric.  ``ModelSpec.qkv_interleaved`` selects the convention.

torch is imported lazily inside functions so importing this module is cheap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# short→substring patterns used by the generic fallback
MATRIX_PATTERNS = {
    "Q": ["q_proj", "query"],
    "K": ["k_proj", "key"],
    "V": ["v_proj", "value"],
    "O": ["o_proj", "out_proj", "dense", "proj"],
    "G": ["gate_proj", "gate"],
    "U": ["up_proj", "fc_in", "dense_h_to_4h", "fc1"],
    "D": ["down_proj", "fc_out", "dense_4h_to_h", "fc2"],
    "QKV": ["query_key_value", "qkv", "Wqkv", "c_attn"],
}

# module-name substrings that are always contiguous [Q|K|V], regardless of the
# spec default (GPT-2 style fused attention).
_CONTIGUOUS_QKV_NAMES = ("c_attn",)


@dataclass
class MatrixRecord:
    name: str
    short: str
    layer_idx: int
    weight: np.ndarray
    n: int
    m: int


@dataclass
class ModelSpec:
    name: str
    patterns: dict
    fused_qkv_substrings: List[str] = field(default_factory=list)
    layer_index_regexes: List[str] = field(default_factory=lambda: [r"\.layers?\.(\d+)\."])
    fused_qkv_order: Tuple[str, str, str] = ("Q", "K", "V")
    # True  → rows are [h0_Q, h0_K, h0_V, h1_Q, …] (GPT-NeoX)
    # False → rows are [all_Q | all_K | all_V]      (GPT-2 c_attn)
    qkv_interleaved: bool = False


_LLAMA = ModelSpec(
    name="llama",
    patterns={
        "Q": ["q_proj"], "K": ["k_proj"], "V": ["v_proj"], "O": ["o_proj"],
        "G": ["gate_proj"], "U": ["up_proj"], "D": ["down_proj"],
    },
    fused_qkv_substrings=[],
    layer_index_regexes=[r"(?:^|\.)layers\.(\d+)(?:\.|$)"],
)

_PYTHIA = ModelSpec(
    name="gpt_neox",
    patterns={
        "QKV": ["query_key_value"], "O": ["attention.dense"],
        "U": ["dense_h_to_4h"], "D": ["dense_4h_to_h"],
    },
    fused_qkv_substrings=["query_key_value"],
    layer_index_regexes=[r"(?:^|\.)layers\.(\d+)(?:\.|$)"],
    qkv_interleaved=True,
)

_QWEN = ModelSpec(
    name="qwen2",
    patterns={
        "Q": ["q_proj"], "K": ["k_proj"], "V": ["v_proj"], "O": ["o_proj"],
        "G": ["gate_proj"], "U": ["up_proj"], "D": ["down_proj"],
    },
    layer_index_regexes=[r"(?:^|\.)layers\.(\d+)(?:\.|$)"],
)

_BERT = ModelSpec(
    name="bert",
    patterns={
        "Q": ["attention.self.query"], "K": ["attention.self.key"],
        "V": ["attention.self.value"], "O": ["attention.output.dense"],
        "U": ["intermediate.dense"], "D": ["output.dense"],
    },
    layer_index_regexes=[r"(?:^|\.)layer\.(\d+)(?:\.|$)"],
)

_GPT2 = ModelSpec(
    name="gpt2",
    patterns={
        "QKV": ["attn.c_attn"], "O": ["attn.c_proj"],
        "U": ["mlp.c_fc"], "D": ["mlp.c_proj"],
    },
    fused_qkv_substrings=["c_attn"],
    layer_index_regexes=[r"(?:^|\.)h\.(\d+)(?:\.|$)"],
    qkv_interleaved=False,
)

_GENERIC = ModelSpec(
    name="generic",
    patterns=MATRIX_PATTERNS,
    fused_qkv_substrings=["query_key_value", "qkv", "Wqkv", "c_attn"],
    layer_index_regexes=[r"(?:^|\.)layers?\.(\d+)(?:\.|$)",
                         r"(?:^|\.)layer\.(\d+)(?:\.|$)",
                         r"(?:^|\.)h\.(\d+)(?:\.|$)",
                         r"(?:^|\.)(\d+)(?:\.|$)"],
    qkv_interleaved=True,
)

_REGISTRY = {
    "llama": _LLAMA, "mistral": _LLAMA, "mixtral": _LLAMA,
    "gpt_neox": _PYTHIA, "pythia": _PYTHIA, "gptneox": _PYTHIA,
    "qwen2": _QWEN, "qwen3": _QWEN, "qwen": _QWEN,
    "bert": _BERT, "roberta": _BERT,
    "gpt2": _GPT2,
}


def get_model_spec(model_or_config_or_name) -> ModelSpec:
    """Resolve a ModelSpec from a model, a config, or a name string. Never None."""
    mt = None
    obj = model_or_config_or_name
    if isinstance(obj, str):
        mt = obj.lower()
    else:
        cfg = getattr(obj, "config", obj)
        mt = (getattr(cfg, "model_type", None) or "").lower()
        if not mt:
            archs = getattr(cfg, "architectures", None) or []
            if archs:
                mt = str(archs[0]).lower()
    mt = mt or ""
    for key, spec in _REGISTRY.items():
        if key in mt:
            return spec
    return _GENERIC


def get_num_heads(model_or_config) -> Optional[int]:
    """Number of attention heads from a model or config, or None."""
    cfg = getattr(model_or_config, "config", model_or_config)
    for attr in ("num_attention_heads", "n_head", "num_heads", "n_heads"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return None


def classify(name: str, spec: ModelSpec) -> Optional[str]:
    """Return the short role for a module name, or None if not analyzable."""
    # fused QKV first (so 'query_key_value' isn't caught by Q/K/V substrings)
    for sub in spec.fused_qkv_substrings:
        if sub in name:
            return "QKV"
    # order matters: most-specific keys first
    # Prefer context-specific attention output patterns over BERT's broad
    # MLP ``output.dense`` pattern.
    order = ["QKV", "Q", "K", "V", "G", "U", "O", "D"]
    keys = [k for k in order if k in spec.patterns] + \
           [k for k in spec.patterns if k not in order]
    matches = [
        (len(pat), -keys.index(short), short)
        for short in keys
        for pat in spec.patterns[short]
        if pat in name
    ]
    return max(matches)[2] if matches else None


def extract_layer_index(name: str, spec: ModelSpec) -> int:
    """First integer matched by the spec's layer regexes, else −1."""
    for rx in spec.layer_index_regexes:
        m = re.search(rx, name)
        if m:
            return int(m.group(1))
    return -1


def _weight_2d(module, dtype="float64"):
    """Return a 2-D numpy weight (out, in). Conv1D (gpt2) is transposed.

    REPORT §0: materialise at float64 (not ``.float()`` → float32), so a weight
    value carrying more than 23 mantissa bits is not silently downcast before the
    SVD ever runs. ``dtype="float32"`` is still accepted for callers that
    explicitly want the smaller copy, but the default is float64.
    """
    import torch
    w = getattr(module, "weight", None)
    if w is None or w.dim() != 2:
        return None
    arr = w.detach().to("cpu").to(torch.float64).numpy()
    # HF Conv1D stores weight as (in, out); detect by class name and transpose.
    cls = type(module).__name__
    if cls == "Conv1D":
        arr = arr.T
    return np.asarray(arr, dtype=np.float32 if dtype == "float32" else np.float64)


# --------------------------------------------------------------------------- #
# fused QKV                                                                    #
# --------------------------------------------------------------------------- #
def qkv_is_interleaved(name: str, spec: ModelSpec) -> bool:
    """Whether this fused-QKV module uses the head-interleaved row layout."""
    if any(sub in name for sub in _CONTIGUOUS_QKV_NAMES):
        return False
    return bool(spec.qkv_interleaved)


def _qkv_view(mat, num_heads):
    """(3·d, d_in) → (num_heads, 3, head_dim, d_in). A view when mat is C-contiguous."""
    n_rows, d_in = mat.shape
    d = n_rows // 3
    if num_heads <= 0 or d % num_heads:
        raise ValueError(
            f"fused QKV of {n_rows} rows is not divisible into {num_heads} heads")
    head_dim = d // num_heads
    return mat.reshape(num_heads, 3, head_dim, d_in)


def extract_qkv_block(mat, idx: int, *, num_heads=None, interleaved=True):
    """Return the Q/K/V block (idx = 0/1/2) of a fused weight as a (d, d_in) array."""
    mat = np.asarray(mat)
    d = mat.shape[0] // 3
    if interleaved:
        if num_heads is None:
            raise ValueError("num_heads is required for head-interleaved QKV")
        return np.ascontiguousarray(_qkv_view(mat, num_heads)[:, idx].reshape(d, -1))
    return mat[idx * d:(idx + 1) * d, :]


def assign_qkv_block(mat, idx: int, block, *, num_heads=None, interleaved=True) -> None:
    """Write ``block`` back into the idx-th Q/K/V slot of ``mat``, in place."""
    mat = np.asarray(mat)
    d = mat.shape[0] // 3
    block = np.asarray(block)
    if interleaved:
        if num_heads is None:
            raise ValueError("num_heads is required for head-interleaved QKV")
        view = _qkv_view(mat, num_heads)
        view[:, idx] = block.reshape(num_heads, d // num_heads, -1)
    else:
        mat[idx * d:(idx + 1) * d, :] = block


def split_fused_qkv(weight, name, layer_idx, spec, *, num_heads=None,
                    interleaved=None) -> List[MatrixRecord]:
    """Split a fused QKV weight (rows = 3·d) into Q/K/V records.

    ``interleaved`` defaults to ``qkv_is_interleaved(name, spec)``.  If an
    interleaved layout is indicated, ``num_heads`` is mandatory; unknown layouts
    are rejected rather than silently relabeled.
    """
    W = np.asarray(weight)
    n_rows = W.shape[0]
    assert n_rows % 3 == 0, f"fused QKV rows {n_rows} not divisible by 3 ({name})"

    if interleaved is None:
        interleaved = qkv_is_interleaved(name, spec)
    if interleaved and num_heads is None:
        raise ValueError(
            f"{name}: num_heads is required to split a head-interleaved fused QKV; "
            "refusing to guess an incompatible layout"
        )

    recs = []
    for i, tag in enumerate(spec.fused_qkv_order):
        block = extract_qkv_block(W, i, num_heads=num_heads, interleaved=interleaved)
        recs.append(MatrixRecord(
            name=f"{name}[{tag}]", short=tag, layer_idx=layer_idx,
            weight=block, n=block.shape[0], m=block.shape[1]))
    return recs


_SKIP_SUBSTRINGS = ("embed", "lm_head", "embed_out", "embed_in", "embed_tokens",
                    "wte", "wpe", "shared", "rotary", "norm", "ln_", "layernorm")


def discover_weight_matrices(model, layer_indices=None, *, spec=None,
                             dtype="float64", num_heads=None) -> List[MatrixRecord]:
    """Walk named_modules(), classify 2-D Linears/Conv1D, skip embeddings/head,
    split fused QKV. Optionally filter to a set of layer indices."""
    if spec is None:
        spec = get_model_spec(model)
    if num_heads is None:
        num_heads = get_num_heads(model)
    want = set(layer_indices) if layer_indices is not None else None
    records: List[MatrixRecord] = []
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        lname = name.lower()
        if any(sk in lname for sk in _SKIP_SUBSTRINGS):
            continue
        short = classify(name, spec)
        if short is None:
            continue
        W = _weight_2d(module, dtype=dtype)
        if W is None:
            continue
        layer_idx = extract_layer_index(name, spec)
        if want is not None and layer_idx not in want:
            continue
        wname = name + ".weight"
        if short == "QKV":
            records.extend(split_fused_qkv(W, wname, layer_idx, spec,
                                           num_heads=num_heads))
        else:
            records.append(MatrixRecord(name=wname, short=short, layer_idx=layer_idx,
                                        weight=W, n=W.shape[0], m=W.shape[1]))
    return records
