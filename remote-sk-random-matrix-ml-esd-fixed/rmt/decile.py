"""rmt.decile — singular-value decile ablation (Paper 3 §5).

``set_layer_svd_decile`` reconstructs a weight with one ascending decile of its
singular values zeroed (decile 1 = smallest 10%), in place.  For fused QKV it
edits only the corresponding row block.  ``perplexity_vs_decile`` rebuilds a
fresh model for each decile and measures the perplexity impact.

torch imported lazily inside functions.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import tempfile
from pathlib import Path
import numpy as np

from .scalars import decile_index_ranges
from .discovery import (get_model_spec, discover_weight_metadata,
                        get_num_heads, qkv_is_interleaved, extract_qkv_block,
                        assign_qkv_block)


class _DiskTensorStore:
    """Temporary disk-backed tensor mapping used to bound host RAM."""
    def __init__(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="rmt-tensors-")
        self._paths = {}

    def _path(self, key):
        if key not in self._paths:
            self._paths[key] = Path(self._temporary.name) / f"{len(self._paths)}.pt"
        return self._paths[key]

    def __setitem__(self, key, value):
        import torch
        if isinstance(value, tuple):
            value = tuple(item.detach().cpu() if hasattr(item, "detach")
                          else np.asarray(item) for item in value)
        else:
            value = value.detach().cpu() if hasattr(value, "detach") else np.asarray(value)
        torch.save(value, self._path(key))

    def get(self, key, default=None):
        path = self._paths.get(key)
        if path is None or not path.is_file():
            return default
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)

    def close(self):
        self._temporary.cleanup()


def _reconstruct_zeroed(W, lo, hi, *, factors=None, backend="numpy", gpu_min_dim=1024):
    """Zero an ascending singular range, optionally reusing pristine factors."""
    if factors is None:
        from .linalg import cached_svd
        result = cached_svd(W, full_matrices=False, backend=backend,
                            gpu_min_dim=gpu_min_dim)
        U, s, Vh = result.U, result.s, result.Vh
    else:
        U, s, Vh = factors
    order = np.argsort(s)                       # ascending
    s_new = s.copy()
    zero_idx = order[lo:hi]
    s_new[zero_idx] = 0.0
    return (U * s_new) @ Vh


def set_layer_svd_decile(model, records, decile, *, n_deciles=10, spec=None,
                         factor_cache=None, backend="numpy", gpu_min_dim=1024) -> None:
    """Zero the ``decile``-th ascending SV decile of each record's matrix in place.

    ``decile`` is 1-based (1 = smallest 10%).  Raises ValueError if out of range.
    For a fused-QKV record (name like ``...query_key_value.weight[Q]``) only that
    block's rows are modified.
    """
    import torch
    if decile < 1 or decile > n_deciles:
        raise ValueError(f"decile {decile} out of range 1..{n_deciles}")
    if spec is None:
        spec = get_model_spec(model)

    for rec in records:
        name = rec.name
        is_fused = name.endswith("]") and "[" in name
        base = name[: name.index("[")] if is_fused else name
        modname = base[: -len(".weight")] if base.endswith(".weight") else base
        module = model.get_submodule(modname)
        w = module.weight                          # (out, in)
        cls = type(module).__name__
        with torch.no_grad():
            # Read at float64 so the reconstruction is not pre-floored (REPORT §0).
            full = w.detach().cpu().to(torch.float64).numpy()
            mat = full.T if cls == "Conv1D" else full
            if is_fused:
                tag = name[name.index("[") + 1: -1]
                order = list(spec.fused_qkv_order)
                idx = order.index(tag)
                # Q/K/V rows may be head-interleaved (GPT-NeoX); slicing
                # contiguous thirds would ablate the wrong rows.
                interleaved = qkv_is_interleaved(base, spec)
                nh = get_num_heads(model) if interleaved else None
                if interleaved and nh is None:
                    raise ValueError("num_heads is required for interleaved fused QKV")
                mat = np.ascontiguousarray(mat)
                block = extract_qkv_block(mat, idx, num_heads=nh,
                                          interleaved=interleaved)
                k = min(block.shape)
                lo, hi = decile_index_ranges(k, n_deciles, ascending=True)[decile - 1]
                factors = None if factor_cache is None else factor_cache.get(name)
                if factors is None and factor_cache is not None:
                    from .linalg import cached_svd
                    result = cached_svd(block, backend=backend, gpu_min_dim=gpu_min_dim)
                    factors = (result.U, result.s, result.Vh)
                    factor_cache[name] = factors
                assign_qkv_block(
                    mat, idx, _reconstruct_zeroed(
                        block, lo, hi, factors=factors, backend=backend,
                        gpu_min_dim=gpu_min_dim,
                    ), num_heads=nh, interleaved=interleaved,
                )
            else:
                k = min(mat.shape)
                lo, hi = decile_index_ranges(k, n_deciles, ascending=True)[decile - 1]
                factors = None if factor_cache is None else factor_cache.get(name)
                if factors is None and factor_cache is not None:
                    from .linalg import cached_svd
                    result = cached_svd(mat, backend=backend, gpu_min_dim=gpu_min_dim)
                    factors = (result.U, result.s, result.Vh)
                    factor_cache[name] = factors
                mat = _reconstruct_zeroed(
                    mat, lo, hi, factors=factors, backend=backend,
                    gpu_min_dim=gpu_min_dim,
                )
            out = mat.T if cls == "Conv1D" else mat
            # Preserve the live Parameter object and execution dtype.  Replacing
            # only a half-precision weight with float32 breaks F.linear against
            # half inputs/biases and also destroys ties/optimizer references.
            # The SVD is still evaluated in float64; the intervention is then
            # explicitly quantized back to the model's execution precision.
            new_w = torch.as_tensor(out, dtype=w.dtype, device=w.device)
            w.copy_(new_w)


def perplexity_vs_decile(model_factory, tokenizer, records, device, *,
                         n_tokens=4096, decile_scope="all", n_deciles=10,
                         spec=None, text_path="./wikitext-2-raw/wiki.test.raw",
                         stride=512, backend="numpy", gpu_min_dim=1024,
                         allow_fallback=False,
                         allow_tokenizer_fallback=False) -> Dict[str, List[float]]:
    """Perplexity after zeroing each decile, with each decile measured against
    the *pristine* weights (deciles never accumulate).

    Because ``set_layer_svd_decile`` mutates weights in place, a snapshot of the
    original ``state_dict`` is taken on the first model and restored before each
    decile's ablation.  This is correct whether ``model_factory`` returns a fresh
    model every call or the same instance repeatedly.

    ``decile_scope='all'`` ablates every matrix of the analyzed *types* across all
    layers (re-discovered on the model); ``'analyzed'`` ablates only the passed
    ``records``.  Returns {"deciles": [...], "perplexity": [...]}.
    """
    from .perplexity import perplexity_wikitext

    if int(n_deciles) < 1:
        raise ValueError("n_deciles must be positive")
    if decile_scope not in {"all", "analyzed"}:
        raise ValueError("decile_scope must be all or analyzed")
    deciles = list(range(1, int(n_deciles) + 1))
    ppls = []
    # Use exactly one pristine model for the baseline and every reversible
    # intervention.  Factories are not assumed to initialize identical weights.
    model = model_factory()
    sp = spec or get_model_spec(model)
    baseline_result = perplexity_wikitext(
        model, tokenizer, device, n_tokens=n_tokens, stride=stride,
        text_path=text_path, allow_fallback=allow_fallback,
        allow_tokenizer_fallback=allow_tokenizer_fallback, return_details=True,
    )
    baseline = float(baseline_result["perplexity"])
    baseline_count = int(baseline_result["scored_tokens"])
    if baseline_count <= 0 or not np.isfinite(baseline):
        raise ValueError("pristine perplexity is unavailable or non-finite")
    selected_roles = {r.short for r in records}
    selected_layers = {r.layer_idx for r in records}
    if decile_scope == "all":
        recs = [r for r in discover_weight_metadata(model, spec=sp)
                if r.short in selected_roles]
    else:
        recs = _rebind_records(model, records, spec=sp)
    if not recs:
        raise ValueError("decile scope selected no matrices")
    actual_names = sorted(r.name for r in recs)
    actual_layers = sorted({r.layer_idx for r in recs})
    actual_roles = sorted({r.short for r in recs})
    parameters = {}
    for rec in recs:
        base = rec.name.split("[", 1)[0]
        modname = base[:-len(".weight")] if base.endswith(".weight") else base
        parameter = model.get_submodule(modname).weight
        parameters[id(parameter)] = parameter
        rec.weight = None  # mutation re-reads live weights; release discovery copies
    snapshots = _DiskTensorStore()
    factor_cache = _DiskTensorStore()
    for key, parameter in parameters.items():
        snapshots[key] = parameter
    import torch
    try:
        for d in deciles:
            with torch.no_grad():
                for key, parameter in parameters.items():
                    parameter.copy_(snapshots.get(key).to(parameter.device))
            try:
                set_layer_svd_decile(
                    model, recs, d, n_deciles=n_deciles, spec=sp,
                    factor_cache=factor_cache, backend=backend,
                    gpu_min_dim=gpu_min_dim,
                )
                measured = perplexity_wikitext(
                    model, tokenizer, device, n_tokens=n_tokens, stride=stride,
                    text_path=text_path, allow_fallback=allow_fallback,
                    allow_tokenizer_fallback=allow_tokenizer_fallback,
                    return_details=True,
                )
                value = float(measured["perplexity"])
                if int(measured["scored_tokens"]) != baseline_count:
                    raise RuntimeError("decile and pristine perplexity scored different token counts")
                if not np.isfinite(value):
                    raise ValueError("decile perplexity is non-finite")
                ppls.append(value)
            finally:
                with torch.no_grad():
                    for key, parameter in parameters.items():
                        parameter.copy_(snapshots.get(key).to(parameter.device))
    finally:
        snapshots.close()
        factor_cache.close()
    return {
        "status": "complete",
        "deciles": deciles,
        "perplexity": ppls,
        "baseline_perplexity": baseline,
        "delta_perplexity": [float(value - baseline) for value in ppls],
        "requested_roles": sorted(selected_roles),
        "requested_layers": sorted(selected_layers),
        "decile_scope": decile_scope,
        "actual_roles": actual_roles,
        "actual_layers": actual_layers,
        "actual_matrix_names": actual_names,
        "actual_matrix_count": len(actual_names),
        # Backward-compatible aliases now describe actual intervention scope.
        "roles": actual_roles,
        "layers": actual_layers,
        "n_tokens_requested": int(n_tokens),
        "n_tokens_scored": baseline_count,
        "stride": int(stride),
        "requested_context_length": int(baseline_result.get("requested_max_length", 1024)),
        "effective_context_length": int(baseline_result.get("effective_max_length", 1024)),
        "requested_stride": int(baseline_result.get("requested_stride", stride)),
        "effective_stride": int(baseline_result.get("effective_stride", stride)),
        "execution_precision": str(next(model.parameters()).dtype),
    }


def _rebind_records(model, records, *, spec=None):
    """Re-discover records by name on a fresh model while preserving its spec."""
    wanted = {r.name for r in records}
    resolved = spec or get_model_spec(model)
    fresh = discover_weight_metadata(model, spec=resolved)
    return [r for r in fresh if r.name in wanted]
