"""rmt.decile — singular-value decile ablation (Paper 3 §5).

``set_layer_svd_decile`` reconstructs a weight with one ascending decile of its
singular values zeroed (decile 1 = smallest 10%), in place.  For fused QKV it
edits only the corresponding row block.  ``perplexity_vs_decile`` rebuilds a
fresh model for each decile and measures the perplexity impact.

torch imported lazily inside functions.
"""
from __future__ import annotations

from typing import Dict, List
import hashlib
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


def _weight_fingerprint(W):
    array = np.ascontiguousarray(np.asarray(W))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _qualified_factors(W, *, backend="numpy", gpu_min_dim=1024):
    """Return factors only when the decile precision contract is satisfied."""
    from .linalg import cached_svd
    result = cached_svd(W, full_matrices=False, backend=backend,
                        gpu_min_dim=gpu_min_dim)
    if result.degraded or result.factorization_dtype != "float64":
        raise RuntimeError(
            "decile lesion requires a non-degraded float64 SVD "
            f"(backend={result.backend}, dtype={result.factorization_dtype})")
    # Keep the qualification marker in every memory/disk factor cache entry so
    # a bare low-precision tuple cannot bypass the guard on reuse.
    return result.U, result.s, result.Vh, "float64", _weight_fingerprint(W)


def _reconstruct_zeroed(W, lo, hi, *, factors=None, backend="numpy", gpu_min_dim=1024):
    """Zero an ascending singular range, optionally reusing qualified factors."""
    if factors is None:
        factors = _qualified_factors(
            W, backend=backend, gpu_min_dim=gpu_min_dim)
    if (len(factors) != 5 or str(np.asarray(factors[3]).item()) != "float64"
            or str(factors[4]) != _weight_fingerprint(W)):
        raise RuntimeError("cached decile factors are stale or lack float64 qualification")
    U, s, Vh = factors[:3]
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
    n_deciles = int(n_deciles)
    if n_deciles < 1:
        raise ValueError("n_deciles must be positive")

    # Resolve the complete scope using metadata only.  In particular, do not
    # retain one float64 host copy per selected projection during prevalidation.
    groups = {}
    seen_scopes = set()
    for rec in records:
        name = rec.name
        is_fused = name.endswith("]") and "[" in name
        base = name[: name.index("[")] if is_fused else name
        modname = base[: -len(".weight")] if base.endswith(".weight") else base
        module = model.get_submodule(modname)
        w = module.weight
        cls = type(module).__name__
        logical_shape = tuple(reversed(tuple(w.shape))) if cls == "Conv1D" else tuple(w.shape)
        tag = name[name.index("[") + 1: -1] if is_fused else None
        scope = (id(w), tag if is_fused else "__full_parameter__")
        if scope in seen_scopes:
            continue
        seen_scopes.add(scope)
        idx = interleaved = nh = None
        block_shape = logical_shape
        if is_fused:
            if logical_shape[0] % 3:
                raise ValueError(f"fused QKV rows are not divisible by three for {name}")
            idx = list(spec.fused_qkv_order).index(tag)
            interleaved = (rec.qkv_interleaved
                           if getattr(rec, "qkv_interleaved", None) is not None
                           else qkv_is_interleaved(base, spec))
            nh = (rec.num_heads if getattr(rec, "num_heads", None) is not None
                  else get_num_heads(model)) if interleaved else None
            if interleaved and nh is None:
                raise ValueError("num_heads is required for interleaved fused QKV")
            block_shape = (logical_shape[0] // 3, logical_shape[1])
            if interleaved and block_shape[0] % int(nh):
                raise ValueError(
                    f"fused QKV of {logical_shape[0]} rows is not divisible into {nh} heads"
                )
        if (int(rec.n), int(rec.m)) != tuple(map(int, block_shape)):
            raise ValueError(f"matrix shape changed for {name}")
        rank = min(block_shape)
        if n_deciles > rank:
            raise ValueError(
                f"n_deciles={n_deciles} exceeds singular count {rank} for {name}"
            )
        groups.setdefault(id(w), {"weight": w, "class": cls, "items": []})["items"].append(
            (scope, is_fused, idx, interleaved, nh)
        )

    # Scope conflicts are metadata errors and must be rejected before any
    # physical parameter is changed.
    for group in groups.values():
        items = group["items"]
        if any(not item[1] for item in items) and len(items) > 1:
            raise ValueError("a parameter cannot be selected as both full and fused scopes")

    with torch.no_grad():
        # Materialize and release one physical parameter at a time.  Q/K/V
        # records sharing a fused parameter are reconstructed from this one
        # pristine copy and committed together.
        for group in groups.values():
            w, cls, items = group["weight"], group["class"], group["items"]
            full = w.detach().cpu().to(torch.float64).numpy()
            pristine_mat = np.ascontiguousarray(full.T if cls == "Conv1D" else full)
            mat = pristine_mat.copy()
            for scope, is_fused, idx, interleaved, nh in items:
                pristine_block = (extract_qkv_block(
                    pristine_mat, idx, num_heads=nh, interleaved=interleaved)
                    if is_fused else pristine_mat)
                lo, hi = decile_index_ranges(
                    min(pristine_block.shape), n_deciles, ascending=True
                )[decile - 1]
                factors = None if factor_cache is None else factor_cache.get(scope)
                if (factors is not None
                        and (len(factors) != 5 or str(factors[4]) != _weight_fingerprint(pristine_block))):
                    factors = None
                if factors is None:
                    factors = _qualified_factors(
                        pristine_block, backend=backend, gpu_min_dim=gpu_min_dim)
                    if factor_cache is not None:
                        factor_cache[scope] = factors
                reconstructed = _reconstruct_zeroed(
                    pristine_block, lo, hi, factors=factors, backend=backend,
                    gpu_min_dim=gpu_min_dim,
                )
                if is_fused:
                    assign_qkv_block(mat, idx, reconstructed, num_heads=nh,
                                     interleaved=interleaved)
                else:
                    mat = reconstructed
            out = mat.T if cls == "Conv1D" else mat
            w.copy_(torch.as_tensor(out, dtype=w.dtype, device=w.device))


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

    records = list(records)
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
    head_overrides = {r.num_heads for r in records if getattr(r, "num_heads", None) is not None}
    if len(head_overrides) > 1:
        raise ValueError("records contain inconsistent fused-QKV head counts")
    head_override = next(iter(head_overrides), None)
    if decile_scope == "all":
        recs = [r for r in discover_weight_metadata(model, spec=sp, num_heads=head_override)
                if r.short in selected_roles]
    else:
        recs = _rebind_records(model, records, spec=sp, num_heads=head_override)
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
        "decile_svd_factorization_dtype": "float64",
        "decile_svd_degraded": False,
        "decile_precision_status": "complete",
    }


def _rebind_records(model, records, *, spec=None, num_heads=None):
    """Re-discover every requested record by exact name on the active model."""
    wanted = {r.name for r in records}
    resolved = spec or get_model_spec(model)
    fresh = discover_weight_metadata(model, spec=resolved, num_heads=num_heads)
    by_name = {r.name: r for r in fresh}
    missing = sorted(wanted - by_name.keys())
    if missing:
        raise ValueError(f"requested analyzed decile matrices are missing: {missing}")
    return [by_name[name] for name in sorted(wanted)]
