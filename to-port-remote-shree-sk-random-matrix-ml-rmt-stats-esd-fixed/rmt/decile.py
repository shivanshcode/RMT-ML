"""rmt.decile — singular-value decile ablation (Paper 3 §5).

``set_layer_svd_decile`` reconstructs a weight with one ascending decile of its
singular values zeroed (decile 1 = smallest 10%), in place.  For fused QKV it
edits only the corresponding row block.  ``perplexity_vs_decile`` rebuilds a
fresh model for each decile and measures the perplexity impact.

torch imported lazily inside functions.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from .scalars import decile_index_ranges
from .discovery import (get_model_spec, discover_weight_matrices, split_fused_qkv,
                        get_num_heads, qkv_is_interleaved, extract_qkv_block,
                        assign_qkv_block)


def _reconstruct_zeroed(W, lo, hi):
    """SVD of W, zero singular values in ascending index range [lo, hi), rebuild."""
    U, s, Vh = np.linalg.svd(W, full_matrices=False)
    order = np.argsort(s)                       # ascending
    s_new = s.copy()
    zero_idx = order[lo:hi]
    s_new[zero_idx] = 0.0
    return (U * s_new) @ Vh


def set_layer_svd_decile(model, records, decile, *, n_deciles=10, spec=None) -> None:
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
                    interleaved = False
                mat = np.ascontiguousarray(mat)
                block = extract_qkv_block(mat, idx, num_heads=nh,
                                          interleaved=interleaved)
                k = min(block.shape)
                lo, hi = decile_index_ranges(k, n_deciles, ascending=True)[decile - 1]
                assign_qkv_block(mat, idx, _reconstruct_zeroed(block, lo, hi),
                                 num_heads=nh, interleaved=interleaved)
            else:
                k = min(mat.shape)
                lo, hi = decile_index_ranges(k, n_deciles, ascending=True)[decile - 1]
                mat = _reconstruct_zeroed(mat, lo, hi)
            out = mat.T if cls == "Conv1D" else mat
            # Store back at >= float32 (REPORT §0): an fp16 store floors the
            # reconstruction at ~1e-3 rel-err and swamps the small-SV signal.
            store_dtype = w.dtype
            if torch.finfo(w.dtype).bits < 32:
                store_dtype = torch.float32
            new_w = torch.as_tensor(out, dtype=store_dtype, device=w.device)
            if store_dtype == w.dtype:
                w.copy_(new_w)
            else:
                module.weight = torch.nn.Parameter(new_w, requires_grad=w.requires_grad)


def perplexity_vs_decile(model_factory, tokenizer, records, device, *,
                         n_tokens=4096, decile_scope="all", n_deciles=10,
                         spec=None, text_path="./wikitext-2-raw/wiki.test.raw"
                         ) -> Dict[str, List[float]]:
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

    deciles = list(range(1, n_deciles + 1))
    ppls = []
    snapshot = None
    for d in deciles:
        model = model_factory()
        model.eval()
        if snapshot is None:
            # REPORT §1 N2: snapshot WEIGHTS ONLY. A full state_dict() deepcopy
            # would also clone every (d_in x d_in) float64 activation-covariance
            # buffer left on wrapped layers (1.64 GB per MLP on the 8B) — a real
            # OOM. The ablation only ever mutates weights, so weights suffice.
            snapshot = {k: v.detach().clone()
                        for k, v in model.state_dict().items()
                        if k.endswith(".weight")}      # pristine baseline
        else:
            model.load_state_dict(snapshot, strict=False)   # restore before ablating
        sp = spec or get_model_spec(model)
        if decile_scope == "all":
            recs = discover_weight_matrices(model, spec=sp)
        else:
            recs = _rebind_records(model, records)
        set_layer_svd_decile(model, recs, d, n_deciles=n_deciles, spec=sp)
        ppls.append(perplexity_wikitext(model, tokenizer, device, n_tokens=n_tokens,
                                        text_path=text_path))
    # leave the last model pristine (un-ablated) for any downstream use
    if snapshot is not None:
        try:
            model.load_state_dict(snapshot, strict=False)
        except Exception:
            pass
    return {"deciles": deciles, "perplexity": ppls}


def _rebind_records(model, records):
    """Re-discover records by name on a fresh model instance (scope='analyzed')."""
    wanted = {r.name for r in records}
    spec = get_model_spec(model)
    fresh = discover_weight_matrices(model, spec=spec)
    return [r for r in fresh if r.name in wanted]
