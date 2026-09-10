"""rmt.activations — centered activation-covariance capture (Paper 3 Eq. 6).

A ``FeatureLayer`` wraps a target Linear: in "mean" mode it accumulates the
running mean of inputs; in "FM" mode it accumulates the *centered* running
covariance C = ⟨xcᵀxc⟩, xc = x − mean.  The FM accumulator uses a SEPARATE
counter (``fm_computation``) from the mean counter (``computation``) — sharing
one counter would scale the covariance by (mean_passes)/(total_passes) and
corrupt it.  Forward is an exact identity to ``F.linear``.

torch imported lazily at module top is fine here (these are torch-only paths).
"""
from __future__ import annotations

from typing import Dict, List, Optional
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .discovery import get_model_spec, classify
from .config import get_logger

_log = get_logger("rmt.activations")


class _Counter:
    """Float64 running first/second-moment accumulator with its own count."""
    def __init__(self, dim, device):
        self.mean = torch.zeros(dim, dtype=torch.float64, device=device)
        self.cov = torch.zeros((dim, dim), dtype=torch.float64, device=device)
        self.count = 0
        self.compute_mean = True


class FeatureLayer(nn.Module):
    """Wrap a Linear; capture centered input covariance while staying identity."""

    def __init__(self, linear: nn.Module, device=None, *, token_weighted=True):
        super().__init__()
        self.weight = linear.weight
        self.bias = getattr(linear, "bias", None)
        # HF Conv1D stores (in, out), unlike nn.Linear's (out, in).
        self.kernel_dim = (linear.weight.shape[0] if type(linear).__name__ == "Conv1D"
                           else linear.weight.shape[1])
        # Keep a non-registered handle to the original module.  This supports
        # both nn.Linear and transformers.pytorch_utils.Conv1D without changing
        # their forward implementation.
        self.__dict__["_orig_linear"] = linear
        dev = device or linear.weight.device
        # mean accumulator + counter
        self.register_buffer("_mean", torch.zeros(self.kernel_dim,
                                                  dtype=torch.float64, device=dev))
        self.computation = 0                              # mean-pass counter
        # SEPARATE FM accumulator + counter
        self.register_buffer("_cov", torch.zeros((self.kernel_dim, self.kernel_dim),
                                                 dtype=torch.float64, device=dev))
        self.register_buffer("_fm_mean", torch.zeros(self.kernel_dim,
                                                    dtype=torch.float64, device=dev))
        self.fm_computation = 0                           # FM-pass counter
        self.mode = "mean"
        self.token_weighted = bool(token_weighted)

    # -- accumulation ------------------------------------------------------- #
    def _flatten(self, x):
        return x.reshape(-1, x.shape[-1]).to(torch.float64)

    def _update_mean(self, x2d):
        # Batched streaming mean (REPORT §5.4): mathematically identical to the
        # per-token loop but O(1) Python ops instead of O(rows). The counter is
        # still incremented by the row count so downstream code that reads
        # ``computation`` sees the per-token total.
        b = x2d.shape[0]
        if b == 0:
            return
        if not self.token_weighted:
            x2d = x2d.mean(dim=0, keepdim=True)
            b = 1
        c0 = self.computation
        self.computation = c0 + b
        # new_mean = (c0*old_mean + sum(x)) / (c0 + b)
        self._mean = (c0 * self._mean + x2d.sum(dim=0)) / self.computation

    def _update_fm(self, x2d):
        # Batched centered covariance with its OWN counter (REPORT §5.4). The
        # mean is frozen during the FM pass, so this equals the per-token loop:
        # cov = average over all FM rows of (x-mean)(x-mean)^T.
        b = x2d.shape[0]
        if b == 0:
            return
        xc = x2d - self._mean                         # (b, d)
        batch_sum = xc.T @ xc                         # (d, d) sum of outer products
        if not self.token_weighted:
            batch_sum = batch_sum / b
            b = 1
        f0 = self.fm_computation
        self.fm_computation = f0 + b
        self._cov = (f0 * self._cov + batch_sum) / self.fm_computation

    # -- forward = identity to F.linear ------------------------------------- #
    def forward(self, x):
        with torch.no_grad():
            x2d = self._flatten(x)
            if self.mode == "mean":
                self._update_mean(x2d)
            elif self.mode == "FM":
                self._update_fm(x2d)
        original = self.__dict__["_orig_linear"]
        return original(x)

    # -- accessors ---------------------------------------------------------- #
    @property
    def mean_(self):
        return self._mean.detach().cpu().numpy()

    @property
    def cov_(self):
        return self._cov.detach().cpu().numpy()


_EXCLUDED_COMPONENTS = {
    "embed", "embedding", "embeddings", "embed_out", "embed_in",
    "embed_tokens", "lm_head", "pooler", "wte", "wpe", "shared",
}


def _excluded_projection(name: str) -> bool:
    """Exclude actual embedding/output-head components, not ancestor substrings."""

    components = {component.lower() for component in name.split(".")}
    return bool(components & _EXCLUDED_COMPONENTS)


def replace_with_feature_layers(model, layer_indices, device, *, spec=None,
                                target_names=None, token_weighted=True) -> List[str]:
    """Replace targeted projection modules transactionally; return their names."""
    if spec is None:
        spec = get_model_spec(model)
    from .discovery import extract_layer_index
    want = set(layer_indices) if layer_indices is not None else None
    explicit = None if target_names is None else set(target_names)
    wrapped = []
    # collect first to avoid mutating during iteration
    targets = []
    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) or type(module).__name__ == "Conv1D"):
            continue
        if explicit is not None and name not in explicit:
            continue
        if _excluded_projection(name):
            continue
        if classify(name, spec) is None:
            continue
        li = extract_layer_index(name, spec)
        if want is not None and li not in want:
            continue
        targets.append(name)
    try:
        for name in targets:
            parent, _, child = name.rpartition(".")
            pmod = model.get_submodule(parent) if parent else model
            linear = getattr(pmod, child)
            setattr(pmod, child, FeatureLayer(
                linear, device=device, token_weighted=token_weighted
            ))
            wrapped.append(name)
    except BaseException:
        # Allocation can fail halfway through a large model.  Never leave a
        # partially wrapped model behind.
        restore_linears(model)
        raise
    return wrapped


def set_feature_mode(model, mode: str) -> None:
    """Set every FeatureLayer to 'mean' or 'FM'."""
    assert mode in ("mean", "FM")
    for module in model.modules():
        if isinstance(module, FeatureLayer):
            module.mode = mode


def restore_linears(model) -> int:
    """Undo :func:`replace_with_feature_layers`: swap every FeatureLayer back for
    its original ``nn.Linear`` (REPORT §1 N1).

    After capture the model must be left so that ordinary forward passes (e.g. the
    perplexity phase) do NOT keep re-running the per-token covariance loop, and so
    that ``state_dict()`` no longer carries the giant ``_cov`` buffers (which would
    OOM the weight-only snapshot in ``perplexity_vs_decile``). Returns the number
    of layers restored. Idempotent.
    """
    # collect first to avoid mutating during iteration
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, FeatureLayer):
            targets.append(name)
    for name in targets:
        parent, _, child = name.rpartition(".")
        pmod = model.get_submodule(parent) if parent else model
        fl = getattr(pmod, child)
        orig = fl.__dict__.get("_orig_linear", None)
        if orig is None:                                  # pragma: no cover
            continue
        # the wrapped weight/bias are the SAME Parameters as the original's, so
        # restoring the original module is lossless.
        setattr(pmod, child, orig)
    return len(targets)


def collect_feature_matrices(model) -> Dict[str, dict]:
    """Collect covariance data and observation counts for wrapped projections."""
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, FeatureLayer):
            out[name] = {
                "FM": module.cov_,
                "mean": module.mean_,
                "mean_count": int(module.computation),
                "fm_count": int(module.fm_computation),
            }
    return out


def compute_activation_covariance(model, tokenizer, layer_indices, device, *,
                                  dataset_name="wikitext", split="train",
                                  n_text_batches=5, max_length=2048, stride=1024,
                                  max_oom=3, token_weighted=False, spec=None,
                                  text_path="./wikitext-2-raw/wiki.test.raw",
                                  allow_fallback=False,
                                  allow_tokenizer_fallback=False,
                                  target_names=None,
                                  window_plan=None) -> dict:
    """Two-pass (mean then centered FM) activation covariance on local text.

    Fully offline: text is read from ``text_path`` (the local wikitext file).
    If ``tokenizer`` is None (tiny in-process test models), a deterministic
    hash tokenizer against the model vocab is used so the path still runs without
    HF assets.  Returns the dict from ``collect_feature_matrices``.
    """
    spec = spec or get_model_spec(model)
    if n_text_batches < 1 or max_length < 2 or stride < 1:
        raise ValueError("capture counts, max_length, and stride must be positive")
    from .config import effective_context_length
    requested_max_length = int(max_length)
    max_length = effective_context_length(model, requested_max_length)
    module_modes = {module: bool(module.training) for module in model.modules()}
    batches = _load_text_batches(model, tokenizer, text_path, n_text_batches,
                                 max_length, stride, device,
                                 allow_fallback=allow_fallback,
                                 allow_tokenizer_fallback=allow_tokenizer_fallback)
    if window_plan is not None and window_plan.get("lengths") is not None:
        lengths = list(window_plan["lengths"])
        if len(lengths) != len(batches):
            raise RuntimeError("activation window plan does not match the text corpus")
        batches = [batch[:, :int(length)] for batch, length in zip(batches, lengths)]
    if not batches:
        raise ValueError("activation text produced no usable token windows")

    # Capture one projection at a time.  Dense d_in² buffers for every MLP at
    # once can exceed model memory by many GiB.
    from .discovery import extract_layer_index
    want = set(layer_indices) if layer_indices is not None else None
    explicit_targets = None if target_names is None else set(target_names)
    targets = []
    for name, module in model.named_modules():
        if not (isinstance(module, nn.Linear) or type(module).__name__ == "Conv1D"):
            continue
        if _excluded_projection(name) or classify(name, spec) is None:
            continue
        if explicit_targets is not None and name not in explicit_targets:
            continue
        if want is None or extract_layer_index(name, spec) in want:
            targets.append(name)
    if not targets:
        raise ValueError("no supported projection modules selected for activation capture")

    result = {}
    try:
        model.eval()
        replay_batches = list(batches)
        # If any later projection requires shorter windows, discard all earlier
        # results and restart so every covariance describes identical tokens.
        while True:
            result = {}
            restart = False
            for target_name in targets:
                try:
                    replace_with_feature_layers(
                        model, layer_indices, device, spec=spec,
                        target_names=[target_name], token_weighted=token_weighted,
                    )
                    set_feature_mode(model, "mean")
                    successful = []
                    with torch.no_grad():
                        for ids in replay_batches:
                            successful.append(_safe_forward(model, ids, max_oom))
                    if any(got.shape[1] != sent.shape[1]
                           for got, sent in zip(successful, replay_batches)):
                        replay_batches = successful
                        restart = True
                        break
                    set_feature_mode(model, "FM")
                    with torch.no_grad():
                        for ids in successful:
                            _safe_forward(model, ids, 0)
                    captured = collect_feature_matrices(model)
                    for value in captured.values():
                        if (value.get("FM") is None or value.get("mean_count", 0) <= 0
                                or value.get("fm_count", 0) <= 0):
                            raise ValueError(
                                f"projection {target_name} was discovered but not executed")
                    result.update(captured)
                finally:
                    restore_linears(model)
            if not restart:
                break
        successful_lengths = [int(batch.shape[1]) for batch in replay_batches]
        previous_lengths = None if window_plan is None else window_plan.get("lengths")
        if window_plan is not None:
            if previous_lengths is not None and list(previous_lengths) != successful_lengths:
                window_plan["revision"] = int(window_plan.get("revision", 0)) + 1
            window_plan["lengths"] = successful_lengths
        import hashlib as _hashlib
        identity = _hashlib.sha256()
        for batch in replay_batches:
            identity.update(batch.detach().to("cpu").numpy().tobytes())
        for value in result.values():
            value.update({
                "requested_max_length": requested_max_length,
                "effective_max_length": int(max_length),
                "captured_window_lengths": list(successful_lengths),
                "window_identity_sha256": identity.hexdigest(),
            })
    finally:
        restore_linears(model)
        # Assign flags directly: ``model.train(root_mode)`` recursively flattens
        # intentionally mixed train/eval submodule state.
        for module, training in module_modes.items():
            module.training = training
    return result


def _safe_forward(model, input_ids, max_oom):
    """Run one complete capture forward and return the exact successful window.

    Accumulator state is rolled back before an OOM retry, preventing early
    wrapped layers from counting a failed prefix twice.
    """
    oom = 0
    while True:
        layers = [module for module in model.modules() if isinstance(module, FeatureLayer)]
        snapshots = [
            (layer, layer._mean.clone(), layer._cov.clone(),
             layer.computation, layer.fm_computation)
            for layer in layers
        ]
        try:
            model(input_ids=input_ids)
            return input_ids
        except RuntimeError as e:                          # pragma: no cover
            for layer, mean, cov, mean_count, cov_count in snapshots:
                layer._mean.copy_(mean)
                layer._cov.copy_(cov)
                layer.computation = mean_count
                layer.fm_computation = cov_count
            if "out of memory" in str(e).lower() and oom < max_oom and input_ids.shape[1] > 2:
                oom += 1
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
                input_ids = input_ids[:, : max(2, input_ids.shape[1] // 2)]
                continue
            raise


def _load_text_batches(model, tokenizer, text_path, n_text_batches,
                       max_length, stride, device, *, allow_fallback=False,
                       allow_tokenizer_fallback=False):
    """Tokenize windows of a local text file (offline). Falls back to a
    deterministic hash tokenizer when ``tokenizer`` is None. Only the portion
    of text needed for ``n_text_batches`` windows is tokenized."""
    text = _read_local_text(text_path, allow_fallback=allow_fallback)
    # how many tokens we actually need, plus headroom; avoids tokenizing the
    # whole file (which triggers the "sequence longer than max length" warning).
    need_tokens = (max(0, n_text_batches - 1) * stride) + max_length + 8
    text = text[: need_tokens * 8]                 # ~8 chars/token upper bound
    if tokenizer is not None:
        enc = tokenizer(text, return_tensors="pt", truncation=False,
                        add_special_tokens=False)
        ids = enc["input_ids"][0]
    else:
        if not allow_tokenizer_fallback:
            raise RuntimeError("a matching tokenizer is required; synthetic token IDs are disabled")
        vocab = int(getattr(model, "vocab",
                            getattr(getattr(model, "config", None), "vocab_size", 50)))
        toks = [int.from_bytes(hashlib.sha256(w.encode("utf-8")).digest()[:8], "big") % vocab
                for w in text.split()]
        ids = torch.tensor(toks, dtype=torch.long)
    batches = []
    for i in range(n_text_batches):
        start = i * stride
        chunk = ids[start:start + max_length]
        if chunk.numel() < 2:
            break
        batches.append(chunk.unsqueeze(0).to(device))
    return batches


def _read_local_text(text_path, *, allow_fallback=False) -> str:
    import os
    if text_path and os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if not allow_fallback:
        raise FileNotFoundError(f"local activation text is missing: {text_path}")
    return ("the quick brown fox jumps over the lazy dog . " * 400).strip()
