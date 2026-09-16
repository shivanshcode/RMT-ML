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

    def __init__(self, linear: nn.Linear, device=None):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.kernel_dim = linear.weight.shape[1]          # in_features (d_in)
        # Keep a handle to the original module so capture is REVERSIBLE
        # (REPORT §1 N1). Stash it via __dict__ to bypass nn.Module.__setattr__,
        # so it is NOT registered as a submodule (its weight must not reappear in
        # state_dict() / discovery while wrapped).
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
        return F.linear(x, self.weight, self.bias)

    # -- accessors ---------------------------------------------------------- #
    @property
    def mean_(self):
        return self._mean.detach().cpu().numpy()

    @property
    def cov_(self):
        return self._cov.detach().cpu().numpy()


_SKIP = ("embed", "lm_head", "embed_out", "embed_in", "pooler", "head", "norm")


def replace_with_feature_layers(model, layer_indices, device, *, spec=None) -> List[str]:
    """Replace targeted Linears with FeatureLayers; return the wrapped names."""
    if spec is None:
        spec = get_model_spec(model)
    from .discovery import extract_layer_index
    want = set(layer_indices) if layer_indices is not None else None
    wrapped = []
    # collect first to avoid mutating during iteration
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lname = name.lower()
        if any(sk in lname for sk in _SKIP):
            continue
        if classify(name, spec) is None:
            continue
        li = extract_layer_index(name, spec)
        if want is not None and li not in want:
            continue
        targets.append(name)
    for name in targets:
        parent, _, child = name.rpartition(".")
        pmod = model.get_submodule(parent) if parent else model
        linear = getattr(pmod, child)
        setattr(pmod, child, FeatureLayer(linear, device=device))
        wrapped.append(name)
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
    """name → {weight, FM, mean}; FM is the (d_in × d_in) centered covariance."""
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, FeatureLayer):
            out[name] = {
                "weight": module.weight.detach().cpu().float().numpy(),
                "FM": module.cov_,
                "mean": module.mean_,
            }
    return out


def compute_activation_covariance(model, tokenizer, layer_indices, device, *,
                                  dataset_name="wikitext", split="train",
                                  n_text_batches=5, max_length=2048, stride=1024,
                                  max_oom=3, token_weighted=False, spec=None,
                                  text_path="./wikitext-2-raw/wiki.test.raw") -> dict:
    """Two-pass (mean then centered FM) activation covariance on local text.

    Fully offline: text is read from ``text_path`` (the local wikitext file).
    If ``tokenizer`` is None (tiny in-process test models), a deterministic
    hash tokenizer against the model vocab is used so the path still runs without
    HF assets.  Returns the dict from ``collect_feature_matrices``.
    """
    spec = spec or get_model_spec(model)
    replace_with_feature_layers(model, layer_indices, device, spec=spec)
    model.eval()

    batches = _load_text_batches(model, tokenizer, text_path, n_text_batches,
                                 max_length, stride, device)

    # pass 1: means
    set_feature_mode(model, "mean")
    with torch.no_grad():
        for ids in batches:
            _safe_forward(model, ids, max_oom)
    # pass 2: centered covariance
    set_feature_mode(model, "FM")
    with torch.no_grad():
        for ids in batches:
            _safe_forward(model, ids, max_oom)
    fms = collect_feature_matrices(model)
    # REPORT §1 N1: capture is reversible — put the original Linears back so later
    # forward passes (perplexity) don't keep accumulating the covariance and the
    # _cov buffers don't leak into snapshots.
    restore_linears(model)
    return fms


def _safe_forward(model, input_ids, max_oom):
    oom = 0
    while True:
        try:
            model(input_ids=input_ids)
            return
        except RuntimeError as e:                          # pragma: no cover
            if "out of memory" in str(e).lower() and oom < max_oom:
                oom += 1
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
                input_ids = input_ids[:, : input_ids.shape[1] // 2]
                continue
            raise


def _load_text_batches(model, tokenizer, text_path, n_text_batches,
                       max_length, stride, device):
    """Tokenize windows of a local text file (offline). Falls back to a
    deterministic hash tokenizer when ``tokenizer`` is None. Only the portion
    of text needed for ``n_text_batches`` windows is tokenized."""
    text = _read_local_text(text_path)
    # how many tokens we actually need, plus headroom; avoids tokenizing the
    # whole file (which triggers the "sequence longer than max length" warning).
    need_tokens = (max(0, n_text_batches - 1) * stride) + max_length + 8
    text = text[: need_tokens * 8]                 # ~8 chars/token upper bound
    if tokenizer is not None:
        enc = tokenizer(text, return_tensors="pt", truncation=False,
                        add_special_tokens=False)
        ids = enc["input_ids"][0]
    else:
        vocab = int(getattr(model, "vocab",
                            getattr(getattr(model, "config", None), "vocab_size", 50)))
        toks = [(abs(hash(w)) % vocab) for w in text.split()]
        ids = torch.tensor(toks, dtype=torch.long)
    batches = []
    for i in range(n_text_batches):
        start = i * stride
        chunk = ids[start:start + max_length]
        if chunk.numel() < 2:
            break
        batches.append(chunk.unsqueeze(0).to(device))
    return batches


def _read_local_text(text_path) -> str:
    import os
    if text_path and os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    # deterministic offline fallback so tiny-model runs still exercise the path
    return ("the quick brown fox jumps over the lazy dog . " * 400).strip()
