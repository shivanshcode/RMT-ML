"""rmt.perplexity — strided negative-log-likelihood perplexity on local text.

Fully offline: reads a local raw text file (default the wikitext-2 test split)
and computes a sliding-window perplexity.  Unusable/non-finite evaluations fail
explicitly rather than being serialized as successful NaN experiments.
"""
from __future__ import annotations

import hashlib
import math
import os
import numpy as np


def perplexity_wikitext(model, tokenizer, device, *, n_tokens=4096, stride=512,
                        text_path="./wikitext-2-raw/wiki.test.raw",
                        max_length=1024, allow_fallback=False,
                        allow_tokenizer_fallback=False, return_details=False):
    """Sliding-window perplexity over the first ~n_tokens tokens of a local file.

    If ``tokenizer`` is None (tiny test models), a trivial whitespace/byte
    tokenizer is used against the model's vocab so the path stays runnable
    offline without HF assets.
    """
    import torch

    def result(value, count=0):
        return ({"perplexity": float(value), "scored_tokens": int(count)}
                if return_details else float(value))

    n_tokens, stride, max_length = int(n_tokens), int(stride), int(max_length)
    if n_tokens < 2 or stride < 1 or max_length < 2:
        raise ValueError("require n_tokens/max_length >= 2 and a positive stride")
    from .config import effective_context_length
    requested_max_length = max_length
    requested_stride = stride
    max_length = effective_context_length(model, requested_max_length)
    stride = min(stride, max_length - 1)
    text = _read_text(text_path, allow_fallback=allow_fallback)
    if not text.strip():
        raise ValueError("perplexity text is empty")

    # only tokenize the portion we need (avoids the "sequence too long" warning)
    text = text[: (n_tokens + 8) * 8]
    if tokenizer is not None:
        enc = tokenizer(text, return_tensors="pt", truncation=False,
                        add_special_tokens=False)
        input_ids = enc["input_ids"][0]
    else:
        if not allow_tokenizer_fallback:
            raise RuntimeError("a matching tokenizer is required; synthetic token IDs are disabled")
        vocab = int(getattr(model, "vocab", getattr(getattr(model, "config", None),
                                                     "vocab_size", 50)))
        toks = [int.from_bytes(hashlib.sha256(w.encode("utf-8")).digest()[:8], "big") % vocab
                for w in text.split()]
        input_ids = torch.tensor(toks, dtype=torch.long)

    input_ids = input_ids[:n_tokens]
    if input_ids.numel() < 2:
        raise ValueError("tokenizer produced fewer than two tokens")
    # always place inputs on the model's actual device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        pass
    input_ids = input_ids.to(device)
    mode_snapshot = [(module, bool(module.training)) for module in model.modules()]
    model.eval()

    nll_sum = 0.0
    n_tok = 0
    seq_len = input_ids.size(0)
    previous_end = 0
    try:
        with torch.no_grad():
            # Each iteration adds at most ``stride`` new targets, retaining up
            # to max_length-stride context tokens.  Prefix labels are ignored.
            for end in range(min(stride, seq_len), seq_len + stride, stride):
                end = min(end, seq_len)
                begin = max(0, end - max_length)
                ids = input_ids[begin:end].unsqueeze(0)
                if ids.size(1) < 2:
                    if end == seq_len:
                        break
                    continue
                labels = ids.clone()
                target_start = max(previous_end, begin + 1)
                ignore_count = max(0, target_start - begin)
                labels[:, :ignore_count] = -100
                valid = int(torch.count_nonzero(labels[:, 1:] != -100).item())
                if valid:
                    out = model(input_ids=ids, labels=labels)
                    loss = getattr(out, "loss", None)
                    if loss is None:
                        raise ValueError("model output has no loss")
                    loss_value = float(loss)
                    if not math.isfinite(loss_value):
                        raise ValueError("model produced a non-finite loss")
                    nll_sum += loss_value * valid
                    n_tok += valid
                previous_end = end
                if end == seq_len:
                    break
    finally:
        # Restore mixed per-submodule modes exactly; model.train(root_mode)
        # would flatten a deliberately mixed train/eval tree.
        for module, training in mode_snapshot:
            module.training = training
    if n_tok <= 0:
        raise ValueError("perplexity evaluation scored no targets")
    value = math.exp(nll_sum / n_tok)
    if not math.isfinite(value):
        raise ValueError("perplexity is non-finite")
    if return_details:
        return {
            "perplexity": float(value),
            "scored_tokens": int(n_tok),
            "requested_max_length": int(requested_max_length),
            "effective_max_length": int(max_length),
            "requested_stride": int(requested_stride),
            "effective_stride": int(stride),
        }
    return result(value, n_tok)


def _read_text(text_path, *, allow_fallback=False) -> str:
    if text_path and os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if not allow_fallback:
        raise FileNotFoundError(f"local perplexity text is missing: {text_path}")
    return ("the quick brown fox jumps over the lazy dog . " * 200).strip()
