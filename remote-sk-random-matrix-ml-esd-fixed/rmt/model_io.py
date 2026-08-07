"""rmt.model_io — offline loading of local model snapshots (lazy HF import)."""
from __future__ import annotations
import os
from .config import OfflineGuard, get_logger

_log = get_logger("rmt.model_io")


def load_model(name_or_path, *, model_path=None, dtype="fp16", device=None):
    """Load a local HF causal-LM snapshot fully offline and place it on a device.

    Resolves a local directory (``model_path`` or ``./models/<name>``); never
    reaches the network (HF offline env vars are set first). The model is moved
    to ``device`` (default: cuda if available, else cpu) so that downstream
    activation / perplexity inputs land on the same device.
    """
    OfflineGuard.enable()
    import torch
    from transformers import AutoModelForCausalLM

    path = _resolve_path(name_or_path, model_path)
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16,
          "fp32": torch.float32}.get(dtype, torch.float16)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _log.info("loading local model from %s (dtype=%s, device=%s)", path, dtype, device)

    # transformers renamed `torch_dtype` -> `dtype`; try the new name, fall back.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dt, local_files_only=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dt, local_files_only=True)

    model.to(device)
    model.eval()
    return model


def load_tokenizer(name_or_path, *, model_path=None):
    """Load the matching tokenizer fully offline; return None if unavailable."""
    OfflineGuard.enable()
    try:
        from transformers import AutoTokenizer
        path = _resolve_path(name_or_path, model_path)
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as e:                                      # pragma: no cover
        _log.warning("tokenizer load failed (%s); using offline fallback", e)
        return None


def _resolve_path(name_or_path, model_path):
    path = model_path or name_or_path
    if not os.path.isdir(path):
        cand = os.path.join("./models", os.path.basename(name_or_path))
        if os.path.isdir(cand):
            path = cand
    return path
