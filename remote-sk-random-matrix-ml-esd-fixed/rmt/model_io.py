"""rmt.model_io — offline loading of local model snapshots (lazy HF import)."""
from __future__ import annotations
import os
from .config import OfflineGuard, get_logger

_log = get_logger("rmt.model_io")


def load_model(name_or_path, *, model_path=None, dtype="fp32", device=None):
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
    model._rmt_resolved_path = os.path.realpath(path)
    return model


def load_tokenizer(name_or_path, *, model_path=None):
    """Load the matching tokenizer fully offline; return None if unavailable.

    The pipeline fails closed on None unless ``allow_fallback_tokenizer`` was
    explicitly enabled for a synthetic/test run.
    """
    OfflineGuard.enable()
    try:
        from transformers import AutoTokenizer
        path = _resolve_path(name_or_path, model_path)
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as e:                                      # pragma: no cover
        _log.warning("tokenizer load failed (%s); production text analyses will be unavailable", e)
        return None


def _resolve_path(name_or_path, model_path):
    if model_path is not None:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"explicit model_path does not exist: {model_path}")
        return os.path.realpath(model_path)
    if os.path.isdir(name_or_path):
        return os.path.realpath(name_or_path)
    cand = os.path.join("./models", os.path.basename(name_or_path))
    if os.path.isdir(cand):
        return os.path.realpath(cand)
    raise FileNotFoundError(f"no offline model snapshot found for {name_or_path!r}")
