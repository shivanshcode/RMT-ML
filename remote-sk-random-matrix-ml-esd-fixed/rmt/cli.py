"""rmt.cli — command-line entry point.

``python -m rmt --selftest``         run the fast analytic self-check (exit 0/1)
``python -m rmt --models <path> ...`` run the full offline analysis pipeline

The argument parser is generated automatically from the fields of
:class:`~rmt.config.RunConfig`, so every documented option is always accepted
and the CLI can never drift out of sync with the config.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import os
import re
import typing
from typing import Optional, get_args, get_origin

from .config import RunConfig, OfflineGuard, get_logger

_log = get_logger("rmt.cli")

# constrained flags get explicit choices
_LIBRARY_ONLY = {"epoch_probe_fracs", "epoch_checkpoint_every_frac"}

_CHOICES = {
    "backend": ["auto", "numpy", "torch"],
    "alpha_estimator": ["csn", "hill", "hill_windowed", "all"],
    "N_cov_mode": ["cols", "rows", "max"],
    "decile_scope": ["all", "analyzed"],
    "dtype": ["fp16", "bf16", "fp32"],
    # REPORT §2: only gd_median is actually dispatched downstream; the former
    # 'median_raw'/'usvt_threshold' choices had no implementation, so advertising
    # them in the CLI silently misled about what ran.
    "sigma_estimator": ["gd_median"],
}


def _scalar_type(field_type):
    """Resolve the element/scalar python type for an (optionally Optional) field."""
    origin = get_origin(field_type)
    if origin is typing.Union:                      # Optional[X] == Union[X, None]
        args = [a for a in get_args(field_type) if a is not type(None)]
        return args[0] if args else str
    return field_type


def build_parser() -> argparse.ArgumentParser:
    d = RunConfig()
    # config.py uses `from __future__ import annotations`, so dataclass field
    # annotations are strings; resolve them to real types here.
    hints = typing.get_type_hints(RunConfig)
    p = argparse.ArgumentParser("rmt", description="RMT analysis of LLM weight matrices")
    p.add_argument("--selftest", action="store_true",
                   help="run analytic self-check and exit")

    for f in dataclasses.fields(d):
        name = f.name
        if name == "selftest" or name in _LIBRARY_ONLY:
            continue
        default = getattr(d, name)
        ftype = hints.get(name, str)
        origin = get_origin(ftype)
        flag = f"--{name}"

        # list fields -> nargs
        if origin in (list, typing.List):
            elem = get_args(ftype)[0] if get_args(ftype) else str
            elem_t = int if elem is int else (float if elem is float else str)
            nargs = "+" if name == "models" else "*"
            p.add_argument(flag, nargs=nargs, type=elem_t, default=default)
            continue

        scalar = _scalar_type(ftype)

        # bool fields -> --flag / --no-flag
        if scalar is bool:
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
            continue

        kwargs = {"default": default}
        if name in _CHOICES:
            kwargs["choices"] = _CHOICES[name]
        if scalar is int:
            kwargs["type"] = int
        elif scalar is float:
            kwargs["type"] = float
        else:
            kwargs["type"] = str
        p.add_argument(flag, **kwargs)

    return p


def _config_from_args(args) -> RunConfig:
    fields = {f.name for f in dataclasses.fields(RunConfig)}
    kw = {k: getattr(args, k) for k in fields if hasattr(args, k)}
    return RunConfig(**kw)


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "offline", True):
        OfflineGuard.enable()

    if args.selftest:
        from .selftest import run
        from .config import SEED
        ok = run(seed=SEED)
        return 0 if ok else 1

    cfg = _config_from_args(args)
    # gate the real run on the selftest passing
    from .selftest import run as selftest_run
    from .config import SEED
    if not selftest_run(seed=SEED, verbose=True):
        _log.error("selftest FAILED — aborting before touching real matrices")
        return 1

    from .pipeline import analyze_one_model
    from .model_io import load_model, load_tokenizer, _resolve_path

    # `output_dir`, `dtype`, `model_path`, `models` are consumed by the loop /
    # loader (and `output_dir` is passed positionally), so they must NOT also be
    # forwarded as keyword overrides — that would collide with analyze_one_model.
    _skip = {"models", "model_path", "dtype", "output_dir", "selftest"}
    overrides = {f.name: getattr(cfg, f.name)
                 for f in dataclasses.fields(cfg) if f.name not in _skip}
    if cfg.model_path is not None and len(args.models) != 1:
        _log.error("an explicit model_path can only be used with one model tag")
        return 2
    had_failure = False
    for tag in args.models:
        model = tokenizer = None
        try:
            model = load_model(tag, model_path=cfg.model_path, dtype=cfg.dtype)
            tokenizer = load_tokenizer(tag, model_path=cfg.model_path)
            resolved = getattr(model, "_rmt_resolved_path", None)
            if resolved is None:
                try:
                    resolved = _resolve_path(tag, cfg.model_path)
                except FileNotFoundError:
                    # In-process/custom loaders may not expose a filesystem path.
                    resolved = cfg.model_path or tag
            safe = _safe_tag(tag, identity=os.path.realpath(resolved))
            out = os.path.join(cfg.output_dir, safe)
            csv_path, rows = analyze_one_model(model, safe, out,
                                               tokenizer=tokenizer, **overrides)
            _log.info("wrote %s (%d rows)", csv_path, len(rows))
        except Exception as exc:
            had_failure = True
            _log.exception("analysis failed for %s: %s", tag, exc)
        finally:
            del tokenizer, model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    return 1 if had_failure else 0


def _safe_tag(name: str, *, identity: str | None = None) -> str:
    raw = str(name)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_.-") or "model"
    # Human-readable names are not identities: even simple tags receive the
    # digest of the loader's resolved snapshot path.
    digest = hashlib.sha256(str(identity or raw).encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}"
