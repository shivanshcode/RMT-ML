"""rmt.pipeline — end-to-end per-model analysis.

``analyze_one_model`` discovers analyzable matrices, optionally captures
activation covariances, runs the single-SVD ``per_matrix_analysis`` on each
record, writes the CSV / summary JSON / plots, and (optionally) runs the
perplexity-vs-decile ablation and the WeightWatcher baseline.

Epoch tracking (``analyze_checkpoints``) probes a settable list of layers across
training checkpoints (every ~10% of iterations) and writes a per-epoch
stable-rank CSV.  torch is imported lazily.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional

import numpy as np

from .config import RunConfig, get_logger, OfflineGuard
from .discovery import (get_model_spec, discover_weight_matrices,
                        discover_weight_metadata, materialize_record)
from .per_matrix import per_matrix_analysis, CSV_COLUMNS

_log = get_logger("rmt.pipeline")


def analyze_one_model(model, model_tag, output_dir, *, tokenizer=None,
                      **cfg_flags) -> Tuple[str, List[dict]]:
    """Run in a fresh owned directory and finalize status after every stage."""
    _acquire_output_directory(output_dir)
    status_path = os.path.join(output_dir, f"{model_tag}_run_status.json")
    _write_json(status_path, {"status": "running", "failures": []})
    try:
        csv_path, rows, failures = _analyze_one_model(
            model, model_tag, output_dir, tokenizer=tokenizer, **cfg_flags)
        state = "complete" if not failures else "partial"
        allow_tokenizer = bool(cfg_flags.get("allow_fallback_tokenizer", False))
        text_path = cfg_flags.get("text_path", RunConfig().text_path)
        provenance = {
            "tokenizer_source": ("real" if tokenizer is not None else
                                 ("synthetic_hash_opt_in" if allow_tokenizer else "unavailable")),
            "tokenizer_identity": (None if tokenizer is None else
                                   getattr(tokenizer, "name_or_path", type(tokenizer).__name__)),
            "text_source": ("file" if text_path and os.path.isfile(text_path)
                            else ("fallback_opt_in" if cfg_flags.get("allow_fallback_text", False)
                                  else "unavailable")),
            "text_path": text_path,
        }
        _write_json(status_path, {"status": state, "usable_matrices": len(rows),
                                  "failures": failures, "provenance": provenance})
        return csv_path, rows
    except BaseException as exc:
        current = _read_status(status_path)
        failures = list(current.get("failures", []))
        failures.append({"stage": "run", "error": repr(exc)})
        _write_json(status_path, {"status": "failed", "failures": failures})
        raise


def _analyze_one_model(model, model_tag, output_dir, *, tokenizer=None,
                       **cfg_flags):
    """Analyze one model → (csv_path, rows, failures).

    ``model`` may be an in-process module (tests) or a loaded local snapshot.
    ``tokenizer`` is used for text passes; synthetic token IDs require the
    explicit ``allow_fallback_tokenizer`` test-only option.
    Extra keyword flags override :class:`RunConfig` defaults.
    """
    cfg = RunConfig(**{k: v for k, v in cfg_flags.items() if hasattr(RunConfig, k)
                       or k in RunConfig().__dict__})
    if cfg.offline:
        OfflineGuard.enable()

    spec = get_model_spec(model)
    layer_filter = cfg.layers if cfg.layers else None
    # REPORT §0: materialise weights at float64 (the discovery default), not the
    # old hardcoded float32 which dropped the smallest singular values to noise.
    records = discover_weight_metadata(model, layer_indices=layer_filter, spec=spec)
    _log.info("[%s] discovered %d matrices", model_tag, len(records))
    if not records:
        raise ValueError("no analyzable matrices matched the model/layer selection")

    failures = []
    # Activation covariances and overlap heatmaps are projection-at-a-time.
    # A shared plan makes OOM-shortened windows a run-wide contract.  If a later
    # projection tightens it, discard/replay earlier rows rather than comparing
    # covariances captured from different token windows.
    activation_plan = {"lengths": None, "revision": 0}
    restart_count = 0
    while True:
        rows = []
        svals = {}            # singular values are small O(min(n,m)) plot inputs
        restart = False
        pass_revision = int(activation_plan["revision"])
        for rec in records:
            live = None
            try:
                live = materialize_record(model, rec, spec=spec)
                fm_dict = (_maybe_activation_cov(
                    model, [rec], cfg, spec, tokenizer, failures=failures,
                    window_plan=activation_plan)
                    if cfg.do_overlap else None)
                if rows and int(activation_plan["revision"]) != pass_revision:
                    restart = True
                    break
                local_overlap = {}
                row = per_matrix_analysis(live, fm_dict=fm_dict, cfg=cfg,
                                          svals_out=svals, ovmat_out=local_overlap)
                rows.append(row)
                if cfg.do_overlap and local_overlap:
                    _emit_overlap_plots(output_dir, local_overlap, cfg, failures)
            except Exception as e:                              # pragma: no cover
                failures.append({"matrix": rec.name, "stage": "per_matrix", "error": repr(e)})
                _log.warning("per_matrix failed for %s: %s", rec.name, e)
            finally:
                if live is not None:
                    live.weight = None
        if not restart:
            break
        restart_count += 1
        if restart_count > max(1, len(records) * cfg.max_oom):
            raise RuntimeError("activation window plan did not stabilize")
        _log.warning("activation window plan shortened; replaying all matrix analyses")

    if not rows:
        _write_json(os.path.join(output_dir, f"{model_tag}_run_status.json"),
                    {"status": "failed", "usable_matrices": 0,
                     "failures": failures})
        raise RuntimeError("all discovered matrices failed analysis")
    csv_path = os.path.join(output_dir, f"{model_tag}_matrix_metrics.csv")
    _write_csv(csv_path, rows, n_deciles=cfg.n_deciles)
    _write_summary(os.path.join(output_dir, f"{model_tag}_summary.json"), rows, model_tag)
    if failures and cfg.strict:
        _write_json(os.path.join(output_dir, f"{model_tag}_run_status.json"),
                    {"status": "failed", "usable_matrices": len(rows),
                     "failures": failures})
        raise RuntimeError(f"strict analysis failed for {len(failures)} matrix/stage(s)")
    _maybe_plots(output_dir, model_tag, rows, records, cfg, svals=svals,
                 ovmats=None, failures=failures)

    # optional perplexity-vs-decile ablation
    if cfg.do_perplexity:
        _maybe_perplexity(model, records, output_dir, model_tag, cfg, spec, tokenizer,
                          failures=failures)

    # Optional WeightWatcher remains best-effort, but requested/unavailable is
    # explicit and makes the overall run partial rather than silently complete.
    if cfg.do_ww:
        baseline_status = {"stage": "weightwatcher", "status": "unavailable"}
        try:
            from .baselines import run_weightwatcher
            ww = run_weightwatcher(model, normalize=cfg.ww_normalize,
                                   glorot_fix=cfg.ww_glorot_fix)
            if ww is not None and ww.get("status", "complete") == "complete":
                _write_json(
                    os.path.join(output_dir, f"{model_tag}_weightwatcher.json"),
                    ww.get("summary", {}),
                )
                baseline_status = {"stage": "weightwatcher", "status": "complete"}
            else:
                baseline_status.update({
                    "status": "unavailable" if ww is None else ww.get("status", "unavailable"),
                    "reason": "adapter returned no result" if ww is None else ww.get("reason", "unspecified"),
                })
                failures.append(dict(baseline_status))
        except Exception as e:                                  # pragma: no cover
            baseline_status.update({"status": "failed", "reason": repr(e)})
            failures.append(dict(baseline_status))
            _log.warning("weightwatcher baseline failed: %s", e)
        _write_json(
            os.path.join(output_dir, f"{model_tag}_weightwatcher_status.json"),
            baseline_status,
        )

    return csv_path, rows, failures


def analyze_checkpoints(checkpoint_loader, checkpoint_fracs, probe_layers,
                        output_dir, model_tag, **cfg_flags) -> str:
    """Track stable rank of ``probe_layers`` across training checkpoints.

    ``checkpoint_loader(frac)`` returns a model at that training fraction.
    ``checkpoint_fracs`` are e.g. [0.0, 0.1, ..., 1.0] (every ~10% of iters).
    Writes ``<tag>_stable_rank_per_epoch.csv`` and returns its path.
    """
    cfg = RunConfig(**{k: v for k, v in cfg_flags.items()
                       if k in RunConfig().__dict__})
    os.makedirs(output_dir, exist_ok=True)
    srk_by_layer: Dict[int, List[float]] = {L: [] for L in probe_layers}
    backend_by_layer: Dict[int, List[str]] = {L: [] for L in probe_layers}
    dtype_by_layer: Dict[int, List[str]] = {L: [] for L in probe_layers}
    for frac in checkpoint_fracs:
        model = checkpoint_loader(frac)
        spec = get_model_spec(model)
        recs = discover_weight_metadata(model, layer_indices=list(probe_layers),
                                         spec=spec)
        from .scalars import stable_rank
        from .linalg import cached_svd
        per_layer: Dict[int, List[float]] = {L: [] for L in probe_layers}
        per_layer_backends: Dict[int, set] = {L: set() for L in probe_layers}
        per_layer_dtypes: Dict[int, set] = {L: set() for L in probe_layers}
        for meta in recs:
            if meta.layer_idx in per_layer:
                r = materialize_record(model, meta, spec=spec)
                factorization = cached_svd(
                    np.asarray(r.weight, float), full_matrices=False,
                    backend=cfg.backend, gpu_min_dim=cfg.gpu_svd_min_dim,
                )
                if factorization.degraded and cfg.strict:
                    raise RuntimeError("checkpoint SVD precision contract was not satisfied")
                per_layer[r.layer_idx].append(stable_rank(s=factorization.s))
                per_layer_backends[r.layer_idx].add(factorization.backend)
                per_layer_dtypes[r.layer_idx].add(factorization.factorization_dtype)
                r.weight = None
        for L in probe_layers:
            vals = per_layer.get(L, [])
            srk_by_layer[L].append(float(np.mean(vals)) if vals else float("nan"))
            backend_by_layer[L].append("|".join(sorted(per_layer_backends[L])) or "unavailable")
            dtype_by_layer[L].append("|".join(sorted(per_layer_dtypes[L])) or "unavailable")
        del recs, model

    path = os.path.join(output_dir, f"{model_tag}_stable_rank_per_epoch.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["training_fraction"]
            + [f"layer_{L}_stable_rank" for L in probe_layers]
            + [f"layer_{L}_svd_backend" for L in probe_layers]
            + [f"layer_{L}_svd_dtype" for L in probe_layers]
        )
        for i, frac in enumerate(checkpoint_fracs):
            w.writerow(
                [frac]
                + [srk_by_layer[L][i] for L in probe_layers]
                + [backend_by_layer[L][i] for L in probe_layers]
                + [dtype_by_layer[L][i] for L in probe_layers]
            )
    return path


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _model_device(model):
    """The device the model's parameters actually live on (not an assumption)."""
    try:
        import torch
        return next(model.parameters()).device
    except StopIteration:                                       # pragma: no cover
        return "cpu"


def _maybe_activation_cov(model, records, cfg, spec, tokenizer=None, failures=None,
                          window_plan=None):
    try:
        from .activations import compute_activation_covariance
        device = _model_device(model)
        numeric_layers = sorted({r.layer_idx for r in records if r.layer_idx >= 0})
        # Exact names are authoritative.  If any selected target has no parsed
        # numeric index, omitting the numeric filter keeps discovery and capture
        # consistent instead of turning it into an empty allowed set.
        layer_idxs = (None if any(r.layer_idx < 0 for r in records)
                      else numeric_layers)
        target_names = sorted({
            r.name.split("[", 1)[0].removesuffix(".weight") for r in records
        })
        return compute_activation_covariance(
            model, tokenizer=tokenizer, layer_indices=layer_idxs, device=device,
            dataset_name=cfg.fm_dataset, n_text_batches=cfg.n_text_batches,
            max_length=cfg.fm_max_length, stride=cfg.fm_stride,
            max_oom=cfg.max_oom, token_weighted=cfg.fm_token_weighted, spec=spec,
            text_path=cfg.text_path, allow_fallback=cfg.allow_fallback_text,
            allow_tokenizer_fallback=cfg.allow_fallback_tokenizer,
            target_names=target_names, window_plan=window_plan)
    except Exception as e:                                      # pragma: no cover
        if failures is not None:
            failures.append({"stage": "activation", "error": repr(e)})
        _log.warning("activation covariance failed (%s); overlap will be unavailable", e)
        if cfg.strict:
            raise
        return None


def _maybe_perplexity(model, records, output_dir, model_tag, cfg, spec, tokenizer=None,
                      failures=None):
    try:
        from .decile import perplexity_vs_decile
        device = _model_device(model)
        res = perplexity_vs_decile(lambda: model, tokenizer, records, device,
                                   n_tokens=cfg.perplexity_tokens,
                                   decile_scope=cfg.decile_scope,
                                   n_deciles=cfg.n_deciles, spec=spec,
                                   text_path=cfg.text_path, stride=cfg.ppl_stride,
                                   backend=cfg.backend,
                                   gpu_min_dim=cfg.gpu_svd_min_dim,
                                   allow_fallback=cfg.allow_fallback_text,
                                   allow_tokenizer_fallback=cfg.allow_fallback_tokenizer)
        # REPORT §3 A8: a missing text file silently falls back to a repeated
        # pangram and still returns a *finite* perplexity. Make that explicit so
        # the numbers are never mistaken for real wikitext perplexity.
        have_text = bool(cfg.text_path) and os.path.exists(cfg.text_path)
        res["text_source"] = "file" if have_text else "fallback"
        res["text_path"] = cfg.text_path
        if not have_text:
            _log.warning("perplexity text file %r not found — using the offline "
                         "fallback corpus; perplexity numbers are NOT wikitext "
                         "(tagged text_source=fallback)", cfg.text_path)
        res["tokenizer_source"] = ("real" if tokenizer is not None
                                   else "synthetic_hash_opt_in")
        _write_json(os.path.join(output_dir, f"{model_tag}_perplexity.json"), res)
        # the decile-vs-perplexity plot (REPORT §2 plot wiring)
        try:
            from .plots import plot_perplexity_vs_decile
            plot_perplexity_vs_decile(
                res, os.path.join(output_dir, "perplexity",
                                  f"{model_tag}_perplexity_vs_decile.png"))
        except Exception as pe:                                 # pragma: no cover
            if failures is not None:
                failures.append({"stage": "perplexity_plot", "error": repr(pe)})
            _log.warning("perplexity plot skipped: %s", pe)
            if cfg.strict:
                raise
    except Exception as e:                                      # pragma: no cover
        if failures is not None:
            failures.append({"stage": "perplexity", "error": repr(e)})
        _log.warning("perplexity-vs-decile failed: %s", e)
        if cfg.strict:
            _write_json(os.path.join(output_dir, f"{model_tag}_run_status.json"),
                        {"status": "failed", "failures": list(failures or [])})
            raise


def _acquire_output_directory(output_dir):
    """Atomically claim an absent or intentionally pre-created empty directory."""

    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        raise FileExistsError(
            f"output directory is not empty: {output_dir}; choose a fresh run directory"
        )
    owner = os.path.join(output_dir, ".run-owner.json")
    try:
        descriptor = os.open(owner, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"output directory is already owned: {output_dir}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "acquired_at_utc": datetime.now(timezone.utc).isoformat()}, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path, value):
    """Atomically write JSON; unavailable numeric values become null."""
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
            return None
        if isinstance(obj, np.integer):
            return int(obj)
        return obj
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".json-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as f:
            json.dump(clean(value), f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_status(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_plot_name(name: str) -> str:
    """Filesystem-safe stem for a matrix name (handles dots/slashes/brackets)."""
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch in "-_") else "_")
    stem = "".join(out).strip("_")
    return stem or "matrix"


def _emit_overlap_plots(output_dir, matrices, cfg, failures=None):
    """Write and release each dense overlap artifact immediately."""
    from .plots import plot_overlap_heatmap
    for name, matrix in matrices.items():
        try:
            plot_overlap_heatmap(
                matrix, os.path.join(output_dir, "overlap", f"{_safe_plot_name(name)}.png"))
        except Exception as error:                              # pragma: no cover
            if failures is not None:
                failures.append({"stage": "overlap_plot", "matrix": name,
                                 "error": repr(error)})
            _log.warning("overlap plot skipped for %s: %s", name, error)
            if cfg.strict:
                raise


def _maybe_plots(output_dir, model_tag, rows, records, cfg, *, svals=None,
                 ovmats=None, failures=None):
    """Emit the per-matrix plots the analyses imply (REPORT §2 plot wiring).

    Previously only ``summary.png`` was produced; ``per_matrix_analysis`` discarded
    the singular-value arrays so ESD/Hill/spacing/QKV/overlap plots never ran. We
    now stash the (small) singular values + overlap matrices during analysis and
    draw from them here — no second SVD, weights already released.
    """
    svals = svals or {}
    ovmats = ovmats or {}

    def failed(stage, error):
        if failures is not None:
            failures.append({"stage": stage, "error": repr(error)})
        if cfg.strict:
            raise error
    try:
        from .plots import plot_model_summary
        plot_model_summary(rows, os.path.join(output_dir, f"{model_tag}_summary.png"))
    except Exception as e:                                      # pragma: no cover
        _log.warning("summary plot skipped: %s", e)
        failed("summary_plot", e)

    try:
        from .plots import (plot_esd, plot_hill, plot_nn_spacing,
                            plot_qkv_heatmap, plot_overlap_heatmap)
    except Exception as e:                                      # pragma: no cover
        _log.warning("per-matrix plots unavailable: %s", e)
        failed("plot_import", e)
        return

    row_by_name = {r.get("name"): r for r in rows}
    qkv_by_layer: Dict[int, Dict[str, np.ndarray]] = {}

    for name, s in svals.items():
        r = row_by_name.get(name, {})
        stem = _safe_plot_name(name)
        n = int(r.get("n", len(s)))
        m = int(r.get("m", len(s)))
        # Use the same fit serialized by the row's MP edges/counts.
        sigma = r.get("sigma_med", float("nan"))
        N_cov = r.get("N_cov", m)
        # ESD (always, when we have the spectrum). The MP curve is rescaled by
        # the count fraction inside the MP support *inside* plot_esd; we no
        # longer pass the energy-based bulk_mass_frac (which crushed the curve
        # on heavy-tailed spectra).
        try:
            plot_esd(s, n, m, sigma,
                     os.path.join(output_dir, "esd", f"{stem}.png"),
                     domain="nu", N=N_cov)
        except Exception as e:                                  # pragma: no cover
            _log.warning("esd plot skipped for %s: %s", name, e)
            failed("esd_plot", e)
        # Hill (gated on do_powerlaw)
        if cfg.do_powerlaw:
            try:
                levels = (s ** 2) / float(N_cov)
                plot_hill(levels, os.path.join(output_dir, "hill", f"{stem}.png"),
                          window=cfg.hill_window)
            except Exception as e:                              # pragma: no cover
                _log.warning("hill plot skipped for %s: %s", name, e)
                failed("hill_plot", e)
        # NN spacing (gated on do_spacing and enough levels)
        if cfg.do_spacing and len(s) >= 50 and r.get("spacing_available") == 1:
            try:
                levels = (s ** 2) / float(N_cov)
                levels = levels[(levels >= float(r.get("mp_minus_eig", -np.inf)))
                                & (levels <= float(r.get("mp_plus_eig", np.inf)))]
                if levels.size >= 3:
                    plot_nn_spacing(levels,
                                    os.path.join(output_dir, "spacing", f"{stem}.png"),
                                    deg=cfg.unfold_deg)
            except Exception as e:                              # pragma: no cover
                _log.warning("spacing plot skipped for %s: %s", name, e)
                failed("spacing_plot", e)
        # collect Q/K/V spectra per layer for the QKV heatmap
        short = r.get("short")
        li = r.get("layer_idx", -1)
        if short in ("Q", "K", "V"):
            qkv_by_layer.setdefault(int(li), {})[short] = s

    # QKV heatmap (gated on do_qkv_heatmap) — one per layer that has Q/K/V
    if cfg.do_qkv_heatmap:
        for li, by_tag in qkv_by_layer.items():
            if not by_tag:
                continue
            try:
                plot_qkv_heatmap(by_tag,
                                 os.path.join(output_dir, "qkv",
                                              f"layer{li:02d}_qkv.png"))
            except Exception as e:                              # pragma: no cover
                _log.warning("qkv heatmap skipped for layer %s: %s", li, e)
                failed("qkv_plot", e)

    # overlap heatmaps (gated on do_overlap) — one per overlap matrix
    if cfg.do_overlap:
        for name, mat in ovmats.items():
            try:
                plot_overlap_heatmap(
                    mat, os.path.join(output_dir, "overlap",
                                      f"{_safe_plot_name(name)}.png"))
            except Exception as e:                              # pragma: no cover
                _log.warning("overlap heatmap skipped for %s: %s", name, e)
                failed("overlap_plot", e)


def _write_csv(path, rows, n_deciles=10):
    groups = int(n_deciles)
    if groups < 1:
        raise ValueError("n_deciles must be positive")
    base = [name for name in CSV_COLUMNS
            if not name.startswith("entropy_decile_") and not name.startswith("srk_decile_")]
    columns = (base + [f"entropy_decile_{i}" for i in range(1, groups + 1)]
               + [f"srk_decile_{i}" for i in range(1, groups + 1)])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def _write_summary(path, rows, tag):
    def _col(name):
        vals = [r[name] for r in rows if name in r and _isfinite(r[name])]
        return vals
    summary = {"model_tag": tag, "n_matrices": len(rows)}
    for name in ("alpha", "stable_rank", "spectral_entropy", "mp_softrank",
                 "sigma_med", "bulk_mass_frac"):
        vals = _col(name)
        if vals:
            summary[f"{name}_mean"] = float(np.mean(vals))
            summary[f"{name}_median"] = float(np.median(vals))
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


def _isfinite(x):
    try:
        return np.isfinite(float(x))
    except Exception:
        return False
