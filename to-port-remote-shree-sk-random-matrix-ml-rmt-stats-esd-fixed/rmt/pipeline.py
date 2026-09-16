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
from typing import List, Tuple, Dict, Optional

import numpy as np

from .config import RunConfig, get_logger, OfflineGuard
from .discovery import get_model_spec, discover_weight_matrices
from .per_matrix import per_matrix_analysis, CSV_COLUMNS

_log = get_logger("rmt.pipeline")


def analyze_one_model(model, model_tag, output_dir, *, tokenizer=None,
                      **cfg_flags) -> Tuple[str, List[dict]]:
    """Analyze one model → (csv_path, rows).

    ``model`` may be an in-process module (tests) or a loaded local snapshot.
    ``tokenizer`` (optional) is used for the activation-covariance and
    perplexity text passes; if None a deterministic offline fallback is used.
    Extra keyword flags override :class:`RunConfig` defaults.
    """
    cfg = RunConfig(**{k: v for k, v in cfg_flags.items() if hasattr(RunConfig, k)
                       or k in RunConfig().__dict__})
    if cfg.offline:
        OfflineGuard.enable()

    os.makedirs(output_dir, exist_ok=True)
    spec = get_model_spec(model)
    layer_filter = cfg.layers if cfg.layers else None
    # REPORT §0: materialise weights at float64 (the discovery default), not the
    # old hardcoded float32 which dropped the smallest singular values to noise.
    records = discover_weight_matrices(model, layer_indices=layer_filter, spec=spec)
    _log.info("[%s] discovered %d matrices", model_tag, len(records))

    # optional activation covariance (overlap / coincidence block)
    fm_dict = None
    if cfg.do_overlap:
        fm_dict = _maybe_activation_cov(model, records, cfg, spec, tokenizer)

    rows = []
    svals = {}            # name -> descending singular values (for the plots)
    ovmats = {}           # name -> overlap matrix (for the overlap heatmaps)
    ptvals = {}           # name -> (svals, Porter-Thomas p-values) per vector
    for rec in records:
        try:
            row = per_matrix_analysis(rec, fm_dict=fm_dict, cfg=cfg,
                                      svals_out=svals, ovmat_out=ovmats,
                                      ptvals_out=ptvals)
            rows.append(row)
        except Exception as e:                                  # pragma: no cover
            _log.warning("per_matrix failed for %s: %s", rec.name, e)
        finally:
            # release the materialized numpy weight; per_matrix already used it
            # and the plots read the (small) stashed singular values instead.
            rec.weight = None

    csv_path = os.path.join(output_dir, f"{model_tag}_matrix_metrics.csv")
    _write_csv(csv_path, rows)
    _write_summary(os.path.join(output_dir, f"{model_tag}_summary.json"), rows, model_tag)
    _maybe_plots(output_dir, model_tag, rows, records, cfg, svals=svals,
                 ovmats=ovmats, ptvals=ptvals)

    # optional perplexity-vs-decile ablation
    if cfg.do_perplexity:
        _maybe_perplexity(model, records, output_dir, model_tag, cfg, spec, tokenizer)

    # optional WeightWatcher baseline (best-effort)
    if cfg.do_ww:
        try:
            from .baselines import run_weightwatcher
            ww = run_weightwatcher(model, normalize=cfg.ww_normalize,
                                   glorot_fix=cfg.ww_glorot_fix)
            if ww is not None:
                with open(os.path.join(output_dir, f"{model_tag}_weightwatcher.json"),
                          "w") as f:
                    json.dump(ww.get("summary", {}), f, indent=2, default=str)
        except Exception as e:                                  # pragma: no cover
            _log.warning("weightwatcher baseline skipped: %s", e)

    return csv_path, rows


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
    for frac in checkpoint_fracs:
        model = checkpoint_loader(frac)
        spec = get_model_spec(model)
        recs = discover_weight_matrices(model, layer_indices=list(probe_layers),
                                        spec=spec)
        from .scalars import stable_rank
        per_layer: Dict[int, List[float]] = {L: [] for L in probe_layers}
        for r in recs:
            if r.layer_idx in per_layer:
                per_layer[r.layer_idx].append(stable_rank(s=np.linalg.svd(
                    np.asarray(r.weight, float), compute_uv=False)))
        for L in probe_layers:
            vals = per_layer.get(L, [])
            srk_by_layer[L].append(float(np.mean(vals)) if vals else float("nan"))

    path = os.path.join(output_dir, f"{model_tag}_stable_rank_per_epoch.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["training_fraction"] + [f"layer_{L}_stable_rank" for L in probe_layers])
        for i, frac in enumerate(checkpoint_fracs):
            w.writerow([frac] + [srk_by_layer[L][i] for L in probe_layers])
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


def _maybe_activation_cov(model, records, cfg, spec, tokenizer=None):
    try:
        from .activations import compute_activation_covariance
        device = _model_device(model)
        layer_idxs = sorted({r.layer_idx for r in records if r.layer_idx >= 0})
        return compute_activation_covariance(
            model, tokenizer=tokenizer, layer_indices=layer_idxs, device=device,
            dataset_name=cfg.fm_dataset, n_text_batches=cfg.n_text_batches,
            max_length=cfg.fm_max_length, stride=cfg.fm_stride,
            max_oom=cfg.max_oom, token_weighted=cfg.fm_token_weighted, spec=spec,
            text_path=cfg.text_path)
    except Exception as e:                                      # pragma: no cover
        _log.warning("activation covariance skipped (%s); overlap will be NaN", e)
        return None


def _maybe_perplexity(model, records, output_dir, model_tag, cfg, spec, tokenizer=None):
    try:
        from .decile import perplexity_vs_decile
        device = _model_device(model)
        res = perplexity_vs_decile(lambda: model, tokenizer, records, device,
                                   n_tokens=cfg.perplexity_tokens,
                                   decile_scope=cfg.decile_scope,
                                   n_deciles=cfg.n_deciles, spec=spec,
                                   text_path=cfg.text_path)
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
        with open(os.path.join(output_dir, f"{model_tag}_perplexity.json"), "w") as f:
            json.dump(res, f, indent=2)
        # the decile-vs-perplexity plot (REPORT §2 plot wiring)
        try:
            from .plots import plot_perplexity_vs_decile
            plot_perplexity_vs_decile(
                res, os.path.join(output_dir, "perplexity",
                                  f"{model_tag}_perplexity_vs_decile.png"))
        except Exception as pe:                                 # pragma: no cover
            _log.warning("perplexity plot skipped: %s", pe)
    except Exception as e:                                      # pragma: no cover
        _log.warning("perplexity-vs-decile skipped: %s", e)


def _safe_plot_name(name: str) -> str:
    """Filesystem-safe stem for a matrix name (handles dots/slashes/brackets)."""
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch in "-_") else "_")
    stem = "".join(out).strip("_")
    return stem or "matrix"


def _maybe_plots(output_dir, model_tag, rows, records, cfg, *, svals=None,
                 ovmats=None, ptvals=None):
    """Emit the per-matrix plots the analyses imply (REPORT §2 plot wiring).

    Previously only ``summary.png`` was produced; ``per_matrix_analysis`` discarded
    the singular-value arrays so ESD/Hill/spacing/QKV/overlap plots never ran. We
    now stash the (small) singular values + overlap matrices during analysis and
    draw from them here — no second SVD, weights already released.
    """
    svals = svals or {}
    ovmats = ovmats or {}
    ptvals = ptvals or {}
    try:
        from .plots import plot_model_summary
        plot_model_summary(rows, os.path.join(output_dir, f"{model_tag}_summary.png"))
    except Exception as e:                                      # pragma: no cover
        _log.warning("summary plot skipped: %s", e)

    try:
        from .plots import (plot_esd, plot_hill, plot_nn_spacing,
                            plot_rigidity, plot_porter_thomas,
                            plot_porter_thomas_deciles, plot_qkv_heatmap,
                            plot_overlap_heatmap)
    except Exception as e:                                      # pragma: no cover
        _log.warning("per-matrix plots unavailable: %s", e)
        return

    row_by_name = {r.get("name"): r for r in rows}
    qkv_by_layer: Dict[int, Dict[str, np.ndarray]] = {}

    for name, s in svals.items():
        r = row_by_name.get(name, {})
        stem = _safe_plot_name(name)
        n = int(r.get("n", len(s)))
        m = int(r.get("m", len(s)))
        # Secondary fix: feed the outlier-trimmed σ to the overlay so the right
        # edge ν₊ is fit to the noise bulk, not to a spectrum that includes the
        # large outliers. Fall back to sigma_med if the refined value is absent
        # or non-finite.
        sigma = r.get("sigma_med_refined", float("nan"))
        if not _isfinite(sigma):
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
        # Hill (gated on do_powerlaw)
        if cfg.do_powerlaw:
            try:
                plot_hill(s, os.path.join(output_dir, "hill", f"{stem}.png"),
                          window=cfg.hill_window)
            except Exception as e:                              # pragma: no cover
                _log.warning("hill plot skipped for %s: %s", name, e)
        # NN spacing (gated on do_spacing and enough levels)
        # NN spacing + rigidity. Feed the *singular values* (cfg.spacing_domain
        # 'sval', Thamm's convention) rather than lambda = s^2/N: unfolding makes
        # the two equivalent, but s is far better conditioned numerically.
        levels = s if cfg.spacing_domain == "sval" else (s ** 2) / float(m)
        if cfg.do_spacing and len(levels) >= max(50, 4 * cfg.unfold_win + 4):
            try:
                plot_nn_spacing(levels,
                                os.path.join(output_dir, "spacing", f"{stem}.png"),
                                deg=cfg.unfold_deg, method=cfg.unfold_method,
                                win_size=cfg.unfold_win, title=name)
            except Exception as e:                              # pragma: no cover
                _log.warning("spacing plot skipped for %s: %s", name, e)
            try:
                plot_rigidity(levels,
                              os.path.join(output_dir, "rigidity", f"{stem}.png"),
                              deg=cfg.unfold_deg, method=cfg.unfold_method,
                              title=name)
            except Exception as e:                              # pragma: no cover
                _log.warning("rigidity plot skipped for %s: %s", name, e)
        # Porter-Thomas: per-vector and per-decile, from the p-values already
        # computed in per_matrix_analysis (no second SVD, no Vh kept alive).
        if cfg.do_porter_thomas and name in ptvals:
            pt_s, pt_p = ptvals[name]
            try:
                plot_porter_thomas(pt_s, pt_p,
                                   os.path.join(output_dir, "porter_thomas",
                                                f"{stem}.png"),
                                   alpha=cfg.pt_alpha, title=name)
            except Exception as e:                              # pragma: no cover
                _log.warning("porter-thomas plot skipped for %s: %s", name, e)
            try:
                plot_porter_thomas_deciles(
                    pt_p, os.path.join(output_dir, "porter_thomas_decile",
                                       f"{stem}.png"),
                    alpha=cfg.pt_alpha, n_deciles=cfg.n_deciles, title=name)
            except Exception as e:                              # pragma: no cover
                _log.warning("pt-decile plot skipped for %s: %s", name, e)
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

    # overlap heatmaps (gated on do_overlap) — one per overlap matrix
    if cfg.do_overlap:
        for name, mat in ovmats.items():
            try:
                plot_overlap_heatmap(
                    mat, os.path.join(output_dir, "overlap",
                                      f"{_safe_plot_name(name)}.png"))
            except Exception as e:                              # pragma: no cover
                _log.warning("overlap heatmap skipped for %s: %s", name, e)


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


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
