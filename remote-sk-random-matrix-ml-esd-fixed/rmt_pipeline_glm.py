# Archived legacy reference script.  Kept parseable for tooling; the production
# entry point is ``python -m rmt``.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 rmt_analysis_v2.py
 Random-Matrix-Theory analysis of LLM weight matrices
================================================================================

This script combines and supersedes:

  (A) The American "WeightWatcher" pipeline (Charles Martin / Michael Mahoney)
      which fits a power-law to the empirical spectral density (ESD) of every
      weight matrix and reports the tail exponent `alpha`, with the
      interpretation  alpha < 2  ->  heavy-tailed ->  well-trained.

  (B) The German pipeline from
        Staats, Thamm, Rosenow,
        "Small Singular Values Matter: A Random Matrix Analysis of Transformer
         Models", NeurIPS 2025  (arXiv:2410.17770v3)
      and its predecessor
        Thamm, Staats, Rosenow,
        "Random matrix analysis of deep neural network weight matrices",
        Phys. Rev. E 106, 054124 (2022).
      The Germans argue the spectrum is  MP-bulk + outliers  (NOT a power law).

  (C) Additional analyses requested by the supervisor:
        * Heat-maps of the Query / Key / Value weight matrices.
        * Hill estimator of the upper-tail index (used in the 2022 paper to
          test the American power-law claim on a per-layer basis).
        * Row-wise average entropy of every weight matrix.
        * Stable rank of every weight matrix (and, if Pythia-style training
          checkpoints are passed, per-epoch stable rank).
        * Eigenvector-eigenvalue coincidence: does the eigenvector with the
          largest eigenvalue (of the activation covariance) coincide with the
          singular vector carrying the largest singular value?  Plotted both
          as a heat-map (Fig. 15 of the German paper) and as a scalar
          rank-correlation score per layer.

The script is designed to run on a free Google Colab T4 (15 GB GPU) and
analyses several small / mid-size LLMs in one pass.

--------------------------------------------------------------------------------
HOW TO RUN ON GOOGLE COLAB
--------------------------------------------------------------------------------

1.  Open a new Colab notebook, set runtime -> T4 GPU.
2.  In the first cell install everything:

        !pip -q install transformers accelerate weightwatcher torch datasets
        !pip -q install powerlaw scipy matplotlib pandas seaborn

3.  Upload this file (rmt_analysis_v2.py) to Colab, OR mount your Google Drive
    and point to it.
4.  Run the analysis for a single small model:

        !python rmt_analysis_v2.py \
            --models EleutherAI/pythia-410m \
            --output_dir ./rmt_results

    Multiple models in one go:

        !python rmt_analysis_v2.py \
            --models EleutherAI/pythia-410m \
                      EleutherAI/pythia-1b \
                      Qwen/Qwen3-4B-Instruct-2507 \
                      meta-llama/Llama-3.2-1B \
            --output_dir ./rmt_results \
            --layers 0 4 9 14 19 \
            --n_text_batches 5 \
            --do_perplexity \
            --do_ww

    Notes
    -----
    * On a T4, ~4 B-parameter models in bfloat16 fit comfortably.  7 B models
      also work in bfloat16 but you should pass `--max_text_tokens 2048` and
      reduce `--n_text_batches` to 2-3.
    * Use `--do_perplexity` ONLY if you have time: it requires 10 forward
      passes per matrix type per model (one per decile).
    * Use `--do_ww` to ALSO run the American WeightWatcher baseline so the
      two methodologies produce side-by-side figures.  WeightWatcher itself
      is optional and can be slow on >3B models.
    * For per-epoch stable rank on Pythia, pass
      `--pythia_steps 1000 20000 143000` and the script will load each
      checkpoint and compute stable rank on the same layer set.

================================================================================
"""

from __future__ import annotations

# ----------------------------------------------------------------------
#  Standard library / numerical imports
# ----------------------------------------------------------------------
import os
import sys
import argparse
import json
import math
import pickle
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Per-glyph font fallback so we can use Greek letters in figures
try:
    import matplotlib.font_manager as fm
    for _f in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]:
        if os.path.exists(_f):
            fm.fontManager.addfont(_f)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
import scipy.special as sp

# ----------------------------------------------------------------------
#  Torch / Hugging Face
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn.functional import linear

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)

# ----------------------------------------------------------------------
#  Optional WeightWatcher (American baseline)
# ----------------------------------------------------------------------
try:
    import weightwatcher as ww
    _HAS_WW = True
except Exception:
    _HAS_WW = False

# ----------------------------------------------------------------------
#  Plot style -- mimic the German paper's clean Phys.-Rev. look
# ----------------------------------------------------------------------
PLOT_COLORS = [
    "#0C5DA5",  # blue
    "#00B945",  # green
    "#FF9500",  # orange
    "#FF2C00",  # red
    "#845B97",  # purple
    "#474747",  # dark grey
    "#9e9e9e",  # light grey
]


def _set_plot_style():
    from matplotlib import cycler
    plt.rcParams.update({
        "axes.formatter.use_mathtext": True,
        "axes.linewidth": 1.0,
        "axes.prop_cycle": cycler("color", PLOT_COLORS),
        "figure.figsize": [6.6, 3.0],
        "font.family": ["DejaVu Sans"],
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "font.size": 10,
    })


_set_plot_style()

# ----------------------------------------------------------------------
#  Logging helper
# ----------------------------------------------------------------------
def log(msg: str):
    print(f"[rmt] {msg}", flush=True)


# ======================================================================
#  PART 1 -- Marchenko-Pastur theory
# ======================================================================
#  All formulas follow  Marcenko & Pastur (1967)  and  Gavish & Donoho
#  (2014, IEEE Trans. Inf. Theory 60(8):5040-5053)  and  Azadkia (2018,
#  arXiv:1801.10015)  -- exactly as used in the German paper code.
# ======================================================================

def mp_pdf(x: np.ndarray, n: int, m: int, sigma: float) -> np.ndarray:
    """
    Marchenko-Pastur density for an n x m matrix with i.i.d. entries of
    standard deviation `sigma`.

    Returns the density evaluated at `x`.  Outside the support the
    density is zero.
    """
    x = np.asarray(x, dtype=np.float64)
    larger = max(n, m)
    smaller = min(n, m)
    sigma_tilde = sigma * np.sqrt(larger)
    lam_plus = sigma_tilde * (1.0 + np.sqrt(smaller / larger))
    lam_minus = sigma_tilde * (1.0 - np.sqrt(smaller / larger))

    # density = sqrt((lam_plus^2 - x^2)(x^2 - lam_minus^2)) /
    #           (pi * sigma_tilde^2 * x) * (larger/smaller)
    denom = np.pi * sigma_tilde ** 2 * x
    num = np.sqrt(np.maximum(lam_plus ** 2 - x ** 2, 0.0) *
                  np.maximum(x ** 2 - lam_minus ** 2, 0.0))
    density = (larger / smaller) * num / np.where(denom == 0, np.inf, denom)
    density = np.where(x <= 0, 0.0, density)
    return density


def mp_bounds(n: int, m: int, sigma: float) -> Tuple[float, float]:
    """Return the (lambda_minus, lambda_plus) MP support bounds."""
    larger = max(n, m)
    smaller = min(n, m)
    sigma_tilde = sigma * np.sqrt(larger)
    lam_plus = sigma_tilde * (1.0 + np.sqrt(smaller / larger))
    lam_minus = sigma_tilde * (1.0 - np.sqrt(smaller / larger))
    return float(lam_minus), float(lam_plus)


# ----------------------------------------------------------------------
#  Median-of-squares  sigma estimator  (Gavish & Donoho 2014 / Azadkia 2018)
#  This is the estimator the German paper uses (function `estimate_sigma_med`
#  in their generate_figures.ipynb).
# ----------------------------------------------------------------------

def _mp_cdf_normalization(gamma: float, zs: np.ndarray, ws: np.ndarray) -> float:
    lam_minus = (1.0 - np.sqrt(gamma)) ** 2
    lam_plus = (1.0 + np.sqrt(gamma)) ** 2
    a, b = lam_minus, lam_plus
    ts = 0.5 * (b - a) * zs + 0.5 * (b + a)
    fs = np.sqrt(np.maximum(lam_plus - ts, 0.0) *
                 np.maximum(ts - lam_minus, 0.0)) / (2.0 * np.pi * gamma * ts)
    return 0.5 * (b - a) * (ws * fs).sum()


def _mp_cdf_at(x: float, gamma: float, zs: np.ndarray, ws: np.ndarray,
               Z: float) -> float:
    lam_minus = (1.0 - np.sqrt(gamma)) ** 2
    lam_plus = (1.0 + np.sqrt(gamma)) ** 2
    if x <= lam_minus:
        return 0.0
    if x >= lam_plus:
        return 1.0
    a, b = lam_minus, x
    ts = 0.5 * (b - a) * zs + 0.5 * (b + a)
    fs = np.sqrt(np.maximum(lam_plus - ts, 0.0) *
                 np.maximum(ts - lam_minus, 0.0)) / (2.0 * np.pi * gamma * ts)
    integral = 0.5 * (b - a) * (ws * fs).sum()
    return integral / Z


def _find_mp_median(gamma: float, n_leg: int = 128) -> float:
    zs, ws = leggauss(n_leg)
    Z = _mp_cdf_normalization(gamma, zs, ws)
    lam_minus = (1.0 - np.sqrt(gamma)) ** 2
    lam_plus = (1.0 + np.sqrt(gamma)) ** 2
    eps = 1e-6 * (lam_plus - lam_minus)
    return brentq(
        lambda x: _mp_cdf_at(x, gamma, zs, ws, Z) - 0.5,
        lam_minus + eps, lam_plus - eps,
    )


def estimate_sigma_med(weight: np.ndarray, discard_largest: float = 0.0) -> float:
    """
    sigma_med estimator: matches the median of the empirical squared
    singular values to the median of the MP distribution.
    """
    w = np.asarray(weight, dtype=np.float64)
    n, m = w.shape
    scaler = 1.0 / np.sqrt(max(n, m))
    gamma = min(n, m) / max(n, m)
    s = np.linalg.svd(w, compute_uv=False)
    n0 = int(discard_largest * len(s))
    med_s2 = float(np.median(s[n0:] ** 2))
    m_gamma = _find_mp_median(gamma)
    return float(np.sqrt(med_s2 / m_gamma) * scaler)


# ======================================================================
#  PART 2 -- Power-law / Hill estimator
# ======================================================================
#  Used in  Thamm, Staats, Rosenow, PRE 106, 054124 (2022)
#  (the predecessor of the German paper)  to test the American claim
#  that singular values are heavy-tailed.  If the spectrum really is
#  heavy-tailed with tail index alpha, then for the k largest order
#  statistics s_{(n-k+1)}, ..., s_{(n)} the Hill estimator
#
#        H_k = (1/k) * sum_{i=0}^{k-1}  log(s_{(n-i)}) - log(s_{(n-k)})
#
#  estimates  1 / alpha.   Plotted as a function of k it should be
#  roughly constant in the heavy-tail regime; under MP it is NOT.
# ======================================================================

def hill_estimator(svals: np.ndarray, k_min: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (ks, alpha_hat) where alpha_hat[k] = 1 / H_k for the upper
    tail of the singular-value spectrum.
    """
    s = np.sort(np.asarray(svals, dtype=np.float64))
    s = s[s > 0]
    n = len(s)
    if n < 2 * k_min:
        return np.array([]), np.array([])
    ks = np.arange(k_min, n)
    # log(s_{(n-i)}) for i = 0..k-1  -> the top k entries in reverse order
    log_s = np.log(s[::-1])               # length n, largest first
    alpha_hat = np.empty(ks.shape[0])
    for j, k in enumerate(ks):
        # H_k = mean(log_s[0:k]) - log_s[k]   (log_s[k] is s_{(n-k)})
        Hk = np.mean(log_s[:k]) - log_s[k]
        alpha_hat[j] = 1.0 / Hk if Hk > 0 else np.nan
    return ks, alpha_hat


def fit_powerlaw_mle(svals: np.ndarray) -> Tuple[float, float]:
    """
    Clauset-Shalizi-Newman MLE for a discrete (we treat s as continuous)
    power-law tail.  Returns (alpha, xmin).
    Used as a single-number American-style summary per matrix.
    """
    s = np.sort(np.asarray(svals, dtype=np.float64))
    s = s[s > 0]
    n = len(s)
    if n < 50:
        return float("nan"), float("nan")
    best = None
    # search over candidate xmin (the bottom 90 percentiles)
    candidates = np.unique(np.percentile(s, np.linspace(50, 95, 20)))
    for xmin in candidates:
        tail = s[s >= xmin]
        if len(tail) < 10:
            continue
        # continuous power-law MLE
        alpha = 1.0 + len(tail) / np.sum(np.log(tail / xmin))
        # goodness-of-fit (KS) -- cheap version
        n_tail = len(tail)
        cdf_emp = np.arange(1, n_tail + 1) / n_tail
        cdf_theo = 1.0 - (tail / xmin) ** (1.0 - alpha)
        ks = np.max(np.abs(cdf_emp - cdf_theo))
        if best is None or ks < best[1]:
            best = (float(alpha), float(ks), float(xmin))
    if best is None:
        return float("nan"), float("nan")
    return best[0], best[2]


# ======================================================================
#  PART 3 -- Row-wise average entropy  +  stable rank
# ======================================================================

def row_wise_entropy(weight: np.ndarray, base: float = np.e) -> float:
    """
    Treat each row of |W| as a probability distribution and return the
    average Shannon entropy (in nats by default).  Rows that are
    identically zero contribute 0.
    """
    w = np.abs(np.asarray(weight, dtype=np.float64))
    # avoid log(0) -- add tiny epsilon
    eps = 1e-30
    row_sums = w.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > 0
    if not np.any(valid):
        return 0.0
    p = w[valid] / row_sums[valid]
    h = -(p * np.log(p + eps)).sum(axis=1)
    # convert to requested base
    h = h / np.log(base)
    return float(np.mean(h))


def spectral_entropy(svals: np.ndarray, base: float = np.e) -> float:
    """
    Shannon entropy of the (normalised) squared singular-value
    distribution.  Maximal when all singular values are equal; small
    when one singular value dominates.
    """
    s = np.asarray(svals, dtype=np.float64)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    p = (s ** 2) / np.sum(s ** 2)
    h = -np.sum(p * np.log(p))
    return float(h / np.log(base))


def stable_rank(weight: np.ndarray) -> float:
    """
    Stable rank = ||W||_F^2 / ||W||_2^2  =  sum(s_i^2) / max(s_i)^2.
    Bounded between 1 and rank(W); numerically stable, hence the name.
    """
    w = np.asarray(weight, dtype=np.float64)
    s = np.linalg.svd(w, compute_uv=False)
    if s.size == 0 or s.max() == 0:
        return 0.0
    return float((s ** 2).sum() / (s.max() ** 2))


# ======================================================================
#  PART 4 -- Layer discovery for transformer models
# ======================================================================
#  We mirror the German paper's naming convention.  For each model we
#  auto-detect which weight matrix types are present and pick a list
#  of layer indices to actually analyse (the rest is too expensive on
#  a free Colab T4).
# ======================================================================

# canonical substrings for each matrix type
MATRIX_PATTERNS = {
    "Q":  ["self_attn.q_proj", "attention.self.query", "attention.query_key_value"],
    "K":  ["self_attn.k_proj", "attention.self.key"],
    "V":  ["self_attn.v_proj", "attention.self.value"],
    "O":  ["self_attn.o_proj", "attention.output.dense", "attention.dense"],
    "U":  ["mlp.up_proj", "mlp.dense_h_to_4h", "intermediate.dense"],
    "D":  ["mlp.down_proj", "mlp.dense_4h_to_h", "output.dense"],
    "G":  ["mlp.gate_proj"],
}


@dataclass
class MatrixRecord:
    """One weight matrix pulled out of a model."""
    name: str               # full parameter name, e.g. "model.layers.4.self_attn.q_proj.weight"
    short: str              # short tag, e.g. "Q"
    layer_idx: int          # transformer-block index
    weight: np.ndarray      # 2-D float32 numpy array (CPU)
    n: int
    m: int


def _extract_layer_index(name: str) -> int:
    """Find the first integer that appears after 'layers.' or 'encoder.layer.' in
    a parameter name.  Returns -1 if none found."""
    import re
    mt = re.search(r"layers\.(\d+)\.", name)
    if mt:
        return int(mt.group(1))
    mt = re.search(r"encoder\.layer\.(\d+)\.", name)
    if mt:
        return int(mt.group(1))
    mt = re.search(r"h\.(\d+)\.", name)
    if mt:
        return int(mt.group(1))
    return -1


def _classify(name: str) -> Optional[str]:
    """Return the short tag (Q/K/V/O/U/D/G) if `name` matches one of the
    canonical matrix patterns, else None."""
    for tag, pats in MATRIX_PATTERNS.items():
        for p in pats:
            if p in name and tag in ("Q", "K", "V", "O", "U", "D", "G"):
                # extra check for pythia qkv: only the Q tag applies to the
                # whole fused matrix; we will split it later.
                if "query_key_value" in name and tag != "Q":
                    continue
                return tag
    return None


def discover_weight_matrices(model, layer_indices: Optional[List[int]] = None,
                             dtype=torch.float32) -> List[MatrixRecord]:
    """Walk the state-dict and return one MatrixRecord per matching
    weight matrix.  Only 2-D weight tensors (no embedding / lm_head) are
    returned, mirroring the German paper.

    For Pythia-style fused QKV matrices the matrix is split row-wise into
    Q / K / V thirds so that we can analyse them separately (this matches
    the German code's `pythiaInfo["Query"] / ["Key"] / ["Value"]` slicing)."""
    out: List[MatrixRecord] = []
    sd = model.state_dict()
    for name, tensor in sd.items():
        if tensor.ndim != 2:
            continue
        # skip embeddings / lm_head
        if any(s in name for s in ("embed_tokens", "wte", "wpe",
                                   "lm_head", "shared", "embedding")):
            continue
        tag = _classify(name)
        if tag is None:
            continue
        layer_idx = _extract_layer_index(name)
        if layer_indices is not None and layer_idx not in layer_indices:
            # Still keep -1 layers (e.g. some BERT pooler)? No -- skip
            if layer_idx != -1:
                continue
        w = tensor.detach().to(dtype).cpu().numpy()

        # Pythia fused QKV: split into three square blocks
        if "query_key_value" in name:
            n_rows, n_cols = w.shape
            third = n_rows // 3
            for k, ttag in enumerate(("Q", "K", "V")):
                sub = w[k * third:(k + 1) * third, :]
                out.append(MatrixRecord(
                    name=f"{name}[{ttag}]",
                    short=ttag,
                    layer_idx=layer_idx,
                    weight=sub,
                    n=sub.shape[0],
                    m=sub.shape[1],
                ))
            continue

        out.append(MatrixRecord(
            name=name,
            short=tag,
            layer_idx=layer_idx,
            weight=w,
            n=w.shape[0],
            m=w.shape[1],
        ))
    return out


# ======================================================================
#  PART 5 -- Activation covariance (feature matrix)
# ======================================================================
#  Direct port of the German `featureM_utility.py` logic.
#  We replace each nn.Linear of interest with a small wrapper that
#  accumulates  (a) the mean of its input  and  (b) the centred
#  covariance  C = sum_i (h_i - hbar)(h_i - hbar)^T  of its input.
# ======================================================================

class FeatureLayer(nn.Module):
    """Replacement for nn.Linear that records the covariance of its
    input activations.  Mirrors the German FeatureLayer but uses float64
    accumulation for numerical stability."""

    def __init__(self, original: nn.Linear, name: str, device):
        super().__init__()
        # store weight/bias as-is (we don't actually train)
        self.weight = original.weight
        self.bias = original.bias
        self.name = name
        self.device = device
        self.kernel_dim = original.weight.shape[-1]
        self.computation = 0
        self.mean = torch.zeros(self.kernel_dim,
                                dtype=torch.float64, device=device)
        self.cov = torch.zeros(self.kernel_dim, self.kernel_dim,
                               dtype=torch.float64, device=device)
        self.compute_mean = True

    def forward(self, x):
        # upcast for accumulation, then back-cast for the actual matmul
        with torch.no_grad():
            x64 = x.detach().to(torch.float64)
            x2d = x64.reshape(-1, self.kernel_dim)
            if self.compute_mean:
                self.computation += 1
                # running mean
                self.mean += (x2d.mean(dim=0) - self.mean) / self.computation
            else:
                self.computation += 1
                xc = x2d - self.mean
                # running covariance  (DxD)  averaged over batches seen
                self.cov += (xc.t() @ xc - self.cov) / self.computation
        # use the original module's forward to keep dtype / device correct
        return torch.nn.functional.linear(x, self.weight, self.bias)


def _replace_with_feature_layers(model, layer_indices: List[int], device):
    """Replace every nn.Linear whose block index is in `layer_indices`
    with a FeatureLayer.  Returns the list of names that were replaced."""
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if module.weight.ndim != 2:
            continue
        layer_idx = _extract_layer_index(name)
        if layer_idx == -1 or layer_idx not in layer_indices:
            continue
        # skip embeddings etc.
        if any(s in name for s in ("embed", "lm_head", "pooler", "head")):
            continue
        # only the matrix types we care about
        if _classify(name) is None and "query_key_value" not in name:
            continue
        fl = FeatureLayer(module, name, device)
        # navigate to parent and replace
        parent = model
        tokens = name.split(".")
        for t in tokens[:-1]:
            parent = getattr(parent, t)
        setattr(parent, tokens[-1], fl)
        replaced.append(name)
    return replaced


def _set_feature_mode(model, mode: str):
    """mode='mean' for first pass, mode='FM' for covariance pass."""
    compute_mean = (mode == "mean")
    for m in model.modules():
        if isinstance(m, FeatureLayer):
            m.compute_mean = compute_mean


def _collect_feature_matrices(model):
    """Return dict[name] -> (weight np.float32, FM np.float64)."""
    out = {}
    for name, m in model.named_modules():
        if not isinstance(m, FeatureLayer):
            continue
        if m.weight is None or m.weight.ndim != 2:
            continue
        out[name] = {
            "weight": m.weight.detach().to(torch.float32).cpu().numpy().copy(),
            "FM": m.cov.detach().to(torch.float64).cpu().numpy().copy(),
            "mean": m.mean.detach().to(torch.float64).cpu().numpy().copy(),
        }
    return out


def compute_activation_covariance(
    model,
    tokenizer,
    layer_indices: List[int],
    device,
    dataset_name: str = "wikitext",
    split: str = "test",
    n_text_batches: int = 5,
    max_length: int = 1024,
    stride: int = 512,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Two-pass forward over a small text sample.

    Returns a dict keyed by parameter name with sub-keys
        weight  -- 2-D float32 numpy array
        FM      -- DxD float64 activation-covariance matrix
        mean    -- D  float64 mean activation
    """
    from datasets import load_dataset
    log(f"Loading text dataset '{dataset_name}' for activation covariance ...")
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(ds["text"][:5000])
    elif dataset_name == "bookcorpus":
        try:
            ds = load_dataset("bookcorpus", split="train", trust_remote_code=True)
            text = "\n\n".join(ds["text"][:5000])
        except Exception:
            log("BookCorpus unavailable; falling back to wikitext")
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            text = "\n\n".join(ds["text"][:5000])
    else:
        ds = load_dataset(dataset_name, split=split)
        text = "\n\n".join(ds["text"][:5000])

    enc = tokenizer(text, return_tensors="pt")
    seq_len = enc.input_ids.size(1)
    log(f"  total text length: {seq_len} tokens; max_length={max_length}, "
        f"stride={stride}, n_text_batches={n_text_batches}")

    # replace target Linear layers
    replaced = _replace_with_feature_layers(model, layer_indices, device)
    log(f"  replaced {len(replaced)} layers with FeatureLayer: {replaced[:6]} ...")

    # ---- Pass 1: mean ----------------------------------------------
    _set_feature_mode(model, "mean")
    model.eval()
    batch_count = 0
    with torch.no_grad():
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            input_ids = enc.input_ids[:, begin:end].to(device)
            if input_ids.size(1) < 2:
                continue
            try:
                _ = model(input_ids)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            batch_count += 1
            if batch_count >= n_text_batches:
                break

    # ---- Pass 2: covariance ---------------------------------------
    _set_feature_mode(model, "FM")
    batch_count = 0
    with torch.no_grad():
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            input_ids = enc.input_ids[:, begin:end].to(device)
            if input_ids.size(1) < 2:
                continue
            try:
                _ = model(input_ids)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            batch_count += 1
            if batch_count >= n_text_batches:
                break

    fm_dict = _collect_feature_matrices(model)
    log(f"  collected {len(fm_dict)} activation covariance matrices")
    return fm_dict


# ======================================================================
#  PART 6 -- Singular-vector / covariance-eigenvector overlap
# ======================================================================
#  This reproduces the German paper's  Ok  statistic (Eq. 7).
#  For each right singular vector  v_k  of  W  we compute
#       O_k = max_j ( v_k . f_j )
#  where  f_j  are the eigenvectors of the activation covariance  C.
#  In a random matrix the singular vectors are uniformly distributed on
#  the sphere so  O_k  ~ 1/sqrt(N)  with fluctuations of order
#  1/sqrt(N) * sqrt(2 ln N)  -> we draw a 3-sigma band accordingly.
# ======================================================================

def overlap_analysis(weight: np.ndarray, feature_matrix: np.ndarray
                     ) -> Dict[str, np.ndarray]:
    """
    Returns dict with:
        svals       -- singular values of W
        overlap     -- O_k for each singular value (length = min(n,m))
        mp_min, mp_max  -- MP support bounds
        sigma_med   -- estimated sigma
        right_outliers, left_outliers  -- counts
    """
    w = np.asarray(weight, dtype=np.float64)
    n, m = w.shape
    # SVD of weight matrix
    U, s, Vh = np.linalg.svd(w, full_matrices=False)
    # eigendecomp of activation covariance
    evals, evecs = np.linalg.eigh(np.asarray(feature_matrix, dtype=np.float64))
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]

    # MP fit
    sigma = estimate_sigma_med(w)
    lam_minus, lam_plus = mp_bounds(n, m, sigma)

    # overlap matrix  (k, j)  = | v_k . f_j |
    overlap_matrix = np.abs(Vh @ evecs)
    overlap = overlap_matrix.max(axis=1)  # O_k

    right_outliers = int(np.sum(s > lam_plus))
    left_outliers = int(np.sum(s < lam_minus)) if lam_minus > 0 else 0

    return {
        "svals": s,
        "overlap": overlap,
        "overlap_matrix": overlap_matrix,
        "mp_min": float(lam_minus),
        "mp_max": float(lam_plus),
        "sigma_med": float(sigma),
        "right_outliers": right_outliers,
        "left_outliers": left_outliers,
        "evals": evals[idx],
    }


def three_sigma_band(N: int, sigma_level: float = 3.0) -> Tuple[float, float]:
    """
    For a random vector of length N, the expected maximum overlap with a
    fixed orthonormal basis is ~ 1/sqrt(N).  The 3-sigma band (under
    Gaussian i.i.d. assumption) is computed by the German paper via a
    binary search on the CDF of |N(0, 1/N)|.

    We reproduce that calculation here -- see `prob_one_exceeds_x`
    in their generate_figures notebook.
    """
    sigma = 1.0 / np.sqrt(N)

    def p_one_exceeds(x):
        cdf = 0.5 * (1.0 + sp.erf(x / (sigma * np.sqrt(2.0))))
        return 1.0 - cdf ** N

    def p_all_below(x):
        return 1.0 - p_one_exceeds(x)

    # find upper bound: smallest x s.t. p_one_exceeds(x) <= sigma_level/N
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if p_one_exceeds(mid) > 1e-3:  # equivalent to their sigma3=0.003
            lo = mid
        else:
            hi = mid
    upper = (lo + hi) / 2

    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if p_all_below(mid) > 1e-3:
            hi = mid
        else:
            lo = mid
    lower = (lo + hi) / 2
    return float(lower), float(upper)


# ======================================================================
#  PART 7 -- Perplexity vs decile removal (German Fig. 5)
# ======================================================================
#  For each matrix type (Q, K, V, O, U, D, G) we zero out one decile of
#  the singular values in EVERY layer of that type, then measure
#  perplexity on WikiText.  Decile 1 = smallest, decile 10 = largest.
# ======================================================================

def perplexity_wikitext(model, tokenizer, device,
                        n_tokens: int = 4096, stride: int = 1024) -> float:
    """Strided negative-log-likelihood perplexity on WikiText-2-raw test."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"][:2000])
    enc = tokenizer(text, return_tensors="pt")
    seq_len = min(enc.input_ids.size(1), n_tokens)

    max_len = getattr(model.config, "max_position_embeddings", 4096)
    if max_len > 4096:
        max_len = 4096

    nll_sum = 0.0
    n_tok = 0
    prev_end = 0
    model.eval()
    with torch.no_grad():
        for begin in range(0, seq_len, stride):
            end = min(begin + max_len, seq_len)
            trg_len = end - prev_end
            input_ids = enc.input_ids[:, begin:end].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            try:
                out = model(input_ids, labels=target_ids)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                prev_end = end
                continue
            num_valid = (target_ids != -100).sum().item()
            nll_sum += float(out.loss) * (num_valid - 1)
            n_tok += (num_valid - 1)
            prev_end = end
            if end >= seq_len:
                break
    if n_tok == 0:
        return float("nan")
    return float(np.exp(nll_sum / n_tok))


def _set_layer_svd_decile(model, layer_records: List[MatrixRecord],
                          decile: int, n_deciles: int = 10):
    """Replace each weight matrix in `layer_records` by its SVD
    reconstruction with the i-th decile of singular values zeroed."""
    if decile < 1 or decile > n_deciles:
        raise ValueError(f"decile must be in [1,{n_deciles}], got {decile}")
    
    with torch.no_grad():
        for rec in layer_records:
            w = rec.weight.astype(np.float64)
            U, s, Vh = np.linalg.svd(w, full_matrices=False)
            k = len(s)
            
            lo = (decile - 1) * k // n_deciles
            hi = decile * k // n_deciles
            
            # BUG FIX: np.linalg.svd returns 's' in DESCENDING order! 
            # To zero the SMALLEST singular values (decile=1), we must zero from the end.
            s[k - hi : k - lo] = 0.0
            
            new_w = (U * s) @ Vh
            
            # Navigate to the actual parameter in the model
            parent = model
            tensor_name = rec.name.split("[")[0]
            for tok in tensor_name.split(".")[:-1]:
                parent = getattr(parent, tok)
            leaf_name = tensor_name.split(".")[-1]
            param = getattr(parent, leaf_name)

            # BUG FIX: Properly convert NumPy array to PyTorch dtype
            new_w_torch = torch.from_numpy(new_w).to(param.dtype)
            
            # BUG FIX: Handle Pythia fused QKV matrices cleanly without shape mismatch
            if "[" in rec.name:
                tag = rec.name.split("[")[1].strip("]") # "Q", "K", or "V"
                dim = new_w_torch.shape[0]
                if tag == "Q":
                    param.data[0:dim, :] = new_w_torch
                elif tag == "K":
                    param.data[dim:2*dim, :] = new_w_torch
                elif tag == "V":
                    param.data[2*dim:3*dim, :] = new_w_torch
            else:
                param.copy_(new_w_torch)


def perplexity_vs_decile(model_factory, tokenizer, layer_records_all: List[MatrixRecord],
                         device, n_tokens: int = 4096) -> Dict[str, List[float]]:
    """
    model_factory : callable that returns a FRESH model each time
                    (so we don't have to undo the SVD modification).
    layer_records_all : list of MatrixRecord for the FRESH model
    Returns dict[short_tag] -> list of 11 floats (base + 10 deciles)
    """
    # group records by short tag
    by_tag: Dict[str, List[MatrixRecord]] = {}
    for rec in layer_records_all:
        by_tag.setdefault(rec.short, []).append(rec)

    results: Dict[str, List[float]] = {}
    log("Computing base perplexity ...")
    base_model = model_factory()
    base_ppl = perplexity_wikitext(base_model, tokenizer, device, n_tokens=n_tokens)
    del base_model
    torch.cuda.empty_cache()
    log(f"  base ppl = {base_ppl:.4f}")

    for tag, recs in by_tag.items():
        log(f"  decile sweep for matrix type '{tag}' "
            f"({len(recs)} matrices across layers)")
        ppls = []
        for dec in range(1, 11):
            mdl = model_factory()
            # discover matching records for THIS fresh model
            fresh_recs = discover_weight_matrices(mdl, dtype=torch.float32)
            fresh_recs = [r for r in fresh_recs if r.short == tag]
            try:
                _set_layer_svd_decile(mdl, fresh_recs, dec)
                ppl = perplexity_wikitext(mdl, tokenizer, device, n_tokens=n_tokens)
            except Exception as e:
                log(f"    decile {dec} failed: {e}")
                ppl = float("nan")
            del mdl
            torch.cuda.empty_cache()
            ppls.append(ppl)
            log(f"    decile {dec}: ppl={ppl:.4f}")
        results[tag] = [base_ppl] + ppls
    return results


# ======================================================================
#  PART 8 -- Eigenvector-eigenvalue coincidence
# ======================================================================
#  Reproduces the German paper's Appendix I (Fig. 15).  For a given
#  layer's weight matrix W and activation covariance C:
#
#    * singular values of W are sorted DESCENDING -> v_0 is the largest
#    * eigenvalues of C    are sorted DESCENDING -> f_0 is the largest
#
#  Coincidence heat-map = |v_k . f_j|  (k = singular index, j = eigen index)
#  Scalar summary       = Spearman rank correlation between:
#        (|v_k . f_0|)  and  (s_k)       -- does the principal eigen-
#        vector align with the LARGEST singular direction?
#        and also:
#        (|v_0 . f_j|)  and  (lambda_j)
# ======================================================================

def eigenvector_eigenvalue_coincidence(weight: np.ndarray,
                                       feature_matrix: np.ndarray
                                       ) -> Dict[str, Any]:
    w = np.asarray(weight, dtype=np.float64)
    U, s, Vh = np.linalg.svd(w, full_matrices=False)
    # descending singular values: s already descending
    evals, evecs = np.linalg.eigh(np.asarray(feature_matrix, dtype=np.float64))
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # cosine-similarity heat-map (k = singular index, j = eigen index)
    cos_mat = np.abs(Vh @ evecs)

    # scalar summaries
    from scipy.stats import spearmanr
    # 1) does the principal eigenvector f_0 align with the largest singular
    #    directions? -> correlation of cos_mat[:, 0] (col 0) with s
    rho_top_eig, _ = spearmanr(cos_mat[:, 0], s)
    # 2) does the principal singular vector v_0 align with the largest
    #    eigen-directions? -> correlation of cos_mat[0, :] with evals
    rho_top_sv, _ = spearmanr(cos_mat[0, :], evals)
    # 3) diagonal tendency: are the top-k singular vectors aligned with the
    #    top-k eigenvectors?  -> correlation between cos_mat.diagonal() and s
    rho_diag, _ = spearmanr(np.diag(cos_mat), s)

    return {
        "cos_matrix": cos_mat,           # heat-map data
        "svals_desc": s,
        "evals_desc": evals,
        "rho_top_eigenvector_vs_svals": float(rho_top_eig) if not np.isnan(rho_top_eig) else 0.0,
        "rho_top_singular_vs_evals":    float(rho_top_sv) if not np.isnan(rho_top_sv) else 0.0,
        "rho_diag_vs_svals":            float(rho_diag) if not np.isnan(rho_diag) else 0.0,
        "max_overlap_with_top_eigenvector": float(cos_mat[:, 0].max()),
        "argmax_singular_for_top_eigenvector": int(cos_mat[:, 0].argmax()),
    }


# ======================================================================
#  PART 9 -- All-in-one per-matrix analysis
# ======================================================================
#  Run every per-matrix diagnostic on one weight matrix and return a
#  flat dict ready to be appended to a pandas DataFrame.
# ======================================================================

def per_matrix_analysis(rec: MatrixRecord,
                        fm_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None
                        ) -> Dict[str, Any]:
    w = rec.weight.astype(np.float64)
    n, m = w.shape
    s = np.linalg.svd(w, compute_uv=False)

    sigma = estimate_sigma_med(w)
    lam_minus, lam_plus = mp_bounds(n, m, sigma)
    right_out = int(np.sum(s > lam_plus))
    left_out = int(np.sum(s < lam_minus)) if lam_minus > 0 else 0

    # American-style power-law alpha (single number)
    alpha_mle, _ = fit_powerlaw_mle(s)

    # Hill estimator at k = sqrt(N)  (standard choice)
    k_hill = max(5, int(np.sqrt(len(s))))
    ks, alphas = hill_estimator(s, k_min=5)
    alpha_hill_mid = float(alphas[len(alphas) // 2]) if alphas.size > 0 else float("nan")
    alpha_hill_k_sqrtN = float(alphas[max(0, np.searchsorted(ks, k_hill) - 1)]
                               ) if alphas.size > 0 else float("nan")

    # entropy / stable rank
    rwe = row_wise_entropy(w)
    spe = spectral_entropy(s)
    srk = stable_rank(w)

    out = {
        "name": rec.name,
        "short": rec.short,
        "layer_idx": rec.layer_idx,
        "n": n,
        "m": m,
        "sigma_med": sigma,
        "mp_minus": lam_minus,
        "mp_plus": lam_plus,
        "n_right_outliers": right_out,
        "n_left_outliers": left_out,
        "fraction_right_outliers": right_out / len(s),
        "fraction_left_outliers": left_out / len(s),
        "alpha_mle_powerlaw": alpha_mle,
        "alpha_hill_mid": alpha_hill_mid,
        "alpha_hill_k_sqrtN": alpha_hill_k_sqrtN,
        "row_wise_entropy": rwe,
        "spectral_entropy": spe,
        "stable_rank": srk,
        "max_sval": float(s.max()),
        "min_sval": float(s.min()),
        "mean_sval": float(s.mean()),
        "median_sval": float(np.median(s)),
    }

    # overlap + coincidence if we have an activation covariance
    if fm_dict is not None and rec.name in fm_dict:
        fm = fm_dict[rec.name]["FM"]
        # match either exact name or pythia-fused-basename
    elif fm_dict is not None and "query_key_value" in rec.name:
        # rec.name like "...query_key_value.weight[Q]"
        base = rec.name.split("[")[0]
        if base in fm_dict:
            # use the full fused FM (best we can do without re-running
            # activations through the sliced matrix)
            fm = fm_dict[base]["FM"]
        else:
            fm = None
    else:
        fm = None

    if fm is not None and fm.shape[0] == fm.shape[1] and fm.shape[0] == w.shape[1]:
        ov = overlap_analysis(w, fm)
        out["mp_softrank"] = float(np.sum(s[s <= lam_plus] ** 2) / np.sum(s ** 2))
        out["max_overlap"] = float(ov["overlap"].max())
        out["mean_overlap"] = float(ov["overlap"].mean())
        out["overlap_at_top_sval"] = float(ov["overlap"][0])
        out["overlap_at_bottom_sval"] = float(ov["overlap"][-1])

        coi = eigenvector_eigenvalue_coincidence(w, fm)
        out["rho_top_eigenvector_vs_svals"] = coi["rho_top_eigenvector_vs_svals"]
        out["rho_top_singular_vs_evals"] = coi["rho_top_singular_vs_evals"]
        out["rho_diag_vs_svals"] = coi["rho_diag_vs_svals"]
        out["max_overlap_with_top_eigenvector"] = coi["max_overlap_with_top_eigenvector"]
        out["argmax_singular_for_top_eigenvector"] = coi["argmax_singular_for_top_eigenvector"]
    return out


# ======================================================================
#  PART 10 -- Plotting helpers
# ======================================================================
#  Each function makes one PNG.  All images land in <output_dir>/<model_tag>/.
# ======================================================================

def plot_esd_with_mp(rec: MatrixRecord, out_path: str,
                     overlap: Optional[Dict] = None,
                     title: Optional[str] = None):
    """Top: singular-value histogram + MP curve.  Bottom (optional):
    overlap O_k.  Outliers are coloured red."""
    w = rec.weight.astype(np.float64)
    s = np.linalg.svd(w, compute_uv=False)
    n, m = w.shape
    sigma = estimate_sigma_med(w)
    lam_minus, lam_plus = mp_bounds(n, m, sigma)

    has_overlap = overlap is not None and "overlap" in overlap
    nrows = 2 if has_overlap else 1
    fig, axs = plt.subplots(nrows, 1, figsize=(6.6, 2.4 * nrows),
                            sharex=True,
                            gridspec_kw={"height_ratios": [3, 1]} if has_overlap else None)
    if nrows == 1:
        axs = [axs]

    # top: histogram
    ax = axs[0]
    bins = np.linspace(0, max(1.05 * s.max(), 1.05 * lam_plus), 50)
    _, _, patches = ax.hist(s, bins=bins, density=True, color=PLOT_COLORS[0],
                            edgecolor="white", linewidth=0.3)
    # color outliers red
    for p in patches:
        x0 = p.xy[0]
        if x0 > lam_plus or (lam_minus > 0 and x0 + p.get_width() < lam_minus):
            p.set_facecolor(PLOT_COLORS[3])

    x_theo = np.linspace(0, bins[-1], 500)
    y_theo = mp_pdf(x_theo, n, m, sigma)
    ax.plot(x_theo, y_theo, "--k", linewidth=1.2, label="MP theory")
    ax.axvline(lam_plus, color="grey", linestyle=":", linewidth=0.8)
    if lam_minus > 0:
        ax.axvline(lam_minus, color="grey", linestyle=":", linewidth=0.8)
    ax.set_ylabel(r"$p(\nu)$")
    ax.set_title(title or f"{rec.short} layer {rec.layer_idx}  "
                  f"({n}x{m}, sigma={sigma:.3f})", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)

    if has_overlap:
        ax2 = axs[1]
        ov = overlap["overlap"]
        sv = overlap["svals"]
        inside = (sv <= lam_plus)
        if lam_minus > 0:
            inside = inside & (sv >= lam_minus)
        outside = ~inside
        ax2.scatter(sv[inside], ov[inside], s=8, color=PLOT_COLORS[0])
        ax2.scatter(sv[outside], ov[outside], s=8, color=PLOT_COLORS[3])
        N = w.shape[1]
        lo, hi = three_sigma_band(N)
        ax2.axhspan(lo, hi, facecolor="white", alpha=0.5, edgecolor="none")
        ax2.axhline(lo, color="grey", linestyle="-.", linewidth=0.6)
        ax2.axhline(hi, color="grey", linestyle="-.", linewidth=0.6)
        ax2.set_xlabel(r"$\nu$")
        ax2.set_ylabel(r"$O_k$")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hill(rec: MatrixRecord, out_path: str):
    s = np.linalg.svd(rec.weight.astype(np.float64), compute_uv=False)
    ks, alphas = hill_estimator(s, k_min=5)
    if ks.size == 0:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.plot(ks, alphas, color=PLOT_COLORS[0], label="Hill $\\hat\\alpha(k)$")
    ax.axhline(2.0, color="red", linestyle="--", linewidth=0.8,
               label="alpha=2 (power-law boundary)")
    ax.axhline(6.0, color="orange", linestyle="--", linewidth=0.8,
               label="alpha=6 (well-trained per WW)")
    ax.set_xlabel("k (number of upper-tail order statistics)")
    ax.set_ylabel(r"$\hat\alpha(k)$")
    ax.set_title(f"Hill estimator -- {rec.short} layer {rec.layer_idx}",
                 fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_qkv_heatmap(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                     out_path: str, layer_idx: int,
                     max_rows: int = 256, max_cols: int = 256):
    """Three side-by-side heat-maps of the Q, K, V weight matrices."""
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    titles = ["Query (Q)", "Key (K)", "Value (V)"]
    for ax, mat, t in zip(axs, [q, k, v], titles):
        sub = mat[:max_rows, :max_cols]
        vmax = np.percentile(np.abs(sub), 99)
        im = ax.imshow(sub, cmap="seismic", vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        ax.set_title(f"{t}  layer {layer_idx}", fontsize=9)
        ax.set_xlabel("input dim", fontsize=8)
        ax.set_ylabel("output dim", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_overlap_heatmap(cos_mat: np.ndarray, out_path: str,
                         title: str, max_dim: int = 100):
    """Plot the |v_k . f_j| coincidence matrix (Appendix I of paper)."""
    M = cos_mat[:max_dim, :max_dim]
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(M, cmap="magma", aspect="auto",
                   vmin=0, vmax=max(0.05, M.max() * 0.95),
                   interpolation="nearest")
    ax.set_xlabel("eigen-direction index $j$ (descending $\\lambda$)", fontsize=9)
    ax.set_ylabel("singular-direction index $k$ (descending $\\nu$)", fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$|v_k \cdot f_j|$")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_perplexity_deciles(results: Dict[str, List[float]],
                            out_path: str, model_tag: str):
    """Reproduce German Fig. 5: bar chart of delta perplexity vs decile."""
    tags = list(results.keys())
    n_tags = len(tags)
    fig, axs = plt.subplots(1, n_tags, figsize=(2.0 * n_tags, 3.0),
                            sharey=False)
    if n_tags == 1:
        axs = [axs]
    for ax, tag in zip(axs, tags):
        ppls = np.asarray(results[tag], dtype=float)
        base = ppls[0]
        delta = ppls[1:] - base
        # colour smallest-decile bar red if non-square (paper convention)
        # we infer rectangular vs square from the original records passed
        colours = [PLOT_COLORS[0]] * 10
        # heuristic: if the smallest-decile ppl is "much" larger than the
        # next few, mark it red -- but the paper marks ALL non-square
        # smallest deciles red.  We don't have shape here reliably, so use
        # the more conservative behaviour: always blue unless delta is the
        # 2nd largest of the 10 (i.e. "small but important")
        order = np.argsort(-delta)
        if 0 in order[:3]:  # decile 1 is among top 3 most important
            colours[0] = PLOT_COLORS[3]
        ax.bar(np.arange(1, 11), np.maximum(delta, 1e-3),
               color=colours, edgecolor="white", linewidth=0.3)
        ax.set_yscale("log")
        ax.set_title(tag, fontsize=9)
        ax.set_xticks([1, 3, 5, 7, 9])
        if tag == tags[0]:
            ax.set_ylabel(r"$\Delta$ Perplexity")
        ax.set_xlabel("decile removed", fontsize=8)
    fig.suptitle(f"{model_tag}  --  perplexity vs decile removal", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_alpha_comparison(df: pd.DataFrame, out_path: str, model_tag: str):
    """Histogram of WW-style alpha (American) vs Hill alpha (German)
    across all analysed matrices of one model."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    df["alpha_mle_powerlaw"].dropna().hist(bins=40, ax=axs[0],
                                            color=PLOT_COLORS[0], alpha=0.85)
    axs[0].axvline(2.0, color="red", linestyle="--", linewidth=0.8)
    axs[0].axvline(6.0, color="orange", linestyle="--", linewidth=0.8)
    axs[0].set_xlabel("power-law alpha (WW-style)")
    axs[0].set_ylabel("count")
    axs[0].set_title("American: power-law tail exponent", fontsize=9)

    df["alpha_hill_mid"].dropna().hist(bins=40, ax=axs[1],
                                       color=PLOT_COLORS[1], alpha=0.85)
    axs[1].axvline(2.0, color="red", linestyle="--", linewidth=0.8)
    axs[1].axvline(6.0, color="orange", linestyle="--", linewidth=0.8)
    axs[1].set_xlabel("Hill alpha (mid k)")
    axs[1].set_title("German: Hill tail exponent", fontsize=9)

    fig.suptitle(f"{model_tag}  --  alpha distribution across matrices",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_outlier_count_per_layer(df: pd.DataFrame, out_path: str, model_tag: str):
    """Reproduces the German paper's outlier-count style table (their
    Table 3) as a stacked bar chart."""
    pivot = df.groupby(["layer_idx", "short"]).agg(
        right=("n_right_outliers", "sum"),
        left=("n_left_outliers", "sum"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 3))
    width = 0.35
    shorts = sorted(df["short"].unique())
    layers = sorted([l for l in df["layer_idx"].unique() if l >= 0])
    x = np.arange(len(layers))
    for i, sh in enumerate(shorts):
        sub = pivot[pivot["short"] == sh].set_index("layer_idx")
        if sh not in sub.index:
            continue
        r = [sub.loc[l, "right"] if l in sub.index else 0 for l in layers]
        l_ = [sub.loc[l, "left"] if l in sub.index else 0 for l in layers]
        ax.bar(x - width / 2, r, width, label=f"{sh} right",
               color=PLOT_COLORS[i % len(PLOT_COLORS)])
        ax.bar(x + width / 2, l_, width,
               color=PLOT_COLORS[i % len(PLOT_COLORS)], alpha=0.4,
               edgecolor=PLOT_COLORS[i % len(PLOT_COLORS)],
               label=f"{sh} left")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("layer index")
    ax.set_ylabel("outlier count")
    ax.set_title(f"{model_tag} -- MP outlier counts per layer", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stable_rank_per_layer(df: pd.DataFrame, out_path: str, model_tag: str):
    """Line plot of stable rank per layer, one line per matrix type."""
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, sh in enumerate(sorted(df["short"].unique())):
        sub = df[df["short"] == sh].sort_values("layer_idx")
        ax.plot(sub["layer_idx"], sub["stable_rank"],
                marker="o", markersize=3, linewidth=1.0,
                color=PLOT_COLORS[i % len(PLOT_COLORS)], label=sh)
    ax.set_xlabel("layer index")
    ax.set_ylabel("stable rank  $\\|W\\|_F^2 / \\|W\\|_2^2$")
    ax.set_title(f"{model_tag} -- stable rank per layer", fontsize=10)
    ax.legend(fontsize=8, ncol=4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_entropy_per_layer(df: pd.DataFrame, out_path: str, model_tag: str):
    """Row-wise average entropy per layer, one line per matrix type."""
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, sh in enumerate(sorted(df["short"].unique())):
        sub = df[df["short"] == sh].sort_values("layer_idx")
        ax.plot(sub["layer_idx"], sub["row_wise_entropy"],
                marker="o", markersize=3, linewidth=1.0,
                color=PLOT_COLORS[i % len(PLOT_COLORS)], label=sh)
    ax.set_xlabel("layer index")
    ax.set_ylabel("row-wise average entropy (nats)")
    ax.set_title(f"{model_tag} -- row-wise average entropy", fontsize=10)
    ax.legend(fontsize=8, ncol=4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_eigvec_eigval_coincidence_summary(df: pd.DataFrame, out_path: str,
                                           model_tag: str):
    """Three-panel summary of the eigenvector-eigenvalue coincidence
    scalar metrics across layers."""
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    metrics = [
        ("rho_top_eigenvector_vs_svals",
         r"$\rho$( $|v_k\\cdot f_0|$, $s_k$ )"),
        ("rho_top_singular_vs_evals",
         r"$\rho$( $|v_0\\cdot f_j|$, $\\lambda_j$ )"),
        ("rho_diag_vs_svals",
         r"$\rho$( diag, $s_k$ )"),
    ]
    for ax, (col, label) in zip(axs, metrics):
        if col not in df.columns:
            ax.set_title("missing")
            continue
        for i, sh in enumerate(sorted(df["short"].unique())):
            sub = df[df["short"] == sh].sort_values("layer_idx")
            ax.plot(sub["layer_idx"], sub[col],
                    marker="o", markersize=3, linewidth=1.0,
                    color=PLOT_COLORS[i % len(PLOT_COLORS)], label=sh)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
        ax.set_xlabel("layer index")
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=9)
    axs[-1].legend(fontsize=7, ncol=4, loc="upper right")
    fig.suptitle(f"{model_tag} -- eigenvector-eigenvalue coincidence",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ======================================================================
#  PART 11 -- American WeightWatcher baseline (optional)
# ======================================================================
#  Reproduces the WW pipeline from the user's original snippet.  We run
#  it on the SAME model and dump the per-layer WW alpha alongside the
#  German MP metrics, so the two methodologies can be compared side by
#  side.  WeightWatcher is slow on >3B models so this is opt-in.
# ======================================================================

def run_weightwatcher(model, output_dir: str, model_tag: str) -> Optional[pd.DataFrame]:
    if not _HAS_WW:
        log("WeightWatcher not installed -- skipping American baseline.")
        return None
    log("Running WeightWatcher (American power-law baseline) ...")
    try:
        watcher = ww.WeightWatcher(model=model)
        details = watcher.analyze(mp_fit=True, plot=False, randomize=True)
        summary = watcher.get_summary(details)
        csv_path = os.path.join(output_dir, f"{model_tag}_ww_details.csv")
        details.to_csv(csv_path, index=False)
        log(f"  saved WW details -> {csv_path}")
        # also save summary
        with open(os.path.join(output_dir, f"{model_tag}_ww_summary.json"), "w") as f:
            json.dump({k: (float(v) if isinstance(v, (int, float, np.floating))
                           else str(v)) for k, v in summary.items()},
                      f, indent=2)
        # plot WW alpha histogram
        fig, ax = plt.subplots(figsize=(6, 3))
        details["alpha"].dropna().hist(bins=80, ax=ax, color=PLOT_COLORS[0])
        ax.axvline(2.0, color="red", linestyle="--", linewidth=0.8)
        ax.axvline(6.0, color="orange", linestyle="--", linewidth=0.8)
        ax.set_xlabel("weightwatcher alpha")
        ax.set_ylabel("count")
        ax.set_title(f"{model_tag} -- WeightWatcher alpha distribution", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_tag}_ww_alpha_hist.png"), dpi=150)
        plt.close(fig)
        return details
    except Exception as e:
        log(f"  WeightWatcher failed: {e}")
        return None


# ======================================================================
#  PART 12 -- Main driver
# ======================================================================

def _model_tag(model_name: str) -> str:
    return model_name.replace("/", "_").replace(".", "_")


def _load_model(model_name: str, dtype, device: str):
    log(f"Loading model '{model_name}' on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def analyze_one_model(
    model_name: str,
    output_dir: str,
    layer_indices: List[int],
    n_text_batches: int,
    max_length: int,
    do_perplexity: bool,
    do_ww: bool,
    do_qkv_heatmap: bool,
    do_overlap: bool,
    pythia_steps: Optional[List[int]] = None,
    perplexity_tokens: int = 4096,
):
    tag = _model_tag(model_name)
    out_dir = os.path.join(output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    log(f"=== analysing {model_name} -> {out_dir} ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # ---------- 1. load model ----------
    model, tokenizer = _load_model(model_name, dtype, device)
    n_layers = getattr(model.config, "num_hidden_layers", None) or \
               getattr(model.config, "n_layer", None) or 0
    log(f"  num_hidden_layers = {n_layers}")

    # restrict layer_indices to actually available ones
    if n_layers:
        layer_indices = sorted(set([l for l in layer_indices if 0 <= l < n_layers]))
    log(f"  layers analysed: {layer_indices}")

    # ---------- 2. discover weight matrices ----------
    records = discover_weight_matrices(model, layer_indices, dtype=torch.float32)
    log(f"  discovered {len(records)} weight matrices across {len(set(r.layer_idx for r in records))} layers")

    # ---------- 3. activation covariance ----------
    fm_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    if do_overlap or do_qkv_heatmap:
        try:
            fm_dict = compute_activation_covariance(
                model, tokenizer,
                dataset_name="wikitext",
                split="test",
                layer_indices=layer_indices,
                device=device,
                n_text_batches=n_text_batches,
                max_length=max_length,
                stride=max(64, max_length // 2),
            )
            with open(os.path.join(out_dir, "activation_covariance.pkl"), "wb") as f:
                # save only the matrices we actually need to keep file size down
                pickle.dump({k: {"FM": v["FM"], "mean": v["mean"]}
                             for k, v in fm_dict.items()}, f)
        except Exception as e:
            log(f"  activation covariance failed: {e}")
            fm_dict = None

    # ---------- 4. per-matrix analysis ----------
    log("  per-matrix analysis ...")
    rows = []
    for rec in records:
        try:
            rows.append(per_matrix_analysis(rec, fm_dict))
        except Exception as e:
            log(f"    failed {rec.name}: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{tag}_matrix_metrics.csv"), index=False)
    log(f"  wrote {tag}_matrix_metrics.csv ({len(df)} rows)")

    # ---------- 5. plots ----------
    # 5a. ESD + MP for every (layer, short) combination
    esd_dir = os.path.join(out_dir, "esd_plots")
    os.makedirs(esd_dir, exist_ok=True)
    for rec in records:
        ov = None
        if fm_dict is not None and rec.name in fm_dict:
            ov = overlap_analysis(rec.weight.astype(np.float64),
                                  fm_dict[rec.name]["FM"])
        plot_esd_with_mp(
            rec,
            os.path.join(esd_dir, f"layer{rec.layer_idx:02d}_{rec.short}.png"),
            overlap=ov if do_overlap else None,
        )

    # 5b. Hill estimator per matrix
    hill_dir = os.path.join(out_dir, "hill_plots")
    os.makedirs(hill_dir, exist_ok=True)
    for rec in records:
        try:
            plot_hill(rec, os.path.join(hill_dir,
                       f"layer{rec.layer_idx:02d}_{rec.short}.png"))
        except Exception as e:
            log(f"    hill plot failed {rec.name}: {e}")

    # 5c. QKV heatmaps
    if do_qkv_heatmap:
        qkv_dir = os.path.join(out_dir, "qkv_heatmaps")
        os.makedirs(qkv_dir, exist_ok=True)
        # group records by layer
        by_layer: Dict[int, Dict[str, MatrixRecord]] = {}
        for r in records:
            if r.short in ("Q", "K", "V"):
                by_layer.setdefault(r.layer_idx, {})[r.short] = r
        for l, mrs in by_layer.items():
            if all(s in mrs for s in ("Q", "K", "V")):
                plot_qkv_heatmap(
                    mrs["Q"].weight, mrs["K"].weight, mrs["V"].weight,
                    os.path.join(qkv_dir, f"layer{l:02d}_qkv.png"),
                    layer_idx=l,
                )

    # 5d. eigenvector-eigenvalue coincidence heat-maps
    if do_overlap and fm_dict is not None:
        coin_dir = os.path.join(out_dir, "coincidence_heatmaps")
        os.makedirs(coin_dir, exist_ok=True)
        for rec in records:
            if rec.name in fm_dict:
                fm = fm_dict[rec.name]["FM"]
                if fm.shape[0] == rec.weight.shape[1] and fm.shape[0] == fm.shape[1]:
                    coi = eigenvector_eigenvalue_coincidence(
                        rec.weight.astype(np.float64), fm)
                    plot_overlap_heatmap(
                        coi["cos_matrix"],
                        os.path.join(coin_dir,
                                     f"layer{rec.layer_idx:02d}_{rec.short}.png"),
                        title=f"{rec.short} layer {rec.layer_idx} -- "
                              f"$|v_k \\cdot f_j|$",
                    )

    # 5e. summary plots across layers
    try:
        plot_alpha_comparison(df, os.path.join(out_dir, f"{tag}_alpha_compare.png"), tag)
    except Exception as e:
        log(f"  alpha comparison plot failed: {e}")
    try:
        plot_outlier_count_per_layer(df, os.path.join(out_dir, f"{tag}_outliers.png"), tag)
    except Exception as e:
        log(f"  outlier plot failed: {e}")
    try:
        plot_stable_rank_per_layer(df, os.path.join(out_dir, f"{tag}_stable_rank.png"), tag)
    except Exception as e:
        log(f"  stable-rank plot failed: {e}")
    try:
        plot_entropy_per_layer(df, os.path.join(out_dir, f"{tag}_entropy.png"), tag)
    except Exception as e:
        log(f"  entropy plot failed: {e}")
    if do_overlap and "rho_top_eigenvector_vs_svals" in df.columns:
        try:
            plot_eigvec_eigval_coincidence_summary(
                df, os.path.join(out_dir, f"{tag}_coincidence.png"), tag)
        except Exception as e:
            log(f"  coincidence plot failed: {e}")

    # ---------- 6. perplexity vs decile (German Fig. 5) ----------
    if do_perplexity:
        log("  perplexity vs decile removal ...")
        try:
            def _factory():
                m, _ = _load_model(model_name, dtype, device)
                return m
            ppl_results = perplexity_vs_decile(
                _factory, tokenizer,
                records, device, n_tokens=perplexity_tokens,
            )
            with open(os.path.join(out_dir, f"{tag}_perplexity.json"), "w") as f:
                json.dump(ppl_results, f, indent=2)
            plot_perplexity_deciles(ppl_results,
                                    os.path.join(out_dir, f"{tag}_perplexity.png"),
                                    tag)
        except Exception as e:
            log(f"  perplexity experiment failed: {e}")

    # ---------- 7. optional WW baseline ----------
    if do_ww:
        run_weightwatcher(model, out_dir, tag)

    # ---------- 8. per-epoch stable rank for Pythia ----------
    if pythia_steps and "pythia" in model_name.lower():
        log(f"  per-epoch stable rank for Pythia steps {pythia_steps} ...")
        per_epoch_rows = []
        for step in pythia_steps:
            try:
                revision = f"step{step}"
                log(f"    loading {model_name} @ {revision}")
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_name, revision=revision,
                    torch_dtype=dtype, low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                recs = discover_weight_matrices(mdl, layer_indices, dtype=torch.float32)
                for r in recs:
                    per_epoch_rows.append({
                        "step": step,
                        "name": r.name,
                        "short": r.short,
                        "layer_idx": r.layer_idx,
                        "stable_rank": stable_rank(r.weight.astype(np.float64)),
                    })
                del mdl
                torch.cuda.empty_cache()
            except Exception as e:
                log(f"    step {step} failed: {e}")
        if per_epoch_rows:
            ped = pd.DataFrame(per_epoch_rows)
            ped.to_csv(os.path.join(out_dir, f"{tag}_stable_rank_per_epoch.csv"),
                       index=False)
            # plot: one line per matrix type, x=step
            fig, ax = plt.subplots(figsize=(7, 3))
            for i, sh in enumerate(sorted(ped["short"].unique())):
                sub = ped[ped["short"] == sh].groupby("step")["stable_rank"].mean()
                ax.plot(sub.index, sub.values, marker="o", markersize=4,
                        linewidth=1.0, color=PLOT_COLORS[i % len(PLOT_COLORS)],
                        label=sh)
            ax.set_xlabel("training step (epoch proxy)")
            ax.set_ylabel("mean stable rank")
            ax.set_title(f"{tag} -- stable rank across training", fontsize=10)
            ax.legend(fontsize=8, ncol=4)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{tag}_stable_rank_per_epoch.png"),
                        dpi=150)
            plt.close(fig)

    log(f"=== done with {model_name} ===")


# ----------------------------------------------------------------------
#  Argparse entry point
# ----------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="RMT analysis of LLM weight matrices (German + American "
                    "+ extras).  Designed for Google Colab T4 free tier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more HuggingFace model IDs, e.g. "
                        "EleutherAI/pythia-410m Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--output_dir", default="./rmt_results",
                   help="Root output directory")
    p.add_argument("--layers", nargs="+", type=int,
                   default=[0, 4, 9, 14, 19],
                   help="Layer indices to analyse (others skipped to save time)")
    p.add_argument("--n_text_batches", type=int, default=5,
                   help="Number of text chunks to feed the model when "
                        "computing the activation covariance matrix.")
    p.add_argument("--max_length", type=int, default=1024,
                   help="Sequence length per text chunk.")
    p.add_argument("--do_perplexity", action="store_true",
                   help="Run the 10-decile perplexity sweep (slow).")
    p.add_argument("--do_ww", action="store_true",
                   help="Also run the American WeightWatcher baseline.")
    p.add_argument("--do_qkv_heatmap", action="store_true", default=True,
                   help="Generate Q/K/V heat-maps (cheap, on by default).")
    p.add_argument("--do_overlap", action="store_true", default=True,
                   help="Compute singular-vector / activation-eigenvector "
                        "overlap (requires activation covariance).")
    p.add_argument("--no_overlap", action="store_true",
                   help="Skip overlap / activation-covariance step entirely.")
    p.add_argument("--perplexity_tokens", type=int, default=4096,
                   help="Number of tokens to evaluate perplexity on (per "
                        "decile removal step).")
    p.add_argument("--pythia_steps", nargs="+", type=int, default=None,
                   help="If the model is a Pythia checkpoint suite, also "
                        "compute stable rank at these training steps "
                        "(e.g. 1000 20000 143000).")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.no_overlap:
        args.do_overlap = False

    log(f"Output dir : {args.output_dir}")
    log(f"Models     : {args.models}")
    log(f"Layers     : {args.layers}")
    log(f"Perplexity : {args.do_perplexity}")
    log(f"WeightWatcher: {args.do_ww} (installed: {_HAS_WW})")
    log(f"QKV heatmap: {args.do_qkv_heatmap}")
    log(f"Overlap    : {args.do_overlap}")

    for model_name in args.models:
        try:
            analyze_one_model(
                model_name=model_name,
                output_dir=args.output_dir,
                layer_indices=args.layers,
                n_text_batches=args.n_text_batches,
                max_length=args.max_length,
                do_perplexity=args.do_perplexity,
                do_ww=args.do_ww,
                do_qkv_heatmap=args.do_qkv_heatmap,
                do_overlap=args.do_overlap,
                pythia_steps=args.pythia_steps,
                perplexity_tokens=args.perplexity_tokens,
            )
        except Exception as e:
            log(f"!!! model {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    log("All models done.")


if __name__ == "__main__":
    raise SystemExit(
        "This archived monolith is intentionally non-executable. "
        "Use `python -m rmt` so the maintained, tested package is executed."
    )