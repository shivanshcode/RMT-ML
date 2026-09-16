"""rmt.config — RunConfig dataclass, shared tolerances, offline guard, logging.

Pure module (no torch import at module top-level) so the scientific core stays
torch-free and unit-testable in milliseconds.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Shared numeric tolerances (from plan-unittest.md calibration table).         #
# Tests import these so the calibration lives in one place.                    #
# --------------------------------------------------------------------------- #
TOL = {
    "sigma_rel": 0.015,            # estimate_sigma_gd_median on Wishart  (<=1.5%)
    "sigma_rel_outliers": 0.02,    # robustness to planted spikes         (<=2%)
    "r_goe": (0.5307, 0.025),      # <r> GOE
    "r_poisson": (0.3863, 0.025),  # <r> Poisson
    "ginue_abs": (0.738, 0.035),   # GinUE <|z|>
    "ginue_cos": (-0.32, -0.10),   # GinUE <cos arg z> band
    "poi2d_abs": (0.667, 0.03),    # 2D-Poisson <|z|>
    "poi2d_cos_abs": 0.06,         # |<cos>| <= 0.06
    "csn_alpha": (2.8, 3.2),       # CSN alpha on Pareto(3)
    "hill_alpha": (2.0, 0.25),     # standard Hill 1/H on Pareto(3)
    "hill_windowed": (2.0, 0.3),   # windowed Hill plateau on Pareto(3)
    "mp_integral": 1e-3,           # MP pdf normalisation
    "delta3_rtol": 0.15,           # exact Bohigas-Giannoni Delta_3 vs GOE theory
    "sigma2_rtol": 0.15,           # Sigma^2 vs GOE theory, L <= 10
    "unfold_mean_spacing": 0.03,   # |<s> - 1| after a valid unfolding
    "brody_goe": (1.0, 0.15),      # Brody beta on GOE
    "brody_poisson": (0.0, 0.12),  # Brody beta on Poisson
    "pt_p_uniform": 0.08,          # |mean p - 0.5| for Haar vectors
}

SEED = 1234  # default deterministic seed used by selftest & tests


@dataclass
class RunConfig:
    """All CLI flags (plan.md §4) as typed fields with defaults."""
    # model / IO
    models: List[str] = field(default_factory=lambda: ["meta-llama/Llama-3.1-8B"])
    model_path: Optional[str] = None
    output_dir: str = "./rmt_results"
    layers: List[int] = field(default_factory=list)   # empty = all layers
    # Precision contract (REPORT §0): source weights are loaded/analysed in fp32
    # so the smallest singular values are not quantisation noise before any SVD.
    dtype: str = "fp32"
    # SVD backend
    backend: str = "auto"                              # auto|numpy|torch
    gpu_svd_min_dim: int = 1024
    # MP / sigma / eigenvalue domain
    sigma_estimator: str = "gd_median"                 # gd_median|median_raw|usvt_threshold
    N_cov_mode: str = "cols"                           # cols|max|rows
    # activations / FM
    do_overlap: bool = True
    fm_dataset: str = "wikitext"
    do_qkv_heatmap: bool = True
    fm_stride: int = 1024
    fm_max_length: int = 2048
    n_text_batches: int = 5
    fm_token_weighted: bool = False
    max_oom: int = 3
    # tail / RMT
    do_powerlaw: bool = True
    alpha_estimator: str = "all"                       # csn|hill|hill_windowed|all
    hill_window: int = 20
    use_powerlaw_pkg: bool = False
    do_spacing: bool = True
    do_complex_spacing: bool = False
    do_ipr: bool = True
    do_porter_thomas: bool = True
    # Porter-Thomas (calibrated KS, Thamm rmt_utils ks_test_statistic_normedPT)
    pt_n_samples: int = 5000        # Monte-Carlo draws for the null CDFs
    pt_alpha: float = 0.05          # significance level behind pt_frac_random
    pt_max_vectors: Optional[int] = None
    # Unfolding. 'auto' = global Chebyshev fit, falling back to Thamm's adaptive
    # Gaussian broadening if that fit is not monotone / unit-mean. The global fit
    # preserves the long-range count fluctuations that Sigma^2(L) measures; the
    # local kernel does not, but copes with any density.
    unfold_method: str = "auto"     # auto | cheb | gauss | poly
    unfold_win: int = 15            # Thamm's window, used by the gauss branch
    unfold_deg: int = 7             # degree of the global (cheb/poly) fit
    spacing_domain: str = "sval"    # sval | eig  (levels fed to the unfolding)
    # Restrict the level statistics to the MP bulk [nu_-, nu_+]. Outliers are a
    # different ensemble (they are the *signal*), and leaving them in stretches
    # the fit domain so the global unfolding has to spend its degrees of freedom
    # on a handful of levels.
    spacing_bulk_only: bool = True
    # How the bulk is selected (review §4.1). The MP cut is a NO-OP on square
    # matrices -- mp_bounds(4096,4096,sigma) = (0, 2*sigma*64), so nothing is
    # trimmed and the Bessel hard edge at zero stays in the sample, where the
    # statistics are Bessel/Airy rather than GOE. 'center' cuts by RANK instead,
    # which removes both edges at any aspect ratio and removes the same fraction
    # from the null replicas, so the matched band stays comparable.
    #   center | mp | none
    spacing_bulk_mode: str = "center"
    spacing_bulk_center_frac: float = 0.7
    # Confirm every conclusion is stable across these fractions before claiming
    # it; a result that moves is a statement about the edge, not the bulk.
    spacing_bulk_frac_sweep: List[float] = field(
        default_factory=lambda: [0.6, 0.7, 0.8])
    # Sigma^2 estimator (review §4.4). Thamm's adaptive stopping rule halts on a
    # cumulative mean's O(1/k) drift and leaves +/-1.4% of pure seed noise; a
    # fixed window count removes the data-dependent stopping time entirely.
    # 0 = restore the adaptive rule.
    sigma2_n_windows: int = 200_000
    do_brody: bool = True
    brody_bootstrap: int = 200
    # Ints: the L values are also used to build CSV column names
    # (delta3_L5, ...), so a float here would emit 'delta3_L5.0'.
    delta3_L: List[int] = field(default_factory=lambda: [5, 10, 50])
    sigma2_L: List[int] = field(default_factory=lambda: [5, 10, 20])
    # decile / perplexity
    do_perplexity: bool = False
    decile_scope: str = "all"                          # all|analyzed
    n_deciles: int = 10
    ppl_dataset: str = "wikitext"
    ppl_stride: int = 512
    perplexity_tokens: int = 4096
    use_svd_cache: bool = True
    svd_cache_dir: str = "./svd_cache"
    text_path: str = "./wikitext-2-raw/wiki.test.raw"
    # headline / baselines / extras
    do_finetune_recovery: bool = False
    ft_method: str = "lora"
    ft_steps: int = 200
    ft_task: str = "rte"
    do_ww: bool = False
    ww_normalize: bool = False
    ww_glorot_fix: bool = False
    do_randomize: bool = False
    pythia_steps: Optional[int] = None
    # which probe layers across epochs (start / middle / end) – settable from main
    epoch_probe_fracs: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    epoch_checkpoint_every_frac: float = 0.10          # probe every 10% of iterations
    # misc
    offline: bool = True
    selftest: bool = False
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class OfflineGuard:
    """Sets/asserts HF offline environment variables. No network at run time."""

    ENV = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            for k, v in self.ENV.items():
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        return False

    @classmethod
    def assert_offline(cls) -> bool:
        return all(os.environ.get(k) == v for k, v in cls.ENV.items())

    @classmethod
    def enable(cls):
        for k, v in cls.ENV.items():
            os.environ[k] = v


_LOGGER_CONFIGURED = False


def get_logger(name: str = "rmt") -> logging.Logger:
    """Logger for ``name``, with the package root configured exactly once.

    The handler must go on the package root ("rmt"), not on whichever module
    happened to ask first. Attaching it to a child and setting propagate=False
    there silenced every *other* module: with ``python -m rmt``, ``rmt.cli``
    imported first and captured the handler, so ``rmt.selftest`` and
    ``rmt.pipeline`` had no handler and propagated to a root logger sitting at
    WARNING -- which is why ``--selftest`` exited 0 without printing anything.
    """
    global _LOGGER_CONFIGURED
    root = logging.getLogger("rmt")
    if not _LOGGER_CONFIGURED:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _LOGGER_CONFIGURED = True
    return logging.getLogger(name)


# Matplotlib plot style is applied lazily inside rmt/plots to avoid importing
# matplotlib in the pure test path.
def apply_plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
    })
