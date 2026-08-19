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
    "delta3_rtol": 0.4,
    "sigma2_rtol": 0.3,
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
    # Which σ defines the MP support everywhere (CSV bounds, outlier counts,
    # small-SV metrics, IPR and the plotted overlay).
    #   "med"     — Gavish–Donoho median matching. DEFAULT. Verified to agree
    #               to 0.1% with `estimate_sigma_med` in the released code of
    #               Staats/Thamm/Rosenow (arXiv:2410.17770, Zenodo
    #               10.5281/zenodo.14764226), which is the reference method for
    #               these same models. Do not change without re-checking that.
    #   "emp"     — ‖W‖_F/√(nm). Diagnostic only. NOT what the reference uses;
    #               `np.std(weight)` is commented out throughout their code.
    #   "refined" — the outlier-trimmed iterate, used only where it converged;
    #               falls back to "med" per matrix otherwise.
    mp_sigma_source: str = "med"                       # med|emp|refined
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
    do_porter_thomas: bool = False
    unfold_deg: int = 7
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
    global _LOGGER_CONFIGURED
    logger = logging.getLogger(name)
    if not _LOGGER_CONFIGURED:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _LOGGER_CONFIGURED = True
    return logger


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
