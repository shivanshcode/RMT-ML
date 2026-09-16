"""rmt.plots — matplotlib visualizations (Agg backend, imported lazily).

Each function is best-effort: plotting must never break the numeric analysis,
so callers wrap these in try/except. Importing this package does NOT import
matplotlib; that happens inside the functions.
"""
from .esd import plot_esd                       # noqa: F401
from .hill import plot_hill                     # noqa: F401
from .heatmaps import plot_qkv_heatmap, plot_overlap_heatmap  # noqa: F401
from .spacing import (plot_nn_spacing, plot_rigidity,           # noqa: F401
                      plot_porter_thomas, plot_porter_thomas_deciles)
from .summary import plot_model_summary, plot_stable_rank_per_epoch  # noqa: F401
from .perplexity import plot_perplexity_vs_decile  # noqa: F401
