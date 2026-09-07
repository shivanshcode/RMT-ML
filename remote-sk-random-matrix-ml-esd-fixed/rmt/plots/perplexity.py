"""Perplexity-vs-decile ablation plot."""
from __future__ import annotations
import os


def plot_perplexity_vs_decile(result, out_path):
    from ..config import apply_plot_style
    apply_plot_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result["deciles"], result["perplexity"], marker="o")
    ax.set_xlabel("zeroed decile (1=smallest)"); ax.set_ylabel("perplexity")
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    return out_path
