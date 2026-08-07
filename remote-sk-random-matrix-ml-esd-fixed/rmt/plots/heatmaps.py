"""Heatmaps: fused-QKV singular spectra and overlap matrices."""
from __future__ import annotations
import os
import numpy as np


def plot_qkv_heatmap(svals_by_tag, out_path):
    from ..config import apply_plot_style
    apply_plot_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for tag, s in svals_by_tag.items():
        ax.plot(np.sort(s)[::-1], label=tag)
    ax.set_yscale("log"); ax.set_xlabel("index"); ax.set_ylabel("ν"); ax.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_overlap_heatmap(overlap_matrix, out_path):
    from ..config import apply_plot_style
    apply_plot_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(np.asarray(overlap_matrix), aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax); ax.set_xlabel("eigenvector j"); ax.set_ylabel("singular vector k")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    return out_path
