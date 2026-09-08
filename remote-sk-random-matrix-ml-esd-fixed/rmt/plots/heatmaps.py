"""Heatmaps: fused-QKV singular spectra and overlap matrices."""
from __future__ import annotations
import os
import numpy as np


def plot_qkv_heatmap(svals_by_tag, out_path):
    from ..config import apply_plot_style
    apply_plot_style()
    import matplotlib.pyplot as plt
    tags = [tag for tag in ("Q", "K", "V") if tag in svals_by_tag]
    if not tags:
        return None
    width = max(len(np.asarray(svals_by_tag[tag]).ravel()) for tag in tags)
    image = np.full((len(tags), width), np.nan)
    for row, tag in enumerate(tags):
        values = np.sort(np.asarray(svals_by_tag[tag]).ravel())[::-1]
        image[row, :values.size] = np.log10(np.maximum(values, np.finfo(float).tiny))
    fig, ax = plt.subplots(figsize=(7, 2.5))
    try:
        rendered = ax.imshow(image, aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(len(tags)), tags)
        ax.set_xlabel("descending singular-value rank")
        fig.colorbar(rendered, ax=ax, label="log10 singular value")
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(out_path)
    finally:
        plt.close(fig)
    return out_path


def plot_overlap_heatmap(overlap_matrix, out_path):
    from ..config import apply_plot_style
    apply_plot_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        im = ax.imshow(np.asarray(overlap_matrix), aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax); ax.set_xlabel("eigenvector j"); ax.set_ylabel("singular vector k")
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(out_path)
    finally:
        plt.close(fig)
    return out_path
