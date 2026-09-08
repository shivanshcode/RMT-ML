"""Model-level summary plots."""
from __future__ import annotations
import os
import numpy as np


def plot_model_summary(rows, out_path):
    from ..config import apply_plot_style
    apply_plot_style()
    import matplotlib.pyplot as plt
    if not rows:
        return None
    layers = [r.get("layer_idx", -1) for r in rows]
    srk = [r.get("stable_rank", np.nan) for r in rows]
    alpha = [r.get("alpha", np.nan) for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    try:
        ax[0].scatter(layers, srk, s=10); ax[0].set_xlabel("layer"); ax[0].set_ylabel("stable rank")
        ax[1].scatter(layers, alpha, s=10); ax[1].set_xlabel("layer"); ax[1].set_ylabel("selected tail exponent")
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(out_path)
    finally:
        plt.close(fig)
    return out_path


def plot_stable_rank_per_epoch(epoch_fracs, srk_by_layer, out_path):
    from ..config import apply_plot_style
    apply_plot_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        for layer, vals in srk_by_layer.items():
            ax.plot(epoch_fracs, vals, marker="o", label=f"layer {layer}")
        ax.set_xlabel("training fraction"); ax.set_ylabel("stable rank"); ax.legend()
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(out_path)
    finally:
        plt.close(fig)
    return out_path
