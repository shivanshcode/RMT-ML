"""Hill plots: standard Hill curve + windowed-Hill plateau."""
from __future__ import annotations
import os
import numpy as np


def plot_hill(eigenvalues, out_path, *, window=20):
    from ..config import apply_plot_style
    from .. import tail as TAIL
    apply_plot_style()
    import matplotlib.pyplot as plt

    levels = np.asarray(eigenvalues, dtype=np.float64)
    ks, inv = TAIL.hill_estimator(levels)
    kw, aw = TAIL.hill_estimator_windowed(levels, window=window)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    try:
        ax[0].plot(ks, inv, "."); ax[0].set_title("standard Hill 1/H_k")
        ax[0].set_xlabel("k"); ax[0].set_ylabel("α_hill on covariance eigenvalues λ")
        ax[1].plot(kw, aw, "."); ax[1].set_title("windowed Hill")
        ax[1].set_xlabel("k"); ax[1].set_ylabel("α_local on covariance eigenvalues λ")
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(out_path)
    finally:
        plt.close(fig)
    return out_path
