"""Hill plots: standard Hill curve + windowed-Hill plateau."""
from __future__ import annotations
import os
import numpy as np


def plot_hill(svals, out_path, *, window=20):
    from ..config import apply_plot_style
    from .. import tail as TAIL
    apply_plot_style()
    import matplotlib.pyplot as plt

    ks, inv = TAIL.hill_estimator(svals)
    kw, aw = TAIL.hill_estimator_windowed(svals, window=window)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ks, inv, "."); ax[0].set_title("standard Hill 1/H_k")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("α_hill")
    ax[1].plot(kw, aw, "."); ax[1].set_title("windowed Hill")
    ax[1].set_xlabel("k"); ax[1].set_ylabel("α_local")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path); plt.close(fig)
    return out_path
