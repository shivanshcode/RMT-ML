"""NN-spacing histogram vs Wigner-GOE / Poisson."""
from __future__ import annotations
import os
import numpy as np


def plot_nn_spacing(levels, out_path, *, deg=7):
    from ..config import apply_plot_style
    from .. import spacing as SP
    apply_plot_style()
    import matplotlib.pyplot as plt
    s = SP.nn_spacing(levels, deg=deg)
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        ax.hist(s, bins=50, density=True, alpha=0.5, label="P(s)")
        xs = np.linspace(0, max(4, s.max()), 200)
        ax.plot(xs, (np.pi / 2) * xs * np.exp(-np.pi * xs**2 / 4), "r-", label="Wigner-GOE")
        ax.plot(xs, np.exp(-xs), "g--", label="Poisson")
        ax.set_xlabel("s"); ax.set_ylabel("P(s)"); ax.legend()
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(out_path)
    finally:
        plt.close(fig)
    return out_path
