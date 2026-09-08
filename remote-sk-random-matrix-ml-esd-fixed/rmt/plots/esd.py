"""ESD plots: empirical spectral density of nu and lambda with MP overlay."""
from __future__ import annotations
import os
import numpy as np


def _fd_width(x):
    """Freedman-Diaconis *bin width* (not a count)."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return None
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 2 * iqr * x.size ** (-1 / 3)
    return h if h > 0 else None


def _bin_edges(vals, lo, hi):
    """Edges spanning the *plotted* range at the bulk's FD width.

    Previously the FD width was computed from the bulk but converted to a bin
    *count*, which matplotlib then spread over the full data range (bulk plus
    right-tail outliers). That silently inflated the bin width by the ratio
    max(s)/nu_plus and smeared the small-nu region.
    """
    vals = np.asarray(vals, dtype=float)
    bulk = vals[(vals >= lo) & (vals <= hi)]
    h = _fd_width(bulk if bulk.size >= 2 else vals)
    left, right = float(min(vals.min(), lo)), float(max(vals.max(), hi))
    if h is None or not np.isfinite(h):
        return np.linspace(left, right, 51)
    requested = int(np.ceil((right - left) / h))
    if requested <= 400:
        return np.linspace(left, right, max(50, requested) + 1)
    # Preserve the bulk FD resolution and cover sparse tails separately rather
    # than spreading a capped count over the outlier-inflated full range.
    bulk_left, bulk_right = max(left, float(lo)), min(right, float(hi))
    bulk_edges = np.arange(bulk_left, bulk_right + h, h)
    tails = vals[(vals < bulk_left) | (vals > bulk_right)]
    tail_edges = (np.quantile(tails, np.linspace(0.0, 1.0, min(100, tails.size) + 1))
                  if tails.size else np.asarray([]))
    return np.unique(np.concatenate(([left], bulk_edges, tail_edges, [right])))


def _count_frac(vals, lo, hi):
    """Fraction of values (by count) inside [lo, hi]."""
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return 1.0
    return int(np.sum((vals >= lo) & (vals <= hi))) / vals.size


def _open_grid(lo, hi, n=400):
    """Grid on the OPEN interval (lo, hi).

    The MP density is defined by strict inequalities, so evaluating it at lo or
    hi returns exactly 0. A closed grid therefore draws a segment from (lo, 0)
    up to the true edge value -- for a square matrix lo = 0 and that segment is
    the spurious vertical line at nu = 0.
    """
    return np.linspace(lo, hi, n + 2)[1:-1]


def plot_esd(svals, n, m, sigma, out_path, *, domain="nu", N=None,
             bulk_mass=1.0, mp_mode="theory", broaden_win=15):
    """ESD with an MP overlay.

    mp_mode:
      "theory" -- strict Eq. (4) law from ``sigma``, unit area rescaled by the
                  count fraction inside the support. Correct reference for a
                  randomly initialised matrix.
      "fit"    -- the paper's three-parameter *modified* MP (nu_min pinned to
                  the smallest empirical sval, free amplitude and nu_max),
                  fitted to a Gaussian-broadened density. This is what
                  Thamm/Staats/Rosenow overlay on *trained* weights.
      "both"   -- draw both.
    ``bulk_mass`` is accepted for backward compatibility and ignored.
    """
    from ..config import apply_plot_style
    from .. import mp as MP
    from ..mp_fit import fit_modified_mp, modified_mp, gaussian_broaden
    apply_plot_style()
    import matplotlib.pyplot as plt

    s = np.sort(np.asarray(svals, dtype=float))
    fig, ax = plt.subplots(figsize=(6, 4))

    try:
        if domain == "nu":
            vals = s
            lo, hi = MP.mp_bounds(n, m, sigma)
            xlabel = "singular value ν"
            pdf_theory = lambda xs: MP.mp_pdf(xs, n, m, sigma)
        else:
            N = N or m
            vals = s ** 2 / N
            lo, hi = MP.mp_bounds_eig(n, m, sigma, N)
            xlabel = "eigenvalue λ=ν²/N"
            pdf_theory = lambda xs: MP.mp_pdf_eig(xs, n, m, sigma, N)

        ax.hist(vals, bins=_bin_edges(vals, lo, hi), density=True, alpha=0.5,
                label=f"empirical {'ν' if domain == 'nu' else 'λ'}")

        if mp_mode in ("theory", "both"):
            xs = _open_grid(max(lo, 1e-12) if domain != "nu" else lo, hi)
            ax.plot(xs, pdf_theory(xs) * _count_frac(vals, lo, hi),
                    "r-", lw=2, label="MP (theory, σ̂)")

        if mp_mode in ("fit", "both"):
            a, nu_min, nu_max = fit_modified_mp(s, win=broaden_win)
            nu_grid = _open_grid(nu_min, nu_max)
            nu_density = modified_mp(nu_grid, a, nu_max, nu_min)
            if domain == "nu":
                fit_x, fit_density = nu_grid, nu_density
            else:
                fit_x = nu_grid**2 / N
                fit_density = nu_density * N / (2.0 * np.maximum(nu_grid, np.finfo(float).tiny))
            ax.plot(fit_x, fit_density, "k--", lw=2, label="modified MP (fit)")
            xb = np.linspace(vals[0], vals[-1], 2000)
            ax.plot(xb, gaussian_broaden(xb, vals, win=broaden_win),
                    color="0.35", lw=1.2, label="broadened ESD")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.legend()
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(out_path)
    finally:
        plt.close(fig)
    return out_path
