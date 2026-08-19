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
    nb = int(np.clip(round((right - left) / h), 50, 400))
    return np.linspace(left, right, nb + 1)


def _count_frac(vals, lo, hi):
    """Fraction of values (by count) inside [lo, hi]."""
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return 1.0
    return int(np.sum((vals >= lo) & (vals <= hi))) / vals.size


def _cheb_grid(lo, hi, n=400):
    """Chebyshev-Gauss nodes on the OPEN interval (lo, hi).

    The MP density is defined by strict inequalities, so it must never be
    evaluated *at* an edge -- that returns exactly 0 and draws a segment down to
    the axis. But a uniform open grid is also wrong: the density rises like
    sqrt(distance from edge), so the first uniform node (1/n of the way in) is
    still at a visibly non-zero height and the curve looks chopped off. Cosine
    spacing puts the first node ~pi^2/(8 n^2) of the support from the edge --
    ~1e-5 at n = 400 -- which both resolves the sqrt flank and lands close
    enough to the edge that the curve visually reaches it.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.array([], dtype=float)
    k = np.arange(1, int(n) + 1, dtype=float)
    theta = (k - 0.5) * np.pi / float(n)
    return 0.5 * (lo + hi) - 0.5 * (hi - lo) * np.cos(theta)


def _mp_curve(pdf, lo, hi, *, left="zero", n=400, div_offset=1e-3):
    """(xs, ys) for an MP-type density on [lo, hi], with the edges handled.

    ``left`` describes the behaviour of the density at the *left* edge:

      "zero"      -- density -> 0 there (rectangular case, nu_minus > 0).
                     Append an exact zero at lo so the curve closes.
      "finite"    -- density -> a finite non-zero limit there. This is the
                     square nu-domain case: with nu_minus = 0 the
                     (a/x)*sqrt((nu_plus^2 - x^2)(x^2 - 0)) form cancels to
                     sqrt(nu_plus^2 - x^2)/(pi*sigma_tilde^2), i.e. 2/(pi*sigma_tilde)
                     as x -> 0+. Start just inside and do NOT append a zero --
                     that zero is the spurious vertical line at nu = 0.
      "divergent" -- density -> +inf there (square *lambda*-domain: lambda^-1/2).
                     Start a fixed fraction of the support in, so the drawn peak
                     height is reproducible instead of being set by the grid size.

    The right edge always closes to zero in both domains.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.array([]), np.array([])
    start = lo + div_offset * (hi - lo) if left == "divergent" else lo
    xs = _cheb_grid(start, hi, n)
    if xs.size == 0:
        return np.array([]), np.array([])
    ys = np.asarray(pdf(xs), dtype=float)
    if left == "zero":
        xs = np.concatenate(([lo], xs))
        ys = np.concatenate(([0.0], ys))
    xs = np.concatenate((xs, [hi]))
    ys = np.concatenate((ys, [0.0]))
    return xs, ys


def plot_esd(svals, n, m, sigma, out_path, *, domain="nu", N=None,
             bulk_mass=1.0, mp_mode="theory", broaden_win=15,
             show_edges=True, xlim_mode="data"):
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

    ``show_edges`` draws thin vertical markers at the MP support edges so they
    are visible even where the density there is small.

    ``xlim_mode``:
      "data" -- full data range (default; unchanged behaviour).
      "mp"   -- clip the x-axis to just past the MP support.
      "auto" -- "mp" only when the spectrum extends past 3x nu_plus, i.e. only
                when the overlay would otherwise be squeezed into a few pixels.
      Under "mp"/"auto" the histogram is binned over the clipped range (the
      alternative -- binning over the full range and merely narrowing the view --
      leaves the whole bulk inside one bar whenever max(s) >> nu_plus), and the
      excluded fraction is reported in the title. Densities are then conditional
      on the shown range; the MP overlay is rescaled by the same count fraction
      either way.

    ``bulk_mass`` is accepted for backward compatibility and ignored.
    """
    from ..config import apply_plot_style
    from .. import mp as MP
    from ..mp_fit import fit_modified_mp, modified_mp, gaussian_broaden
    apply_plot_style()
    import matplotlib.pyplot as plt

    s = np.sort(np.asarray(svals, dtype=float))
    fig, ax = plt.subplots(figsize=(6, 4))

    if domain == "nu":
        vals = s
        lo, hi = MP.mp_bounds(n, m, sigma)
        xlabel = "singular value ν"
        edge_lbl = ("ν₋", "ν₊")
        pdf_theory = lambda xs: MP.mp_pdf(xs, n, m, sigma)
        # nu_minus = 0 for a square matrix; the density is finite there.
        left_kind = "zero" if lo > 0 else "finite"
    else:
        N = N or m
        vals = s ** 2 / N
        lo, hi = MP.mp_bounds_eig(n, m, sigma, N)
        xlabel = "eigenvalue λ=ν²/N"
        edge_lbl = ("λ₋", "λ₊")
        pdf_theory = lambda xs: MP.mp_pdf_eig(xs, n, m, sigma, N)
        # lambda_minus = 0 for a square matrix; the density diverges as λ^(-1/2).
        left_kind = "zero" if lo > 0 else "divergent"

    # Decide the view *before* binning. _bin_edges spans the full data range at
    # a fixed 400-bin cap, so on a spectrum whose maximum is orders of magnitude
    # past ν₊ the whole MP support falls inside a single bar. When the view is
    # clipped, bin over the clipped range instead so the bulk is resolved.
    clip = (xlim_mode == "mp") or (
        xlim_mode == "auto" and np.isfinite(hi) and hi > 0
        and float(np.max(vals)) > 3.0 * hi)
    clip = bool(clip and np.isfinite(hi) and hi > 0)
    view_right = hi * 1.15 if clip else None

    hist_vals = vals[vals <= view_right] if clip else vals
    if hist_vals.size < 2:
        hist_vals, clip = vals, False
    ax.hist(hist_vals, bins=_bin_edges(hist_vals, lo, min(hi, float(hist_vals.max()))),
            density=True, alpha=0.5,
            label=f"empirical {'ν' if domain == 'nu' else 'λ'}")

    if mp_mode in ("theory", "both"):
        frac = _count_frac(vals, lo, hi)
        xs, ys = _mp_curve(pdf_theory, lo, hi, left=left_kind)
        if xs.size:
            ax.plot(xs, ys * frac, "r-", lw=2, label="MP (theory, σ̂)")

    if mp_mode in ("fit", "both"):
        a, nu_min, nu_max = fit_modified_mp(vals, win=broaden_win)
        # nu_min is pinned to a positive empirical sval, so the fitted density
        # vanishes at both ends: close the curve on both sides.
        xs, ys = _mp_curve(lambda x: modified_mp(x, a, nu_max, nu_min),
                           nu_min, nu_max, left="zero")
        if xs.size:
            ax.plot(xs, ys, "k--", lw=2, label="modified MP (fit)")
        xb = np.linspace(vals[0], vals[-1], 2000)
        ax.plot(xb, gaussian_broaden(xb, vals, win=broaden_win),
                color="0.35", lw=1.2, label="broadened ESD")

    if show_edges and np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        for x, lab in ((lo, edge_lbl[0]), (hi, edge_lbl[1])):
            ax.axvline(x, color="r", ls=":", lw=1.0, alpha=0.8)
        ax.plot([], [], color="r", ls=":", lw=1.0,
                label=f"MP edges {edge_lbl[0]}={lo:.3g}, {edge_lbl[1]}={hi:.3g}")

    if clip:
        left = min(float(np.min(vals)), lo)
        ax.set_xlim(left, view_right)
        beyond = float(np.mean(np.asarray(vals) > view_right))
        if beyond > 0:
            ax.set_title(f"x-axis clipped to MP support; {beyond:.1%} of the "
                         f"spectrum lies beyond and is not binned here",
                         fontsize=9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
