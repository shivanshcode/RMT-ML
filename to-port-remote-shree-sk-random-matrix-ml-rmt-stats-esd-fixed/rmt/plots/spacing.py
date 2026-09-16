"""NN-spacing histogram + CDF inset, and the Sigma^2(L) / Delta_3(L) panel.

Layout follows Thamm ``src/plot_utils.py``:
  * ``plot_spacingDistribution``  -> histogram of P(s) with a CDF inset
  * ``plot_levelNumberVariance``  -> Sigma^2(L) with the GOE log-law overlay
"""
from __future__ import annotations

import os
import numpy as np


def _ensure_dir(out_path):
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_nn_spacing(levels, out_path, *, deg=7, method="auto", win_size=15,
                    bins=40, title=None):
    """P(s) vs Wigner-GOE / Poisson, with the empirical CDF as an inset."""
    from ..config import apply_plot_style
    from .. import spacing as SP
    apply_plot_style()
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    s = SP.nn_spacing(levels, deg=deg, method=method, win_size=win_size)
    s = s[np.isfinite(s)]
    if s.size < 5:
        raise ValueError("too few spacings to plot")
    ks = SP.nn_spacing_ks(levels, deg=deg, method=method, win_size=win_size)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(s, bins=bins, density=True, alpha=0.55, label="$P(s)$")
    xs = np.linspace(0, max(4.0, float(s.max())), 300)
    ax.plot(xs, SP.wigner_goe_pdf(xs), "r-", lw=1.5, label="Wigner-GOE")
    ax.plot(xs, np.exp(-xs), "g--", lw=1.5, label="Poisson")
    ax.set_xlabel("$s$")
    ax.set_ylabel("$P(s)$")
    ax.set_xlim(0, max(4.0, float(np.percentile(s, 99.5))))
    ax.legend(loc="upper right", fontsize=8)
    sub = (f"$\\langle s\\rangle$={np.mean(s):.3f}  "
           f"$D_{{GOE}}$={ks['nn_KS_GOE']:.3f} ($p$={ks['nn_KS_GOE_p']:.2g})")
    ax.set_title(title or sub, fontsize=9)

    # CDF inset (Thamm plot_spacingDistribution)
    inax = inset_axes(ax, width="42%", height="42%", loc="center right",
                      borderpad=1.1)
    xso = np.sort(s)
    inax.plot(xso, np.arange(xso.size) / xso.size, lw=1.2, label="data")
    inax.plot(xso, SP.wigner_goe_cdf(xso), "r-", lw=1.0)
    inax.plot(xso, SP.poisson_cdf(xso), "g--", lw=1.0)
    inax.set_xlim(0, 3.2)
    inax.set_ylim(-0.05, 1.1)
    inax.set_xlabel("$s$", labelpad=0, fontsize=7)
    inax.set_ylabel("cdf", labelpad=0, fontsize=7)
    inax.tick_params(labelsize=6)

    _ensure_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_rigidity(levels, out_path, *, method="auto", deg=7, win_size=15,
                  L_sigma=(1, 2, 3, 5, 8, 10, 15, 20, 35, 50),
                  L_delta=(2, 5, 10, 20, 35, 50), title=None):
    """Sigma^2(L) and Delta_3(L) against the GOE and Poisson references.

    The Sigma^2 panel is truncated at the L calibrated for the unfolding in use
    (``rmt.spacing.SIGMA2_RELIABLE_LMAX[method]``): a local kernel cannot
    preserve count fluctuations beyond its own window.
    """
    from ..config import apply_plot_style
    from .. import spacing as SP
    apply_plot_style()
    import matplotlib.pyplot as plt

    lmax = SP.SIGMA2_RELIABLE_LMAX.get(method, 15.0)
    Ls = np.array([L for L in L_sigma if L <= lmax], float)
    Ld = np.array(list(L_delta), float)
    s2 = np.array([SP.sigma2(levels, L, deg=deg, method=method,
                             win_size=win_size) for L in Ls])
    d3 = np.array([SP.delta3(levels, L, deg=deg, method=method,
                             win_size=win_size) for L in Ld])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.plot(Ls, s2, "o-", ms=4, label="data")
    ax.plot(Ls, [SP.sigma2_goe_theory(L) for L in Ls], "k--", label="GOE")
    ax.plot(Ls, Ls, "g:", label="Poisson")
    ax.set_xlabel("$L$"); ax.set_ylabel("$\\Sigma^2(L)$")
    ax.set_ylim(0, max(2.0, float(np.nanmax(s2)) * 1.2))
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(Ld, d3, "o-", ms=4, label="data")
    ax.plot(Ld, [SP.delta3_goe_theory(L) for L in Ld], "k--", label="GOE")
    ax.plot(Ld, Ld / 15.0, "g:", label="Poisson")
    ax.set_xlabel("$L$"); ax.set_ylabel("$\\Delta_3(L)$")
    ax.set_ylim(0, max(0.6, float(np.nanmax(d3)) * 1.2))
    ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_porter_thomas(svals, pvalues, out_path, *, alpha=0.05, title=None):
    """Porter-Thomas p-value per singular vector against its singular value.

    Mirrors Thamm ``plot_utils.plot_p_values_for_svecs``. Takes the p-values
    already computed by ``per_matrix_analysis`` (stashed via ``ptvals_out``), so
    nothing is recomputed and the singular vectors need not be kept alive.
    """
    from ..config import apply_plot_style
    apply_plot_style()
    import matplotlib.pyplot as plt

    x = np.asarray(svals, dtype=np.float64)
    p = np.asarray(pvalues, dtype=np.float64)
    k = min(x.size, p.size)
    x, p = x[:k], p[:k]
    if k == 0:
        raise ValueError("no p-values to plot")
    order = np.argsort(x)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    ax.plot(x[order], p[order], ".", ms=3)
    ax.axhline(alpha, color="r", ls="--", lw=1.0, label=f"$\\alpha$={alpha}")
    ax.set_xlabel("singular value $\\nu$")
    ax.set_ylabel("Porter-Thomas $p$-value")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)

    ax = axes[1]
    rank = np.arange(1, k + 1)          # p is ordered by descending sval
    ax.plot(rank, p, ".", ms=3)
    ax.axhline(alpha, color="r", ls="--", lw=1.0)
    ax.set_xlabel("rank (1 = largest $\\nu$)")
    ax.set_ylabel("$p$-value")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)

    fig.suptitle(title or f"non-rejected fraction = {np.mean(p > alpha):.2f}",
                 fontsize=9)
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_porter_thomas_deciles(pvalues, out_path, *, alpha=0.05, n_deciles=10,
                               title=None):
    """Mean p-value and non-rejected fraction per spectral decile.

    Decile 1 is the top 10% of the spectrum. A trained matrix is expected to
    show Porter-Thomas survival across the bulk deciles and rejection in the
    leading ones, i.e. localisation confined to the large singular vectors.
    """
    from ..config import apply_plot_style
    from ..porter_thomas import porter_thomas_by_decile
    apply_plot_style()
    import matplotlib.pyplot as plt

    d = porter_thomas_by_decile(pvalues, alpha=alpha, n_deciles=n_deciles)
    if not d["decile"]:
        raise ValueError("no p-values to plot")
    x = np.array(d["decile"], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(x - 0.19, d["p_mean"], width=0.38, label="mean $p$", color="#4C72B0")
    ax.bar(x + 0.19, d["frac_random"], width=0.38,
           label=f"frac $p>\\alpha$", color="#DD8452")
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.axhline(1 - alpha, color="r", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("spectral decile (1 = largest singular values)")
    ax.set_ylabel("Porter-Thomas statistic")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title or "Porter-Thomas by decile", fontsize=9)
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
