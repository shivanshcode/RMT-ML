"""rmt.spacing — PART 4: bulk RMT universality on level spectra.

Atas r-statistic, polynomial unfolding, NN-spacing KS vs Wigner-GOE/Poisson,
Dyson–Mehta Δ₃(L) and number variance Σ²(L) (standard GOE asymptotics with
prefactor and constant), and the complex spacing ratio for the Ginibre test.
Pure numpy/scipy.
"""
from __future__ import annotations

import numpy as np

GAMMA = 0.5772156649015329       # Euler–Mascheroni


# --------------------------------------------------------------------------- #
# r-statistic (Atas 2013, no unfolding)                                        #
# --------------------------------------------------------------------------- #
def r_statistic(levels) -> float:
    """Mean adjacent-gap ratio, preserving physical zero-gap degeneracies."""
    x = np.sort(np.asarray(levels, dtype=np.float64))
    x = x[np.isfinite(x)]
    d = np.diff(x)
    if d.size < 2:
        return float("nan")
    denominator = np.maximum(d[:-1], d[1:])
    valid = denominator > 0.0  # only 0/0 is undefined; 0/positive contributes 0
    if not np.any(valid):
        return float("nan")
    r = np.minimum(d[:-1][valid], d[1:][valid]) / denominator[valid]
    return float(np.mean(r))


# --------------------------------------------------------------------------- #
# unfolding                                                                    #
# --------------------------------------------------------------------------- #
def unfold(levels, deg=7) -> np.ndarray:
    """Unfold via a polynomial fit to the integrated density (staircase).

    Returns the unfolded levels ξ = N̄(λ); mean NN spacing ≈ 1.
    """
    x = np.sort(np.asarray(levels, dtype=np.float64))
    x = x[np.isfinite(x)]
    N = x.size
    if N < 3:
        raise ValueError("at least three finite levels are required")
    unique, first, counts = np.unique(x, return_index=True, return_counts=True)
    if unique.size < 2:
        raise ValueError("unfolding requires at least two distinct levels")
    # Fit in scaled coordinates, reducing complexity until the fitted density is
    # monotone.  Never substitute the empirical ranks evaluated at themselves:
    # that turns every distinct input spectrum into an artificial picket fence.
    center, scale = float(np.mean(unique)), max(float(np.ptp(unique)), 1.0)
    coords = (unique - center) / scale
    targets = first + 0.5 * (counts + 1)
    smooth_unique = None
    for fitted_deg in range(min(max(1, int(deg)), unique.size - 1), 0, -1):
        coeffs = np.polyfit(coords, targets, fitted_deg)
        candidate = np.polyval(coeffs, coords)
        if np.all(np.isfinite(candidate)) and np.all(np.diff(candidate) > 0.0):
            smooth_unique = candidate
            break
    if smooth_unique is None:
        raise ValueError("no monotone polynomial unfolding could be fitted")
    xi = np.interp(x, unique, smooth_unique)
    gaps = np.diff(xi)
    mean_gap = float(np.mean(gaps))
    if not np.isfinite(mean_gap) or mean_gap <= 0.0:
        raise ValueError("unfolding has no positive mean spacing")
    return (xi - xi[0]) / mean_gap


def nn_spacing(levels, deg=7) -> np.ndarray:
    """Normalised nearest-neighbour spacings (mean 1) of the unfolded spectrum."""
    xi = unfold(levels, deg=deg)
    s = np.diff(xi)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return s
    mean_gap = float(np.mean(s))
    if not np.isfinite(mean_gap) or mean_gap <= 0.0:
        return s
    # Multiplicities are part of the empirical ESD, so normalize over the full
    # spacing sample, including its zero mass.
    return s / mean_gap


def wigner_goe_cdf(s) -> np.ndarray:
    """Wigner surmise GOE CDF: 1 - exp(-π s²/4)."""
    s = np.asarray(s, dtype=np.float64)
    return 1.0 - np.exp(-np.pi * s**2 / 4.0)


def poisson_cdf(s) -> np.ndarray:
    """Poisson NN-spacing CDF: 1 - exp(-s)."""
    s = np.asarray(s, dtype=np.float64)
    return 1.0 - np.exp(-s)


def _ks_against(sample_sorted, cdf_func) -> float:
    n = sample_sorted.size
    theo = cdf_func(sample_sorted)
    emp_hi = np.arange(1, n + 1) / n
    emp_lo = np.arange(0, n) / n
    return float(max(np.max(np.abs(emp_hi - theo)), np.max(np.abs(theo - emp_lo))))


def nn_spacing_ks(levels, deg=7) -> dict:
    """{nn_KS_GOE, nn_KS_Poisson} — KS of P(s) vs Wigner-GOE and Poisson."""
    s = np.sort(nn_spacing(levels, deg=deg))
    s = s[s >= 0]
    if s.size < 5:
        return {"nn_KS_GOE": float("nan"), "nn_KS_Poisson": float("nan")}
    return {"nn_KS_GOE": _ks_against(s, wigner_goe_cdf),
            "nn_KS_Poisson": _ks_against(s, poisson_cdf)}


# --------------------------------------------------------------------------- #
# number variance Σ²(L) and spectral rigidity Δ₃(L)                            #
# --------------------------------------------------------------------------- #
def sigma2(levels, L, deg=7, *, seed=None, rng=None) -> float:
    """Number variance Σ²(L) over random windows.

    ``seed``/``rng`` makes the sampling part of run-level reproducibility.
    GOE: Σ²(L) ≈ (2/π²)[ln(2πL)+γ+1−π²/8].
    """
    L = float(L)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("L must be finite and positive")
    xi = np.sort(unfold(levels, deg=deg))
    lo, hi = xi[0], xi[-1]
    span = hi - lo
    if span <= L:
        return float("nan")
    n_windows = max(200, int(10 * span / L))
    generator = rng if rng is not None else np.random.default_rng(seed)
    starts = generator.uniform(lo, hi - L, size=n_windows)
    counts = np.array([np.count_nonzero((xi >= a) & (xi < a + L)) for a in starts],
                      dtype=np.float64)
    return float(np.var(counts))


def delta3(levels, L, deg=7, *, seed=None, rng=None) -> float:
    """Dyson–Mehta spectral rigidity Δ₃(L) via least-squares staircase fit.

    ``seed``/``rng`` makes the random windows part of run-level reproducibility.
    Δ₃(L) = ⟨ min_{a,b} (1/L)∫_x^{x+L}(N(ξ)−a−bξ)² dξ ⟩.
    GOE: Δ₃(L) ≈ (1/π²)[ln(2πL)+γ−5/4−π²/8].
    """
    L = float(L)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("L must be finite and positive")
    xi = np.sort(unfold(levels, deg=deg))
    lo, hi = xi[0], xi[-1]
    span = hi - lo
    if span <= L:
        return float("nan")
    generator = rng if rng is not None else np.random.default_rng(seed)
    n_windows = max(150, int(8 * span / L))
    starts = generator.uniform(lo, hi - L, size=n_windows)
    vals = []
    for a in starts:
        b = a + L
        pts = xi[(xi >= a) & (xi < b)]
        # least-squares linear fit to the staircase N(ξ) under the L2 integral
        # norm via dense sampling (fast and robust for our L<=50)
        npts = 400
        t = np.linspace(a, b, npts)
        # staircase value at t: number of levels < t
        Nt = np.searchsorted(pts, t, side="right").astype(np.float64)
        A = np.vstack([np.ones_like(t), t]).T
        coef, *_ = np.linalg.lstsq(A, Nt, rcond=None)
        resid = Nt - A @ coef
        integrate = getattr(np, "trapezoid", None)
        if integrate is None:  # NumPy < 2.0
            integrate = getattr(np, "trapz")
        d3 = integrate(resid**2, t) / L
        vals.append(d3)
    return float(np.mean(vals))


# --------------------------------------------------------------------------- #
# complex spacing ratio (Ginibre test)                                         #
# --------------------------------------------------------------------------- #
def complex_spacing_ratio(matrix) -> dict:
    """z_k = (NN − λ_k)/(NNN − λ_k) for complex eigenvalues of ``matrix``.

    Returns {abs_mean = ⟨|z|⟩, cos_mean = ⟨cos arg z⟩}.
    GinUE: 0.738 / −0.24.  Poisson-2D: 0.667 / 0.
    """
    M = np.asarray(matrix)
    ev = np.linalg.eigvals(M)
    n = ev.size
    if n < 4:
        return {"abs_mean": float("nan"), "cos_mean": float("nan")}
    zs = []
    for k in range(n):
        d = ev - ev[k]
        d = np.delete(d, k)
        order = np.argsort(np.abs(d))
        nn = d[order[0]]
        nnn = d[order[1]]
        if nnn == 0:
            continue
        zs.append(nn / nnn)
    zs = np.asarray(zs)
    return {"abs_mean": float(np.mean(np.abs(zs))),
            "cos_mean": float(np.mean(np.cos(np.angle(zs))))}


# --------------------------------------------------------------------------- #
# reference Mehta asymptotics (for plotting / tests)                           #
# --------------------------------------------------------------------------- #
def sigma2_goe_theory(L) -> float:
    return (2.0 / np.pi**2) * (np.log(2 * np.pi * L) + GAMMA + 1.0 - np.pi**2 / 8.0)


def delta3_goe_theory(L) -> float:
    return (1.0 / np.pi**2) * (np.log(2 * np.pi * L) + GAMMA - 5.0 / 4.0 - np.pi**2 / 8.0)
