"""rmt.mp — PART 1: Marchenko–Pastur theory.

Singular-value domain (Paper 3 v3 Eq. 4/5) plus the eigenvalue domain λ = ν²/N
(req 6, C = WWᵀ/N), the Gavish–Donoho median σ estimator (the USVT-cited
estimator of Paper 3), the USVT hard threshold, and the small-SV deviation
metrics (Paper 3's thesis).

Pure numpy/scipy — no torch.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from scipy import integrate, optimize


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _resolve_s_n_m(weight, s, n, m):
    """Return (s, n, m) from either a weight matrix or precomputed (s, n, m)."""
    if weight is not None:
        W = np.asarray(weight, dtype=np.float64)
        n_, m_ = int(W.shape[0]), int(W.shape[1])
        sv = np.linalg.svd(W, compute_uv=False).astype(np.float64)
        return np.sort(sv)[::-1], n_, m_
    if s is None or n is None or m is None:
        raise ValueError("provide either `weight` or all of `s, n, m`")
    return np.asarray(s, dtype=np.float64), int(n), int(m)


def _edges(n, m, sigma):
    mx, mn = max(n, m), min(n, m)
    nu_minus = sigma * (np.sqrt(mx) - np.sqrt(mn))
    nu_plus = sigma * (np.sqrt(mx) + np.sqrt(mn))
    return float(nu_minus), float(nu_plus)


# --------------------------------------------------------------------------- #
# singular-value domain  (Paper 3 v3 Eq. 4/5)                                  #
# --------------------------------------------------------------------------- #
def mp_pdf(x, n, m, sigma) -> np.ndarray:
    """Normalised MP singular-value density.

    P(ν) = q / (π σ̃² ν) · √((ν₊²-ν²)(ν²-ν₋²)),   ν ∈ [ν₋, ν₊]
    q = max/min,  σ̃ = σ·√max.  Zero outside the support; integrates to 1.
    """
    x = np.asarray(x, dtype=np.float64)
    mx, mn = max(n, m), min(n, m)
    q = mx / mn
    sig_t = sigma * np.sqrt(mx)
    nu_minus, nu_plus = _edges(n, m, sigma)

    out = np.zeros_like(x)
    inside = (x > nu_minus) & (x < nu_plus) & (x > 0)
    xi = x[inside]
    rad = (nu_plus**2 - xi**2) * (xi**2 - nu_minus**2)
    rad = np.clip(rad, 0.0, None)
    out[inside] = q / (np.pi * sig_t**2 * xi) * np.sqrt(rad)
    return out if out.shape else float(out)


def mp_bounds(n, m, sigma) -> Tuple[float, float]:
    """Edges (ν₋, ν₊) = σ(√max ∓ √min)."""
    return _edges(n, m, sigma)


def mp_cdf(x, n, m, sigma):
    """CDF of mp_pdf; 0 below ν₋, 1 above ν₊ (Gauss–Legendre quadrature)."""
    nu_minus, nu_plus = _edges(n, m, sigma)
    scalar = np.isscalar(x)
    xa = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(xa)
    # 200-node Gauss-Legendre on [nu_minus, t] per query (vectorised mapping).
    nodes, wts = np.polynomial.legendre.leggauss(200)
    for i, t in enumerate(xa):
        if t <= nu_minus:
            out[i] = 0.0
        elif t >= nu_plus:
            out[i] = 1.0
        else:
            a, b = nu_minus, t
            xm = 0.5 * (b - a) * nodes + 0.5 * (b + a)
            out[i] = 0.5 * (b - a) * np.sum(wts * mp_pdf(xm, n, m, sigma))
    out = np.clip(out, 0.0, 1.0)
    return float(out[0]) if scalar else out


def mp_median(n, m, sigma=1.0) -> float:
    """Median ν_med of the MP singular-value distribution (root-find on CDF=0.5)."""
    nu_minus, nu_plus = _edges(n, m, sigma)
    f = lambda t: float(mp_cdf(t, n, m, sigma)) - 0.5
    # Bracket strictly inside the open support.
    lo = nu_minus + 1e-9 * (nu_plus - nu_minus)
    hi = nu_plus - 1e-9 * (nu_plus - nu_minus)
    return float(optimize.brentq(f, lo, hi, xtol=1e-10, rtol=1e-12))


# --------------------------------------------------------------------------- #
# eigenvalue domain  λ = ν²/N,  C = WWᵀ/N   (req 6)                            #
# --------------------------------------------------------------------------- #
def eigenvalues_of_cov(weight=None, *, s=None, n=None, m=None, N=None) -> np.ndarray:
    """λ = ν²/N. Default N = #columns of W (the contraction dim) so C=WWᵀ/N has
    the standard MP eigenvalue law with variance σ²."""
    s, n_, m_ = _resolve_s_n_m(weight, s, n, m)
    if N is None:
        N = m_
    return (np.asarray(s, dtype=np.float64) ** 2) / float(N)


def mp_bounds_eig(n, m, sigma, N) -> Tuple[float, float]:
    """Exact image of mp_bounds under λ=ν²/N: (ν₋²/N, ν₊²/N)."""
    nu_minus, nu_plus = _edges(n, m, sigma)
    return (nu_minus**2 / float(N), nu_plus**2 / float(N))


def mp_pdf_eig(x, n, m, sigma, N) -> np.ndarray:
    """Eigenvalue MP density: image of mp_pdf under λ=ν²/N.

    P_λ(λ) = P_ν(√(Nλ)) · dν/dλ = P_ν(√(Nλ)) · √N/(2√λ). Integrates to 1.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    pos = x > 0
    nu = np.sqrt(float(N) * x[pos])
    jac = np.sqrt(float(N)) / (2.0 * np.sqrt(x[pos]))
    out[pos] = np.asarray(mp_pdf(nu, n, m, sigma)) * jac
    return out if out.shape else float(out)


def mp_cdf_eig(x, n, m, sigma, N):
    """Eigenvalue MP CDF: F_λ(λ) = F_ν(√(Nλ)) (monotone change of variables)."""
    scalar = np.isscalar(x)
    xa = np.atleast_1d(np.asarray(x, dtype=np.float64))
    nu = np.sqrt(np.clip(float(N) * xa, 0.0, None))
    res = np.atleast_1d(mp_cdf(nu, n, m, sigma))
    return float(res[0]) if scalar else res


# --------------------------------------------------------------------------- #
# σ estimation — Gavish–Donoho median  (= USVT-cited estimator, Paper 3 §3)    #
# --------------------------------------------------------------------------- #
def estimate_sigma_gd_median(weight=None, *, s=None, n=None, m=None,
                             discard_largest=0.0) -> float:
    """σ̂ from singular-value median matching to the MP median (Gavish–Donoho).

    The empirical median ν_med of W = (iid N(0,σ²) factor) equals σ·μ₁ where
    μ₁ = mp_median(n, m, 1.0) is the σ=1 MP singular-value median.  Hence
    σ̂ = ν_med / μ₁.  Equivalently σ̂ = ν_med /(√max·√μ_q) with the eigenvalue
    MP median μ_q = μ₁²/max — the Gavish–Donoho / USVT form Paper 3 cites.
    """
    s, n, m = _resolve_s_n_m(weight, s, n, m)
    s = np.sort(np.asarray(s, dtype=np.float64))[::-1]
    if discard_largest and discard_largest > 0:
        k_drop = int(np.floor(discard_largest * len(s)))
        if k_drop > 0:
            s = s[k_drop:]
    nu_med = float(np.median(s))
    mu1 = mp_median(n, m, 1.0)           # σ=1 MP singular-value median
    return nu_med / mu1


# Backward-compatible alias (locked by test_estimate_sigma_alias).
estimate_sigma_med = estimate_sigma_gd_median


def estimate_sigma_med_refined(weight=None, *, s=None, n=None, m=None,
                               max_iter=3) -> Tuple[float, float, int]:
    """Iterative: estimate σ → drop ν>ν₊ outliers → re-estimate, until stable.

    Returns (sigma, sigma_refined, n_iter).
    """
    s, n, m = _resolve_s_n_m(weight, s, n, m)
    s = np.sort(np.asarray(s, dtype=np.float64))[::-1]
    sigma0 = estimate_sigma_gd_median(s=s, n=n, m=m)
    sigma = sigma0
    kept = s.copy()
    prev_count = len(kept)
    n_iter = 0
    for _ in range(int(max_iter)):
        n_iter += 1
        _, nu_plus = _edges(n, m, sigma)
        kept = s[s <= nu_plus]
        if len(kept) < max(10, 0.05 * len(s)):
            kept = s            # safety: never discard the whole spectrum
            break
        new_sigma = estimate_sigma_gd_median(s=kept, n=n, m=m)
        if len(kept) == prev_count and abs(new_sigma - sigma) < 1e-9:
            sigma = new_sigma
            break
        prev_count = len(kept)
        sigma = new_sigma
    return float(sigma0), float(sigma), int(n_iter)


def usvt_hard_threshold(n, m, sigma, *, square_optimal=True) -> float:
    """USVT / Gavish–Donoho denoising cutoff.

    τ = (2+ε)σ√max(n,m); if square_optimal and n==m, the optimal (4/√3)σ√n.
    """
    mx = max(n, m)
    if square_optimal and n == m:
        return float((4.0 / np.sqrt(3.0)) * sigma * np.sqrt(n))
    eps = 0.0
    return float((2.0 + eps) * sigma * np.sqrt(mx))


def small_sv_deviation(s, n, m, sigma) -> dict:
    """Departures at the *small* singular-value edge (Paper 3's thesis).

    Returns {ks_lower, n_below_minus, frac_mass_below_minus, excess_small_sv}.
    - ks_lower: KS of empirical CDF of ν vs MP CDF on [ν₋, ν_med].
    - n_below_minus: #{ν < ν₋}.
    - frac_mass_below_minus: energy fraction Σν²[ν<ν₋]/Σν².
    - excess_small_sv: #{ν ≤ q10_MP} − 0.1·len(s), q10 = 10th MP percentile.
    """
    s = np.sort(np.asarray(s, dtype=np.float64))   # ascending
    nu_minus, nu_plus = _edges(n, m, sigma)

    n_below_minus = int(np.sum(s < nu_minus))
    total_energy = float(np.sum(s**2))
    frac_mass_below_minus = (float(np.sum(s[s < nu_minus] ** 2)) / total_energy
                             if total_energy > 0 else 0.0)

    # lowest MP decile threshold (q10) via root-find on the CDF
    try:
        f = lambda t: float(mp_cdf(t, n, m, sigma)) - 0.10
        q10 = float(optimize.brentq(f, nu_minus + 1e-9, nu_plus - 1e-9))
    except Exception:
        q10 = nu_minus + 0.1 * (nu_plus - nu_minus)
    n_le_q10 = int(np.sum(s <= q10))
    excess_small_sv = float(n_le_q10 - 0.10 * len(s))

    # KS on the lower half of the bulk [ν₋, median]
    nu_med = mp_median(n, m, sigma)
    band = s[(s >= nu_minus) & (s <= nu_med)]
    if band.size >= 5:
        F0 = float(mp_cdf(nu_minus, n, m, sigma))
        F1 = float(mp_cdf(nu_med, n, m, sigma))
        denom = max(F1 - F0, 1e-12)
        emp = np.arange(1, band.size + 1) / band.size
        theo = (np.asarray(mp_cdf(band, n, m, sigma)) - F0) / denom
        ks_lower = float(np.max(np.abs(emp - theo)))
    else:
        ks_lower = float("nan")

    return {
        "ks_lower": ks_lower,
        "n_below_minus": n_below_minus,
        "frac_mass_below_minus": frac_mass_below_minus,
        "excess_small_sv": excess_small_sv,
    }
