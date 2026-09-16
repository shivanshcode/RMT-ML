"""rmt.nulls — matched-null calibration for the bulk-RMT and Porter-Thomas blocks.

This module fixes three defects in the analysis layer of ``rmt.spacing`` that
are invisible to the existing unit tests because those tests compare against
large-L asymptotics rather than against the pipeline's own null:

1. ``delta3_goe_theory`` / ``sigma2_goe_theory`` are L -> infinity asymptotics.
   The beta = 1 Delta_3 asymptote is ~11% low at L = 5 and ~26% low at L = 3,
   so a *perfect* Gaussian matrix reports a "deviation" at those L. Fixed here
   by :func:`null_band`, which measures what the pipeline actually returns on
   synthetic matrices of the caller's exact shape.

2. ``nn_spacing_ks``'s p-values come from ``scipy.stats.kstest``, whose null
   assumes iid draws from a *fully specified* CDF. Neither holds: unfolded
   spacings are anti-correlated, and the unfolding is fitted to the same data
   (a Lilliefors effect). Measured null medians are 0.5-0.75 instead of 0.5 with
   frac(p < 0.05) = 0.00, i.e. the test has almost no power. Fixed here by
   :func:`ks_pvalue_mc`, a parametric-bootstrap p-value.

3. ``wigner_goe_cdf`` is the 2x2 Wigner surmise, not the exact Gaudin beta = 1
   law; the two differ by 0.0077 in CDF, which exceeds the KS 5% critical value
   once more than ~31000 spacings are pooled. :func:`ks_pvalue_mc` removes this
   too, because the same surmise is used for the observed statistic *and* for
   every null replica, so its bias cancels exactly. That is why no exact Gaudin
   CDF is needed.

Also provides the beta = 2 / beta = 4 Mehta laws and the GSE / circular-ensemble
generators the suite needs to benchmark itself beyond GOE.

Pure numpy/scipy. Imports ``rmt.spacing`` only.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, Optional, Sequence, Union
import warnings

import numpy as np
import scipy.stats

from . import spacing as SP

_RngLike = Union[int, np.random.Generator, None]

GAMMA = 0.5772156649015329


def _as_rng(rng: _RngLike) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


# --------------------------------------------------------------------------- #
# beta-general Mehta asymptotics                                               #
# --------------------------------------------------------------------------- #
#: <r> = <min(d_i,d_i+1)/max(...)>, Atas et al. 2013 Table I. beta=0 is Poisson.
R_THEORY: Dict[int, float] = {0: 0.386294, 1: 0.5307, 2: 0.5996, 4: 0.6744}


def sigma2_theory(L, beta: int = 1) -> float:
    """Number variance Sigma^2(L), Mehta ch. 16.  beta in {0,1,2,4}; 0 = Poisson.

    These are L -> infinity asymptotics. Do NOT use them as the reference for a
    hypothesis test at L < 20 -- use :func:`null_band` instead. They are correct
    for plotting a reference curve.
    """
    L = float(L)
    if beta == 0:
        return L
    if beta == 1:
        return (2.0 / np.pi ** 2) * (np.log(2 * np.pi * L) + GAMMA + 1.0 - np.pi ** 2 / 8.0)
    if beta == 2:
        return (1.0 / np.pi ** 2) * (np.log(2 * np.pi * L) + GAMMA + 1.0)
    if beta == 4:
        return (1.0 / (2 * np.pi ** 2)) * (np.log(4 * np.pi * L) + GAMMA + 1.0 + np.pi ** 2 / 8.0)
    raise ValueError("beta must be 0, 1, 2 or 4")


def delta3_theory(L, beta: int = 1) -> float:
    """Dyson-Mehta rigidity Delta_3(L). Same caveat as :func:`sigma2_theory`."""
    L = float(L)
    if beta == 0:
        return L / 15.0
    if beta == 1:
        return (1.0 / np.pi ** 2) * (np.log(2 * np.pi * L) + GAMMA - 5.0 / 4.0 - np.pi ** 2 / 8.0)
    if beta == 2:
        return (1.0 / (2 * np.pi ** 2)) * (np.log(2 * np.pi * L) + GAMMA - 5.0 / 4.0)
    if beta == 4:
        return (1.0 / (4 * np.pi ** 2)) * (np.log(4 * np.pi * L) + GAMMA - 5.0 / 4.0 + np.pi ** 2 / 8.0)
    raise ValueError("beta must be 0, 1, 2 or 4")


def wigner_surmise_cdf(s, beta: int = 1) -> np.ndarray:
    """2x2 Wigner surmise CDF for beta = 1, 2, 4.

    Approximation to the true Gaudin law: the beta = 1 error is 0.0077 in CDF,
    which matters only if you pool >~31000 spacings against a fixed threshold.
    :func:`ks_pvalue_mc` cancels it; a bare KS distance does not.
    """
    from scipy.special import erf
    s = np.asarray(s, dtype=np.float64)
    if beta == 1:
        return 1.0 - np.exp(-np.pi * s ** 2 / 4.0)
    if beta == 2:
        a = 4.0 / np.pi
        return erf(np.sqrt(a) * s) - 2.0 * np.sqrt(a / np.pi) * s * np.exp(-a * s ** 2)
    if beta == 4:
        a = 64.0 / (9.0 * np.pi)
        r = np.sqrt(a) * s
        # integral of C x^4 exp(-a x^2), C = 2^18/(3^6 pi^3), in closed form
        return (erf(r) - (2.0 / np.sqrt(np.pi)) * np.exp(-r ** 2)
                * (r + (2.0 / 3.0) * r ** 3))
    raise ValueError("beta must be 1, 2 or 4")


# --------------------------------------------------------------------------- #
# Ensembles the shipped rmt.ensembles does not provide                         #
# --------------------------------------------------------------------------- #
def gse(n: int, rng: _RngLike = None) -> np.ndarray:
    """GSE as a 2n x 2n complex Hermitian self-dual matrix (Kramers-degenerate)."""
    g = _as_rng(rng)
    A0 = g.standard_normal((n, n)); A0 = (A0 + A0.T) / 2.0
    A1 = g.standard_normal((n, n)); A1 = (A1 - A1.T) / 2.0
    A2 = g.standard_normal((n, n)); A2 = (A2 - A2.T) / 2.0
    A3 = g.standard_normal((n, n)); A3 = (A3 - A3.T) / 2.0
    I2 = np.eye(2)
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]])
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)
    H = (np.kron(A0, I2) + 1j * np.kron(A1, s1)
         + 1j * np.kron(A2, s2) + 1j * np.kron(A3, s3))
    return (H + H.conj().T) / (2.0 * np.sqrt(n))


def gse_levels(n: int, rng: _RngLike = None) -> np.ndarray:
    """The n distinct GSE levels (every Kramers pair collapsed to one)."""
    return np.linalg.eigvalsh(gse(n, rng))[::2]


def _haar_unitary(n: int, g: np.random.Generator) -> np.ndarray:
    z = (g.standard_normal((n, n)) + 1j * g.standard_normal((n, n))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    return q * (d / np.abs(d))


def circular_levels(n: int, beta: int = 1, rng: _RngLike = None,
                    unfolded: bool = True) -> np.ndarray:
    """Eigenphases of COE (beta=1) / CUE (beta=2) / CSE (beta=4).

    **The point of these ensembles**: the eigenphase density is *exactly*
    uniform, so xi = n * theta / (2 pi) is an exact unfolding. Any discrepancy
    against theory is therefore estimator bias or asymptotic-formula error --
    never unfolding error. This is the only clean way to tell the three apart,
    and it is what shows that the beta = 1 Delta_3 asymptote (not the code) is
    responsible for the L = 5 discrepancy.

    With ``unfolded=True`` the returned levels already have mean spacing 1 and
    should be passed to ``delta3(..., unfolded=...)`` directly.
    """
    g = _as_rng(rng)
    if beta == 1:
        U = _haar_unitary(n, g); M = U.T @ U
    elif beta == 2:
        M = _haar_unitary(n, g)
    elif beta == 4:
        U = _haar_unitary(2 * n, g)
        J = np.kron(np.eye(n), np.array([[0.0, 1.0], [-1.0, 0.0]]))
        M = J @ U.T @ J.T @ U
    else:
        raise ValueError("beta must be 1, 2 or 4")
    a = np.sort(np.angle(np.linalg.eigvals(M)))
    if beta == 4:
        a = a[::2]
    if not unfolded:
        return a
    return a * a.size / (2.0 * np.pi)


# --------------------------------------------------------------------------- #
# Synthetic null generators matched to a real matrix                           #
# --------------------------------------------------------------------------- #
def wishart_levels(n: int, m: int, rng: _RngLike = None,
                   dtype=np.float64) -> np.ndarray:
    """Singular values of an iid Gaussian (n, m) matrix, ascending.

    This -- not GOE eigenvalues -- is the correct null for a weight matrix: it
    has the same aspect ratio, the same number of levels, and the same hard MP
    edges. Local bulk statistics are in the beta = 1 class either way, but the
    finite-size corrections that dominate at L <= 20 are shape-dependent.
    """
    g = _as_rng(rng)
    W = (g.standard_normal((int(n), int(m))) / np.sqrt(max(n, m))).astype(dtype)
    return np.sort(np.linalg.svd(W.astype(np.float64), compute_uv=False))


def goe_levels(n: int, rng: _RngLike = None) -> np.ndarray:
    g = _as_rng(rng)
    A = g.standard_normal((int(n), int(n))) / np.sqrt(n)
    return np.linalg.eigvalsh((A + A.T) / np.sqrt(2.0))


def poisson_levels(n: int, rng: _RngLike = None) -> np.ndarray:
    return np.sort(_as_rng(rng).uniform(0.0, 1.0, size=int(n)))


# --------------------------------------------------------------------------- #
# The statistics vector -- one place, so data and null go through one code path #
# --------------------------------------------------------------------------- #
#: Match RunConfig.delta3_L / RunConfig.sigma2_L so a null band lines up
#: column-for-column with the pipeline CSV.
DEFAULT_DELTA3_L = (5, 10, 50)
DEFAULT_SIGMA2_L = (5, 10, 20)


def spacing_statistics(levels, *, unfold_method: str = "auto",
                       unfold_deg: int = 7, unfold_win: int = 15,
                       delta3_L: Sequence[int] = DEFAULT_DELTA3_L,
                       sigma2_L: Sequence[int] = DEFAULT_SIGMA2_L,
                       beta: int = 1, brody: bool = True,
                       brody_bootstrap: int = 40,
                       seed: int = 0) -> dict:
    """Every bulk-RMT scalar for one level set, via exactly the pipeline's path.

    Identical to what ``rmt.per_matrix.per_matrix_analysis`` computes for the
    spacing block, so a null band produced from this function is directly
    comparable to a CSV row. Returns NaN rather than raising on short spectra.
    """
    lv = np.sort(np.asarray(levels, dtype=np.float64))
    nan = float("nan")
    out: Dict[str, object] = {"n_levels": int(lv.size)}
    min_levels = max(50, 4 * unfold_win + 4)
    if lv.size < min_levels:
        out["branch"] = ""
        out.update({k: nan for k in
                    ["r_statistic_mean", "unfold_mean_spacing",
                     "unfold_frac_nonpositive", "nn_KS_GOE", "brody_beta"]})
        for L in delta3_L:
            out[f"delta3_L{int(L)}"] = nan
        for L in sigma2_L:
            out[f"sigma2_L{int(L)}"] = nan
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if unfold_method == "auto":
            xi, used = SP.unfold_auto(lv, deg=unfold_deg, win_size=unfold_win)
        else:
            xi = SP.unfold(lv, deg=unfold_deg, method=unfold_method,
                           win_size=unfold_win)
            used = unfold_method
        xi = np.sort(xi)
        d = np.diff(xi)
        out["branch"] = used
        out["r_statistic_mean"] = SP.r_statistic(lv)
        out["unfold_mean_spacing"] = float(d.mean()) if d.size else nan
        out["unfold_frac_nonpositive"] = float(np.mean(d <= 0)) if d.size else nan
        out["unfold_n_levels"] = int(xi.size)

        s = d[np.isfinite(d)]
        s = np.sort(s[s >= 0])
        out["nn_KS_GOE"] = (_ks_distance(s, lambda x: wigner_surmise_cdf(x, beta))
                            if s.size >= 5 else nan)
        out["nn_KS_Poisson"] = (_ks_distance(s, SP.poisson_cdf)
                                if s.size >= 5 else nan)
        out["brody_beta"] = (SP.brody_beta(None, n_bootstrap=brody_bootstrap,
                                           rng=seed, unfolded=xi)["brody_beta"]
                             if brody else nan)
        for L in delta3_L:
            out[f"delta3_L{int(L)}"] = SP.delta3(None, L, unfolded=xi, rng=seed + 1)
        for L in sigma2_L:
            out[f"sigma2_L{int(L)}"] = SP.sigma2(None, L, unfolded=xi,
                                                 method=used, rng=seed)
    return out


def _ks_distance(sample_sorted: np.ndarray, cdf: Callable) -> float:
    n = sample_sorted.size
    if n == 0:
        return float("nan")
    theo = np.asarray(cdf(sample_sorted), dtype=np.float64)
    hi = np.arange(1, n + 1) / n
    lo = np.arange(0, n) / n
    return float(max(np.max(np.abs(hi - theo)), np.max(np.abs(theo - lo))))


# --------------------------------------------------------------------------- #
# (1) Null bands -- replaces comparison against the asymptotic constants        #
# --------------------------------------------------------------------------- #
def null_band(shape, *, n_reps: int = 30, kind: str = "wishart",
              seed: int = 0, dtype=np.float64, bulk_frac: float = 1.0,
              **stat_kw) -> dict:
    """Monte-Carlo the pipeline's own null for one matrix shape.

    Parameters
    ----------
    shape : (n, m) for ``kind='wishart'``, or an int level count otherwise.
    kind  : 'wishart' | 'goe' | 'poisson'
    n_reps: >= 30 for usable 2.5/97.5 percentiles; >= 100 for stable tails.
    bulk_frac : keep this centred fraction of the spectrum (see the review's
        note that the MP cut [nu_-, nu_+] is a no-op on square matrices and
        leaves the hard edge at 0 in place).

    Returns
    -------
    {'stat': {'mean','sd','p2.5','p97.5','values'}, ..., 'branches': Counter}
    """
    g = np.random.default_rng(seed)
    rows = []
    for i in range(int(n_reps)):
        if kind == "wishart":
            lv = wishart_levels(shape[0], shape[1], g, dtype=dtype)
        elif kind == "goe":
            lv = goe_levels(int(shape), g)
        elif kind == "poisson":
            lv = poisson_levels(int(shape), g)
        else:
            raise ValueError(f"unknown kind {kind!r}")
        if bulk_frac < 1.0:
            k = int(lv.size * (1.0 - bulk_frac) / 2.0)
            lv = lv[k:lv.size - k]
        rows.append(spacing_statistics(lv, seed=seed + i, **stat_kw))

    keys = [k for k in rows[0] if k not in ("branch",)]
    out: Dict[str, object] = {"n_reps": int(n_reps), "kind": kind,
                              "shape": shape, "bulk_frac": float(bulk_frac)}
    from collections import Counter
    out["branches"] = Counter(r["branch"] for r in rows)
    for k in keys:
        v = np.array([r.get(k, np.nan) for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        out[k] = {"mean": float(v.mean()),
                  "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                  "p2.5": float(np.percentile(v, 2.5)),
                  "p97.5": float(np.percentile(v, 97.5)),
                  "n": int(v.size), "values": v}
    return out


def zscore(value: float, band_entry: dict) -> float:
    """Standardised deviation of a measured value against its own null band.

    This is what belongs in the paper, not ``value / delta3_goe_theory(L) - 1``.

    NOTE the reference distribution: because the band mean and sd are estimated
    from ``n`` replicas, this ratio is **t-distributed with n-1 degrees of
    freedom, scaled by sqrt(1 + 1/n)** -- not standard normal.  Use
    :func:`critical_value` to threshold it, never ``norm.ppf``.
    """
    sd = band_entry.get("sd", 0.0)
    if not np.isfinite(value) or sd <= 0:
        return float("nan")
    return float((value - band_entry["mean"]) / sd)


def critical_value(n_reps: int, n_tests: int = 1, alpha: float = 0.05) -> float:
    """Two-sided Bonferroni critical value for :func:`zscore`, done correctly.

    A band built from ``n_reps`` replicas gives an estimated mean and sd, so
    scoring an out-of-sample matrix against it yields
    ``t_{n-1} * sqrt(1 + 1/n)``, not a standard normal.  Using ``norm.ppf`` is
    anti-conservative by 20 % at n = 20 and 39 % at n = 12 -- which is exactly
    why a 12-replica null run appeared to produce |z| = 4.1 "false positives"
    against a nominal 3.23 threshold.  Under the correct threshold (4.48) it was
    never a false positive at all.

    >>> round(critical_value(20, 40), 2)
    3.88
    >>> round(critical_value(12, 40), 2)
    4.48
    """
    n = int(n_reps)
    if n < 3:
        return float("inf")
    a = float(alpha) / max(1, int(n_tests))
    return float(scipy.stats.t.ppf(1.0 - a / 2.0, n - 1) * np.sqrt(1.0 + 1.0 / n))


# --------------------------------------------------------------------------- #
# (2)+(3) Calibrated KS p-value -- replaces scipy.stats.kstest                  #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=32)
def _ks_null_cached(shape_key, kind: str, n_null: int, seed: int, beta: int,
                    unfold_method: str, unfold_deg: int, unfold_win: int,
                    bulk_frac: float, target: str) -> np.ndarray:
    g = np.random.default_rng(seed)
    Ds = np.empty(int(n_null), dtype=np.float64)
    cdf = ((lambda x: wigner_surmise_cdf(x, beta)) if target == "wigner"
           else SP.poisson_cdf)
    for i in range(int(n_null)):
        if kind == "wishart":
            lv = wishart_levels(shape_key[0], shape_key[1], g)
        elif kind == "goe":
            lv = goe_levels(int(shape_key[0]), g)
        else:
            lv = poisson_levels(int(shape_key[0]), g)
        if bulk_frac < 1.0:
            k = int(lv.size * (1.0 - bulk_frac) / 2.0)
            lv = lv[k:lv.size - k]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if unfold_method == "auto":
                xi, _ = SP.unfold_auto(lv, deg=unfold_deg, win_size=unfold_win)
            else:
                xi = SP.unfold(lv, deg=unfold_deg, method=unfold_method,
                               win_size=unfold_win)
        s = np.diff(np.sort(xi))
        s = np.sort(s[np.isfinite(s) & (s >= 0)])
        Ds[i] = _ks_distance(s, cdf)
    return np.sort(Ds)


def ks_pvalue_mc(levels, shape, *, n_null: int = 200, kind: str = "wishart",
                 beta: int = 1, target: str = "wigner", seed: int = 0,
                 unfold_method: str = "auto", unfold_deg: int = 7,
                 unfold_win: int = 15, bulk_frac: float = 1.0,
                 unfolded: Optional[np.ndarray] = None) -> dict:
    """Parametric-bootstrap p-value for the NN-spacing KS distance.

    p = P(D_null >= D_obs), where the null replicas are synthetic matrices of the
    caller's exact shape pushed through the *same* unfolding and compared to the
    *same* reference CDF. Three biases therefore cancel identically rather than
    being modelled: the Wigner-surmise-vs-Gaudin error, the anti-correlation of
    unfolded spacings, and the Lilliefors effect of fitting the unfolding to the
    data being tested.

    Under a true null this p-value is Uniform(0, 1) by construction, which is
    exactly the property ``scipy.stats.kstest`` does not have here.

    ``shape`` is (n, m) for kind='wishart', else (n_levels,).
    Resolution floor on the p-value is 1/(n_null+1); use n_null >= 200.
    """
    lv = np.sort(np.asarray(levels, dtype=np.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if unfolded is not None:
            xi = np.sort(np.asarray(unfolded, dtype=np.float64))
        elif unfold_method == "auto":
            xi, _ = SP.unfold_auto(lv, deg=unfold_deg, win_size=unfold_win)
        else:
            xi = SP.unfold(lv, deg=unfold_deg, method=unfold_method,
                           win_size=unfold_win)
    s = np.diff(np.sort(xi))
    s = np.sort(s[np.isfinite(s) & (s >= 0)])
    cdf = ((lambda x: wigner_surmise_cdf(x, beta)) if target == "wigner"
           else SP.poisson_cdf)
    D_obs = _ks_distance(s, cdf)

    Ds = _ks_null_cached(tuple(int(v) for v in np.atleast_1d(shape)), kind,
                         int(n_null), int(seed), int(beta), unfold_method,
                         int(unfold_deg), int(unfold_win), float(bulk_frac),
                         target)
    # +1 in numerator and denominator: never reports p = 0 (Davison-Hinkley).
    p = float((np.sum(Ds >= D_obs) + 1.0) / (Ds.size + 1.0))
    return {"D_obs": float(D_obs), "p_mc": p,
            "D_null_median": float(np.median(Ds)),
            "D_null_p95": float(np.percentile(Ds, 95)),
            "n_null": int(Ds.size), "n_spacings": int(s.size)}


def clear_null_cache():
    """Drop the memoised KS nulls (they key on shape/seed/unfolding)."""
    _ks_null_cached.cache_clear()
