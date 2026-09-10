"""rmt.tail — PART 2: power-law tail estimators.

CSN (Clauset–Shalizi–Newman) MLE+KS density exponent, the standard Hill
survival exponent, and the Paper-1 adapted *windowed* Hill with a plateau
diagnostic.  Pure numpy/scipy.

Exponent conventions (plan.md §0.1):
  CSN α = density exponent  p(x) ∝ x^{-α}
  Hill α = 1/H_k            survival exponent P(X>x) ∝ x^{-α}
  Pure power law ⇒ α_csn = α_hill + 1.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _pareto_alpha_and_cdf(tail, xmin, xmax=None):
    """Return the MLE density exponent and fitted (possibly bounded) CDF."""

    logs = np.log(tail / xmin)
    slog = float(np.sum(logs))
    if slog <= 0.0 or not np.isfinite(slog):
        return None, None
    if xmax is None:
        beta = tail.size / slog
        normalization = 1.0
    else:
        upper_log = float(np.log(xmax / xmin))
        if upper_log <= 0.0:
            return None, None

        def objective(log_beta):
            beta_value = float(np.exp(log_beta))
            return float(
                -tail.size * np.log(beta_value)
                + (beta_value + 1.0) * slog
                + tail.size * np.log(-np.expm1(-beta_value * upper_log))
            )

        optimum = minimize_scalar(objective, bounds=(-12.0, 12.0), method="bounded",
                                  options={"xatol": 1e-10, "maxiter": 500})
        if not optimum.success:
            return None, None
        beta = float(np.exp(optimum.x))
        normalization = float(-np.expm1(-beta * upper_log))
    cdf = (1.0 - np.exp(-beta * logs)) / normalization
    return 1.0 + beta, cdf


# --------------------------------------------------------------------------- #
# CSN MLE + KS                                                                 #
# --------------------------------------------------------------------------- #
def fit_powerlaw_csn(values, *, min_tail=50, tail_frac=0.02,
                     max_xmin_candidates=200, xmax=None) -> dict:
    """CSN continuous power-law fit. Returns {alpha, xmin, ks_D, n_tail}.

    ``alpha`` is the **density** exponent (NaN if no admissible xmin).
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if xmax is not None:
        xmax = float(xmax)
        if not np.isfinite(xmax) or xmax <= 0.0:
            raise ValueError("xmax must be finite and positive")
        x = x[x <= xmax]
    x = np.sort(x)
    N = x.size
    nan_res = {"alpha": float("nan"), "xmin": float("nan"),
               "ks_D": float("nan"), "n_tail": 0}
    if N < max(min_tail, 1):
        return nan_res

    min_n_tail = int(max(min_tail, np.ceil(tail_frac * N)))
    # candidate xmin: distinct values, sub-sampled, always include up to 99.9 pct
    uniq = np.unique(x)
    # keep only candidates that can leave >= min_n_tail in the tail
    if uniq.size > 1:
        hi_cut = np.quantile(x, 0.999)
        uniq = uniq[uniq <= hi_cut]
    if uniq.size > max_xmin_candidates:
        idx = np.linspace(0, uniq.size - 1, max_xmin_candidates).astype(int)
        uniq = uniq[idx]

    best = None
    for xmin in uniq:
        tail = x[x >= xmin]
        n_tail = tail.size
        if n_tail < min_n_tail:
            continue
        alpha, theo = _pareto_alpha_and_cdf(tail, xmin, xmax=xmax)
        if alpha is None:
            continue
        emp_hi = np.arange(1, n_tail + 1) / n_tail
        emp_lo = np.arange(0, n_tail) / n_tail
        D = float(max(np.max(np.abs(emp_hi - theo)),
                      np.max(np.abs(theo - emp_lo))))
        if best is None or D < best["ks_D"]:
            best = {"alpha": float(alpha), "xmin": float(xmin),
                    "ks_D": D, "n_tail": int(n_tail)}
    return best if best is not None else nan_res


# --------------------------------------------------------------------------- #
# standard Hill                                                                #
# --------------------------------------------------------------------------- #
def hill_estimator(svals, k_min=5):
    """Standard Hill estimator.

    Returns (ks, inv_H) where inv_H = 1/H_k is the **positive survival**
    exponent; ``ks`` is capped at n//2.
    """
    x = np.sort(np.asarray(svals, dtype=np.float64))[::-1]   # descending
    x = x[x > 0]
    n = x.size
    k_cap = max(k_min + 1, n // 2)
    ks = np.arange(k_min, k_cap)
    inv_H = np.full(ks.shape, np.nan, dtype=np.float64)
    logx = np.log(x)
    for i, k in enumerate(ks):
        if k + 1 > n:
            break
        H = np.mean(logx[:k]) - logx[k]                      # x_(k+1) is logx[k]
        inv_H[i] = 1.0 / H if H > 0 else np.nan
    return ks, inv_H


def hill_alpha_at(svals, k) -> float:
    """Survival exponent at exactly ``k`` upper order statistics."""
    x = np.sort(np.asarray(svals, dtype=np.float64))[::-1]
    x = x[np.isfinite(x) & (x > 0.0)]
    k = int(k)
    if k < 1 or k >= x.size:
        raise ValueError("k must satisfy 1 <= k < number of positive values")
    H = float(np.mean(np.log(x[:k])) - np.log(x[k]))
    return float("nan") if H <= 0.0 else 1.0 / H


# --------------------------------------------------------------------------- #
# Paper-1 adapted windowed Hill + plateau                                      #
# --------------------------------------------------------------------------- #
def hill_estimator_windowed(svals, *, window=20, k_min=1):
    """Local Hill over a sliding window of ``window`` order statistics (Paper 1).

    Uses the Rényi normalized log-spacings g_i = i·(ln x₍ᵢ₎ − ln x₍ᵢ₊₁₎), which
    are ~Exp(mean 1/β) for a power-law tail (β = survival exponent). A windowed
    mean of g estimates 1/β, so α̂_local = 1/⟨g⟩.  Scanned over the extreme tail
    (the largest singular values), where a tail index can exist.

    Returns (ks, alpha_local).
    """
    x = np.sort(np.asarray(svals, dtype=np.float64))[::-1]   # descending
    x = x[x > 0]
    n = x.size
    logx = np.log(x)
    idx = np.arange(1, n)                                     # 1..n-1
    g = idx * (logx[:-1] - logx[1:])                          # normalized spacings
    a = int(window)
    # extreme tail region: top ~12% of the spectrum (>= a few windows)
    K = int(min(n // 2, max(3 * a, np.ceil(0.12 * n))))
    # k is a one-based rank label, while g is zero based.  A point labelled k
    # therefore starts at g[k-1].  Include the terminal exactly-fitting window.
    first_k = max(1, int(k_min))
    last_k = min(K - 1, g.size - a + 1)
    ks = np.arange(first_k, last_k + 1)
    alpha_local = np.full(ks.shape, np.nan, dtype=np.float64)
    for i, k in enumerate(ks):
        start = k - 1
        H = np.mean(g[start:start + a])
        alpha_local[i] = 1.0 / H if H > 1e-12 else np.nan
    return ks, alpha_local


def hill_plateau(svals, *, window=20, flat_tol=0.20) -> dict:
    """Detect a stable plateau in the windowed-Hill curve (Paper-1 diagnostic).

    Returns {hill_plateau_alpha, hill_plateau_width, hill_is_powerlaw}.

    A genuine power law is scale-free: its windowed index is finite and stable
    starting at the *extreme* tail.  The MP hard edge instead shows a very large
    index at the extreme tail that drifts, so it is rejected even if a flat band
    appears deeper in the body.
    """
    ks, a_loc = hill_estimator_windowed(svals, window=window)
    res = {"hill_plateau_alpha": float("nan"),
           "hill_plateau_width": 0,
           "hill_plateau_start_rank": None,
           "hill_plateau_end_rank": None,
           "hill_window": int(window),
           "hill_support_observations": 0,
           "hill_is_powerlaw": False}
    fin = a_loc[np.isfinite(a_loc) & (a_loc > 0)]
    if fin.size < max(5, window // 2):
        return res
    # extreme-tail index = median of the first `window` valid windowed estimates
    extreme_alpha = float(np.median(fin[:window]))
    # MP-edge signature: the extreme tail index is enormous (not a tail index).
    if extreme_alpha > 15.0:
        res["hill_plateau_alpha"] = extreme_alpha
        return res

    # search for the longest contiguous flat band anchored near the extreme tail
    a = a_loc.copy()
    a[~(np.isfinite(a) & (a > 0) & (a < 2.5 * extreme_alpha + 1.0))] = np.nan
    best_lo, best_hi = -1, -1
    i = 0
    while i < a.size:
        if not np.isfinite(a[i]):
            i += 1
            continue
        j = i
        while j + 1 < a.size and np.isfinite(a[j + 1]):
            seg = a[i:j + 2]
            med = np.median(seg)
            if med <= 0 or np.max(np.abs(seg - med) / med) > flat_tol:
                break
            j += 1
        if (j - i) > (best_hi - best_lo):
            best_lo, best_hi = i, j
        i = j + 1
    if best_lo < 0:
        res["hill_plateau_alpha"] = extreme_alpha
        return res
    width = best_hi - best_lo + 1
    plateau_vals = a[best_lo:best_hi + 1]
    plateau_alpha = float(np.median(plateau_vals))
    res["hill_plateau_alpha"] = plateau_alpha
    res["hill_plateau_width"] = int(width)
    start_rank = int(ks[best_lo])
    # ``window`` adjacent log-spacings consume ``window + 1`` observations.
    end_rank = int(ks[best_hi] + int(window))
    res["hill_plateau_start_rank"] = start_rank
    res["hill_plateau_end_rank"] = end_rank
    res["hill_support_observations"] = end_rank - start_rank + 1
    # power law only if a wide flat band exists AND it agrees with the extreme tail
    consistent = abs(plateau_alpha - extreme_alpha) <= 0.5 * extreme_alpha
    res["hill_is_powerlaw"] = bool(width >= window and consistent)
    return res


def powerlaw_pkg_fit(values, xmax=None):
    """Delegate to the local `powerlaw` package if installed (LR test R, p)."""
    try:
        import powerlaw  # local, optional
    except Exception:
        return None
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 50:
        return None
    try:
        fit = powerlaw.Fit(x, xmax=xmax, verbose=False)
        R, p = fit.distribution_compare("truncated_power_law", "power_law",
                                        normalized_ratio=True)
        return {"alpha": float(fit.alpha), "xmin": float(fit.xmin),
                "LR_trunc": float(R), "LR_p": float(p)}
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# thin selector used by per_matrix                                             #
# --------------------------------------------------------------------------- #
def select_alpha(values, *, estimator="all", window=20,
                 csn_kwargs=None) -> dict:
    """Return α(s) per ``cfg.alpha_estimator`` ∈ {csn, hill, hill_windowed, all}.

    Always includes the headline ``alpha`` key; ``all`` additionally returns
    every variant's keys.
    """
    csn_kwargs = csn_kwargs or {}
    csn = fit_powerlaw_csn(values, **csn_kwargs)
    plateau = hill_plateau(values, window=window)
    usable = np.asarray(values, dtype=np.float64)
    usable = usable[np.isfinite(usable) & (usable > 0)]
    hill_k = min(max(1, int(usable.size // 40)), usable.size - 1)
    hill_a = hill_alpha_at(usable, hill_k) if usable.size >= 2 else float("nan")

    if estimator == "csn":
        return {"alpha": csn["alpha"], "source": "csn"}
    if estimator == "hill":
        return {"alpha": hill_a, "source": "hill"}
    if estimator == "hill_windowed":
        return {"alpha": plateau["hill_plateau_alpha"], "source": "hill_windowed"}
    # "all"
    return {
        "alpha": csn["alpha"], "source": "csn",
        "csn_alpha": csn["alpha"], "hill_alpha": hill_a,
        "hill_windowed_alpha": plateau["hill_plateau_alpha"],
        "hill_is_powerlaw": plateau["hill_is_powerlaw"],
    }
