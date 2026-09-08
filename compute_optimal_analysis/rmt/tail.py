"""Continuous CSN power-law fitting and standard/windowed Hill estimators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import erfc


@dataclass(frozen=True)
class PowerLawFit:
    alpha: float
    xmin: float
    ks_distance: float
    n_tail: int
    standard_error: float
    log_likelihood: float

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ks_D"] = result.pop("ks_distance")
        return result


def _clean_positive(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    return np.sort(array[np.isfinite(array) & (array > 0.0)])


def _empty_fit() -> dict[str, float | int]:
    return {
        "alpha": float("nan"),
        "xmin": float("nan"),
        "ks_D": float("nan"),
        "n_tail": 0,
        "standard_error": float("nan"),
        "log_likelihood": float("nan"),
    }


def fit_powerlaw_csn(
    values: np.ndarray,
    *,
    min_tail: int = 50,
    tail_frac: float = 0.02,
    max_xmin_candidates: int = 200,
    xmax: float | None = None,
) -> dict[str, float | int]:
    """Fit a continuous power-law density with CSN MLE and KS selection.

    ``alpha`` is the density exponent.  Candidate lower cutoffs are selected
    from observed values while retaining the configured minimum tail size.
    """

    data = _clean_positive(values)
    if xmax is not None:
        upper = float(xmax)
        if not np.isfinite(upper) or upper <= 0.0:
            raise ValueError("xmax must be finite and positive")
        data = data[data <= upper]
    min_tail = int(min_tail)
    max_candidates = int(max_xmin_candidates)
    tail_frac = float(tail_frac)
    if min_tail < 2 or max_candidates < 1:
        raise ValueError("min_tail and max_xmin_candidates must be positive")
    if not 0.0 <= tail_frac <= 1.0:
        raise ValueError("tail_frac must lie in [0, 1]")
    required = max(min_tail, int(np.ceil(tail_frac * data.size)))
    if data.size < required:
        return _empty_fit()

    last_start = data.size - required
    # Candidate cutoffs are distinct values represented by their first index;
    # a selected tail therefore always contains every observation equal to xmin.
    possible = np.unique(data[:last_start + 1], return_index=True)[1]
    if possible.size > max_candidates:
        sampled = np.linspace(0, possible.size - 1, max_candidates, dtype=int)
        possible = np.unique(possible[sampled])

    best: PowerLawFit | None = None
    for index in possible:
        xmin = float(data[index])
        tail = data[index:]
        logs = np.log(tail / xmin)
        denominator = float(np.sum(logs))
        if denominator <= 0.0 or not np.isfinite(denominator):
            continue
        alpha = 1.0 + tail.size / denominator
        if not np.isfinite(alpha) or alpha <= 1.0:
            continue
        empirical_hi = np.arange(1, tail.size + 1, dtype=np.float64) / tail.size
        empirical_lo = np.arange(0, tail.size, dtype=np.float64) / tail.size
        model = 1.0 - np.power(tail / xmin, 1.0 - alpha)
        ks_distance = float(max(np.max(np.abs(empirical_hi - model)),
                                np.max(np.abs(model - empirical_lo))))
        standard_error = float((alpha - 1.0) / np.sqrt(tail.size))
        log_likelihood = float(
            tail.size * np.log(alpha - 1.0)
            - tail.size * np.log(xmin)
            - alpha * denominator
        )
        candidate = PowerLawFit(
            alpha=float(alpha),
            xmin=xmin,
            ks_distance=ks_distance,
            n_tail=int(tail.size),
            standard_error=standard_error,
            log_likelihood=log_likelihood,
        )
        if best is None or candidate.ks_distance < best.ks_distance:
            best = candidate
    return _empty_fit() if best is None else best.as_dict()


def hill_estimator(
    values: np.ndarray,
    k_min: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Hill positive survival exponents for upper-tail sizes ``k``."""

    data = _clean_positive(values)[::-1]
    k_min = int(k_min)
    if k_min < 1:
        raise ValueError("k_min must be positive")
    k_max = data.size // 2
    if k_max < k_min or data.size < 2:
        return np.asarray([], dtype=int), np.asarray([], dtype=np.float64)
    logs = np.log(data)
    cumulative = np.cumsum(logs)
    ks = np.arange(k_min, k_max + 1, dtype=int)
    estimates = np.empty(ks.size, dtype=np.float64)
    for output_index, k in enumerate(ks):
        hill_mean = cumulative[k - 1] / k - logs[k]
        estimates[output_index] = np.inf if hill_mean <= 0.0 else 1.0 / hill_mean
    return ks, estimates


def hill_alpha_at(values: np.ndarray, k: int) -> float:
    """Return the Hill survival exponent at one upper-tail size."""

    data = _clean_positive(values)[::-1]
    k = int(k)
    if k < 1 or k >= data.size:
        return float("nan")
    hill_mean = float(np.mean(np.log(data[:k])) - np.log(data[k]))
    return float("nan") if hill_mean <= 0.0 else float(1.0 / hill_mean)


def hill_estimator_windowed(
    values: np.ndarray,
    *,
    window: int = 20,
    k_min: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a local Hill curve from Renyi normalized log spacings.

    For an ideal Pareto tail, ``i * (log x_i - log x_(i+1))`` is exponentially
    distributed with a scale independent of rank.  A moving mean therefore
    exposes a plateau while a bounded MP edge generally drifts.
    """

    data = _clean_positive(values)[::-1]
    window = int(window)
    k_min = int(k_min)
    if window < 3 or k_min < 1:
        raise ValueError("window must be at least three and k_min positive")
    if data.size <= k_min + window:
        return np.asarray([], dtype=int), np.asarray([], dtype=np.float64)
    ranks = np.arange(1, data.size, dtype=np.float64)
    renyi = ranks * np.diff(np.log(data)) * -1.0
    starts = np.arange(k_min - 1, renyi.size - window + 1, dtype=int)
    estimates = np.empty(starts.size, dtype=np.float64)
    for output_index, start in enumerate(starts):
        local_scale = float(np.mean(renyi[start : start + window]))
        estimates[output_index] = np.inf if local_scale <= 0.0 else 1.0 / local_scale
    return starts + 1, estimates


def hill_plateau(
    values: np.ndarray,
    *,
    window: int = 20,
    flat_tol: float = 0.15,
) -> dict[str, float | int | bool]:
    """Summarize whether the extreme-tail local Hill curve has a plateau."""

    flat_tol = float(flat_tol)
    if not 0.0 < flat_tol < 1.0:
        raise ValueError("flat_tol must lie in (0, 1)")
    ks, estimates = hill_estimator_windowed(values, window=window, k_min=5)
    finite = np.isfinite(estimates) & (estimates > 0.0)
    ks, estimates = ks[finite], estimates[finite]
    if estimates.size < 8:
        return {
            "hill_plateau_alpha": float("nan"),
            "hill_plateau_width": 0,
            "hill_plateau_start_rank": None,
            "hill_plateau_end_rank": None,
            "hill_window": int(window),
            "hill_support_observations": 0,
            "hill_is_powerlaw": False,
        }
    extreme_count = min(estimates.size, max(3 * int(window), estimates.size // 3, 12))
    extreme = estimates[:extreme_count]
    extreme_ks = ks[:extreme_count].astype(np.float64)
    median = float(np.median(extreme))
    q25, q75 = np.quantile(extreme, [0.25, 0.75])
    relative_iqr = float((q75 - q25) / max(abs(median), np.finfo(float).eps))
    slope = float(np.polyfit(extreme_ks, extreme, 1)[0]) if extreme.size > 1 else float("inf")
    rank_span = max(float(extreme_ks[-1] - extreme_ks[0]), 1.0)
    relative_drift = abs(slope) * rank_span / max(abs(median), np.finfo(float).eps)
    plausible = 0.25 <= median <= 15.0
    stable = relative_iqr <= max(0.25, 1.75 * flat_tol) and relative_drift <= max(0.35, 2.5 * flat_tol)
    start_rank = int(extreme_ks[0])
    end_rank = int(extreme_ks[-1] + int(window) - 1)
    return {
        "hill_plateau_alpha": median,
        "hill_plateau_width": int(extreme_count),
        "hill_plateau_start_rank": start_rank,
        "hill_plateau_end_rank": end_rank,
        "hill_window": int(window),
        "hill_support_observations": end_rank - start_rank + 1,
        "hill_is_powerlaw": bool(plausible and stable),
    }


def fixed_cutoff_mle(
    values: np.ndarray,
    *,
    xmin: float | None = None,
    tail_fraction: float = 0.1,
    xmax: float | None = None,
) -> dict[str, float | int]:
    """Fit a continuous density exponent above one fixed lower cutoff."""

    data = _clean_positive(values)
    fraction = float(tail_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    if xmax is not None:
        upper = float(xmax)
        if not np.isfinite(upper) or upper <= 0.0:
            raise ValueError("xmax must be finite and positive")
        data = data[data <= upper]
    if data.size < 2:
        return _empty_fit()
    if xmin is None:
        tail_count = max(2, int(np.ceil(fraction * data.size)))
        cutoff = float(data[-tail_count])
    else:
        cutoff = float(xmin)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("xmin must be finite and positive")
    tail = data[data >= cutoff]
    if tail.size < 2:
        return _empty_fit()
    denominator = float(np.sum(np.log(tail / cutoff)))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return _empty_fit()
    alpha = float(1.0 + tail.size / denominator)
    empirical_hi = np.arange(1, tail.size + 1, dtype=np.float64) / tail.size
    empirical_lo = np.arange(0, tail.size, dtype=np.float64) / tail.size
    model = 1.0 - np.power(tail / cutoff, 1.0 - alpha)
    ks_distance = float(max(np.max(np.abs(empirical_hi - model)),
                            np.max(np.abs(model - empirical_lo))))
    log_likelihood = float(
        tail.size * np.log(alpha - 1.0)
        - tail.size * np.log(cutoff)
        - alpha * denominator
    )
    return PowerLawFit(
        alpha=alpha,
        xmin=cutoff,
        ks_distance=ks_distance,
        n_tail=int(tail.size),
        standard_error=float((alpha - 1.0) / np.sqrt(tail.size)),
        log_likelihood=log_likelihood,
    ).as_dict()


def rank_ordered_mle(
    values: np.ndarray,
    *,
    xmin: float | None = None,
    tail_fraction: float = 0.1,
) -> dict[str, float | int]:
    """Fit a density exponent from the tail rank-frequency relation.

    For a continuous power law, ``rank / n`` scales as
    ``x**(1-alpha)``.  The returned ``alpha`` is therefore one minus the
    fitted log-log slope, matching the density convention used by CSN.
    """

    data = _clean_positive(values)
    fraction = float(tail_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("tail_fraction must lie in (0, 1]")
    if data.size < 3:
        return {**_empty_fit(), "r_squared": float("nan")}
    if xmin is None:
        tail_count = max(3, int(np.ceil(fraction * data.size)))
        cutoff = float(data[-tail_count])
    else:
        cutoff = float(xmin)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("xmin must be finite and positive")
    tail = data[data >= cutoff][::-1]
    if tail.size < 3 or np.allclose(tail, tail[0]):
        return {**_empty_fit(), "r_squared": float("nan")}
    ranks = np.arange(1, tail.size + 1, dtype=np.float64)
    log_values = np.log(tail)
    log_survival = np.log(ranks / tail.size)
    slope, intercept = np.polyfit(log_values, log_survival, 1)
    fitted = slope * log_values + intercept
    residual = float(np.sum(np.square(log_survival - fitted)))
    total = float(np.sum(np.square(log_survival - np.mean(log_survival))))
    r_squared = 1.0 - residual / total if total > 0.0 else float("nan")
    alpha = float(1.0 - slope)
    empirical = ranks / tail.size
    model = np.power(tail / cutoff, 1.0 - alpha)
    ks_distance = float(np.max(np.abs(empirical - model)))
    return {
        "alpha": alpha,
        "xmin": cutoff,
        "ks_D": ks_distance,
        "n_tail": int(tail.size),
        "standard_error": float("nan"),
        "log_likelihood": float("nan"),
        "r_squared": float(r_squared),
    }


def csn_goodness_of_fit(
    values: np.ndarray,
    *,
    n_bootstrap: int = 250,
    min_tail: int = 50,
    tail_frac: float = 0.02,
    max_xmin_candidates: int = 200,
    rng: np.random.Generator | int | None = 0,
) -> dict[str, Any]:
    """Return the CSN semiparametric bootstrap goodness-of-fit probability."""

    data = _clean_positive(values)
    replicates = int(n_bootstrap)
    if replicates < 1:
        raise ValueError("n_bootstrap must be positive")
    observed = fit_powerlaw_csn(
        data,
        min_tail=min_tail,
        tail_frac=tail_frac,
        max_xmin_candidates=max_xmin_candidates,
    )
    alpha = float(observed["alpha"])
    xmin = float(observed["xmin"])
    observed_distance = float(observed["ks_D"])
    if not np.isfinite(alpha) or not np.isfinite(xmin) or not np.isfinite(observed_distance):
        return {
            **observed,
            "p_value": float("nan"),
            "bootstrap_distances": np.asarray([], dtype=np.float64),
            "n_bootstrap": replicates,
        }
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    body = data[data < xmin]
    tail_probability = float(np.mean(data >= xmin))
    distances = np.empty(replicates, dtype=np.float64)
    distances.fill(np.nan)
    for index in range(replicates):
        tail_count = int(generator.binomial(data.size, tail_probability))
        body_count = data.size - tail_count
        if body_count and body.size:
            simulated_body = generator.choice(body, size=body_count, replace=True)
        elif body_count:
            simulated_body = np.full(body_count, xmin, dtype=np.float64)
        else:
            simulated_body = np.asarray([], dtype=np.float64)
        if tail_count:
            uniforms = generator.uniform(np.finfo(float).eps, 1.0, size=tail_count)
            simulated_tail = xmin * np.power(1.0 - uniforms, -1.0 / (alpha - 1.0))
        else:
            simulated_tail = np.asarray([], dtype=np.float64)
        simulated = np.concatenate((simulated_body, simulated_tail))
        fitted = fit_powerlaw_csn(
            simulated,
            min_tail=min_tail,
            tail_frac=tail_frac,
            max_xmin_candidates=max_xmin_candidates,
        )
        distances[index] = float(fitted["ks_D"])
    finite = distances[np.isfinite(distances)]
    probability = (
        float((1.0 + np.count_nonzero(finite >= observed_distance)) / (finite.size + 1.0))
        if finite.size
        else float("nan")
    )
    return {
        **observed,
        "p_value": probability,
        "bootstrap_distances": distances,
        "n_bootstrap": replicates,
    }


def select_tail_estimator(
    values: np.ndarray,
    estimator: str = "csn",
    **kwargs: Any,
) -> dict[str, Any]:
    """Select a labeled tail estimate without conflating density and survival exponents."""

    aliases = {
        "csn": "clauset_mle",
        "clauset_mle": "clauset_mle",
        "hill": "hill_estimator",
        "hill_estimator": "hill_estimator",
        "hill_windowed": "hill_windowed",
        "fixed_cutoff_mle": "fixed_cutoff_mle",
        "rank_ordered_mle": "rank_ordered_mle",
        "all": "all",
    }
    requested = str(estimator).lower()
    if requested not in aliases:
        raise ValueError(
            "estimator must be clauset_mle, hill_estimator, hill_windowed, "
            "fixed_cutoff_mle, rank_ordered_mle, or all"
        )
    name = aliases[requested]
    csn_kwargs = {
        key: kwargs[key]
        for key in ("min_tail", "tail_frac", "max_xmin_candidates", "xmax")
        if key in kwargs
    }
    window = int(kwargs.get("window", 20))
    if name in {"clauset_mle", "all"}:
        csn = fit_powerlaw_csn(values, **csn_kwargs)
    else:
        csn = _empty_fit()
    k = int(kwargs.get("k", max(5, int(np.sqrt(max(_clean_positive(values).size, 1))))))
    hill = hill_alpha_at(values, k) if name in {"hill_estimator", "all"} else float("nan")
    plateau = (
        hill_plateau(values, window=window)
        if name in {"hill_windowed", "all"}
        else {
            "hill_plateau_alpha": float("nan"),
            "hill_plateau_width": 0,
            "hill_plateau_start_rank": None,
            "hill_plateau_end_rank": None,
            "hill_window": window,
            "hill_support_observations": 0,
            "hill_is_powerlaw": False,
        }
    )
    fixed = (
        fixed_cutoff_mle(
            values,
            xmin=kwargs.get("xmin"),
            tail_fraction=float(kwargs.get("tail_fraction", kwargs.get("tail_frac", 0.1))),
            xmax=kwargs.get("xmax"),
        )
        if name == "fixed_cutoff_mle"
        else _empty_fit()
    )
    rank_ordered = (
        rank_ordered_mle(
            values,
            xmin=kwargs.get("xmin"),
            tail_fraction=float(kwargs.get("tail_fraction", kwargs.get("tail_frac", 0.1))),
        )
        if name == "rank_ordered_mle"
        else {**_empty_fit(), "r_squared": float("nan")}
    )
    if name == "hill_estimator":
        selected, kind = hill, "survival"
        clean = _clean_positive(values)[::-1]
        if 1 <= k < clean.size:
            csn = {**_empty_fit(), "xmin": float(clean[k]), "n_tail": int(k)}
    elif name == "hill_windowed":
        selected, kind = plateau["hill_plateau_alpha"], "survival"
        # A sequence of overlapping local windows is not one Pareto sample;
        # single-cutoff xmin/n_tail fields are intentionally unavailable.
        csn = _empty_fit()
    elif name == "fixed_cutoff_mle":
        selected, kind = fixed["alpha"], "density"
        csn = fixed
    elif name == "rank_ordered_mle":
        selected, kind = rank_ordered["alpha"], "density"
        csn = rank_ordered
    else:
        selected, kind = csn["alpha"], "density"
    return {
        **csn,
        "alpha_hill": hill,
        **plateau,
        "selected_alpha": selected,
        "selected_kind": kind,
        "rank_ordered_r_squared": rank_ordered.get("r_squared", float("nan")),
        "selected_estimator": "clauset_mle" if name == "all" else name,
    }


def powerlaw_pkg_fit(values: np.ndarray, xmax: float | None = None) -> dict[str, float] | None:
    """Compare pure and exponentially truncated tails with a Vuong statistic.

    The historical function name is retained, but the implementation is
    internal so the numerical package keeps its dependency boundary.
    """

    data = _clean_positive(values)
    if xmax is not None:
        upper = float(xmax)
        if not np.isfinite(upper) or upper <= 0.0:
            raise ValueError("xmax must be finite and positive")
        data = data[data <= upper]
    if data.size < 10:
        return None
    fit = fit_powerlaw_csn(
        data,
        min_tail=max(10, min(50, data.size // 4)),
        max_xmin_candidates=100,
    )
    alpha = float(fit["alpha"])
    xmin = float(fit["xmin"])
    if not np.isfinite(alpha) or not np.isfinite(xmin):
        return None
    tail = data[data >= xmin]
    log_ratio = np.log(tail / xmin)
    pure_log_density = np.log(alpha - 1.0) - np.log(xmin) - alpha * log_ratio

    def truncated_normalizer(truncated_alpha: float, cutoff: float) -> float:
        value, _ = quad(
            lambda scaled: scaled ** (-truncated_alpha) * np.exp(-cutoff * (scaled - 1.0)),
            1.0,
            np.inf,
            epsabs=1e-9,
            epsrel=1e-8,
            limit=200,
        )
        return float(value)

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        truncated_alpha = 1.0 + float(np.exp(parameters[0]))
        cutoff = float(np.exp(parameters[1]))
        normalizer = truncated_normalizer(truncated_alpha, cutoff)
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            return float(np.finfo(float).max)
        log_density = (
            -truncated_alpha * log_ratio
            - cutoff * (tail / xmin - 1.0)
            - np.log(xmin)
            - np.log(normalizer)
        )
        return float(-np.sum(log_density))

    optimum = minimize(
        negative_log_likelihood,
        np.log(np.asarray([max(alpha - 1.0, 1e-3), 0.05])),
        method="L-BFGS-B",
        bounds=((-8.0, 6.0), (-12.0, 6.0)),
    )
    truncated_alpha = 1.0 + float(np.exp(optimum.x[0]))
    cutoff = float(np.exp(optimum.x[1]))
    normalizer = truncated_normalizer(truncated_alpha, cutoff)
    truncated_log_density = (
        -truncated_alpha * log_ratio
        - cutoff * (tail / xmin - 1.0)
        - np.log(xmin)
        - np.log(normalizer)
    )
    differences = pure_log_density - truncated_log_density
    likelihood_ratio = float(np.sum(differences))
    deviation = float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
    if deviation > 0.0:
        z_score = likelihood_ratio / (np.sqrt(differences.size) * deviation)
        probability = float(erfc(abs(z_score) / np.sqrt(2.0)))
    else:
        probability = 1.0
    return {
        "LR_trunc": likelihood_ratio,
        "LR_p": probability,
        "alpha": alpha,
        "xmin": xmin,
        "truncated_alpha": truncated_alpha,
        "cutoff_rate": cutoff / xmin,
        "optimizer_success": float(bool(optimum.success)),
    }


__all__ = [
    "PowerLawFit",
    "fixed_cutoff_mle",
    "fit_powerlaw_csn",
    "csn_goodness_of_fit",
    "hill_alpha_at",
    "hill_estimator",
    "hill_estimator_windowed",
    "hill_plateau",
    "powerlaw_pkg_fit",
    "rank_ordered_mle",
    "select_tail_estimator",
]
