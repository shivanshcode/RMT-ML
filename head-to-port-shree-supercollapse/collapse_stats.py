"""Collapse deviation and noise floor for compute-optimally trained models.

Implements the two statistics of arXiv 2507.02119 section 2.4 that
``figures/collapse.ipynb`` does not compute -- the ones behind Figure 1c
(supercollapse, with LR decay) and Figure 3c (ordinary collapse, constant LR):

    reducible loss       L_red(t, p, w) = L(t, p, w) - L0
    normalized curve     ell(x, p, w)   = L_red(x t*(p), p, w) / L_red(t*(p), p, w)

    collapse deviation   Delta(x)  = V_{p,w}[ell]^(1/2) / E_{p,w}[ell]      (Eq. 2)
    noise floor          sigma(x,p) = V_w[L_red]^(1/2) / E_w[L_red]         (Eq. 3)

Three details decide whether these numbers mean anything, and all three are
easy to get wrong:

1. ``Delta`` is the spread of the *self-normalized* curve ``ell``; ``sigma`` is
   the spread of the *raw reducible loss* ``L_red``. They are statistics of
   different objects. Computing both on the same object makes the comparison
   vacuous.
2. ``Delta`` pools model size and seed into one variance (V_{p,w}), it does not
   hold a seed fixed. By the law of total variance the seed noise sits *inside*
   ``Delta`` -- see ``collapse_deviation`` and paper Eq. 4.
3. The denominator of ``ell`` is that run's own *stochastic* final loss, not the
   seed-averaged one. That is the control variate (paper section 3.3) and it is
   the entire mechanism of supercollapse. Normalize by the mean instead and
   Delta reduces exactly to sigma.

Pure numpy / pandas / scipy. No jax, no flax, no torch, no matplotlib -- this
module runs in any environment, including a login node.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "Curves",
    "fit_power_law_constant",
    "fit_power_law",
    "load_runs",
    "fit_L0",
    "make_grid",
    "resample",
    "collapse_deviation",
    "noise_floor",
    "per_model_collapse_deviation",
    "supercollapse_onset",
    "fraction_below_floor",
]


# ---------------------------------------------------------------------------
# 1. Power-law fit
# ---------------------------------------------------------------------------
# VERBATIM from figures/collapse.ipynb cell 2. Do not reformat, do not "clean
# up" -- tests/test_collapse_stats.py::test_vendored_fit_matches_notebook
# compares this source against the notebook cell and will fail on any edit.
# If the authors' notebook changes, update this and let the test confirm.

def fit_power_law_constant(C, L, num_inits=10, L0=None):
    """
    Fit power law with constant: L = a * C^(-b) + L0
    Uses Huber loss in log-log space for robustness
    
    Args:
        C: Compute values (array)
        L: Loss values (array) 
        num_inits: Number of initializations per parameter
    
    Returns:
        dict with fitted parameters and R^2 score
    """
    def power_law_const(params, x):
        if L0 is None:
            a, b, L0_fit = params
            return a * x**(-b) + L0_fit
        else:
            a, b = params
            return a * x**(-b) + L0
    
    def huber_loss(residual, delta=1e-3):
        mask = np.abs(residual) <= delta
        return np.where(mask, 
                       0.5 * residual**2,
                       delta * (np.abs(residual) - 0.5 * delta))
    
    def objective(params):
        pred = power_law_const(params, C)
        residuals = np.log(pred) - np.log(L)
        return np.mean(huber_loss(residuals))
    
    # Try all combinations of parameter initializations
    best_loss = np.inf
    best_params = None
    
    # Parameter ranges for initialization
    a_range = [0.1, 1]
    b_range = [0.01, 0.3]
    
    # Create evenly spaced initializations for each parameter
    a_inits = np.linspace(a_range[0], a_range[1], num_inits)
    b_inits = np.linspace(b_range[0], b_range[1], num_inits)
    
    if L0 is None:
        L0_range = [min(L)*0.1, min(L)]
        L0_inits = np.linspace(L0_range[0], L0_range[1], num_inits)
        # Try all combinations of a, b, and L0
        for a, b, L0_init in product(a_inits, b_inits, L0_inits):
            init_params = [a, b, L0_init]
            result = minimize(
                objective,
                init_params,
                method='L-BFGS-B',
                bounds=[(0, None), (0, None), (0, None)]  # Changed bounds to >= 0
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
    else:
        # Try all combinations of a and b
        for a, b in product(a_inits, b_inits):
            init_params = [a, b]
            result = minimize(
                objective,
                init_params,
                method='L-BFGS-B',
                bounds=[(0, None), (0, None)]  # Changed bounds to >= 0
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
            
    # Compute R^2 score
    y_pred = power_law_const(best_params, C)
    log_L = np.log(L)
    log_pred = np.log(y_pred)
    ss_res = np.sum((log_L - log_pred) ** 2)
    ss_tot = np.sum((log_L - np.mean(log_L)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    if L0 is None:
        return {
            'a': best_params[0],
            'b': best_params[1],
            'L0': best_params[2],
            'r2': r2
        }
    else:
        return {
            'a': best_params[0],
            'b': best_params[1],
            'L0': L0,
            'r2': r2
        }


def fit_power_law(C, L, num_inits: int = 10, L0: Optional[float] = None) -> dict:
    """``fit_power_law_constant`` with the benign log(0) warnings silenced.

    The optimizer probes ``a = 0`` across the ~100 restarts; those candidates
    produce ``log(0)`` and are then discarded on objective value. Muting the
    warning process-wide (``np.seterr``) would hide real numerical trouble
    elsewhere, so it is scoped to this call only.
    """
    C = np.asarray(C, dtype=float)
    L = np.asarray(L, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return fit_power_law_constant(C, L, num_inits=num_inits, L0=L0)


# ---------------------------------------------------------------------------
# 2. Loading
# ---------------------------------------------------------------------------

_REQUIRED_HISTORY_COLS = ("compute", "test_loss")


def _rows_identical(a: pd.Series, b: pd.Series) -> bool:
    """True if two run rows are the same run: same history, same metadata."""
    ha, hb = a["history"], b["history"]
    if ha.shape != hb.shape or list(ha.columns) != list(hb.columns):
        return False
    na, nb = ha.select_dtypes("number"), hb.select_dtypes("number")
    if list(na.columns) != list(nb.columns):
        return False
    if not np.allclose(na.to_numpy(float), nb.to_numpy(float), equal_nan=True):
        return False
    for k in a.index:
        if k == "history":
            continue
        va, vb = a[k], b[k]
        if isinstance(va, float) and isinstance(vb, float):
            if not (np.isclose(va, vb) or (np.isnan(va) and np.isnan(vb))):
                return False
        elif va != vb:
            return False
    return True


def _dedupe_identical(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Drop exact duplicate runs; raise on duplicates that actually differ.

    The authors' own published ``logs/mlp.pkl`` contains 16 duplicated
    (width, seed) pairs, because ``save_logs.py`` iterates a comma-separated
    tag list and ``extend``s the run list per tag -- a run carrying two tags is
    ingested twice. Those copies are identical down to the history, so
    discarding one cannot change any statistic.

    Duplicates that DIFFER are a different matter entirely: two distinct
    training runs wearing one label, which is what a glob collision between
    ``mlp`` and ``mlp_no_mup`` produces. Averaging or arbitrarily picking
    between those silently changes the answer, so that case still raises.
    Use ``inspect_logs.py --split`` to separate them.
    """
    dup_mask = df.duplicated(subset=["num_params", "seed"], keep=False)
    if not dup_mask.any():
        return df

    keep, dropped, conflicting = [], 0, []
    for (p, seed), g in df.groupby(["num_params", "seed"], sort=False):
        keep.append(g.index[0])
        if len(g) == 1:
            continue
        first = df.loc[g.index[0]]
        for idx in g.index[1:]:
            if _rows_identical(first, df.loc[idx]):
                dropped += 1
            else:
                conflicting.append((int(df.loc[idx, "D"]), int(seed)))

    if conflicting:
        raise ValueError(
            "duplicate (num_params, seed) runs that are NOT identical -- two "
            f"different runs share a label: {sorted(set(conflicting))}. "
            "Averaging them or picking one arbitrarily would change the "
            "result. Run `python inspect_logs.py <pickle> --split` to "
            "separate them, or rebuild the pickle with a tag glob that does "
            "not collide (see make_logs.select_csvs)."
        )

    if dropped and verbose:
        print(f"[cs] {dropped} exact duplicate run(s) dropped (identical "
              f"history and metadata; upstream save_logs.py ingests a run "
              f"once per tag it carries)")
    return df.loc[sorted(keep)]


def load_runs(
    path,
    min_D: Optional[float] = None,
    max_D: Optional[float] = None,
    min_seeds: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load a ``logs/*.pkl`` written by ``make_logs.py`` (or the authors' own).

    Normalizes the schema, filters widths, and drops any width that does not
    have ``min_seeds`` seeds -- such a width would contribute to ``Delta`` while
    having no ``sigma``, which makes the two curves quietly incomparable.

    Returns a DataFrame with columns
    ``history, D, num_params, seed, opt_L, opt_C``.
    """
    df = pd.read_pickle(path) if not isinstance(path, pd.DataFrame) else path.copy()
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path}: expected a pandas DataFrame, got {type(df)}")

    # --- model size key ----------------------------------------------------
    # make_logs.py writes num_params; the authors' notebook uses P. Accept both,
    # but refuse to guess if they disagree.
    if "num_params" in df.columns and "P" in df.columns:
        a = pd.to_numeric(df["num_params"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(df["P"], errors="coerce").to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        if both.any() and not np.allclose(a[both], b[both]):
            raise ValueError(
                "'num_params' and 'P' disagree in this pickle; refusing to pick one"
            )
    elif "P" in df.columns:
        df = df.assign(num_params=pd.to_numeric(df["P"], errors="coerce"))
    elif "num_params" not in df.columns:
        raise ValueError("pickle has neither 'num_params' nor 'P'")

    for col in ("history", "seed", "D"):
        if col not in df.columns:
            raise ValueError(f"pickle is missing required column {col!r}")

    # --- history sanity ----------------------------------------------------
    for i, h in df["history"].items():
        if not isinstance(h, pd.DataFrame):
            raise ValueError(f"row {i}: 'history' is {type(h)}, expected a DataFrame")
        missing = [c for c in _REQUIRED_HISTORY_COLS if c not in h.columns]
        if missing:
            raise ValueError(f"row {i}: history is missing {missing}")

    # --- opt_L / opt_C fallback (collapse.ipynb cell 5) --------------------
    if "opt_L" not in df.columns:
        df = df.assign(
            opt_L=df["history"].map(lambda h: float(h["test_loss"].iloc[-1]))
        )
    if "opt_C" not in df.columns:
        df = df.assign(
            opt_C=df["history"].map(lambda h: float(h["compute"].iloc[-1]))
        )

    df = df.copy()
    for c in ("D", "num_params", "seed", "opt_L", "opt_C"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["D"] = df["D"].astype(int)
    df["num_params"] = df["num_params"].astype(np.int64)

    n0 = len(df)
    df = df[np.isfinite(df["opt_L"]) & np.isfinite(df["opt_C"])]
    if min_D is not None:
        df = df[df["D"] >= min_D]
    if max_D is not None:
        df = df[df["D"] <= max_D]
    if verbose and len(df) < n0:
        print(f"[cs] {n0 - len(df)} run(s) dropped by width filter / non-finite opt_L")

    df = _dedupe_identical(df, verbose=verbose)

    # --- min-seeds filter --------------------------------------------------
    counts = df.groupby("num_params")["seed"].nunique()
    thin = counts[counts < min_seeds]
    if len(thin):
        for p in thin.index:
            d = int(df.loc[df["num_params"] == p, "D"].iloc[0])
            print(
                f"[cs] WARNING: dropping D={d} (p={p}) -- {counts[p]} seed(s), "
                f"need {min_seeds}. A width with no noise floor cannot be "
                f"compared against Delta."
            )
        df = df[~df["num_params"].isin(thin.index)]

    if df.empty:
        raise ValueError(
            "no runs left after filtering -- check --min-D and the seed count"
        )

    cols = ["history", "D", "num_params", "seed", "opt_L", "opt_C"]
    df = df[cols].sort_values(["num_params", "seed"]).reset_index(drop=True)
    if verbose:
        print(
            f"[cs] {len(df)} runs, {df['D'].nunique()} widths "
            f"({sorted(df['D'].unique())}), seeds {sorted(df['seed'].unique())}"
        )
    return df


def fit_L0(
    runs: pd.DataFrame, L0: Optional[float] = None, num_inits: int = 10
) -> dict:
    """Estimate the irreducible loss from the compute-optimal Pareto points.

    Mirrors ``collapse.ipynb`` cell 10: average ``opt_L``/``opt_C`` over seeds
    within each width, then fit ``L = a C^-b + L0`` to those points.

    Pass ``L0`` to bypass the fit (sensitivity checks, or a value carried over
    from another ladder). Note the paper fits ``L0`` separately per scaling
    ladder -- muP and no-muP each get their own.
    """
    min_opt_L = float(runs["opt_L"].min())
    if L0 is not None:
        L0 = float(L0)
        if L0 >= min_opt_L:
            raise ValueError(
                f"L0={L0:.6g} >= min(opt_L)={min_opt_L:.6g}; reducible losses "
                "would be non-positive and every ratio meaningless"
            )
        return {"L0": L0, "a": np.nan, "b": np.nan, "r2": np.nan, "source": "user"}

    pareto = runs.groupby("num_params")[["opt_L", "opt_C"]].mean().reset_index()
    if len(pareto) < 3:
        raise ValueError(
            f"only {len(pareto)} width(s); need at least 3 Pareto points to fit "
            "L0 (or pass --L0 explicitly)"
        )
    res = fit_power_law(pareto["opt_C"].to_numpy(), pareto["opt_L"].to_numpy(),
                        num_inits=num_inits)
    res = dict(res)
    res["source"] = "fit"
    if res["L0"] >= min_opt_L:
        raise ValueError(
            f"fitted L0={res['L0']:.6g} >= min(opt_L)={min_opt_L:.6g}. The "
            "power-law fit failed; inspect the Pareto points or pass --L0."
        )
    return res


# ---------------------------------------------------------------------------
# 3. Resampling onto a shared normalized-compute grid
# ---------------------------------------------------------------------------

@dataclass
class Curves:
    """Per-run curves resampled onto one grid of normalized compute."""

    grid: np.ndarray  # (G,)   x in [0, 1]
    ell: np.ndarray   # (R, G) ell(x, p, w)   -- self-normalized
    lred: np.ndarray  # (R, G) L_red(x, p, w) -- raw reducible loss
    p: np.ndarray     # (R,)   num_params
    seed: np.ndarray  # (R,)
    D: np.ndarray     # (R,)
    L0: float

    @property
    def widths(self) -> np.ndarray:
        """Unique model sizes, ascending."""
        return np.unique(self.p)

    def D_of(self, p) -> int:
        return int(self.D[np.argmax(self.p == p)])


def make_grid(
    n: int = 200, x_min: float = 0.0, x_max: float = 1.0, spacing: str = "linear"
) -> np.ndarray:
    """Grid of normalized compute. ``x_max`` is guaranteed to be the last point.

    ``ell(1) == 1`` by construction, so having 1.0 land exactly on a grid point
    keeps ``Delta(1) == 0`` exact rather than an interpolation near-miss.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if spacing == "linear":
        grid = np.linspace(x_min, x_max, n)
    elif spacing == "log":
        if x_min <= 0:
            raise ValueError("log spacing needs x_min > 0")
        grid = np.geomspace(x_min, x_max, n)
    else:
        raise ValueError(f"unknown spacing {spacing!r}")
    grid[-1] = x_max
    return grid


def resample(runs: pd.DataFrame, L0: float, grid: np.ndarray) -> Curves:
    """Interpolate every run's reducible and normalized loss onto ``grid``.

    Necessary, not cosmetic: ``train.py`` builds ``eval_steps`` as the set union
    of a linspace and a geomspace, so the eval grid -- and even the number of
    eval points -- differs from run to run. Nothing downstream may assume the
    raw arrays are aligned.
    """
    grid = np.asarray(grid, dtype=float)
    R, G = len(runs), len(grid)
    ell = np.full((R, G), np.nan)
    lred = np.full((R, G), np.nan)

    for i, (_, row) in enumerate(runs.iterrows()):
        h = row["history"]
        opt_C, opt_L = float(row["opt_C"]), float(row["opt_L"])

        denom = opt_L - L0
        if denom <= 0:
            raise ValueError(
                f"D={int(row['D'])} seed={int(row['seed'])}: opt_L - L0 = "
                f"{denom:.6g} <= 0; L0 is too large for this ladder"
            )

        x = h["compute"].to_numpy(dtype=float) / opt_C
        Lr = h["test_loss"].to_numpy(dtype=float) - L0

        finite = np.isfinite(x) & np.isfinite(Lr)
        x, Lr = x[finite], Lr[finite]
        if x.size < 2:
            raise ValueError(
                f"D={int(row['D'])} seed={int(row['seed'])}: fewer than 2 usable "
                "eval points"
            )

        order = np.argsort(x, kind="stable")
        x, Lr = x[order], Lr[order]
        # Duplicate x (same compute logged twice) breaks np.interp monotonicity;
        # keep the last occurrence.
        keep = np.r_[np.diff(x) > 0, True]
        x, Lr = x[keep], Lr[keep]

        if abs(x[-1] - 1.0) > 1e-6:
            raise ValueError(
                f"D={int(row['D'])} seed={int(row['seed'])}: last logged compute "
                f"gives x={x[-1]:.6f}, not 1. opt_C does not match the history, "
                "so opt_L and the curve refer to different points."
            )

        lred[i] = np.interp(grid, x, Lr, left=np.nan, right=np.nan)
        ell[i] = lred[i] / denom

    # ell(1) == 1 exactly, by definition -- not by interpolation round-off.
    if abs(grid[-1] - 1.0) < 1e-12:
        ell[:, -1] = 1.0

    return Curves(
        grid=grid,
        ell=ell,
        lred=lred,
        p=runs["num_params"].to_numpy(),
        seed=runs["seed"].to_numpy(),
        D=runs["D"].to_numpy(),
        L0=float(L0),
    )


# ---------------------------------------------------------------------------
# 4. The statistics
# ---------------------------------------------------------------------------

def _nan_mean_var(a: np.ndarray, ddof: int):
    """(mean, var) down axis 0, nan-aware, with too-few-samples -> nan."""
    n = np.sum(np.isfinite(a), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(a, axis=0)
        var = np.nanvar(a, axis=0, ddof=ddof)
    var = np.where(n > ddof, var, np.nan)
    mean = np.where(n > 0, mean, np.nan)
    return mean, var


def collapse_deviation(
    curves: Curves, ddof: int = 1, estimator: str = "decomposed"
) -> Dict[str, np.ndarray]:
    """Delta(x) -- paper Eq. 2, with the Eq. 4 decomposition.

    Each *width* is weighted equally regardless of its seed count, which is what
    ``E_p`` over the ladder's (log-uniform) empirical distribution of model sizes
    means:

        mean    = E_p E_w[ell]
        within  = E_p V_w[ell]          seed noise, averaged over widths
        between = V_p E_w[ell]          genuine model-to-model spread
        Delta   = sqrt(within + between) / mean

    ``delta_pooled`` is the naive std/mean over all runs at once; it coincides
    with ``delta`` on a balanced design and is kept as a cross-check.
    """
    if estimator not in ("decomposed", "pooled"):
        raise ValueError(f"unknown estimator {estimator!r}")

    widths = np.unique(curves.p)
    per_width_mean, per_width_var = [], []
    for p in widths:
        m, v = _nan_mean_var(curves.ell[curves.p == p], ddof)
        per_width_mean.append(m)
        per_width_var.append(v)
    per_width_mean = np.asarray(per_width_mean)  # (P, G)
    per_width_var = np.asarray(per_width_var)

    mean, between = _nan_mean_var(per_width_mean, ddof)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        within = np.nanmean(per_width_var, axis=0)

    pooled_mean, pooled_var = _nan_mean_var(curves.ell, ddof)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "mean": mean,
            "delta_seed": np.sqrt(within) / mean,
            "delta_width": np.sqrt(between) / mean,
            "delta_decomposed": np.sqrt(within + between) / mean,
            "delta_pooled": np.sqrt(pooled_var) / pooled_mean,
        }
    out["delta"] = out["delta_decomposed"] if estimator == "decomposed" \
        else out["delta_pooled"]
    return out


def noise_floor(curves: Curves, ddof: int = 1) -> Dict[str, object]:
    """sigma(x, p) -- paper Eq. 3. Per-model relative seed noise.

    Computed on ``curves.lred``, the **raw reducible loss**, NOT on ``ell``.
    This is the whole point of the comparison: ``Delta`` is the spread of the
    self-normalized curve, ``sigma`` the spread of the unnormalized one, and the
    gap between them is the variance reduction from the control variate. Compute
    sigma on ``ell`` and the two become the same statistic (see
    tests T6 and T8).
    """
    per_p: Dict[int, np.ndarray] = {}
    for p in np.unique(curves.p):
        m, v = _nan_mean_var(curves.lred[curves.p == p], ddof)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_p[int(p)] = np.sqrt(v) / m

    stack = np.asarray(list(per_p.values()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        median = np.nanmedian(stack, axis=0)
    return {"per_p": per_p, "mean": mean, "median": median}


def per_model_collapse_deviation(
    curves: Curves, ddof: int = 1
) -> Dict[int, np.ndarray]:
    """Delta-tilde(x, p) = std_w(ell) / mean_w(ell) -- paper section 3.3.

    The like-for-like partner of ``sigma(x, p)``: same seeds, same width, same
    statistic, differing *only* in whether the curve was divided by its own
    final value. The gap between this and ``sigma`` is the control variate,
    isolated.
    """
    out: Dict[int, np.ndarray] = {}
    for p in np.unique(curves.p):
        m, v = _nan_mean_var(curves.ell[curves.p == p], ddof)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[int(p)] = np.sqrt(v) / m
    return out


def supercollapse_onset(
    grid: np.ndarray, delta: np.ndarray, sigma_bar: np.ndarray
) -> float:
    """Largest ``d`` such that ``Delta(x) < sigma_bar(x)`` for all ``x > 1 - d``.

    The paper reports ``d`` as large as 0.5 for LR-decay schedules. Scans from
    the right and stops at the last grid point where the inequality fails.
    ``x = 1`` is excluded (``Delta(1) = 0`` trivially), as are points where
    either curve is nan.
    """
    grid = np.asarray(grid, float)
    delta = np.asarray(delta, float)
    sigma_bar = np.asarray(sigma_bar, float)

    interior = grid < 1.0 - 1e-12
    ok = interior & np.isfinite(delta) & np.isfinite(sigma_bar)
    if not ok.any():
        return float("nan")

    idx = np.flatnonzero(ok)
    fails = idx[delta[idx] >= sigma_bar[idx]]
    if fails.size == 0:
        return 1.0 - float(grid[idx[0]])
    return 1.0 - float(grid[fails[-1]])


def fraction_below_floor(
    grid: np.ndarray, delta: np.ndarray, sigma_bar: np.ndarray
) -> float:
    """Fraction of interior grid points where ``Delta(x) < sigma_bar(x)``.

    Companion to ``supercollapse_onset``, which takes the paper's definition
    literally and therefore stops at the *last* violation -- one noisy grid
    point can cut the reported window in half. This statistic does not care
    about isolated spikes. Report both; if they disagree badly, look at the
    figure before believing either.
    """
    grid = np.asarray(grid, float)
    ok = (grid < 1.0 - 1e-12) & np.isfinite(delta) & np.isfinite(sigma_bar)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.asarray(delta)[ok] < np.asarray(sigma_bar)[ok]))


# ---------------------------------------------------------------------------
# 5. One-call pipeline
# ---------------------------------------------------------------------------

def analyze(
    path,
    min_D: Optional[float] = None,
    max_D: Optional[float] = None,
    L0: Optional[float] = None,
    n_grid: int = 200,
    x_min: float = 0.0,
    spacing: str = "linear",
    ddof: int = 1,
    estimator: str = "decomposed",
    min_seeds: int = 2,
    verbose: bool = True,
) -> dict:
    """pickle -> everything. Returns runs, fit, curves, delta, sigma, onset."""
    runs = load_runs(path, min_D=min_D, max_D=max_D, min_seeds=min_seeds,
                     verbose=verbose)
    fit = fit_L0(runs, L0=L0)
    grid = make_grid(n_grid, x_min=x_min, x_max=1.0, spacing=spacing)
    curves = resample(runs, fit["L0"], grid)
    dev = collapse_deviation(curves, ddof=ddof, estimator=estimator)
    sig = noise_floor(curves, ddof=ddof)
    onset = supercollapse_onset(grid, dev["delta"], sig["mean"])
    frac = fraction_below_floor(grid, dev["delta"], sig["mean"])
    if verbose:
        print(f"[cs] L0 = {fit['L0']:.6g} ({fit['source']}), r2 = {fit['r2']}")
        print(f"[cs] supercollapse onset delta = {onset:.3f}, "
              f"fraction below floor = {frac:.3f}")
    return {
        "runs": runs,
        "fit": fit,
        "curves": curves,
        "delta": dev,
        "sigma": sig,
        "onset": onset,
        "frac_below": frac,
        "grid": grid,
    }
