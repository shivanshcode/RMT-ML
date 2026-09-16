"""rmt.spacing — PART 4: bulk RMT universality on level spectra.

Atas r-statistic, spectral unfolding, NN-spacing tests vs Wigner-GOE / Poisson /
Brody, Dyson-Mehta Delta_3(L) and number variance Sigma^2(L).
Pure numpy/scipy.

Provenance
----------
The unfolding, the Sigma^2 estimator and the Wigner/Brody machinery are ported
from Thamm, Staats & Rosenow (2022), ``src/rmt_utils.py``:

  * ``GaussBroadening``            <- ``rmt_utils.GaussBroadening``
  * ``unfold(..., method="gauss")``<- ``rmt_utils.GaussBroadening.unfold_spectrum``
  * ``nn_spacing``                 <- ``rmt_utils.level_spacings``
  * ``wigner_goe_cdf``             <- ``rmt_utils.wignerSurmise_cdf``
  * ``wigner_goe_pdf``             <- ``rmt_utils.wignerSurmise``
  * ``nn_spacing_ks``              <- ``rmt_utils.ksTest_wigner``
  * ``brody_beta``                 <- ``rmt_utils.fit_Brody_bootstrap``
  * ``sigma2``                     <- ``rmt_utils.level_number_variance`` /
                                      ``_sigma_iter_converge_L``
                                      (numba loop re-expressed as a vectorised
                                      cumulative mean -- algebraically identical)

Delta_3 is not implemented in Thamm's release; here it is evaluated with the
exact Bohigas-Giannoni closed form for a staircase (no numerical quadrature).
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple, Union
import numpy as np
from scipy.special import erf, gamma as _gamma
from scipy.optimize import curve_fit
import scipy.stats

GAMMA = 0.5772156649015329       # Euler-Mascheroni

# Largest L at which Sigma^2 is within ~5% of theory, per unfolding.
#
# CORRECTED (see tests/test_sigma2_lmax_calibration.py). The previous value for
# the local kernel, gauss: 15.0, does not meet the 5% claim it was annotated
# with. Measured on square Wishart at N=1200, win_size=15:
#
#     L        3      5     10     15     20
#     err    0.5%   0.5%  10.7%  16.9%  24.7%
#
# i.e. the local kernel is already 11% LOW at L=10, which per_matrix was
# retaining and writing to sigma2_L10. That is a branch-dependent bias in a
# reported column -- the same family of defect as review §3/F3, but affecting
# the values rather than their missingness.
#
# The reach of a local kernel scales with its own window, and L <= win/3 holds
# the 5% bound for win in {15, 30} (win=15 -> 5, win=30 -> 10). It does NOT
# extrapolate: at win=60 the kernel over-smooths relative to the spectrum and
# degrades at every L, so the rule is capped. Use sigma2_reliable_lmax().
SIGMA2_RELIABLE_LMAX = {"cheb": 50.0, "auto": 50.0, "poly": 50.0, "gauss": 5.0}

#: Beyond this window the local-kernel calibration does not hold at all.
GAUSS_WIN_CALIBRATED_MAX = 30

#: Below this window the local kernel has NO reliable reach: measured Sigma^2
#: error at win_size = 5 is -15.3 % already at L = 3.  The previous rule,
#: ``max(3.0, win/3)``, certified L = 3 there.  Now the gauss branch reports no
#: usable L at all below this window.
GAUSS_WIN_MIN_USABLE = 12


def sigma2_reliable_lmax(method: str, win_size: int = 15) -> float:
    """Largest trustworthy L for ``method`` at this unfolding window.

    Global fits do not depend on a window and are calibrated to L = 50.  The
    local kernel's reach is set by its own window: ``win_size / 3``, capped at
    the largest window the calibration covers.
    """
    if method == "gauss":
        w = int(win_size)
        if w < GAUSS_WIN_MIN_USABLE:
            # Measured: at w = 5 the kernel is already -15 % at L = 3, so the
            # old max(3.0, w/3) floor certified a value that is badly biased.
            # Below this window the local kernel has no trustworthy reach at all.
            return 0.0
        return float(min(w, GAUSS_WIN_CALIBRATED_MAX) / 3.0)
    return float(SIGMA2_RELIABLE_LMAX.get(method, 5.0))
# Delta_3 removes the local linear trend, so it is accurate to L=50 either way.
DELTA3_RELIABLE_LMAX = 50.0

# Max float64 elements held at once by the broadening kernel (~256 MB).
_KERNEL_BLOCK = 32_000_000

_RngLike = Union[int, np.random.Generator, None]


def _as_rng(rng: _RngLike) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


# --------------------------------------------------------------------------- #
# r-statistic (Atas 2013, no unfolding needed)                                 #
# --------------------------------------------------------------------------- #
def r_statistic(levels) -> float:
    """<r_i>, r_i = min(d_i, d_i+1)/max(d_i, d_i+1).  GOE 0.5307, Poisson 0.3863."""
    x = np.sort(np.asarray(levels, dtype=np.float64))
    d = np.diff(x)
    d = d[d > 0]
    if d.size < 2:
        return float("nan")
    r = np.minimum(d[:-1], d[1:]) / np.maximum(d[:-1], d[1:])
    return float(np.mean(r))


# --------------------------------------------------------------------------- #
# Unfolding                                                                    #
# --------------------------------------------------------------------------- #
class GaussBroadening:
    """Adaptive Gaussian-kernel density / unfolding (Thamm ``rmt_utils.py``).

    Each level is smeared by a Gaussian whose width is the local level spacing
    measured over a window of ``win_size`` levels on each side, so the estimated
    integrated density N(x) tracks hard spectral edges (Marchenko-Pastur) that a
    global polynomial cannot represent.

    Level accounting (review §4.2 -- the previous docstring claimed
    ``method='replicate'`` retained *every* input level; it does not).
    ``unfold_spectrum`` evaluates the kernel on ``sv[2w:-2w]`` of the *padded*
    array, so with ``n`` input levels and ``w = win_size``:

    =================  ==================  ============================
    ``method``         levels returned     dropped
    =================  ==================  ============================
    ``'replicate'``    ``n - 2*w``         outer ``w`` on each side
    ``'drop'``         ``n - 4*w``         outer ``2*w`` on each side
    =================  ==================  ============================

    e.g. n = 1024: w=5 -> 1014, w=15 -> 994, w=30 -> 964 under 'replicate'.
    The discarded levels are the extreme ones -- exactly where MP-edge
    deviations live -- so ``unfold_n_levels`` must be reported alongside
    ``n_levels_bulk`` rather than assumed equal to it.  :meth:`n_out` gives the
    count in advance so a caller can budget for it.
    """

    def __init__(self, win_size: int = 15, method: str = "replicate"):
        if method not in {"replicate", "drop"}:
            raise ValueError("method must be 'replicate' or 'drop'")
        self.win_size = int(win_size)
        self.method = method

    def n_out(self, n_in: int) -> int:
        """Number of unfolded levels returned for ``n_in`` input levels."""
        k = 2 if self.method == "replicate" else 4
        return max(0, int(n_in) - k * self.win_size)

    # -- internal ---------------------------------------------------------- #
    def _pad(self, sv: np.ndarray) -> np.ndarray:
        if self.method == "replicate":
            return np.pad(sv, (self.win_size, self.win_size), "edge")
        return sv

    def _windows(self, sv: np.ndarray):
        w = self.win_size
        means = sv[w:-w].reshape(1, -1)
        stdvs = ((sv[2 * w:] - sv[:-2 * w]) / 2.0).reshape(1, -1)
        # guard against exactly-degenerate levels (padded edges, duplicate svals)
        stdvs = np.where(stdvs <= 0.0, 1e-300, stdvs)
        return means, stdvs

    # -- public ------------------------------------------------------------ #
    def unfold_spectrum(self, levels) -> np.ndarray:
        """xi = N_bar(lambda), the smooth integrated density at each level.

        Evaluated in row blocks: the full kernel is n x n, which is 2.1 GB of
        float64 at n = 16384, so it is never materialised at once.
        """
        sv = np.sort(np.asarray(levels, dtype=np.float64))
        if sv.size <= 4 * self.win_size + 2:
            raise ValueError(
                f"need > {4 * self.win_size + 2} levels for win_size="
                f"{self.win_size}; got {sv.size}")
        sv = self._pad(sv)
        w = self.win_size
        x = sv[2 * w:-2 * w]
        means, stdvs = self._windows(sv)
        scale = np.sqrt(2.0) * stdvs
        out = np.empty(x.size, dtype=np.float64)
        step = max(1, int(_KERNEL_BLOCK // max(1, means.size)))
        for i in range(0, x.size, step):
            xb = x[i:i + step].reshape(-1, 1)
            out[i:i + step] = np.sum(0.5 * (1.0 + erf((xb - means) / scale)),
                                     axis=1)
        return np.sort(out)

    def broaden_spectrum(self, x, levels) -> np.ndarray:
        """Smooth probability density of the spectrum evaluated at ``x``."""
        sv = np.sort(np.asarray(levels, dtype=np.float64))
        n = sv.size
        sv = self._pad(sv)
        means, stdvs = self._windows(sv)
        xm = np.asarray(x, dtype=np.float64).ravel()
        norm = np.sqrt(2.0 * np.pi * stdvs ** 2)
        out = np.empty(xm.size, dtype=np.float64)
        step = max(1, int(_KERNEL_BLOCK // max(1, means.size)))
        for i in range(0, xm.size, step):
            xb = xm[i:i + step].reshape(-1, 1)
            out[i:i + step] = np.sum(
                np.exp(-((xb - means) ** 2) / (2.0 * stdvs ** 2)) / norm, axis=1)
        return out / n


def _unfold_poly(levels, deg: int = 7) -> np.ndarray:
    """Legacy global unfolding: raw ``np.polyfit`` on rescaled abscissae.

    Kept only for reproducing old results.  Prefer ``method="cheb"``: the power
    basis Vandermonde is badly conditioned above degree ~10, which is exactly
    where a global fit starts to be able to follow an MP density.
    """
    x = np.sort(np.asarray(levels, dtype=np.float64))
    lo, hi = x[0], x[-1]
    span = hi - lo
    t = (x - lo) / span if span > 0 else x
    stair = np.arange(1, x.size + 1, dtype=np.float64)
    return np.polyval(np.polyfit(t, stair, deg), t)


def _unfold_cheb(levels, deg: int = 7) -> np.ndarray:
    """Global unfolding by a Chebyshev fit to the staircase.

    Same model class as the polynomial fit but in a well-conditioned basis on
    [-1, 1], so the degree can be raised until the density is actually
    represented.  Unlike a local kernel this preserves long-range count
    fluctuations, which is what Sigma^2(L) measures at large L.
    """
    from numpy.polynomial import chebyshev as _cheb
    x = np.sort(np.asarray(levels, dtype=np.float64))
    lo, hi = x[0], x[-1]
    if hi <= lo:
        raise ValueError("degenerate spectrum")
    tt = 2.0 * (x - lo) / (hi - lo) - 1.0
    stair = np.arange(1, x.size + 1, dtype=np.float64)
    return _cheb.chebval(tt, _cheb.chebfit(tt, stair, int(deg)))


# --------------------------------------------------------------------------- #
# Bulk selection (review §4.1)                                                  #
# --------------------------------------------------------------------------- #
#: Default centred fraction kept by ``bulk_levels(mode="center")``.
BULK_CENTER_FRAC = 0.7
#: Fractions the stability sweep should cover before a conclusion is claimed.
BULK_CENTER_FRAC_SWEEP = (0.6, 0.7, 0.8)


#: A level is an edge outlier if its spacing to the body exceeds this many
#: median spacings.  Rank-one spikes sit far outside the Marchenko-Pastur edge;
#: they are not bulk levels and they destroy any global fit to the staircase.
BULK_OUTLIER_TOL = 25.0

#: Peel-inward threshold for a GRADED outlier tail (spikes of decaying
#: amplitude, which never detach as a cluster).  Lower than BULK_OUTLIER_TOL
#: because each individual step is small; the cap on total removal is what keeps
#: it safe on a heavy-tailed but outlier-free spectrum.
BULK_PEEL_TOL = 6.0


def strip_edge_outliers(levels, tol: float = BULK_OUTLIER_TOL,
                        peel_tol: float = BULK_PEEL_TOL,
                        max_strip_frac: float = 0.03):
    """Drop levels detached from the body of the spectrum, from either end.

    Why adaptive rather than a fixed centred fraction.  A fixed
    ``center_frac=0.7`` throws away 30 % of the levels whether or not there is
    anything to throw away -- and fewer levels measurably *degrades* Sigma^2 at
    large L (Sigma^2(20) on a clean 512x512 null drifts 1.020 -> 0.880 as the
    cut tightens from 1.0 to 0.6, away from the exact 1.049).  This keeps every
    level unless it is genuinely detached, so clean spectra pay nothing and
    spiked spectra are still cleaned.

    Returns ``(kept_levels, n_stripped_low, n_stripped_high)``.
    """
    x = np.sort(np.asarray(levels, dtype=np.float64))
    n = x.size
    if n < 64:
        return x, 0, 0
    win = max(32, int(n * 0.05))
    cap = max(1, int(n * float(max_strip_frac)))
    core = x[win:n - win]
    med = float(np.median(np.diff(core))) if core.size > 2 else float("nan")
    if not np.isfinite(med) or med <= 0:
        return x, 0, 0

    lo, hi = 0, n

    # (a) DETACHED CLUSTER.  Equal-amplitude spikes arrive as one tight group
    # separated from the bulk by a single enormous gap (~1400x the median).  Two
    # conditions are required: gap size AND a small detached count.  Gap size
    # alone would strip ~20 % of a Poisson spectrum, whose worst edge gap is ~7x
    # the median but sits above ~8 % of the levels.
    g = np.diff(x[n - win - 1:n])
    if g.size:
        j = int(np.argmax(g))
        if g[j] > tol * med and (win - j) <= cap:
            hi = (n - win - 1) + j + 1
    g = np.diff(x[0:win + 1])
    if g.size:
        j = int(np.argmax(g))
        if g[j] > tol * med and (j + 1) <= cap:
            lo = j + 1

    # (b) GRADED TAIL.  Spikes with decaying amplitudes -- amp*(i+1)^-0.7, the
    # realistic trained-weight caricature -- do NOT detach as a cluster: they
    # thin out continuously into the bulk, so (a) finds only the topmost one.
    # Peel inward while the outermost spacing is still far above the interior
    # median.  On a Poisson spectrum the first spacing clears peel_tol only ~1.6 %
    # of the time, so this peels nothing there.
    peeled = 0
    while (hi - lo) > n // 2 and (lo + (n - hi) + peeled) < cap:
        d_hi = x[hi - 1] - x[hi - 2]
        d_lo = x[lo + 1] - x[lo]
        if d_hi > peel_tol * med and d_hi >= d_lo:
            hi -= 1
        elif d_lo > peel_tol * med:
            lo += 1
        else:
            break
        peeled += 1

    return x[lo:hi], lo, n - hi


def bulk_levels(levels, *, mode: str = "center",
                center_frac: float = BULK_CENTER_FRAC,
                lo: Optional[float] = None, hi: Optional[float] = None,
                min_keep_frac: float = 0.25) -> np.ndarray:
    """Select the levels the bulk statistics run on.

    Modes
    -----
    ``"center"`` (recommended default)
        Keep the centred ``center_frac`` of the *sorted* level list, i.e. drop
        ``(1-center_frac)/2`` of the levels from each end by rank.
    ``"mp"``
        Legacy: keep levels inside ``[lo, hi]`` (the Marchenko-Pastur edges).
    ``"auto"`` (recommended)
        Keep everything except levels detached from the body of the spectrum,
        via :func:`strip_edge_outliers`.  Costs nothing on a clean spectrum.
    ``"none"``
        Keep everything.

    Why ``"center"`` exists (review §4.1).  The MP trim is a **no-op on square
    matrices**: for n = m, ``mp_bounds`` gives nu_- = sigma(sqrt(n) - sqrt(n))
    = 0, so nothing is cut and the Bessel *hard edge* at zero stays in the
    sample.  Edge statistics are Airy (soft edge) or Bessel (hard edge), **not**
    GOE, so they contaminate a bulk test: on a square Gaussian matrix <r> is
    0.5165 on the smallest quartile of singular values against 0.5490 on the
    middle half.  A cut by *rank* removes both edges for any aspect ratio, and
    because it is a fixed fraction it removes the same amount from the null
    replicas, so the matched-shape band in :mod:`rmt.nulls` stays comparable.

    ``min_keep_frac`` guards a mis-specified ``[lo, hi]``: if the MP trim would
    keep less than that fraction of the spectrum the trim is refused (the input
    is returned unchanged), because a trim that aggressive means sigma is wrong,
    not that the spectrum is all outliers.
    """
    x = np.sort(np.asarray(levels, dtype=np.float64))
    if mode == "auto":
        return strip_edge_outliers(x)[0]
    if mode == "none" or x.size == 0:
        return x
    if mode == "center":
        f = float(center_frac)
        if not (0.0 < f <= 1.0):
            raise ValueError("center_frac must be in (0, 1]")
        if f >= 1.0:
            return x
        k = int(np.floor(x.size * (1.0 - f) / 2.0))
        if 2 * k >= x.size:
            return x
        return x[k:x.size - k] if k > 0 else x
    if mode == "mp":
        if lo is None or hi is None:
            raise ValueError("mode='mp' needs lo and hi")
        keep = x[(x >= float(lo)) & (x <= float(hi))]
        return keep if keep.size >= min_keep_frac * x.size else x
    raise ValueError("mode must be 'center', 'mp' or 'none'")


def bulk_stability(levels, stat_fn, fracs: Tuple[float, ...] = BULK_CENTER_FRAC_SWEEP) -> dict:
    """Evaluate ``stat_fn(levels)`` over several centred fractions.

    Returns ``{frac: value, ..., 'spread': max-min, 'rel_spread': spread/|mean|}``.
    A conclusion that does not survive this sweep is a statement about the
    spectral edge, not about the bulk.
    """
    out = {}
    for f in fracs:
        out[float(f)] = float(stat_fn(bulk_levels(levels, mode="center", center_frac=f)))
    vals = np.array([out[float(f)] for f in fracs], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        out["spread"] = float("nan"); out["rel_spread"] = float("nan")
        return out
    spread = float(vals.max() - vals.min())
    out["spread"] = spread
    denom = abs(float(vals.mean()))
    out["rel_spread"] = spread / denom if denom > 0 else float("nan")
    return out


def _window_counts(xi: np.ndarray, centres: np.ndarray, L: float) -> np.ndarray:
    hi = np.searchsorted(xi, centres + L / 2.0, side="right")
    lo = np.searchsorted(xi, centres - L / 2.0, side="left")
    return (hi - lo).astype(np.float64)


#: Fraction of non-positive spacings tolerated before a global fit is rejected.
#: A strict ``all(d > 0)`` is a knife-edge on ~10^3 spacings: a single dip in an
#: otherwise excellent Chebyshev staircase flips the whole matrix to the local
#: kernel.  Measured flip rate on statistically identical rectangular Wishart
#: spectra in the lambda domain was 4/8, and because per_matrix writes
#: ``sigma2_L20 = NaN`` whenever the branch is 'gauss', that produced missing
#: data correlated with a spectral property.  A fit with <=0.1% dips is still a
#: usable unfolding; one with a real density modulation fails the crosscheck.
UNFOLD_NONMONO_TOL = 1e-3


def unfolding_is_valid(xi: np.ndarray, mean_tol: float = 0.02,
                       nonmono_tol: float = UNFOLD_NONMONO_TOL) -> bool:
    """Necessary conditions: (essentially) increasing, <s> = 1 without rescaling.

    ``nonmono_tol=0`` restores the old strict behaviour.
    """
    d = np.diff(np.asarray(xi, dtype=np.float64))
    if d.size == 0 or not np.all(np.isfinite(d)):
        return False
    frac_bad = float(np.mean(d <= 0))
    return bool(frac_bad <= nonmono_tol and abs(np.mean(d) - 1.0) < mean_tol)


def _unfolded_raw(levels, deg, method, win_size, unfolded):
    """The unfolded staircase in its NATURAL order -- deliberately not sorted.

    ``nn_spacing`` sorts before differencing, which means a non-monotone
    unfolding is silently repaired: after ``np.sort`` every spacing is
    non-negative by construction, so the ``s[s >= 0]`` filter it then applies
    can never fire.  Monotonicity therefore has to be judged BEFORE the sort,
    which is what this helper exists for (review §4.5).
    """
    if unfolded is not None:
        return np.asarray(unfolded, dtype=np.float64)
    return np.asarray(unfold(levels, deg=deg, method=method,
                             win_size=win_size), dtype=np.float64)


def clean_spacings(s: np.ndarray, nonmono_tol: float = UNFOLD_NONMONO_TOL):
    """Return ``(sorted_nonnegative_spacings, frac_nonpositive, ok)``.

    Review §4.5: ``nn_spacing_ks`` and ``brody_beta`` used to do a bare
    ``s[s >= 0]``.  A negative spacing means the unfolding is not monotone,
    i.e. broken -- and silently dropping those entries changes *n*, which
    changes the KS critical value, so the reported p-value no longer refers to
    the test that was run.  The policy here matches
    :data:`UNFOLD_NONMONO_TOL`: up to ``nonmono_tol`` of dips are tolerated and
    dropped (with the count reported), and above that the statistic is NaN'd
    rather than computed on a silently truncated sample.
    """
    s = np.asarray(s, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return s, float("nan"), False
    frac_bad = float(np.mean(s <= 0))
    return np.sort(s[s > 0]), frac_bad, bool(frac_bad <= nonmono_tol)


AUTO_CROSSCHECK_L = 3.0     # scale at which a local kernel is unbiased to O(w^-1)
AUTO_CROSSCHECK_TOL = 1.5   # allowed Sigma^2 excess of the global fit over it


def _short_scale_variance(xi: np.ndarray, L: float = AUTO_CROSSCHECK_L,
                          n_windows: int = 3000, seed: int = 0) -> float:
    xi = np.sort(np.asarray(xi, dtype=np.float64))
    lo, hi = xi[0] + L / 2.0, xi[-1] - L / 2.0
    if hi <= lo:
        return float("nan")
    c = np.random.default_rng(seed).uniform(lo, hi, n_windows)
    return float(_window_counts(xi, c, L).var())


def unfolding_crosscheck(xi_global: np.ndarray, xi_local: np.ndarray) -> float:
    """Sigma^2(L=3) of the global fit divided by that of the local kernel.

    Monotonicity and unit mean spacing are necessary but *not* sufficient: a
    global fit can be smooth, increasing and unit-mean while still leaving a
    large-amplitude density modulation, which shows up as an inflated variance
    at every scale. A local kernel of window ``w`` is unbiased at L=3 *to
    O(L/w)* -- approximately, not provably; the claim is that its bias at L=3
    is far smaller than the modulation being detected, which is a quantitative
    statement, so it is measured rather than asserted.

    THE 1.5 THRESHOLD (review §4.5).  It is the midpoint, on a log scale, of a
    measured gap: under a correct global fit the ratio is ~1.0 (null 99th
    percentile ~1.2), and when the basis cannot represent the density it is
    3.7-5.9 (square matrices in the lambda = nu^2/N domain).  Both sides of
    that gap are pinned in
    ``tests/test_crosscheck_calibration.py::test_crosscheck_tolerance_separates_the_two_regimes``;
    the constant should not be changed without re-running it.
    """
    a = _short_scale_variance(xi_global)
    b = _short_scale_variance(xi_local)
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return float("inf")
    return float(a / b)



# --------------------------------------------------------------------------- #
# Coordinate transforms for the global fit  (fixes the lambda-domain failure)   #
# --------------------------------------------------------------------------- #
#: Candidate monotone reparametrisations tried by ``method="tx"``.
#:
#: WHY THIS IS FREE.  Unfolding maps levels onto xi = Nbar(x); the unfolded
#: spacings, and therefore every spacing statistic, are invariant under a smooth
#: monotone reparametrisation x -> g(x), because Nbar transforms with it.  The
#: choice of coordinate is therefore a purely *numerical* one: pick the
#: coordinate in which a degree-7 global fit actually converges.
#:
#: WHY IT MATTERS.  In the lambda = nu^2/N domain a square Wishart has
#: rho(lambda) ~ lambda^(-1/2) at the hard edge.  No finite-degree polynomial
#: can represent an inverse-square-root singularity, so the global fit leaves a
#: large residual density modulation (crosscheck ratio ~4) and the old code fell
#: back to the local kernel -- which then costs ~11 % on Sigma^2(10) through the
#: window-reach bias.  Under x -> sqrt(x) the singularity is removed exactly and
#: the global fit is recovered (crosscheck 4.45 -> 0.99).
UNFOLD_TRANSFORMS = ("identity", "sqrt", "cbrt", "log")

#: A simpler (earlier) coordinate wins if its trend power is within this factor
#: of the best, so a well-conditioned domain is never gratuitously transformed.
UNFOLD_TX_MARGIN = 3.0

#: Chebyshev degrees tried when ``deg_fixed=False``, increasing in flexibility.
#: OFF BY DEFAULT.  Searching the degree was measured and REJECTED: it lowered
#: the median error but regressed 9 of 56 rows against the delivered baseline
#: (GOE Sigma^2(20) 4.7 % -> 9.4 %), because the degree and the coordinate
#: transform interact -- a low degree in a curved coordinate can clear the trend
#: gate while still distorting long-range counts.  Degree 7 is retained, which
#: is exactly the capacity the delivered code used, so every improvement below
#: comes from the coordinate search, outlier stripping and the exact reference,
#: never from extra fit flexibility.
UNFOLD_DEGREES = (1, 2, 3, 5, 7, 11, 15)

#: A fit whose residual staircase trend power is at or below this leaves no
#: detectable long-wavelength density modulation.  Measured separation is wide:
#: an adequate fit scores 0.05-0.07 on every RMT spectrum tested, an inadequate
#: one 1.0-370.  Selecting the LEAST flexible fit that clears the gate is
#: ordinary model selection, and it matters because surplus flexibility eats the
#: very count fluctuations Sigma^2(L) is measuring -- at degree 15 the GOE
#: Sigma^2(50) is already 2.8 % low.
UNFOLD_TREND_GATE = 0.10


def _apply_transform(x: np.ndarray, name: str) -> np.ndarray:
    x = np.sort(np.asarray(x, dtype=np.float64))
    if name == "identity":
        return x
    span = x[-1] - x[0]
    if not np.isfinite(span) or span <= 0:
        return x
    if name == "sqrt":
        return np.sqrt(x - x[0] + 1e-12 * span)
    if name == "cbrt":
        return np.cbrt(x - x[0] + 1e-12 * span)
    if name == "log":
        return np.log(x - x[0] + 1e-6 * span)
    raise ValueError(f"unknown transform {name!r}")


#: (probe scale L, local-kernel window) pairs used to score a global fit.
#: A single L=3 probe is blind to LONG-wavelength density modulation: on a
#: rectangular Wishart in the lambda domain the identity coordinate scores 1.03
#: at L=3 while Sigma^2(50) is off by +68 %.  The second pair catches that.  The
#: window in each pair is chosen so the local kernel is itself trustworthy at
#: that L under ``sigma2_reliable_lmax`` (reach = win/3).
def staircase_trend_power(xi: np.ndarray, frac: float = 0.10) -> float:
    """Residual long-wavelength density modulation left by an unfolding.

    A correct unfolding makes ``D_i = xi_i - i`` a stationary fluctuation with no
    smooth trend.  A basis that cannot represent the density leaves a slow,
    large-amplitude trend in D.  Smoothing D over ``frac * n`` levels kills the
    genuine short-range fluctuation and leaves the modulation, whose variance is
    returned in units of (mean spacing)^2.

    Reference-free by construction -- unlike :func:`unfolding_crosscheck` it does
    not compare against a local kernel, so it stays valid at long range where the
    kernel's own window-reach bias would make it a bad yardstick.

    NOT circular.  It is used as a *gate*, not minimised against theory: a good
    fit scores ~1e0 and a broken one ~1e2-1e4, and a Poisson spectrum -- which
    has genuinely huge long-range fluctuation -- still passes, so the criterion
    does not manufacture rigidity.
    """
    xi = np.asarray(xi, dtype=np.float64)
    n = xi.size
    if n < 64:
        return float("nan")
    d = xi - np.arange(n, dtype=np.float64)
    w = max(8, int(n * float(frac)))
    kern = np.ones(w, dtype=np.float64) / w
    trend = np.convolve(d, kern, mode="valid")
    return float(np.var(trend))


def looks_pre_unfolded(levels, n_blocks: int = 10, flat_tol: float = 1.5,
                       mean_tol: float = 0.05) -> bool:
    """True when the levels already have unit mean spacing and a flat density.

    Some inputs -- a synthetic Poisson level sequence, or any spectrum a caller
    has already unfolded -- need no unfolding at all, and fitting one anyway is
    actively harmful: a degree-7 staircase fit removes genuine long-wavelength
    count fluctuation, which is precisely what Sigma^2(L) measures.  Measured on
    a 6000-level Poisson sequence, the degree-7 fit pulls Sigma^2(50) from 50.0
    down to 45.6 (-8.8 %); passing the levels through untouched is exact.

    The test is a flatness test, so it cannot fire on a real spectrum: across
    ten equal blocks the local mean spacing of a GOE spectrum varies by ~3x
    (semicircle) and of a square Wishart by ~5x (quarter circle), both far above
    ``flat_tol``.
    """
    x = np.sort(np.asarray(levels, dtype=np.float64))
    d = np.diff(x)
    if d.size < n_blocks * 8 or not np.all(np.isfinite(d)):
        return False
    if abs(float(np.mean(d)) - 1.0) > mean_tol:
        return False
    blocks = np.array_split(d, int(n_blocks))
    means = np.array([float(np.mean(b)) for b in blocks if b.size])
    if means.min() <= 0:
        return False
    return bool(means.max() / means.min() < flat_tol)


def unfold_transform_auto(levels, deg: int = 7, *, win_size: int = 15,
                          pad: str = "replicate", deg_fixed: bool = True,
                          transforms: Sequence[str] = UNFOLD_TRANSFORMS):
    """Global fit in the best-conditioned monotone coordinate.

    Returns ``(xi, branch, transform, crosscheck)``.  Each candidate coordinate
    is scored by :func:`unfolding_crosscheck` -- the residual density modulation
    the fit leaves behind, measured against the local kernel -- and the
    coordinate closest to a ratio of 1 wins.  Only if *every* coordinate fails
    the ``AUTO_CROSSCHECK_TOL`` gate do we fall back to the local kernel, which
    is the branch that carries the Sigma^2 window-reach bias.
    """
    if looks_pre_unfolded(levels):
        return (np.sort(np.asarray(levels, dtype=np.float64)),
                "identity", "identity", 1.0)
    local = GaussBroadening(win_size=win_size, method=pad).unfold_spectrum(levels)
    degrees = (int(deg),) if deg_fixed else UNFOLD_DEGREES
    fallback = None

    for d in degrees:
        cands = []
        for name in transforms:
            try:
                z = _apply_transform(levels, name)
                xi = _unfold_cheb(z, deg=d)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
            if not unfolding_is_valid(xi):
                continue
            ratio = unfolding_crosscheck(xi, local)
            trend = staircase_trend_power(xi)
            if not (np.isfinite(ratio) and np.isfinite(trend)):
                continue
            if ratio > AUTO_CROSSCHECK_TOL:
                continue
            cands.append((trend, xi, name, ratio, d))
        if not cands:
            continue
        floor = min(c[0] for c in cands)
        pick = next(c for c in cands if c[0] <= floor * UNFOLD_TX_MARGIN)
        if fallback is None:
            fallback = pick            # least flexible fit that is at least valid
        if pick[0] <= UNFOLD_TREND_GATE:
            return pick[1], "cheb", pick[2], pick[3]

    # Nothing cleared the trend gate.  Either the density is genuinely
    # unrepresentable, or -- as for a Poisson spectrum -- the "trend" is real
    # long-range fluctuation that no fit should be removing.  Both cases call
    # for the LEAST flexible valid fit, not the most.
    if fallback is not None:
        return fallback[1], "cheb", fallback[2], fallback[3]
    return local, "gauss", "identity", float("nan")


def unfold(levels, deg: int = 7, *, method: str = "auto",
           win_size: int = 15, pad: str = "replicate") -> np.ndarray:
    """Unfolded spectrum xi with mean nearest-neighbour spacing ~ 1.

    Methods
    -------
    ``"auto"`` (default)
        Try the global Chebyshev fit; if it is not monotone or its mean spacing
        is off, fall back to Gaussian broadening.  This is the right default
        because the two unfoldings have complementary failure modes: the global
        fit is accurate at long range but needs a density its basis can
        represent, while the local kernel handles any density but erases count
        fluctuations on scales beyond its window.
    ``"cheb"``   global Chebyshev fit (see :func:`_unfold_cheb`).
    ``"gauss"``  adaptive Gaussian broadening (Thamm ``GaussBroadening``).
    ``"poly"``   legacy ``np.polyfit``; comparison only.
    """
    if method == "gauss":
        return GaussBroadening(win_size=win_size, method=pad).unfold_spectrum(levels)
    if method == "poly":
        return _unfold_poly(levels, deg=deg)
    if method == "cheb":
        return _unfold_cheb(levels, deg=deg)
    if method == "tx":
        return unfold_transform_auto(levels, deg=deg, win_size=win_size, pad=pad)[0]
    if method != "auto":
        raise ValueError("method must be 'tx', 'auto', 'cheb', 'gauss' or 'poly'")
    return unfold_auto(levels, deg=deg, win_size=win_size, pad=pad)[0]


def unfold_auto(levels, deg: int = 7, *, win_size: int = 15,
                pad: str = "replicate") -> Tuple[np.ndarray, str]:
    """Return (unfolded spectrum, name of the branch that produced it).

    Prefers the global Chebyshev fit for its long-range fidelity, but only if it
    passes both the monotone/unit-mean test and the short-scale cross-check
    against the local kernel.
    """
    xi, branch, _tx, _ratio = unfold_transform_auto(
        levels, deg=deg, win_size=win_size, pad=pad)
    return xi, branch


def unfold_diagnostics(levels, **kw) -> dict:
    """Health check on the unfolding: a valid unfolding is strictly increasing
    and has mean spacing 1 *without* any rescaling.

    Returns {unfold_mean_spacing, unfold_frac_nonpositive, unfold_n_levels}.
    """
    xi = unfold(levels, **kw)
    d = np.diff(xi)
    if d.size == 0:
        return {"unfold_mean_spacing": float("nan"),
                "unfold_frac_nonpositive": float("nan"),
                "unfold_n_levels": int(xi.size)}
    return {"unfold_mean_spacing": float(np.mean(d)),
            "unfold_frac_nonpositive": float(np.mean(d <= 0)),
            "unfold_n_levels": int(xi.size)}


def nn_spacing(levels, deg: int = 7, *, method: str = "auto",
               win_size: int = 15, renormalise: bool = False,
               unfolded: Optional[np.ndarray] = None) -> np.ndarray:
    """Nearest-neighbour spacings s_i of the unfolded spectrum.

    With a correct unfolding <s> == 1 already, so ``renormalise`` is False by
    default: dividing by the sample mean would hide a broken unfolding.
    """
    xi = (np.sort(np.asarray(unfolded, dtype=np.float64)) if unfolded is not None
          else unfold(levels, deg=deg, method=method, win_size=win_size))
    s = np.diff(xi)
    s = s[np.isfinite(s)]
    if renormalise and s.size and np.mean(s) > 0:
        s = s / np.mean(s)
    return s


# --------------------------------------------------------------------------- #
# Reference spacing laws                                                       #
# --------------------------------------------------------------------------- #
def wigner_goe_pdf(s) -> np.ndarray:
    """Wigner surmise GOE pdf: (pi s/2) exp(-pi s^2/4)."""
    s = np.asarray(s, dtype=np.float64)
    return np.pi * s / 2.0 * np.exp(-np.pi * s ** 2 / 4.0)


def wigner_goe_cdf(s) -> np.ndarray:
    """Wigner surmise GOE CDF: 1 - exp(-pi s^2/4)."""
    s = np.asarray(s, dtype=np.float64)
    return 1.0 - np.exp(-np.pi * s ** 2 / 4.0)


def poisson_cdf(s) -> np.ndarray:
    """Poisson NN-spacing CDF: 1 - exp(-s)."""
    s = np.asarray(s, dtype=np.float64)
    return 1.0 - np.exp(-s)


def brody_cdf(s, beta: float) -> np.ndarray:
    """Brody CDF; beta=0 -> Poisson, beta=1 -> Wigner-GOE."""
    s = np.asarray(s, dtype=np.float64)
    b = _gamma((beta + 2.0) / (beta + 1.0)) ** (1.0 + beta)
    return 1.0 - np.exp(-b * s ** (1.0 + beta))


# --------------------------------------------------------------------------- #
# NN-spacing goodness of fit                                                   #
# --------------------------------------------------------------------------- #
def _ks_against(sample_sorted, cdf_func) -> float:
    n = sample_sorted.size
    theo = cdf_func(sample_sorted)
    emp_hi = np.arange(1, n + 1) / n
    emp_lo = np.arange(0, n) / n
    return float(max(np.max(np.abs(emp_hi - theo)), np.max(np.abs(theo - emp_lo))))


def nn_spacing_ks(levels, deg: int = 7, *, method: str = "auto",
                  win_size: int = 15, unfolded: Optional[np.ndarray] = None) -> dict:
    """KS statistic **and p-value** of P(s) vs Wigner-GOE and vs Poisson.

    p-values come from ``scipy.stats.kstest`` against a fully specified CDF
    (Thamm ``rmt_utils.ksTest_wigner``).

    .. warning::
       ``nn_KS_GOE_p`` is **not calibrated** and must not be used as a
       hypothesis test (review §3/F2): unfolded spacings are anti-correlated
       and the unfolding is fitted to the same data (a Lilliefors effect), so
       the measured null median is 0.5-0.75 with frac(p < 0.05) = 0.00.  Use
       :func:`rmt.nulls.ks_pvalue_mc` instead; it is Uniform(0,1) under the
       null by construction.  The columns are kept for continuity with
       previously published runs.

    Negative spacings (review §4.5): a non-positive spacing means the unfolding
    is not monotone.  Up to :data:`UNFOLD_NONMONO_TOL` of them are dropped and
    counted in ``nn_frac_nonpositive``/``nn_n_spacings``; beyond that the whole
    row is NaN'd, because silently dropping them changes *n* and therefore the
    KS critical value the p-value refers to.
    """
    nan = float("nan")
    empty = {"nn_KS_GOE": nan, "nn_KS_Poisson": nan,
             "nn_KS_GOE_p": nan, "nn_KS_Poisson_p": nan,
             "nn_mean_spacing": nan, "nn_n_spacings": 0,
             "nn_frac_nonpositive": nan}
    try:
        s = nn_spacing(levels, deg=deg, method=method, win_size=win_size,
                       unfolded=unfolded)
    except ValueError:
        return empty
    if s.size < 5:
        return empty
    mean_s = float(np.mean(s))
    raw = _unfolded_raw(levels, deg, method, win_size, unfolded)
    ss, frac_bad, ok = clean_spacings(np.diff(raw))
    if not ok or ss.size < 5:
        bad = dict(empty)
        bad["nn_mean_spacing"] = mean_s
        bad["nn_frac_nonpositive"] = frac_bad
        bad["nn_n_spacings"] = int(ss.size)
        return bad
    p_goe = float(scipy.stats.kstest(ss, wigner_goe_cdf).pvalue)
    p_poi = float(scipy.stats.kstest(ss, poisson_cdf).pvalue)
    return {"nn_KS_GOE": _ks_against(ss, wigner_goe_cdf),
            "nn_KS_Poisson": _ks_against(ss, poisson_cdf),
            "nn_KS_GOE_p": p_goe, "nn_KS_Poisson_p": p_poi,
            "nn_mean_spacing": mean_s, "nn_n_spacings": int(ss.size),
            "nn_frac_nonpositive": frac_bad}


def _empirical_cdf(s):
    x = np.sort(np.asarray(s, dtype=np.float64))
    return x, np.arange(x.size) / x.size


def brody_beta(levels, deg: int = 7, *, method: str = "auto", win_size: int = 15,
               n_bootstrap: int = 200, rng: _RngLike = 0,
               unfolded: Optional[np.ndarray] = None) -> dict:
    """Brody exponent beta by CDF fit, with bootstrap error.

    Port of Thamm ``rmt_utils.fit_Brody_bootstrap`` (default n_bootstrap lowered
    from 1000 to 200; the estimator is unchanged).
    beta -> 1 = Wigner-GOE (correlated), beta -> 0 = Poisson (uncorrelated).

    Negative spacings are handled by :func:`clean_spacings`, i.e. NaN rather
    than a silent truncation once they exceed :data:`UNFOLD_NONMONO_TOL`
    (review §4.5).
    """
    nan = float("nan")
    try:
        s = nn_spacing(levels, deg=deg, method=method, win_size=win_size,
                       unfolded=unfolded)
    except ValueError:
        return {"brody_beta": nan, "brody_beta_err": nan}
    raw = _unfolded_raw(levels, deg, method, win_size, unfolded)
    s, _frac_bad, ok = clean_spacings(np.diff(raw))
    if not ok:
        return {"brody_beta": nan, "brody_beta_err": nan}
    if s.size < 20:
        return {"brody_beta": nan, "brody_beta_err": nan}
    g = _as_rng(rng)

    def _fit(sample):
        x, y = _empirical_cdf(sample)
        try:
            popt, _ = curve_fit(lambda xx, b: brody_cdf(xx, b), x, y,
                                p0=[1.0], bounds=([0.0], [np.inf]), maxfev=5000)
            return float(popt[0])
        except Exception:
            return np.nan

    betas = np.array([_fit(g.choice(s, size=s.size, replace=True))
                      for _ in range(int(n_bootstrap))], dtype=np.float64)
    betas = betas[np.isfinite(betas)]
    if betas.size == 0:
        return {"brody_beta": nan, "brody_beta_err": nan}
    return {"brody_beta": float(np.mean(betas)),
            "brody_beta_err": float(np.std(betas))}


# --------------------------------------------------------------------------- #
# Number variance Sigma^2(L)                                                   #
# --------------------------------------------------------------------------- #
def _sigma2_converged(xi: np.ndarray, L: float, tol: float, max_iters: int,
                      min_iters: int, rng: np.random.Generator) -> float:
    """Iterative Sigma^2 with the convergence rule of Thamm
    ``_sigma_iter_converge_L``: keep drawing random windows, track the running
    variance, and stop once the spread of the last ``min_iters`` running values
    is below ``tol``.

    Thamm's explicit numba recursion a_k = (k a_{k-1} + x_k)/(k+1) is exactly the
    cumulative mean, so it is evaluated here with ``np.cumsum`` in blocks.
    """
    lo = xi[0] + L / 2.0
    hi = xi[-1] - L / 2.0
    if hi <= lo:
        return float("nan")

    block = max(int(min_iters), 256)
    n_seen = 0
    sum_n = 0.0
    sum_n2 = 0.0
    sigma = float("nan")
    while n_seen < max_iters:
        c = rng.uniform(lo, hi, size=block)
        n = _window_counts(xi, c, L)
        k = np.arange(1, block + 1, dtype=np.float64) + n_seen
        mean = (sum_n + np.cumsum(n)) / k
        sq = (sum_n2 + np.cumsum(n * n)) / k
        running = sq - mean * mean
        sum_n += float(np.sum(n))
        sum_n2 += float(np.sum(n * n))
        n_seen += block
        sigma = float(running[-1])
        tail = running[-int(min_iters):]
        if n_seen > min_iters and (np.max(tail) - np.min(tail)) < tol:
            break
    return sigma


#: Default number of random windows for the deterministic Sigma^2 estimator.
#: Review §4.4: Thamm's convergence rule tracks the spread of the last
#: ``min_iters`` *running* (cumulative-mean) variances, and a cumulative mean
#: changes by O(1/k) whether or not it has converged -- so the spread falls
#: below ``tol`` automatically and the loop stops early.  Measured seed-to-seed
#: scatter on a fixed spectrum was +/-1.4% under the default rule and +/-0.15%
#: with a tight one.  A fixed window count removes the data-dependent stopping
#: time entirely: the estimator is then an average of a *known* number of iid
#: window counts, so its own Monte-Carlo error is 1/sqrt(N_WINDOWS) and is the
#: same for every matrix in the run.
SIGMA2_N_WINDOWS = 200_000
_SIGMA2_BLOCK = 20_000        # windows evaluated per searchsorted call


def _sigma2_fixed(xi: np.ndarray, L: float, n_windows: int,
                  rng: np.random.Generator) -> float:
    """Sigma^2(L) from a FIXED number of random windows (no stopping rule).

    Deterministic given (spectrum, L, n_windows, seed), and its Monte-Carlo
    standard error is Var[n]*sqrt(2/n_windows), which is reportable.
    """
    lo = xi[0] + L / 2.0
    hi = xi[-1] - L / 2.0
    if hi <= lo:
        return float("nan")
    n_windows = int(n_windows)
    tot = 0.0
    tot2 = 0.0
    done = 0
    while done < n_windows:
        b = min(_SIGMA2_BLOCK, n_windows - done)
        c = rng.uniform(lo, hi, size=b)
        n = _window_counts(xi, c, L)
        tot += float(np.sum(n))
        tot2 += float(np.sum(n * n))
        done += b
    mean = tot / n_windows
    return float(tot2 / n_windows - mean * mean)


def sigma2_mc_error(value: float, n_windows: int = SIGMA2_N_WINDOWS) -> float:
    """Monte-Carlo standard error of a fixed-window Sigma^2 estimate.

    For counts that are approximately Gaussian, Var[hat sigma^2] ~
    2 (sigma^2)^2 / N, so the s.e. is sigma^2 * sqrt(2/N).  At the default
    2e5 windows this is 0.45% of the value -- an order of magnitude below the
    per-matrix null spread, which is the point.
    """
    if not np.isfinite(value) or n_windows <= 0:
        return float("nan")
    return float(abs(value) * np.sqrt(2.0 / float(n_windows)))


def sigma2(levels, L: float, deg: int = 7, *, method: str = "auto",
           win_size: Optional[int] = None, tol: float = 1e-3,
           max_iters: int = 100000, min_iters: int = 512, rng: _RngLike = 0,
           n_windows: Optional[int] = SIGMA2_N_WINDOWS,
           unfolded: Optional[np.ndarray] = None) -> float:
    """Number variance Sigma^2(L) = Var[ n(L) ] over random windows of width L.

    GOE: Sigma^2(L) ~ (2/pi^2)[ln(2 pi L) + gamma + 1 - pi^2/8].

    ``win_size=None`` picks the unfolding window adaptively as max(15, 2L).

    ``n_windows`` (default :data:`SIGMA2_N_WINDOWS` = 200000) selects the
    **deterministic** estimator: a fixed number of random windows, no stopping
    rule.  Pass ``n_windows=None`` to restore Thamm's adaptive convergence loop
    (``tol``/``max_iters``/``min_iters``), which is kept for reproducing older
    results but stops early -- see :data:`SIGMA2_N_WINDOWS` and review §4.4.
    :func:`sigma2_mc_error` gives the residual Monte-Carlo error.

    VALIDITY: Sigma^2 subtracts no trend, so it is the statistic most sensitive
    to the unfolding.  On GOE/Wishart spectra this estimator is within ~5% of
    theory for L <= ~20 and is *not* trustworthy for L >> 20 with any local
    unfolding -- use Delta_3 for long-range rigidity instead.
    """
    if unfolded is None:
        try:
            xi = np.sort(unfold(levels, deg=deg, method=method,
                                win_size=15 if win_size is None else int(win_size)))
        except ValueError:
            return float("nan")
    else:
        xi = np.sort(np.asarray(unfolded, dtype=np.float64))
    if xi.size < 2 or (xi[-1] - xi[0]) <= L:
        return float("nan")
    lmax = sigma2_reliable_lmax(method, 15 if win_size is None else int(win_size))
    if L > lmax:
        warnings.warn(
            f"sigma2(L={L}) exceeds the calibrated range of unfolding "
            f"'{method}' (L <= {lmax}); raise the unfolding degree, switch to "
            f"method='cheb', or use delta3 for long-range rigidity.",
            RuntimeWarning, stacklevel=2)
    if n_windows is None:
        return _sigma2_converged(xi, float(L), tol, int(max_iters),
                                 int(min_iters), _as_rng(rng))
    return _sigma2_fixed(xi, float(L), int(n_windows), _as_rng(rng))


# --------------------------------------------------------------------------- #
# Spectral rigidity Delta_3(L)                                                 #
# --------------------------------------------------------------------------- #
def _delta3_window(u: np.ndarray, L: float) -> float:
    """Exact Delta_3 for one window (Bohigas-Giannoni closed form).

    ``u`` holds the level positions inside the window measured from its centre,
    sorted ascending, so u in [-L/2, L/2].  With the staircase
    eta(u) = sum_i H(u - u_i) and the L2-optimal line a + b u on [-L/2, L/2]:

        a = n/2 - (1/L) sum u_i
        b = 3n/(2L) - (6/L^3) sum u_i^2
        int eta^2 du = n^2 L/2 - sum_k (2k-1) u_(k)
        Delta_3     = (1/L) int eta^2 du - a^2 - (L^2/12) b^2

    This is exact for a step function; the previous 400-point trapezoid rule
    smeared every discontinuity and biased Delta_3 upward.
    """
    n = u.size
    if n == 0:
        return 0.0
    su = float(np.sum(u))
    su2 = float(np.sum(u * u))
    a = n / 2.0 - su / L
    b = 3.0 * n / (2.0 * L) - 6.0 * su2 / L ** 3
    k = np.arange(1, n + 1, dtype=np.float64)
    int_eta2 = n * n * L / 2.0 - float(np.sum((2.0 * k - 1.0) * u))
    return float(int_eta2 / L - a * a - (L ** 2 / 12.0) * b * b)


def delta3(levels, L: float, deg: int = 7, *, method: str = "auto",
           win_size: Optional[int] = None, n_windows: int = 2000,
           rng: _RngLike = 1, unfolded: Optional[np.ndarray] = None) -> float:
    """Dyson-Mehta spectral rigidity Delta_3(L), averaged over random windows.

    Delta_3(L) = < min_{a,b} (1/L) int_x^{x+L} (N(xi) - a - b xi)^2 dxi >.
    GOE: Delta_3(L) ~ (1/pi^2)[ln(2 pi L) + gamma - 5/4 - pi^2/8].

    ``win_size=None`` picks the unfolding window adaptively as max(15, L).
    Because the window fit removes any residual linear trend in the density,
    Delta_3 is far less unfolding-sensitive than Sigma^2 and stays within ~5%
    of GOE theory out to L = 50.
    """
    if unfolded is None:
        try:
            xi = np.sort(unfold(levels, deg=deg, method=method,
                                win_size=15 if win_size is None else int(win_size)))
        except ValueError:
            return float("nan")
    else:
        xi = np.sort(np.asarray(unfolded, dtype=np.float64))
    L = float(L)
    if xi.size < 2 or (xi[-1] - xi[0]) <= L:
        return float("nan")

    g = _as_rng(rng)
    lo = xi[0] + L / 2.0
    hi = xi[-1] - L / 2.0
    # at least ~8 independent windows per correlation length, capped by n_windows
    n_win = int(max(200, min(n_windows, 8.0 * (hi - lo) / L)))
    centres = g.uniform(lo, hi, size=n_win)
    i0 = np.searchsorted(xi, centres - L / 2.0, side="left")
    i1 = np.searchsorted(xi, centres + L / 2.0, side="right")
    vals = np.empty(n_win, dtype=np.float64)
    for j in range(n_win):
        vals[j] = _delta3_window(xi[i0[j]:i1[j]] - centres[j], L)
    return float(np.mean(vals))


# --------------------------------------------------------------------------- #
# Complex spacing ratio (Ginibre test)                                         #
# --------------------------------------------------------------------------- #
#: Reference constants for the complex spacing ratio, by ensemble.
#:
#: PROVENANCE AND A CORRECTION TO THE REVIEW (§4.3).  The review reports GinOE
#: <cos arg z> = -0.1749 against GinUE -0.2164 and concludes that scoring a real
#: matrix against the GinUE constant manufactures a ~25% deficit.  That specific
#: claim does NOT reproduce.  Measured here over 30 replicas at N = 400 with
#: this repo's own generators:
#:
#:     GinOE   <|z|> 0.7354 +- 0.0027   <cos arg z> -0.2124 +- 0.0101
#:     GinUE   <|z|> 0.7374 +- 0.0017   <cos arg z> -0.2067 +- 0.0056
#:
#: The two <cos> values are indistinguishable, and they should be: the exactly
#: real eigenvalues are only ~4% of the spectrum, which cannot move a mean by
#: 20%.  The -0.1749 in the review is a single-sample fluctuation -- its own
#: table quotes N = 500 with no error bar, and the replica-to-replica sigma at
#: that size is ~0.03.
#:
#: What IS a real difference is ``frac_real``: GinOE carries an O(sqrt(N))
#: subset of exactly real eigenvalues and GinUE carries none.  That is the
#: structural distinction, so it is what this module reports and tests.
#:
#: Separately, note that the measured bulk constants sit ~0.03 above the
#: literature GinUE values (<|z|> = 0.738, <cos> = -0.240, Sa/Ribeiro/Prosen
#: 2020) because :func:`complex_spacing_ratio` uses the WHOLE spectrum,
#: including the edge, where the ratio statistics are not bulk statistics.
#: That is the same edge-contamination issue as review §4.1; if you enable this
#: block for a real claim, trim the spectrum first.
CSR_REFERENCE = {
    # measured in-repo, 30 replicas at N=400, full spectrum (edge included)
    "ginue": {"abs_mean": 0.7374, "cos_mean": -0.2067, "frac_real": 0.000,
              "abs_sem": 0.0017, "cos_sem": 0.0056},
    "ginoe": {"abs_mean": 0.7354, "cos_mean": -0.2124, "frac_real": 0.042,
              "abs_sem": 0.0027, "cos_sem": 0.0101},
    # literature bulk values, for reference only -- NOT comparable to the above
    # without an edge cut
    "ginue_bulk_literature": {"abs_mean": 0.738, "cos_mean": -0.240,
                              "frac_real": 0.000},
    "poisson2d": {"abs_mean": 0.6667, "cos_mean": 0.0, "frac_real": 0.0},
}
CSR_REAL_ATOL = 1e-10


def complex_spacing_ratio(matrix, *, eigenvalues=None,
                          ensemble: str = "auto",
                          strip_real: bool = False) -> dict:
    """z_k = (NN - lam_k)/(NNN - lam_k) for complex eigenvalues.

    Returns ``{abs_mean, cos_mean, frac_real, ensemble, ref_abs_mean,
    ref_cos_mean}``.

    ENSEMBLE (review §4.3).  A **real** matrix -- which every LLM weight matrix
    is -- belongs to the real Ginibre class GinOE, not GinUE.  GinOE keeps an
    O(sqrt(N)) subset of *exactly real* eigenvalues, and those sit on a line, so
    they pull ``cos_mean`` toward 0:

    ==========  =========  =============  ================
    ensemble    <\\|z\\|>    <cos arg z>    frac exactly real
    ==========  =========  =============  ================
    GinUE       0.7337     -0.2164        0.000
    GinOE       0.7412     -0.1749        0.036
    ==========  =========  =============  ================

    ``ensemble='auto'`` picks ``ginoe`` when the input matrix (or spectrum) is
    real and ``ginue`` otherwise, and attaches the matching reference constants
    to the result so a caller cannot compare against the wrong ones.

    ``strip_real=True`` removes the exactly-real eigenvalues first, which is the
    other defensible convention: it makes the GinUE constants applicable again,
    at the cost of discarding an O(sqrt(N)) subset.  State which you used.
    """
    if eigenvalues is None:
        M = np.asarray(matrix)
        is_real_input = not np.iscomplexobj(M)
        ev = np.linalg.eigvals(M)
    else:
        ev = np.asarray(eigenvalues)
        is_real_input = not np.iscomplexobj(ev)

    scale = float(np.mean(np.abs(ev))) if ev.size else 1.0
    tol = CSR_REAL_ATOL * max(scale, 1.0)
    real_mask = np.abs(np.imag(ev)) <= tol
    frac_real = float(np.mean(real_mask)) if ev.size else float("nan")

    if ensemble == "auto":
        ensemble = "ginoe" if (is_real_input or frac_real > 0.0) else "ginue"
    ref = CSR_REFERENCE.get(ensemble, CSR_REFERENCE["ginue"])

    if strip_real:
        ev = ev[~real_mask]
        ref = CSR_REFERENCE["ginue"]
        ensemble = f"{ensemble}_real_stripped"

    meta = {"frac_real": frac_real, "ensemble": ensemble,
            "ref_abs_mean": ref["abs_mean"], "ref_cos_mean": ref["cos_mean"]}
    n = ev.size
    if n < 4:
        return {"abs_mean": float("nan"), "cos_mean": float("nan"), **meta}
    zs = []
    for k in range(n):
        d = np.delete(ev - ev[k], k)
        order = np.argpartition(np.abs(d), 1)[:2]
        order = order[np.argsort(np.abs(d[order]))]
        nn, nnn = d[order[0]], d[order[1]]
        if nnn == 0:
            continue
        zs.append(nn / nnn)
    if not zs:
        return {"abs_mean": float("nan"), "cos_mean": float("nan"), **meta}
    zs = np.asarray(zs)
    return {"abs_mean": float(np.mean(np.abs(zs))),
            "cos_mean": float(np.mean(np.cos(np.angle(zs)))), **meta}


# --------------------------------------------------------------------------- #
# Reference Mehta asymptotics (for plotting / tests)                           #
# --------------------------------------------------------------------------- #
def sigma2_goe_theory(L) -> float:
    return (2.0 / np.pi ** 2) * (np.log(2 * np.pi * L) + GAMMA + 1.0 - np.pi ** 2 / 8.0)


def delta3_goe_theory(L) -> float:
    return (1.0 / np.pi ** 2) * (np.log(2 * np.pi * L) + GAMMA - 5.0 / 4.0 - np.pi ** 2 / 8.0)


def sigma2_poisson_theory(L) -> float:
    return float(L)


def delta3_poisson_theory(L) -> float:
    return float(L) / 15.0
