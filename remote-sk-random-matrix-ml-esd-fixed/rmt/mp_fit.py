"""rmt.mp_fit — the *paper's* modified Marchenko-Pastur estimator.

Additive module. It does not modify ``rmt.mp``: ``mp.mp_pdf`` stays the strict,
parameter-free Eq. (4) law (correct for a randomly *initialised* matrix), and
this module adds the three-parameter *modified* MP that Thamm/Staats/Rosenow
actually use for *trained* matrices (their ``rmt_utils.marcenkoPastur`` /
``fit_marcenkoPastur``).

Differences from the strict law, all deliberate:
  * nu_min is fixed to the smallest empirical singular value, not to the
    theoretical edge sigma*(sqrt(max)-sqrt(min)) -- which is exactly 0 for a
    square matrix and is what makes the overlay start at nu=0.
  * the amplitude ``a`` is a free fit parameter, not a unit-area normalisation
    times an ad-hoc rescale. ``a`` absorbs the fraction of the spectrum that
    still follows MP.
  * the density it is fitted against is a Gaussian-broadened spectrum, not a
    histogram.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.optimize import curve_fit


# --------------------------------------------------------------------------- #
# density estimate: adaptive Gaussian broadening (paper Eq. 7, window a)       #
# --------------------------------------------------------------------------- #
def gaussian_broaden(x, svals, win: int = 15) -> np.ndarray:
    """P(nu) ~= (1/m) sum_k N(nu; nu_k, sigma_k), sigma_k = (nu_{k+a}-nu_{k-a})/2.

    Edge values are replicated, matching the paper's ``method='replicate'``.
    """
    x = np.asarray(x, dtype=np.float64)
    s = np.sort(np.asarray(svals, dtype=np.float64))
    m = s.size
    pad = np.pad(s, (win, win), mode="edge")
    widths = (pad[2 * win:] - pad[:-2 * win]) / 2.0
    scale = float(np.max(np.abs(s))) if s.size else 0.0
    floor = max(np.finfo(float).eps * max(scale, np.finfo(float).tiny),
                np.finfo(float).tiny)
    widths = np.maximum(widths, floor)
    z = (x[:, None] - s[None, :]) / widths[None, :]
    dens = np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * widths[None, :])
    return dens.sum(axis=1) / m


# --------------------------------------------------------------------------- #
# the three-parameter modified MP                                             #
# --------------------------------------------------------------------------- #
def modified_mp(x, a: float, nu_max: float, nu_min: float) -> np.ndarray:
    """y = (a/nu) * sqrt((nu_max^2 - nu^2)(nu^2 - nu_min^2)), 0 outside support."""
    x = np.asarray(x, dtype=np.float64)
    scale = float(nu_max)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("nu_max must be finite and positive")
    lower = float(nu_min) / scale
    normalized = x / scale
    out = np.zeros_like(x)
    inside = (normalized > lower) & (normalized < 1.0)
    xi = normalized[inside]
    rad = (1.0 - xi ** 2) * (xi ** 2 - lower ** 2)
    out[inside] = float(a) * scale / xi * np.sqrt(np.clip(rad, 0.0, None))
    return out


def fit_modified_mp(svals, *, win: int = 15, n_grid: int = 4000,
                    range_of_y_to_fit: float = 0.7, i_nu_min: int = 0,
                    x_min: float = 0.0) -> Tuple[float, float, float]:
    """Fit (a, nu_max) with nu_min pinned to the i_nu_min-th smallest sval.

    Returns (a, nu_min, nu_max). Mirrors ``rmt_utils.fit_marcenkoPastur``:
    only the rising flank and the region where the density is still above
    ``range_of_y_to_fit`` * peak is used, so right-tail outliers do not drag
    the fit.
    """
    s = np.sort(np.asarray(svals, dtype=np.float64))
    if (s.ndim != 1 or s.size < 4 or not np.all(np.isfinite(s))
            or np.any(s < 0.0)):
        raise ValueError("modified-MP fitting requires at least four finite nonnegative values")
    if not 0 <= int(i_nu_min) < s.size:
        raise ValueError("i_nu_min is outside the singular-value sample")
    spectral_scale = float(np.max(np.abs(s)))
    if spectral_scale <= 0.0:
        raise ValueError("modified-MP fitting requires a nonzero spectrum")
    normalized = s / spectral_scale
    span = float(normalized[-1] - normalized[0])
    if span <= np.sqrt(np.finfo(float).eps):
        raise ValueError("modified-MP fitting requires a nondegenerate spectrum")
    normalized_x_min = float(x_min) / spectral_scale
    nu_min = float(max(normalized[i_nu_min], normalized_x_min))
    if nu_min >= normalized[-1]:
        raise ValueError("modified-MP lower support edge leaves no positive width")

    x = np.linspace(max(normalized[0], normalized_x_min), normalized[-1], n_grid)
    pdf = gaussian_broaden(x, normalized, win=win)

    keep = x > normalized_x_min
    x, pdf = x[keep], pdf[keep]
    if x.size < 3:
        raise ValueError("modified-MP fit interval is empty")

    x_peak = x[int(np.argmax(pdf))]
    keep = (x <= x_peak) | (pdf > range_of_y_to_fit * pdf.max())
    x_fit, pdf_fit = x[keep], pdf[keep]

    min_width = max(np.finfo(float).eps * max(abs(nu_min), 1.0), np.finfo(float).tiny)
    initial_width = max(float(np.percentile(normalized, 95)) - nu_min, 10.0 * min_width)
    p0 = np.array([0.5, initial_width])
    try:
        (a_normalized, width), _ = curve_fit(
            lambda xx, aa, ww: modified_mp(xx, aa, nu_min + ww, nu_min),
            x_fit, pdf_fit, p0=p0,
            bounds=((0.0, min_width), (np.inf, np.inf)), maxfev=20000,
        )
    except Exception as error:
        raise RuntimeError(f"modified-MP optimizer failed: {error}") from error
    nu_max = nu_min + float(width)
    if not np.isfinite(a_normalized) or not np.isfinite(nu_max) or nu_max <= nu_min:
        raise RuntimeError("modified-MP optimizer returned an invalid support")
    # P_x(x)=P_y(x/scale)/scale and the curve factor contributes one
    # power of scale, hence amplitude transforms as 1/scale^2.
    return (float(a_normalized / spectral_scale**2),
            float(nu_min * spectral_scale), float(nu_max * spectral_scale))


def mp_curve_grid(nu_min: float, nu_max: float, n: int = 400) -> np.ndarray:
    """Open-interval grid: never evaluates the density *at* an edge.

    Evaluating the MP density exactly at nu_min / nu_max returns 0, so a closed
    grid draws a spurious vertical segment down to the axis -- the artifact at
    nu = 0 in the square-matrix ESD plots.
    """
    return np.linspace(nu_min, nu_max, n + 2)[1:-1]
