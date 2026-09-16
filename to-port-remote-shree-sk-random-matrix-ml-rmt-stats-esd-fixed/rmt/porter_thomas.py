"""rmt.porter_thomas — calibrated Porter-Thomas test for singular vectors.

Direct port of the "KS test statistics for Porter-Thomas test" section of
Thamm, Staats & Rosenow (2022), ``src/rmt_utils.py``:

  ``CDF``                       <- ``rmt_utils.CDF``
  ``normed_pt_cdf``             <- ``rmt_utils.ks_Cbar``
  ``_ks_distances``             <- ``rmt_utils.ks_D``
  ``pt_test_statistic``         <- ``rmt_utils.ks_test_statistic_normedPT``
  ``pt_pvalue``                 <- ``rmt_utils.ks_test_normedPT``
  ``pt_test_statistic_pooled``  <- ``rmt_utils.ks_test_statistic_normedPT_pooled``

Why the null has to be built by Monte-Carlo
-------------------------------------------
A singular vector is L2-normalised, so its N entries are neither N(0, 1/N) nor
independent -- they live on the unit sphere S^(N-1) and obey sum_i v_i^2 = 1.
Testing ``v * sqrt(N)`` against a standard normal with the textbook KS p-value
is therefore mis-specified, and comparing the raw KS distance D against a fixed
threshold is worse still: the null distribution of D shrinks like 1/sqrt(N), so
one threshold cannot serve matrices of different width.

Both the reference CDF ``Cbar`` (entries of Haar vectors on S^(N-1)) and the null
distribution ``C`` of the KS distance D are sampled at the *same* N as the data,
giving p = 1 - C(D), uniform on [0,1] under the Porter-Thomas null.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple, Union
import numpy as np

_RngLike = Union[int, np.random.Generator, None]


def _as_rng(rng: _RngLike) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


class CDF:
    """Discrete CDF C(C_x) = C_y, linearly interpolated when called."""

    def __init__(self, C_x, C_y):
        self.x = np.asarray(C_x, dtype=np.float64)
        self.y = np.asarray(C_y, dtype=np.float64)

    def __call__(self, x):
        return np.interp(x, self.x, self.y)


def _haar_vectors(N: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """(N, n_samples) columns uniform on the unit sphere S^(N-1)."""
    xi = rng.standard_normal((N, n_samples))
    xi /= np.sqrt(np.sum(xi ** 2, axis=0))
    return xi


def normed_pt_cdf(N: int, n_samples: int, rng: _RngLike = 0) -> CDF:
    """Reference CDF Cbar of the *entries* of L2-normalised Porter-Thomas vectors."""
    xi = _haar_vectors(N, n_samples, _as_rng(rng)).reshape(-1)
    x = np.sort(xi)
    return CDF(x, np.arange(x.size) / (x.size - 1))


def _ks_distances(cdf: CDF, N: int, n_samples: int,
                  rng: np.random.Generator) -> np.ndarray:
    """KS distances of fresh Porter-Thomas vectors against ``cdf`` (the null)."""
    xi = _haar_vectors(N, n_samples, rng)
    xi = np.sort(xi, axis=0)
    emp = (np.arange(N) / (N - 1)).reshape(-1, 1)
    return np.max(np.abs(cdf(xi) - emp), axis=0)


@lru_cache(maxsize=16)
def _pt_test_statistic_cached(N: int, n_samples: int, seed: int):
    g = np.random.default_rng(seed)
    Cbar = normed_pt_cdf(N, n_samples, g)
    D = np.sort(_ks_distances(Cbar, N, n_samples, g))
    return Cbar, CDF(D, np.arange(D.size) / (D.size - 1)), D


def pt_test_statistic(N: int, n_samples: int = 5000,
                      rng: _RngLike = 0) -> Tuple[CDF, CDF, np.ndarray]:
    """Return (Cbar, C, D) for dimension ``N``.

    ``Cbar`` is the entry CDF of normalised PT vectors; ``C`` is the CDF of the
    KS distance under the null, so a sample with distance D has p = 1 - C(D).

    The null depends only on (N, n_samples, seed), never on the data, so it is
    memoised: a transformer has O(100) matrices sharing a handful of widths, and
    rebuilding the Monte-Carlo null for each of them costs ~12 s at N=4096 for
    no information gain.  Pass a ``Generator`` to bypass the cache.
    """
    if isinstance(rng, np.random.Generator):
        Cbar = normed_pt_cdf(N, n_samples, rng)
        D = np.sort(_ks_distances(Cbar, N, n_samples, rng))
        return Cbar, CDF(D, np.arange(D.size) / (D.size - 1)), D
    return _pt_test_statistic_cached(int(N), int(n_samples), int(rng or 0))


def pt_test_statistic_pooled(N: int, n_samples: int = 2000, pooling_window: int = 5,
                             rng: _RngLike = 0) -> Tuple[CDF, CDF, np.ndarray]:
    """Same statistic for a *pool* of ``2 * pooling_window`` vectors tested jointly
    (Thamm ``ks_test_statistic_normedPT_pooled``); use when single vectors are too
    short to give the KS test power."""
    g = _as_rng(rng)
    k = 2 * int(pooling_window)
    pool = _haar_vectors(N, n_samples * k, g).reshape(-1)
    x = np.sort(pool)
    Cbar = CDF(x, np.arange(x.size) / (x.size - 1))

    M = N * k
    emp = (np.arange(M) / (M - 1)).reshape(-1, 1)
    xi = _haar_vectors(N, n_samples * k, g).reshape(N * k, n_samples, order="F")
    xi = np.sort(xi, axis=0)
    D = np.sort(np.max(np.abs(Cbar(xi) - emp), axis=0))
    return Cbar, CDF(D, np.arange(D.size) / (D.size - 1)), D


def pt_pvalue(v, C: CDF, Cbar: CDF) -> float:
    """p-value that ``v`` is a Porter-Thomas (Haar) vector."""
    x = np.asarray(v, dtype=np.float64).reshape(-1)
    nrm = np.linalg.norm(x)
    if nrm == 0 or not np.isfinite(nrm):
        return float("nan")
    x = np.sort(x / nrm)
    emp = np.arange(x.size) / (x.size - 1)
    D = float(np.max(np.abs(Cbar(x) - emp)))
    return float(1.0 - C(D))


def porter_thomas_pvalues(vectors, *, axis: int = 0, n_samples: int = 5000,
                          max_vectors: Optional[int] = None,
                          rng: _RngLike = 0) -> np.ndarray:
    """Per-vector Porter-Thomas p-values.

    ``vectors`` is a 2-D array; ``axis=0`` treats columns as vectors (e.g. U),
    ``axis=1`` treats rows as vectors (e.g. the rows of Vh).
    """
    V = np.asarray(vectors, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError("vectors must be 2-D")
    M = V if axis == 0 else V.T          # columns are now the vectors
    N, n_vec = M.shape
    if N < 8:
        return np.array([], dtype=np.float64)
    if max_vectors is not None:
        n_vec = min(int(max_vectors), n_vec)
    Cbar, C, _ = pt_test_statistic(N, n_samples=n_samples, rng=rng)
    return np.array([pt_pvalue(M[:, i], C, Cbar) for i in range(n_vec)],
                    dtype=np.float64)


def porter_thomas_summary(Vh, *, n_vectors: Optional[int] = None,
                          n_samples: int = 5000, alpha: float = 0.05,
                          rng: _RngLike = 0, pvals_out: Optional[list] = None) -> dict:
    """Summary row for the CSV.

    Returns {pt_p_mean, pt_frac_random, pt_ks_mean, pt_p_top10_mean,
             pt_p_bulk_mean}, where ``pt_frac_random`` is the fraction of
    singular vectors whose Porter-Thomas null is *not* rejected at ``alpha``.
    Rows of ``Vh`` are the right singular vectors, ordered by descending
    singular value.
    """
    nan = float("nan")
    V = np.asarray(Vh, dtype=np.float64)
    if V.ndim != 2 or V.shape[1] < 8:
        if pvals_out is not None:
            pvals_out.append(np.array([], dtype=np.float64))
        return {"pt_p_mean": nan, "pt_frac_random": nan, "pt_ks_mean": nan,
                "pt_p_top10_mean": nan, "pt_p_bulk_mean": nan}
    N = V.shape[1]
    n_vec = V.shape[0] if n_vectors is None else min(int(n_vectors), V.shape[0])
    Cbar, C, _ = pt_test_statistic(N, n_samples=n_samples, rng=rng)

    emp = np.arange(N) / (N - 1)
    ps = np.empty(n_vec, dtype=np.float64)
    Ds = np.empty(n_vec, dtype=np.float64)
    for i in range(n_vec):
        v = V[i]
        nrm = np.linalg.norm(v)
        if nrm == 0:
            ps[i] = Ds[i] = np.nan
            continue
        x = np.sort(v / nrm)
        Ds[i] = np.max(np.abs(Cbar(x) - emp))
        ps[i] = 1.0 - C(Ds[i])

    if pvals_out is not None:
        pvals_out.append(ps)
    good = np.isfinite(ps)
    if not good.any():
        return {"pt_p_mean": nan, "pt_frac_random": nan, "pt_ks_mean": nan,
                "pt_p_top10_mean": nan, "pt_p_bulk_mean": nan}
    top = max(1, n_vec // 10)
    return {"pt_p_mean": float(np.nanmean(ps)),
            "pt_frac_random": float(np.mean(ps[good] > alpha)),
            "pt_ks_mean": float(np.nanmean(Ds)),
            "pt_p_top10_mean": float(np.nanmean(ps[:top])),
            "pt_p_bulk_mean": float(np.nanmean(ps[top:])) if n_vec > top else nan}


def porter_thomas_by_decile(pvalues, alpha: float = 0.05, n_deciles: int = 10) -> dict:
    """Split per-vector p-values into deciles of *descending* singular value.

    Decile 1 is the top 10% of the spectrum. Returns
    {decile, p_mean, frac_random, n} as parallel arrays, which is the
    localisation-vs-rank picture: in a trained network the Porter-Thomas null
    survives in the bulk and fails in the leading deciles.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return {"decile": [], "p_mean": [], "frac_random": [], "n": []}
    edges = np.linspace(0, p.size, n_deciles + 1).astype(int)
    dec, pm, fr, nn = [], [], [], []
    for d in range(n_deciles):
        seg = p[edges[d]:edges[d + 1]]
        if seg.size == 0:
            continue
        dec.append(d + 1)
        pm.append(float(np.mean(seg)))
        fr.append(float(np.mean(seg > alpha)))
        nn.append(int(seg.size))
    return {"decile": dec, "p_mean": pm, "frac_random": fr, "n": nn}
