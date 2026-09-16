"""rmt.scalars — PART 3: scalar spectral summaries.

stable rank, spectral/row-wise entropy, IPR (Paper 1 Eq. 12), Porter–Thomas
(Paper 1 §V, optional), Paper-2 MP soft-rank, bulk-mass fraction, and the
ascending per-decile partition shared with rmt.decile.  Pure numpy/scipy.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np


def _svals(weight=None, *, s=None) -> np.ndarray:
    if s is not None:
        return np.asarray(s, dtype=np.float64)
    if weight is None:
        raise ValueError("provide weight or s")
    W = np.asarray(weight, dtype=np.float64)
    return np.linalg.svd(W, compute_uv=False).astype(np.float64)


def stable_rank(weight=None, *, s=None) -> float:
    """‖W‖_F² / ‖W‖_2²  ∈ [1, rank]; scale-invariant."""
    sv = _svals(weight, s=s)
    smax = float(np.max(sv))
    if smax <= 0:
        return 0.0
    return float(np.sum(sv**2) / smax**2)


def spectral_entropy(svals, base=np.e) -> float:
    """Entropy of pᵢ = sᵢ²/Σsᵢ²; max ln k (uniform), min 0 (dominant)."""
    sv = np.asarray(svals, dtype=np.float64)
    e = sv**2
    tot = float(np.sum(e))
    if tot <= 0:
        return 0.0
    p = e / tot
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(H / np.log(base))


def row_wise_entropy(weight, base=np.e) -> float:
    """Mean per-row entropy of the squared-entry distribution of W."""
    W = np.asarray(weight, dtype=np.float64)
    sq = W**2
    rs = sq.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    p = sq / rs
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log(p), 0.0)
    H = terms.sum(axis=1) / np.log(base)
    return float(np.mean(H))


def ipr(vectors, axis=0) -> np.ndarray:
    """Inverse participation ratio Σ v⁴ per column (axis=0) or row (Paper 1 Eq. 12).

    Vectors are L2-normalised along ``axis`` first.
    """
    V = np.asarray(vectors, dtype=np.float64)
    if V.ndim == 1:
        v = V / (np.linalg.norm(V) + 1e-300)
        return np.array(np.sum(v**4))
    norm = np.linalg.norm(V, axis=axis, keepdims=True)
    norm[norm == 0] = 1.0
    Vn = V / norm
    return np.sum(Vn**4, axis=axis)


def ipr_summary(Vh, s, n, m, sigma) -> dict:
    """{ipr_top10_mean, ipr_bulk_mean}; split by MP upper edge ν₊."""
    from .mp import mp_bounds
    Vh = np.asarray(Vh, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    # right singular vectors are the rows of Vh -> compute IPR along columns of Vh.T
    iprs = ipr(Vh.T, axis=0)            # one per singular vector
    _, nu_plus = mp_bounds(n, m, sigma)
    k = min(len(s), iprs.size)
    s = s[:k]; iprs = iprs[:k]
    top_n = max(1, k // 10)
    ipr_top10_mean = float(np.mean(iprs[:top_n]))   # s descending -> first = largest
    bulk_mask = s <= nu_plus
    ipr_bulk_mean = float(np.mean(iprs[bulk_mask])) if bulk_mask.any() else float("nan")
    return {"ipr_top10_mean": ipr_top10_mean, "ipr_bulk_mean": ipr_bulk_mean}


def porter_thomas_ks(Vh, *, n_vectors=None, n_samples=5000, alpha=0.05,
                     rng=0, pvals_out=None) -> dict:
    """Calibrated Porter-Thomas test of the singular-vector entries.

    Thin wrapper over :func:`rmt.porter_thomas.porter_thomas_summary`, which
    ports Thamm ``rmt_utils.ks_test_statistic_normedPT`` /
    ``ks_test_normedPT``.  The null CDF of the entries *and* the null
    distribution of the KS distance are Monte-Carlo sampled at the same
    dimension N as ``Vh``, because the entries of an L2-normalised vector are
    neither N(0,1/N) nor independent.

    Returns {pt_p_mean, pt_frac_random, pt_ks_mean, pt_p_top10_mean,
    pt_p_bulk_mean}; ``pt_frac_random`` is the fraction of singular vectors for
    which the Porter-Thomas null survives at level ``alpha`` (delocalised /
    RMT-like), computed from real p-values rather than a fixed KS threshold.
    """
    from .porter_thomas import porter_thomas_summary
    return porter_thomas_summary(Vh, n_vectors=n_vectors, n_samples=n_samples,
                                 alpha=alpha, rng=rng, pvals_out=pvals_out)


def mp_softrank(s, nu_plus) -> float:
    """Paper 2 Eq. 11 soft-rank R_mp = ν₊ / max(s)  (a ratio, not energy)."""
    sv = np.asarray(s, dtype=np.float64)
    smax = float(np.max(sv))
    return float(nu_plus / smax) if smax > 0 else float("nan")


def bulk_mass_frac(s, nu_plus) -> float:
    """Σs²[s≤ν₊] / Σs²  ∈ [0,1]  (energy fraction inside the bulk)."""
    sv = np.asarray(s, dtype=np.float64)
    tot = float(np.sum(sv**2))
    if tot <= 0:
        return float("nan")
    return float(np.sum(sv[sv <= nu_plus] ** 2) / tot)


def decile_index_ranges(k, n_deciles=10, ascending=True) -> List[Tuple[int, int]]:
    """Half-open, disjoint ranges that cover [0, k), ascending.

    With ``ascending=True`` the ranges index the spectrum from smallest to
    largest, so decile 1 (range index 0) is the smallest 10%.
    """
    k = int(k)
    edges = np.linspace(0, k, n_deciles + 1).astype(int)
    ranges = [(int(edges[i]), int(edges[i + 1])) for i in range(n_deciles)]
    if not ascending:
        ranges = ranges[::-1]
    return ranges


def per_decile(s, n_deciles=10) -> dict:
    """{entropy_decile_1..10, srk_decile_1..10}; decile 1 = smallest 10%.

    Singular values are sorted ascending then partitioned; each decile's
    spectral entropy and stable-rank contribution are reported.
    """
    sv = np.sort(np.asarray(s, dtype=np.float64))      # ascending
    k = sv.size
    ranges = decile_index_ranges(k, n_deciles, ascending=True)
    out = {}
    for d, (lo, hi) in enumerate(ranges, start=1):
        seg = sv[lo:hi]
        if seg.size == 0:
            out[f"entropy_decile_{d}"] = 0.0
            out[f"srk_decile_{d}"] = 0.0
            continue
        out[f"entropy_decile_{d}"] = spectral_entropy(seg)
        smax = float(np.max(seg))
        out[f"srk_decile_{d}"] = float(np.sum(seg**2) / smax**2) if smax > 0 else 0.0
    return out
