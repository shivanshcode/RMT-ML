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
        if np.iscomplexobj(np.asarray(s)):
            raise TypeError("complex singular values are not supported")
        return np.asarray(s, dtype=np.float64)
    if weight is None:
        raise ValueError("provide weight or s")
    raw = np.asarray(weight)
    if np.iscomplexobj(raw):
        raise TypeError("complex weights are not supported by real scalar metrics")
    W = np.asarray(raw, dtype=np.float64)
    return np.linalg.svd(W, compute_uv=False).astype(np.float64)


def stable_rank(weight=None, *, s=None) -> float:
    """‖W‖_F² / ‖W‖_2²  ∈ [1, rank]; scale-invariant."""
    sv = _svals(weight, s=s)
    smax = float(np.max(sv))
    if smax <= 0:
        return 0.0
    return float(np.sum(np.square(sv / smax)))


def spectral_entropy(svals, base=np.e) -> float:
    """Entropy of pᵢ = sᵢ²/Σsᵢ²; max ln k (uniform), min 0 (dominant)."""
    sv = np.asarray(svals, dtype=np.float64)
    scale = float(np.max(np.abs(sv))) if sv.size else 0.0
    if scale <= 0:
        return 0.0
    e = np.square(sv / scale)
    tot = float(np.sum(e))
    if tot <= 0:
        return 0.0
    p = e / tot
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(H / np.log(base))


def row_wise_entropy(weight, base=np.e) -> float:
    """Mean per-row entropy of the squared-entry distribution of W."""
    raw = np.asarray(weight)
    if np.iscomplexobj(raw):
        raise TypeError("complex weights are not supported by real scalar metrics")
    W = np.asarray(raw, dtype=np.float64)
    scales = np.max(np.abs(W), axis=1, keepdims=True)
    normalized = np.divide(W, scales, out=np.zeros_like(W), where=scales > 0)
    sq = np.square(normalized)
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
    """IPR summaries with ``ipr_bulk_mean`` inside both MP support edges."""
    from .mp import mp_bounds
    Vh = np.asarray(Vh, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    # right singular vectors are the rows of Vh -> compute IPR along columns of Vh.T
    iprs = ipr(Vh.T, axis=0)            # one per singular vector
    nu_minus, nu_plus = mp_bounds(n, m, sigma)
    k = min(len(s), iprs.size)
    s = s[:k]; iprs = iprs[:k]
    top_n = max(1, k // 10)
    ipr_top10_mean = float(np.mean(iprs[:top_n]))   # s descending -> first = largest
    bulk_mask = (s >= nu_minus) & (s <= nu_plus)
    ipr_bulk_mean = float(np.mean(iprs[bulk_mask])) if bulk_mask.any() else float("nan")
    return {"ipr_top10_mean": ipr_top10_mean, "ipr_bulk_mean": ipr_bulk_mean}


def porter_thomas_ks(Vh, *, n_vectors=None) -> dict:
    """KS of singular-vector entries vs the Gaussian Porter–Thomas prediction.

    Returns {pt_ks_mean, pt_frac_random}. For a Haar-random basis the entries
    look Gaussian (high frac_random); for a localized basis they do not.
    """
    from scipy import stats
    V = np.asarray(Vh, dtype=np.float64)
    nvec = V.shape[0] if n_vectors is None else min(n_vectors, V.shape[0])
    dim = V.shape[1]
    ks_vals = []
    n_random = 0
    for i in range(nvec):
        v = V[i]
        nv = np.linalg.norm(v)
        if nv == 0:
            continue
        z = v / nv * np.sqrt(dim)                 # standardise to ~N(0,1) if random
        D, _ = stats.kstest(z, "norm")
        ks_vals.append(D)
        if D < 0.1:
            n_random += 1
    if not ks_vals:
        return {"pt_ks_mean": float("nan"), "pt_frac_random": 0.0}
    return {"pt_ks_mean": float(np.mean(ks_vals)),
            "pt_frac_random": float(n_random / len(ks_vals))}


def mp_softrank(s, nu_plus) -> float:
    """Paper 2 Eq. 11 soft-rank R_mp = ν₊ / max(s)  (a ratio, not energy)."""
    sv = np.asarray(s, dtype=np.float64)
    smax = float(np.max(sv))
    return float(nu_plus / smax) if smax > 0 else float("nan")


def bulk_mass_frac(s, nu_plus, *, nu_minus=0.0) -> float:
    """Energy fraction inside the two-sided support ``[nu_minus, nu_plus]``."""
    sv = np.asarray(s, dtype=np.float64)
    scale = float(np.max(np.abs(sv))) if sv.size else 0.0
    if scale <= 0:
        return float("nan")
    energy = np.square(sv / scale)
    tot = float(np.sum(energy))
    mask = (sv >= float(nu_minus)) & (sv <= float(nu_plus))
    return float(np.sum(energy[mask]) / tot)


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
    n_deciles = int(n_deciles)
    if n_deciles < 1:
        raise ValueError("n_deciles must be positive")
    if n_deciles > k:
        raise ValueError(
            f"n_deciles={n_deciles} exceeds singular count {k}; empty groups are not measurements"
        )
    ranges = decile_index_ranges(k, n_deciles, ascending=True)
    out = {}
    scale = float(np.max(np.abs(sv))) if sv.size else 0.0
    energy = np.square(sv / scale) if scale > 0.0 else np.zeros_like(sv)
    total = float(np.sum(energy))
    global_max = float(np.max(energy)) if energy.size else 0.0
    probabilities = energy / total if total > 0.0 else np.zeros_like(energy)
    for d, (lo, hi) in enumerate(ranges, start=1):
        p = probabilities[lo:hi]
        positive = p[p > 0.0]
        # Additive contributions: sums recover global entropy and stable rank.
        out[f"entropy_decile_{d}"] = float(-np.sum(positive * np.log(positive)))
        out[f"srk_decile_{d}"] = (
            float(np.sum(energy[lo:hi]) / global_max) if global_max > 0.0 else 0.0
        )
    return out
