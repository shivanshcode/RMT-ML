"""rmt.overlap — PART 7: activation-covariance overlap (Paper 3 Eq. 7),
eigenvector/eigenvalue coincidence (Fig. 15), the random-basis kσ band, and
fused-QKV feature-matrix key resolution.  Pure numpy/scipy.
"""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from scipy import special, stats


def _svd_of(weight, svd):
    if svd is not None:
        return np.asarray(svd.Vh, dtype=np.float64), np.asarray(svd.s, dtype=np.float64)
    W = np.asarray(weight, dtype=np.float64)
    _, s, Vh = np.linalg.svd(W, full_matrices=False)
    return Vh.astype(np.float64), s.astype(np.float64)


def overlap_analysis(weight, feature_matrix, *, svd=None, eig=None) -> dict:
    """Paper 3 Eq. 7: O_k = maxⱼ |v_k·f_j|.

    v_k = right singular vectors of W (rows of Vh); f_j = eigenvectors of the
    centered activation covariance C (descending λ).  Both live in in-feature space.

    ``eig`` (optional ``(evals, evecs)`` from ``np.linalg.eigh(C)``) lets the caller
    factor C once and share it with :func:`eigenvector_eigenvalue_coincidence`
    (REPORT §2 eigh dedup).
    """
    from .mp import estimate_sigma_gd_median, mp_bounds
    Vh, s = _svd_of(weight, svd)
    C = np.asarray(feature_matrix, dtype=np.float64)
    evals, evecs = np.linalg.eigh(C) if eig is None else eig   # ascending
    order = np.argsort(evals)[::-1]               # descending λ
    evals = evals[order]
    evecs = evecs[:, order]                       # columns = f_j

    # overlap matrix |v_k · f_j| ; Vh rows length = in = dim of f_j
    k = min(Vh.shape[0], evecs.shape[0])
    ov_mat = np.abs(Vh[:k] @ evecs)               # (k_sv, dim_f)
    overlap = np.max(ov_mat, axis=1)              # O_k

    n = Vh.shape[1] if Vh.ndim == 2 else len(Vh)  # in-features
    # n_rows (out-features): prefer the weight, else the SVD's U, else min-dim
    if weight is not None:
        n_rows = weight.shape[0]
    elif svd is not None and getattr(svd, "U", None) is not None:
        n_rows = svd.U.shape[0]
    else:
        n_rows = len(s)
    try:
        sigma = estimate_sigma_gd_median(s=s, n=int(n_rows), m=int(Vh.shape[1]))
        mp_min, mp_max = mp_bounds(int(n_rows), int(Vh.shape[1]), sigma)
    except Exception:
        sigma, mp_min, mp_max = float("nan"), float("nan"), float("nan")
    right_outliers = int(np.sum(s > mp_max)) if np.isfinite(mp_max) else 0
    left_outliers = int(np.sum(s < mp_min)) if np.isfinite(mp_min) else 0

    return {
        "svals": s[:k], "overlap": overlap, "overlap_matrix": ov_mat,
        "mp_min": float(mp_min), "mp_max": float(mp_max),
        "sigma_med": float(sigma),
        "right_outliers": right_outliers, "left_outliers": left_outliers,
        "evals": evals,
    }


def eigenvector_eigenvalue_coincidence(weight, feature_matrix, *, svd=None,
                                       eig=None) -> dict:
    """Fig. 15 cosine-similarity map + scalar summaries (NaN → 0.0).

    ``eig`` (optional ``(evals, evecs)``) shares a single ``eigh(C)`` factorisation
    with :func:`overlap_analysis` (REPORT §2 eigh dedup).
    """
    Vh, s = _svd_of(weight, svd)
    C = np.asarray(feature_matrix, dtype=np.float64)
    evals, evecs = np.linalg.eigh(C) if eig is None else eig
    order = np.argsort(evals)[::-1]
    evals_desc = evals[order]
    evecs = evecs[:, order]

    k = min(Vh.shape[0], evecs.shape[0])
    # Keep every activation direction for rectangular weights.  Truncating to
    # :k columns can force a right singular vector away from its true best match.
    cos_matrix = np.abs(Vh[:k] @ evecs)
    svals_desc = s[:k]

    argmax_per_sv = np.argmax(cos_matrix, axis=1)   # best eigvec for each sv
    diag_hits = np.array([argmax_per_sv[i] == i for i in range(k)], dtype=float)
    diagonal_coincidence = float(np.mean(diag_hits))

    # which singular vector best matches the *top* eigenvector (column 0)
    argmax_singular_for_top_eigenvector = int(np.argmax(cos_matrix[:, 0]))
    max_overlap_with_top_eigenvector = float(np.max(cos_matrix[:, 0]))

    def _safe_spear(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        rho, _ = stats.spearmanr(a, b)
        return 0.0 if not np.isfinite(rho) else float(rho)

    rho_top_eigenvector_vs_svals = _safe_spear(cos_matrix[:, 0], svals_desc)
    rho_top_singular_vs_evals = _safe_spear(cos_matrix[0, :], evals_desc)
    rho_diag_vs_svals = _safe_spear(np.diag(cos_matrix[:, :k]), svals_desc)

    return {
        "cos_matrix": cos_matrix, "svals_desc": svals_desc, "evals_desc": evals_desc,
        "rho_top_eigenvector_vs_svals": rho_top_eigenvector_vs_svals,
        "rho_top_singular_vs_evals": rho_top_singular_vs_evals,
        "rho_diag_vs_svals": rho_diag_vs_svals,
        "max_overlap_with_top_eigenvector": max_overlap_with_top_eigenvector,
        "argmax_singular_for_top_eigenvector": argmax_singular_for_top_eigenvector,
        "diagonal_coincidence": diagonal_coincidence,
    }


def three_sigma_band(N, sigma_level=3.0) -> Tuple[float, float]:
    """One-sided null band [0, upper] for the max row-overlap of a random basis.

    per_entry_threshold = sigma_level/√N (diagnostic);
    upper = (1/√N)·√2·erfcinv(p₁), p₁ = 1−(1−T)^{1/N}, T = erfc(sigma_level/√2).
    """
    N = int(N)
    per_entry = sigma_level / np.sqrt(N)
    T = special.erfc(sigma_level / np.sqrt(2.0))
    p1 = 1.0 - (1.0 - T) ** (1.0 / N)
    upper = (1.0 / np.sqrt(N)) * np.sqrt(2.0) * special.erfcinv(p1)
    return float(per_entry), float(upper)


def resolve_fm_key(record_name: str, fm_keys) -> Optional[str]:
    """Map a MatrixRecord name to a feature-matrix key.

    Strips a trailing ``.weight``; maps fused ``...query_key_value.weight[Q]``
    to the base ``...query_key_value`` key.  None if unresolved.
    """
    keys = list(fm_keys)
    name = record_name
    # fused-QKV tag like  ...query_key_value.weight[Q]
    if "[" in name and name.endswith("]"):
        base = name[: name.index("[")]
        if base.endswith(".weight"):
            base = base[: -len(".weight")]
        for k in keys:
            kk = k[: -len(".weight")] if k.endswith(".weight") else k
            if kk == base or k == base:
                return k
        return None
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    for k in keys:
        kk = k[: -len(".weight")] if k.endswith(".weight") else k
        if kk == name or k == name:
            return k
    return None
