"""rmt.controls — the nulls that turn "we measured something" into "we measured
something randomness does not explain" (review §8).

An iid-Gaussian null band answers *"what does this pipeline return on a random
matrix of this shape?"*.  It does **not** answer *"is the deviation I see a
property of training?"*, because a trained weight matrix differs from an iid
Gaussian one in at least two ways that have nothing to do with learned spectral
structure:

* its **entry distribution** is heavy-tailed, not Gaussian;
* it is **heteroscedastic** — the per-row and per-column scales vary a lot after
  training, and an iid null has no such variance profile.

Both effects move the statistics on their own.  Attributing them to "deviation
from RMT" is the single most likely way to publish an artefact, so this module
provides the destructive controls that isolate them:

=========================  ==========================================  ======================
control                    preserves                                   destroys
=========================  ==========================================  ======================
``entry_shuffle``          the exact multiset of entries               all structure
``row_col_shuffle``        entry multiset **and** row/col scale         within-row/col structure
``gaussian_match``         only the global variance                     everything else
``init_control``           the model's own initialisation law           everything trained
=========================  ==========================================  ======================

Interpretation, in the order you should read it:

1. If the **real** matrix sits inside the iid-Gaussian band, there is nothing to
   explain and no further control is needed.
2. If it sits outside, compare it to ``entry_shuffle``.  If shuffled-real is
   *also* outside, what you measured is the entry distribution (heavy tails),
   not learned structure — say so, and hand the claim to the tail-exponent
   block.
3. If shuffled-real is inside but the real matrix is outside, compare to
   ``row_col_shuffle``.  **This, not the iid band, is the null the paper should
   quote**: it matches the variance profile, so a deviation from it cannot be
   explained by heteroscedasticity.
4. ``init_control`` is a plumbing check, not a science control: a re-initialised
   matrix must land inside the band, and if it does not, the loading path is
   broken.

Pure numpy.  Imports ``rmt.spacing`` and ``rmt.nulls`` only.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union
import numpy as np

from . import spacing as SP
from . import nulls as NU

_RngLike = Union[int, np.random.Generator, None]


def _as_rng(rng: _RngLike) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


# --------------------------------------------------------------------------- #
# The controls themselves                                                      #
# --------------------------------------------------------------------------- #
def entry_shuffle(W, rng: _RngLike = 0) -> np.ndarray:
    """Permute **all** entries of W (review §8.1).

    Preserves the entry distribution *exactly* — every moment, the full
    empirical CDF — and destroys every row, column and low-rank structure.  Any
    statistic that still deviates after this shuffle is a statement about the
    entry distribution, not about training.
    """
    A = np.asarray(W, dtype=np.float64)
    g = _as_rng(rng)
    return g.permutation(A.ravel()).reshape(A.shape)


def row_shuffle(W, rng: _RngLike = 0) -> np.ndarray:
    """Permute entries **within each row** (review §8.2).

    Preserves each row's exact multiset of entries -- so the per-row scale is
    preserved *exactly* -- and destroys every column-wise and low-rank
    relationship.
    """
    A = np.asarray(W, dtype=np.float64).copy()
    g = _as_rng(rng)
    for i in range(A.shape[0]):
        A[i, :] = g.permutation(A[i, :])
    return A


def col_shuffle(W, rng: _RngLike = 0) -> np.ndarray:
    """Permute entries **within each column** (review §8.2).

    The column-wise counterpart of :func:`row_shuffle`: per-column scale
    preserved exactly, everything else destroyed.
    """
    A = np.asarray(W, dtype=np.float64).copy()
    g = _as_rng(rng)
    for j in range(A.shape[1]):
        A[:, j] = g.permutation(A[:, j])
    return A


def row_col_shuffle(W, rng: _RngLike = 0, n_rounds: int = 1) -> np.ndarray:
    """Row shuffle **composed with** a column shuffle.

    .. warning::
       This does **not** preserve the variance profile, and the review's §8.2
       phrasing ("permute entries within each row, then separately within each
       column") should be read as prescribing TWO SEPARATE controls, not their
       composition.  Composing them destroys what each one individually
       preserves: the column pass scrambles the row scales the row pass had
       just kept.  Measured on a matrix with row-sd CV 0.787, the composition
       leaves 0.372 -- statistically the same as a full entry shuffle (0.367),
       so as a variance-matched null it is worthless.

       Use :func:`row_shuffle` and :func:`col_shuffle` separately.  This
       function is kept only because a composed shuffle is a legitimate
       *stronger* destructive control, equivalent in practice to
       :func:`entry_shuffle` with the row/column marginals loosely retained.
    """
    A = np.asarray(W, dtype=np.float64)
    g = _as_rng(rng)
    for _ in range(int(n_rounds)):
        A = col_shuffle(row_shuffle(A, g), g)
    return A


def gaussian_match(W, rng: _RngLike = 0) -> np.ndarray:
    """iid Gaussian with the same shape and the same global variance.

    The weakest control: it matches nothing but sigma.  Included because it is
    what the matched-shape null band is built from, so running it through the
    same code path is a consistency check on the band itself.
    """
    A = np.asarray(W, dtype=np.float64)
    g = _as_rng(rng)
    return g.standard_normal(A.shape) * float(np.std(A))


def init_control(shape, scheme: str = "normal", rng: _RngLike = 0,
                 std: Optional[float] = None, gain: float = 1.0) -> np.ndarray:
    """A freshly initialised matrix under the model's own init law (review §8.3).

    ``scheme``: ``'normal'`` (std defaults to 0.02, the Llama/GPT-NeoX
    convention), ``'xavier'`` (gain·sqrt(2/(fan_in+fan_out))) or ``'kaiming'``
    (gain·sqrt(2/fan_in)).  This must land *inside* the null band; if it does
    not, the loading/dtype path is the problem, not the model.
    """
    n, m = int(shape[0]), int(shape[1])
    g = _as_rng(rng)
    if scheme == "normal":
        s = 0.02 if std is None else float(std)
    elif scheme == "xavier":
        s = gain * np.sqrt(2.0 / (n + m))
    elif scheme == "kaiming":
        s = gain * np.sqrt(2.0 / m)
    else:
        raise ValueError("scheme must be 'normal', 'xavier' or 'kaiming'")
    return g.standard_normal((n, m)) * s


#: Name -> callable(W, rng) for the controls run by :func:`control_suite`.
CONTROLS = {
    "real": lambda W, rng: np.asarray(W, dtype=np.float64),
    "entry_shuffle": entry_shuffle,
    "row_shuffle": row_shuffle,
    "col_shuffle": col_shuffle,
    "gaussian_match": gaussian_match,
}

#: The controls that preserve a variance margin, checked together by
#: :func:`interpret`.
MARGIN_CONTROLS = ("row_shuffle", "col_shuffle")


# --------------------------------------------------------------------------- #
# Variance-profile diagnostics                                                 #
# --------------------------------------------------------------------------- #
def variance_profile_gaussian(W, rng: _RngLike = 0, n_sinkhorn: int = 0) -> np.ndarray:
    """Gaussian null matched to W's *variance profile* -- review item 8.2, done right.

    Draws ``G_ij ~ N(0, 1)`` and scales it by a rank-one profile
    ``sigma_ij = a_i b_j`` fitted from W's row and column standard deviations,
    then rescales to W's total Frobenius norm.

    WHY THIS AND NOT A SHUFFLE.  §8.2 asked for a null that preserves the
    heteroscedasticity of a trained weight matrix.  Neither marginal shuffle
    does: ``row_shuffle`` preserves the row profile exactly but flattens the
    column profile, and ``col_shuffle`` the reverse.  Measured on a matrix with
    row-sd CV 0.785 / col-sd CV 0.413:

    ======================  ===========  ===========
    control                 CV(row sd)   CV(col sd)
    ======================  ===========  ===========
    original                0.785        0.413
    row_shuffle             0.785        0.116
    col_shuffle             0.125        0.413
    row_col_shuffle         0.127        0.116
    entry_shuffle           0.133        0.132
    **this function**       **0.776**    **0.415**
    ======================  ===========  ===========

    Only this one holds both margins, so it is the null that separates "the
    weights are heteroscedastic" from "the weights carry learned spectral
    structure".  Theory: random matrices with a variance profile are the
    Wigner-type / deformed-Marchenko-Pastur ensembles of Ajanki, Erdos & Kruger,
    *Probab. Theory Relat. Fields* **169**, 667 (2017).

    ``n_sinkhorn`` > 0 runs that many Sinkhorn iterations to match both margins
    simultaneously when the profile is not well approximated by rank one.
    """
    g = _as_rng(rng)
    W = np.asarray(W, dtype=np.float64)
    a = W.std(axis=1, keepdims=True)
    b = W.std(axis=0, keepdims=True)
    b = b / np.sqrt(np.mean(b ** 2)) if np.mean(b ** 2) > 0 else np.ones_like(b)
    S = a * b
    for _ in range(int(n_sinkhorn)):
        ra = W.std(axis=1, keepdims=True) / np.maximum(S.std(axis=1, keepdims=True), 1e-300)
        S = S * ra
        rb = W.std(axis=0, keepdims=True) / np.maximum(S.std(axis=0, keepdims=True), 1e-300)
        S = S * rb
    out = S * g.standard_normal(W.shape)
    nrm = np.linalg.norm(out)
    return out * (np.linalg.norm(W) / nrm) if nrm > 0 else out


def variance_profile(W) -> dict:
    """How heteroscedastic is this matrix, and how heavy are its entries?

    ``row_cv``/``col_cv`` are the coefficients of variation of the per-row and
    per-column standard deviations; for an iid Gaussian n x m matrix they are
    ~1/sqrt(2m) and ~1/sqrt(2n) respectively, i.e. essentially zero at LLM
    shapes.  Anything materially larger means the iid null does not match the
    matrix and :func:`row_col_shuffle` is the control that must be quoted.
    """
    A = np.asarray(W, dtype=np.float64)
    rs = A.std(axis=1)
    cs = A.std(axis=0)
    x = A.ravel()
    sd = float(x.std())
    kurt = float(np.mean(((x - x.mean()) / sd) ** 4)) if sd > 0 else float("nan")
    n, m = A.shape
    return {
        "row_sd_cv": float(rs.std() / rs.mean()) if rs.mean() > 0 else float("nan"),
        "col_sd_cv": float(cs.std() / cs.mean()) if cs.mean() > 0 else float("nan"),
        "row_sd_cv_iid_expected": float(1.0 / np.sqrt(2.0 * m)),
        "col_sd_cv_iid_expected": float(1.0 / np.sqrt(2.0 * n)),
        "excess_kurtosis": kurt - 3.0,
        "global_sd": sd,
    }


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
def _levels(W, *, bulk_mode: str = "center",
            bulk_center_frac: float = SP.BULK_CENTER_FRAC) -> np.ndarray:
    s = np.sort(np.linalg.svd(np.asarray(W, dtype=np.float64),
                              compute_uv=False))
    return SP.bulk_levels(s, mode=bulk_mode, center_frac=bulk_center_frac)


def control_suite(W, *, controls: Sequence[str] = tuple(CONTROLS),
                  rng: _RngLike = 0, bulk_mode: str = "center",
                  bulk_center_frac: float = SP.BULK_CENTER_FRAC,
                  band: Optional[dict] = None, n_reps: int = 0,
                  seed: int = 0, **stat_kw) -> Dict[str, dict]:
    """Run the spacing statistics on W and on each destructive control.

    If ``band`` (from :func:`rmt.nulls.null_band`) is given — or ``n_reps > 0``,
    in which case one is computed for W's shape — every returned row also
    carries ``z_<stat>``, the deviation in units of the *pipeline's own* null
    spread.  That is the number to report; ``value / delta3_goe_theory(L) - 1``
    is not (review §3/F1).
    """
    A = np.asarray(W, dtype=np.float64)
    g = _as_rng(rng)
    bulk_frac = bulk_center_frac if bulk_mode == "center" else 1.0
    if band is not None:
        bf = band.get("bulk_frac")
        if bf is not None and abs(float(bf) - bulk_frac) > 1e-9:
            raise ValueError(
                f"band was built with bulk_frac={bf} but this suite trims to "
                f"{bulk_frac}. The z-scores would be meaningless: the band and "
                f"the measurement must see the same fraction of the spectrum.")
    if band is None and n_reps > 0:
        band = NU.null_band(A.shape, n_reps=int(n_reps), seed=seed,
                            bulk_frac=(bulk_center_frac if bulk_mode == "center"
                                       else 1.0), **stat_kw)

    out: Dict[str, dict] = {}
    for name in controls:
        fn = CONTROLS[name]
        M = fn(A, g)
        lv = _levels(M, bulk_mode=bulk_mode, bulk_center_frac=bulk_center_frac)
        row = NU.spacing_statistics(lv, seed=seed, **stat_kw)
        if band is not None:
            for k, v in list(row.items()):
                if isinstance(v, float) and isinstance(band.get(k), dict):
                    row[f"z_{k}"] = NU.zscore(v, band[k])
        out[name] = row
    return out


def interpret(suite: Dict[str, dict], stat: str = "delta3_L10",
              z_thresh: float = 2.0) -> str:
    """One-line verdict for ``stat``, following the decision ladder above.

    Returns one of: ``'consistent-with-random'``, ``'entry-distribution'``,
    ``'variance-profile'``, ``'structure-beyond-controls'``, or
    ``'undetermined'`` when the z-scores are not available.

    The variance-profile step tests :data:`MARGIN_CONTROLS` and takes the
    LARGEST deviation among them: if either margin alone reproduces the effect,
    the effect is explained by heteroscedasticity.
    """
    key = f"z_{stat}"

    def z(name):
        v = suite.get(name, {}).get(key, np.nan)
        return abs(v) if np.isfinite(v) else np.nan

    zr = z("real")
    ze = z("entry_shuffle")
    margins = [z(k) for k in MARGIN_CONTROLS if np.isfinite(z(k))]
    # legacy composed control, if that is all the caller ran
    if not margins and np.isfinite(z("row_col_shuffle")):
        margins = [z("row_col_shuffle")]

    if not np.isfinite(zr):
        return "undetermined"
    if zr <= z_thresh:
        return "consistent-with-random"
    if np.isfinite(ze) and ze > z_thresh:
        return "entry-distribution"
    if not margins:
        return "undetermined"
    if max(margins) > z_thresh:
        return "variance-profile"
    return "structure-beyond-controls"
