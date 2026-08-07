"""rmt.per_matrix — the single-SVD aggregator.

``per_matrix_analysis`` computes **exactly one** SVD for a matrix and threads the
resulting :class:`~rmt.linalg.SVDResult` to every sub-analysis (MP bulk, σ,
tail exponents, scalars, spacing, overlap, per-decile), producing one flat dict
that matches the CSV schema in plan.md §5.

Pure numpy/scipy at run time (cached_svd may use torch internally but returns
numpy); this module never imports torch.
"""
from __future__ import annotations

from typing import Optional, Dict
import numpy as np

from .linalg import cached_svd, SVDResult
from . import mp as MP
from . import tail as TAIL
from . import scalars as SC
from . import spacing as SP
from . import overlap as OV
from .config import RunConfig


def _nan():
    return float("nan")


def per_matrix_analysis(record, fm_dict=None, *, cfg: Optional[RunConfig] = None,
                        svals_out=None, ovmat_out=None) -> dict:
    """Analyze one MatrixRecord → one flat row dict (plan.md §5 schema).

    Exactly one SVD (via :func:`cached_svd`) is computed and threaded to all
    sub-analyses.  ``fm_dict`` (optional) maps record names → feature-matrix
    dicts ``{weight, FM, mean}`` for the overlap/coincidence block.

    ``svals_out``/``ovmat_out`` (optional dicts): if given, the descending
    singular-value array and the overlap matrix for this record are stashed under
    ``record.name`` so the pipeline can emit the per-matrix plots without a second
    SVD and without keeping the weight alive (REPORT §2 plot wiring).
    """
    cfg = cfg or RunConfig()
    W = np.asarray(record.weight, dtype=np.float64)
    n, m = int(record.n), int(record.m)

    # --- THE single SVD ---------------------------------------------------- #
    svd: SVDResult = cached_svd(W, full_matrices=False, backend=cfg.backend,
                                gpu_min_dim=cfg.gpu_svd_min_dim)
    s = np.sort(svd.s)[::-1]                       # descending
    if svals_out is not None:
        svals_out[record.name] = s
    Vh = svd.Vh

    # eigenvalue-domain covariance dimension N (default = #cols)
    N_cov = {"cols": m, "rows": n, "max": max(n, m)}.get(cfg.N_cov_mode, m)
    lam = (s ** 2) / float(N_cov)

    row: Dict[str, object] = {
        "name": record.name, "short": record.short, "layer_idx": record.layer_idx,
        "n": n, "m": m, "is_square": int(n == m), "N_cov": int(N_cov),
    }

    # --- MP bulk + sigma --------------------------------------------------- #
    sigma_med = MP.estimate_sigma_gd_median(s=s, n=n, m=m)
    sig0, sig_ref, n_iter = MP.estimate_sigma_med_refined(s=s, n=n, m=m)
    mp_minus, mp_plus = MP.mp_bounds(n, m, sigma_med)
    mp_minus_eig, mp_plus_eig = MP.mp_bounds_eig(n, m, sigma_med, N_cov)
    n_right = int(np.sum(s > mp_plus))
    n_left = int(np.sum(s < mp_minus))
    row.update({
        "sigma_med": sigma_med, "sigma_med_refined": sig_ref, "n_iter_sigma": n_iter,
        "mp_minus": mp_minus, "mp_plus": mp_plus,
        "mp_minus_eig": mp_minus_eig, "mp_plus_eig": mp_plus_eig,
        "n_right_outliers": n_right, "n_left_outliers": n_left,
        "frac_right_outliers": n_right / len(s), "frac_left_outliers": n_left / len(s),
    })

    # --- small-SV deviation ------------------------------------------------ #
    # small_sv_deviation re-sorts internally; pass s directly (no extra sort).
    dev = MP.small_sv_deviation(s, n, m, sigma_med)
    row.update(dev)

    # --- tail exponents (labelled) — gated on cfg.do_powerlaw (REPORT §2) --- #
    if cfg.do_powerlaw:
        csn_lambda = TAIL.fit_powerlaw_csn(lam)            # CSN density-α on λ
        csn_nu = TAIL.fit_powerlaw_csn(s)                  # CSN density-α on ν
        k_hill = max(5, len(s) // 40)
        alpha_hill_nu = TAIL.hill_alpha_at(s, k_hill)
        alpha_hill_lambda = TAIL.hill_alpha_at(lam, k_hill)
        plateau = TAIL.hill_plateau(s, window=cfg.hill_window)
        row.update({
            "alpha": csn_lambda["alpha"], "xmin": csn_lambda["xmin"],
            "ks_D": csn_lambda["ks_D"], "n_tail": csn_lambda["n_tail"],
            "alpha_on_nu": csn_nu["alpha"],
            "alpha_hill_nu": alpha_hill_nu, "alpha_hill_lambda": alpha_hill_lambda,
            "hill_plateau_alpha": plateau["hill_plateau_alpha"],
            "hill_plateau_width": plateau["hill_plateau_width"],
            "hill_is_powerlaw": int(bool(plateau["hill_is_powerlaw"])),
        })
        # honor cfg.alpha_estimator for the headline 'alpha'
        if cfg.alpha_estimator != "all":
            sel = TAIL.select_alpha(lam, estimator=cfg.alpha_estimator,
                                    window=cfg.hill_window)
            row["alpha"] = sel["alpha"]

        # optional powerlaw-pkg LR test
        if cfg.use_powerlaw_pkg:
            pk = TAIL.powerlaw_pkg_fit(lam)
            row["LR_trunc"] = pk["LR_trunc"] if pk else _nan()
            row["LR_p"] = pk["LR_p"] if pk else _nan()
        else:
            row["LR_trunc"] = _nan(); row["LR_p"] = _nan()
        # optional randomized-control
        row["alpha_rand"] = _nan(); row["max_ev_rand"] = _nan()
        if cfg.do_randomize:
            rng = np.random.default_rng(cfg.seed)
            Wr = rng.standard_normal(W.shape) * np.std(W)
            sr = np.linalg.svd(Wr, compute_uv=False)
            row["alpha_rand"] = TAIL.fit_powerlaw_csn((sr ** 2) / N_cov)["alpha"]
            row["max_ev_rand"] = float(np.max(sr ** 2) / N_cov)
    else:
        for kx in ("alpha", "xmin", "ks_D", "n_tail", "alpha_on_nu",
                   "alpha_hill_nu", "alpha_hill_lambda", "hill_plateau_alpha",
                   "hill_plateau_width", "hill_is_powerlaw",
                   "LR_trunc", "LR_p", "alpha_rand", "max_ev_rand"):
            row[kx] = _nan()

    # --- scalars ----------------------------------------------------------- #
    row["row_wise_entropy"] = SC.row_wise_entropy(W)
    row["spectral_entropy"] = SC.spectral_entropy(s)
    row["stable_rank"] = SC.stable_rank(s=s)
    row["mp_softrank"] = SC.mp_softrank(s, mp_plus)
    row["bulk_mass_frac"] = SC.bulk_mass_frac(s, mp_plus)
    row.update({
        "max_sval": float(np.max(s)), "min_sval": float(np.min(s)),
        "mean_sval": float(np.mean(s)), "median_sval": float(np.median(s)),
    })
    # --- IPR — gated on cfg.do_ipr (REPORT §2) ----------------------------- #
    if cfg.do_ipr:
        ipr = SC.ipr_summary(Vh, s, n, m, sigma_med)
        row.update(ipr)
    else:
        row["ipr_top10_mean"] = _nan(); row["ipr_bulk_mean"] = _nan()
    if cfg.do_porter_thomas:
        pt = SC.porter_thomas_ks(Vh)
        row.update(pt)
    else:
        row["pt_ks_mean"] = _nan(); row["pt_frac_random"] = _nan()

    # --- RMT bulk (on eigenvalues of the covariance) ----------------------- #
    if cfg.do_spacing and len(lam) >= 50:
        row["r_statistic_mean"] = SP.r_statistic(lam)
        ks = SP.nn_spacing_ks(lam, deg=cfg.unfold_deg)
        row["nn_KS_GOE"] = ks["nn_KS_GOE"]; row["nn_KS_Poisson"] = ks["nn_KS_Poisson"]
        row["delta3_L10"] = SP.delta3(lam, 10, deg=cfg.unfold_deg)
        row["delta3_L50"] = SP.delta3(lam, 50, deg=cfg.unfold_deg)
        row["sigma2_L10"] = SP.sigma2(lam, 10, deg=cfg.unfold_deg)
        row["sigma2_L50"] = SP.sigma2(lam, 50, deg=cfg.unfold_deg)
    else:
        for kx in ("r_statistic_mean", "nn_KS_GOE", "nn_KS_Poisson",
                   "delta3_L10", "delta3_L50", "sigma2_L10", "sigma2_L50"):
            row[kx] = _nan()

    row["complex_r_abs_mean"] = _nan(); row["complex_r_cos_mean"] = _nan()
    if cfg.do_complex_spacing and n == m:
        cs = SP.complex_spacing_ratio(W)
        row["complex_r_abs_mean"] = cs["abs_mean"]
        row["complex_r_cos_mean"] = cs["cos_mean"]

    # --- overlap / coincidence (only if a feature matrix is available) ----- #
    _set_overlap_nans(row)
    if fm_dict is not None:
        key = OV.resolve_fm_key(record.name, list(fm_dict.keys()))
        if key is not None:
            C = np.asarray(fm_dict[key]["FM"], dtype=np.float64)
            if C.shape[0] == m:                       # dims must match in-features
                # REPORT §2/Flaw 3.4: factor the covariance ONCE and feed both
                # overlap functions, instead of two identical eigh on (d_in x d_in).
                eig = np.linalg.eigh(C)
                ov = OV.overlap_analysis(W, C, svd=svd, eig=eig)
                coi = OV.eigenvector_eigenvalue_coincidence(W, C, svd=svd, eig=eig)
                if ovmat_out is not None:
                    ovmat_out[record.name] = ov["overlap_matrix"]
                o = ov["overlap"]
                row["max_overlap"] = float(np.max(o))
                row["mean_overlap"] = float(np.mean(o))
                row["overlap_at_top_sval"] = float(o[0])
                row["overlap_at_bottom_sval"] = float(o[-1])
                row["rho_top_eigenvector_vs_svals"] = coi["rho_top_eigenvector_vs_svals"]
                row["rho_top_singular_vs_evals"] = coi["rho_top_singular_vs_evals"]
                row["rho_diag_vs_svals"] = coi["rho_diag_vs_svals"]
                row["max_overlap_with_top_eigenvector"] = coi["max_overlap_with_top_eigenvector"]
                row["argmax_singular_for_top_eigenvector"] = coi["argmax_singular_for_top_eigenvector"]
                row["diagonal_coincidence"] = coi["diagonal_coincidence"]

    # --- per-decile (ascending) -------------------------------------------- #
    row.update(SC.per_decile(s, cfg.n_deciles))
    return row


def _set_overlap_nans(row):
    for k in ("max_overlap", "mean_overlap", "overlap_at_top_sval",
              "overlap_at_bottom_sval", "rho_top_eigenvector_vs_svals",
              "rho_top_singular_vs_evals", "rho_diag_vs_svals",
              "max_overlap_with_top_eigenvector", "diagonal_coincidence"):
        row[k] = _nan()
    row["argmax_singular_for_top_eigenvector"] = -1


# canonical column order (plan.md §5)
CSV_COLUMNS = (
    ["name", "short", "layer_idx", "n", "m", "is_square", "N_cov",
     "sigma_med", "sigma_med_refined", "n_iter_sigma", "mp_minus", "mp_plus",
     "mp_minus_eig", "mp_plus_eig", "n_right_outliers", "n_left_outliers",
     "frac_right_outliers", "frac_left_outliers",
     "ks_lower", "n_below_minus", "frac_mass_below_minus", "excess_small_sv",
     "alpha", "xmin", "ks_D", "n_tail", "alpha_on_nu",
     "alpha_hill_nu", "alpha_hill_lambda",
     "hill_plateau_alpha", "hill_plateau_width", "hill_is_powerlaw",
     "LR_trunc", "LR_p", "alpha_rand", "max_ev_rand",
     "row_wise_entropy", "spectral_entropy", "stable_rank", "mp_softrank",
     "bulk_mass_frac", "max_sval", "min_sval", "mean_sval", "median_sval",
     "ipr_top10_mean", "ipr_bulk_mean", "pt_ks_mean", "pt_frac_random",
     "r_statistic_mean", "nn_KS_GOE", "nn_KS_Poisson",
     "delta3_L10", "delta3_L50", "sigma2_L10", "sigma2_L50",
     "complex_r_abs_mean", "complex_r_cos_mean",
     "max_overlap", "mean_overlap", "overlap_at_top_sval", "overlap_at_bottom_sval",
     "rho_top_eigenvector_vs_svals", "rho_top_singular_vs_evals", "rho_diag_vs_svals",
     "max_overlap_with_top_eigenvector", "argmax_singular_for_top_eigenvector",
     "diagonal_coincidence"]
    + [f"entropy_decile_{i}" for i in range(1, 11)]
    + [f"srk_decile_{i}" for i in range(1, 11)]
)
