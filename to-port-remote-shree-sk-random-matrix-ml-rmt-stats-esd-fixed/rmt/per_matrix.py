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
                        svals_out=None, ovmat_out=None, ptvals_out=None) -> dict:
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
        _pv = []
        row.update(SC.porter_thomas_ks(Vh, n_vectors=cfg.pt_max_vectors,
                                       n_samples=cfg.pt_n_samples,
                                       alpha=cfg.pt_alpha, rng=cfg.seed,
                                       pvals_out=_pv))
        pvals = _pv[0] if _pv else np.array([])
        if ptvals_out is not None:
            ptvals_out[record.name] = (s[:pvals.size], pvals)
        from .porter_thomas import porter_thomas_by_decile
        dec = porter_thomas_by_decile(pvals, alpha=cfg.pt_alpha,
                                      n_deciles=cfg.n_deciles)
        for d in range(1, cfg.n_deciles + 1):
            row[f"pt_p_decile_{d}"] = _nan()
            row[f"pt_frac_random_decile_{d}"] = _nan()
        for d, pm, fr in zip(dec["decile"], dec["p_mean"], dec["frac_random"]):
            row[f"pt_p_decile_{d}"] = pm
            row[f"pt_frac_random_decile_{d}"] = fr
    else:
        for kx in ("pt_ks_mean", "pt_p_mean", "pt_frac_random",
                   "pt_p_top10_mean", "pt_p_bulk_mean"):
            row[kx] = _nan()
        for d in range(1, cfg.n_deciles + 1):
            row[f"pt_p_decile_{d}"] = _nan()
            row[f"pt_frac_random_decile_{d}"] = _nan()

    # --- RMT bulk (unfolded level statistics) ------------------------------ #
    # Which levels the spacing statistics run on. Unfolding makes the two
    # equivalent in exact arithmetic; singular values (Thamm's convention) are
    # numerically better conditioned than lambda = s^2/N.
    levels = s if cfg.spacing_domain == "sval" else lam
    # Bulk selection (REPORT §4.1). The MP cut is a no-op on square matrices --
    # nu_- = sigma(sqrt(n)-sqrt(n)) = 0 -- so for q_proj/o_proj it trims nothing
    # and the Bessel hard edge at 0 stays in the sample, contaminating a test
    # that assumes GOE bulk statistics. 'center' trims by rank instead.
    bulk_mode = getattr(cfg, "spacing_bulk_mode", "center")
    if not cfg.spacing_bulk_only:
        bulk_mode = "none"
    if bulk_mode == "mp":
        lo_b, hi_b = ((mp_minus, mp_plus) if cfg.spacing_domain == "sval"
                      else (mp_minus_eig, mp_plus_eig))
        levels = SP.bulk_levels(levels, mode="mp", lo=lo_b, hi=hi_b,
                                min_keep_frac=0.5)
    else:
        levels = SP.bulk_levels(
            levels, mode=bulk_mode,
            center_frac=getattr(cfg, "spacing_bulk_center_frac",
                                SP.BULK_CENTER_FRAC))
    row["bulk_mode"] = bulk_mode
    row["bulk_center_frac"] = (float(getattr(cfg, "spacing_bulk_center_frac",
                                             SP.BULK_CENTER_FRAC))
                               if bulk_mode == "center" else _nan())
    row["n_levels_bulk"] = int(levels.size)
    min_levels = max(50, 4 * cfg.unfold_win + 4)
    if cfg.do_spacing and len(levels) >= min_levels:
        uf = dict(method=cfg.unfold_method, deg=cfg.unfold_deg,
                  win_size=cfg.unfold_win)
        row["r_statistic_mean"] = SP.r_statistic(levels)
        # Unfold ONCE and thread the result to every level statistic: the
        # 'auto' branch builds both candidate unfoldings to cross-check them,
        # so re-unfolding per L would repeat that work eight times over.
        if cfg.unfold_method == "auto":
            xi, used = SP.unfold_auto(levels, deg=cfg.unfold_deg,
                                      win_size=cfg.unfold_win)
        else:
            xi = SP.unfold(levels, deg=cfg.unfold_deg, method=cfg.unfold_method,
                           win_size=cfg.unfold_win)
            used = cfg.unfold_method
        row["unfold_method_used"] = used
        d = np.diff(np.sort(xi))
        row["unfold_mean_spacing"] = float(np.mean(d)) if d.size else _nan()
        row["unfold_frac_nonpositive"] = float(np.mean(d <= 0)) if d.size else _nan()
        row["unfold_n_levels"] = int(xi.size)
        row.update(SP.nn_spacing_ks(levels, unfolded=xi))
        if cfg.do_brody:
            row.update(SP.brody_beta(levels, n_bootstrap=cfg.brody_bootstrap,
                                     rng=cfg.seed, unfolded=xi))
        else:
            row["brody_beta"] = _nan(); row["brody_beta_err"] = _nan()
        # Delta_3 is reliable to L=50 with either unfolding; Sigma^2 only to
        # L~15 when the local kernel had to be used (rmt.spacing.SIGMA2_RELIABLE_LMAX).
        for L in cfg.delta3_L:
            row[f"delta3_L{int(L)}"] = SP.delta3(levels, L, unfolded=xi,
                                                 rng=cfg.seed + 1)
        # Sigma^2 with a FIXED window count (REPORT §4.4): the adaptive rule's
        # stopping time depends on the data, so its seed noise (+-1.4%) is
        # neither reportable nor equal across matrices. 0 -> adaptive.
        nw = int(getattr(cfg, "sigma2_n_windows", SP.SIGMA2_N_WINDOWS))
        nw_arg = None if nw <= 0 else nw
        row["sigma2_n_windows"] = nw if nw > 0 else -1
        for L in cfg.sigma2_L:
            if L > SP.sigma2_reliable_lmax(used, cfg.unfold_win):
                row[f"sigma2_L{int(L)}"] = _nan()
                row[f"sigma2_L{int(L)}_mcerr"] = _nan()
                continue
            v = SP.sigma2(levels, L, unfolded=xi, method=used, rng=cfg.seed,
                          n_windows=nw_arg)
            row[f"sigma2_L{int(L)}"] = v
            row[f"sigma2_L{int(L)}_mcerr"] = (SP.sigma2_mc_error(v, nw)
                                              if nw_arg else _nan())
    else:
        for kx in ("r_statistic_mean", "nn_KS_GOE", "nn_KS_Poisson",
                   "nn_KS_GOE_p", "nn_KS_Poisson_p", "nn_mean_spacing",
                   "unfold_mean_spacing", "unfold_frac_nonpositive",
                   "unfold_n_levels", "brody_beta", "brody_beta_err",
                   "nn_n_spacings", "nn_frac_nonpositive"):
            row[kx] = _nan()
        row["unfold_method_used"] = ""
        for L in cfg.delta3_L:
            row[f"delta3_L{int(L)}"] = _nan()
        for L in cfg.sigma2_L:
            row[f"sigma2_L{int(L)}"] = _nan()
            row[f"sigma2_L{int(L)}_mcerr"] = _nan()
        row["bulk_mode"] = row.get("bulk_mode", "")
        row["sigma2_n_windows"] = row.get("sigma2_n_windows", -1)

    row["complex_r_abs_mean"] = _nan(); row["complex_r_cos_mean"] = _nan()
    row["complex_ensemble"] = ""; row["complex_frac_real"] = _nan()
    if cfg.do_complex_spacing and n == m:
        # REPORT §4.3: W is REAL, so the reference is GinOE (<cos arg z> =
        # -0.175), not GinUE (-0.216). 'auto' detects that and attaches the
        # matching constants, so the CSV can never be scored against the wrong
        # ensemble.
        cs = SP.complex_spacing_ratio(W, ensemble="auto")
        row["complex_r_abs_mean"] = cs["abs_mean"]
        row["complex_r_cos_mean"] = cs["cos_mean"]
        row["complex_ensemble"] = cs["ensemble"]
        row["complex_frac_real"] = cs["frac_real"]

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
     "ipr_top10_mean", "ipr_bulk_mean",
     "pt_ks_mean", "pt_p_mean", "pt_frac_random",
     "pt_p_top10_mean", "pt_p_bulk_mean",
     "r_statistic_mean", "n_levels_bulk", "bulk_mode", "bulk_center_frac",
     "unfold_method_used",
     "unfold_mean_spacing", "unfold_frac_nonpositive", "unfold_n_levels",
     "nn_mean_spacing", "nn_n_spacings", "nn_frac_nonpositive",
     "nn_KS_GOE", "nn_KS_GOE_p",
     "nn_KS_Poisson", "nn_KS_Poisson_p", "brody_beta", "brody_beta_err",
     "delta3_L5", "delta3_L10", "delta3_L50",
     "sigma2_L5", "sigma2_L10", "sigma2_L20",
     "sigma2_L5_mcerr", "sigma2_L10_mcerr", "sigma2_L20_mcerr",
     "sigma2_n_windows",
     "complex_r_abs_mean", "complex_r_cos_mean",
     "complex_ensemble", "complex_frac_real",
     "max_overlap", "mean_overlap", "overlap_at_top_sval", "overlap_at_bottom_sval",
     "rho_top_eigenvector_vs_svals", "rho_top_singular_vs_evals", "rho_diag_vs_svals",
     "max_overlap_with_top_eigenvector", "argmax_singular_for_top_eigenvector",
     "diagonal_coincidence"]
    + [f"pt_p_decile_{i}" for i in range(1, 11)]
    + [f"pt_frac_random_decile_{i}" for i in range(1, 11)]
    + [f"entropy_decile_{i}" for i in range(1, 11)]
    + [f"srk_decile_{i}" for i in range(1, 11)]
)
