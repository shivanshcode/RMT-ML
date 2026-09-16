"""rmt.selftest — fast analytic self-check (no torch, no model).

Verifies the scientific core against known RMT ground truth before any expensive
real-matrix run.  ``run()`` returns True iff every check is within ``config.TOL``.
"""
from __future__ import annotations

import numpy as np

from . import ensembles as E
from . import mp as MP
from . import tail as TAIL
from . import spacing as SP
from . import overlap as OV
from .config import TOL, get_logger

_log = get_logger("rmt.selftest")


def run(seed: int = 1234, verbose: bool = True) -> bool:
    rng = np.random.default_rng(seed)
    checks = []

    # 1. MP pdf normalisation
    lo, hi = MP.mp_bounds(800, 1600, 1.0)
    nodes, wts = np.polynomial.legendre.leggauss(300)
    xm = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    I = 0.5 * (hi - lo) * np.sum(wts * MP.mp_pdf(xm, 800, 1600, 1.0))
    checks.append(("mp_pdf_norm", abs(I - 1.0) < TOL["mp_integral"]))

    # 2. sigma recovery
    W = E.wishart_factor(2000, 500, sigma=1.3, rng=rng)
    sig = MP.estimate_sigma_gd_median(W)
    checks.append(("sigma_recovery", abs(sig - 1.3) / 1.3 < TOL["sigma_rel"]))

    # 3. CSN on Pareto
    x = E.pareto(20000, 3.0, rng=rng)
    a_csn = TAIL.fit_powerlaw_csn(x)["alpha"]
    lo_c, hi_c = TOL["csn_alpha"]
    checks.append(("csn_pareto", lo_c <= a_csn <= hi_c))

    # 4. windowed Hill plateau on Pareto vs MP bulk
    pl_pareto = TAIL.hill_plateau(x)["hill_is_powerlaw"]
    s_w = np.linalg.svd(E.wishart_factor(1200, 1200, rng=rng), compute_uv=False)
    pl_mp = TAIL.hill_plateau(s_w)["hill_is_powerlaw"]
    checks.append(("hill_windowed_discriminates", bool(pl_pareto) and not bool(pl_mp)))

    # 5. r-statistic GOE vs Poisson
    rg = SP.r_statistic(np.linalg.eigvalsh(E.goe(600, rng)))
    rp = SP.r_statistic(E.poisson_levels(4000, rng))
    mid_g, tol_g = TOL["r_goe"]; mid_p, tol_p = TOL["r_poisson"]
    checks.append(("r_goe", abs(rg - mid_g) < tol_g))
    checks.append(("r_poisson", abs(rp - mid_p) < tol_p))

    # 6. NN-spacing discrimination
    ev = np.linalg.eigvalsh(E.goe(600, rng))
    d = SP.nn_spacing_ks(ev)
    checks.append(("nn_spacing_goe", d["nn_KS_GOE"] < d["nn_KS_Poisson"]))
    # 6b. unfolding health: strictly increasing, unit mean spacing, no rescaling
    uf = SP.unfold_diagnostics(ev)
    checks.append(("unfolding_valid",
                   uf["unfold_frac_nonpositive"] == 0.0
                   and abs(uf["unfold_mean_spacing"] - 1.0) < 0.05))
    # 6c. Delta_3 reproduces the GOE log-law
    d3, th3 = SP.delta3(ev, 10), SP.delta3_goe_theory(10)
    checks.append(("delta3_goe_log_law", abs(d3 - th3) / th3 < 0.2))
    # 6d. Porter-Thomas p-values are uniform under the null
    from . import porter_thomas as PT
    Q, _ = np.linalg.qr(rng.standard_normal((256, 256)))
    Cbar, C, _ = PT.pt_test_statistic(256, n_samples=1500, rng=0)
    pv = np.array([PT.pt_pvalue(Q[:, i], C, Cbar) for i in range(100)])
    checks.append(("porter_thomas_calibrated", abs(pv.mean() - 0.5) < 0.12))

    # 7. fm-key resolution
    key = OV.resolve_fm_key("m.layers.0.attn.query_key_value.weight[Q]",
                            ["m.layers.0.attn.query_key_value"])
    checks.append(("fm_key", key == "m.layers.0.attn.query_key_value"))

    ok = all(p for _, p in checks)
    if verbose:
        for name, passed in checks:
            _log.info("selftest %-28s %s", name, "PASS" if passed else "FAIL")
        _log.info("selftest overall: %s", "PASS" if ok else "FAIL")
    return ok
