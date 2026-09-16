#!/usr/bin/env python
"""rmt_null_calibration.py — validate the RMT / Porter-Thomas suite before use.

Run this from the repo root (the directory containing ``rmt/``).

Five stages, in the order you should run them:

  1. ``ensembles``   GOE / GUE / GSE / Poisson + COE / CUE / CSE vs Mehta theory.
                     Answers "are the estimators right?"
  2. ``nulls``       Matched-shape Monte-Carlo null bands for every Llama matrix
                     shape.  Answers "what does this pipeline return when the
                     matrix really is random?"  This is the artifact the paper
                     needs; the asymptotic constants are NOT a valid reference
                     below L ~ 20.
  3. ``power``       Planted deviations.  Answers "what can this suite detect?"
                     -- and, just as importantly, what it cannot.
  4. ``invariance``  Exact controls (transpose, sign flip, permutation, rescale,
                     precision).  Catches plumbing bugs, runs in seconds.
  5. ``score``       Take the pipeline's ``_matrix_metrics.csv`` and the bands
                     from stage 2 and emit z-scores + calibrated p-values.
                     This is what you report.

Every stage writes a CSV into ``--out`` and prints a human-readable table.

Examples
--------
    python rmt_null_calibration.py ensembles --out cal/
    python rmt_null_calibration.py nulls --shapes 4096x4096 1024x4096 \
        14336x4096 4096x14336 --reps 40 --out cal/
    python rmt_null_calibration.py power --shape 1024x1024 --out cal/
    python rmt_null_calibration.py invariance --shape 512x512 --out cal/
    python rmt_null_calibration.py score \
        --csv rmt_results/meta-llama_Llama-3.1-8B/_matrix_metrics.csv \
        --bands cal/null_bands.csv --out cal/
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from rmt import nulls as NU          # noqa: E402
from rmt import spacing as SP        # noqa: E402
from rmt import porter_thomas as PT  # noqa: E402
from rmt import controls as CT       # noqa: E402


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _parse_shape(txt):
    """'4096x4096' -> (4096, 4096);  '2048' -> (2048,)"""
    if "x" in txt.lower():
        a, b = txt.lower().split("x")
        return (int(a), int(b))
    return (int(txt),)


def _write_csv(rows, path):
    import csv
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    keys = list(OrderedDict((k, None) for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nwrote {path}  ({len(rows)} rows)")


def _fmt(x, w=9, p=4):
    return f"{x:{w}.{p}f}" if isinstance(x, float) and np.isfinite(x) else f"{'nan':>{w}}"


# --------------------------------------------------------------------------- #
# STAGE 1 — ensemble benchmark                                                 #
# --------------------------------------------------------------------------- #
def stage_ensembles(a):
    """GOE / GUE / GSE / Poisson through the pipeline, plus the circular
    ensembles as an unfolding-free control.

    Why the circular ensembles matter: their eigenphase density is *exactly*
    uniform, so xi = n*theta/2pi is an exact unfolding.  A discrepancy there is
    estimator bias or asymptotic-formula error, never unfolding error.  Comparing
    the two blocks separates the three causes -- which is how one discovers that
    the beta = 1 Delta_3 asymptote (not the code) is ~11% low at L = 5.
    """
    Ls = [int(x) for x in a.L]
    rng = np.random.default_rng(a.seed)
    rows = []

    def run(label, gen, beta, reps, pre_unfolded):
        t0 = time.time()
        d3 = {L: [] for L in Ls}
        s2 = {L: [] for L in Ls}
        rr, ms, branches = [], [], []
        for _ in range(reps):
            lv = gen()
            if pre_unfolded:
                xi, used = np.sort(lv), "exact"
            else:
                xi, used = SP.unfold_auto(np.sort(lv))
                xi = np.sort(xi)
            branches.append(used)
            rr.append(SP.r_statistic(lv))
            ms.append(float(np.diff(xi).mean()))
            for L in Ls:
                d3[L].append(SP.delta3(None, L, unfolded=xi))
                s2[L].append(SP.sigma2(None, L, unfolded=xi,
                                       method=("cheb" if used == "exact" else used)))
        rt = NU.R_THEORY[beta]
        print(f"\n### {label}   beta={beta}  reps={reps}  "
              f"branch={sorted(set(branches))}  ({time.time()-t0:.0f}s)")
        print(f"  <r> = {np.mean(rr):.4f}   theory {rt:.4f}   "
              f"({100*(np.mean(rr)/rt-1):+.1f} %)     <s> = {np.mean(ms):.4f}")
        print(f"  {'L':>4s} {'Delta3':>9s} {'theory':>9s} {'err%':>7s}   "
              f"{'Sigma2':>9s} {'theory':>9s} {'err%':>7s}")
        rows.append({"ensemble": label, "beta": beta, "reps": reps,
                     "stat": "r", "L": "", "measured": float(np.mean(rr)),
                     "theory": rt, "err_pct": 100*(np.mean(rr)/rt-1),
                     "branch": "|".join(sorted(set(branches)))})
        for L in Ls:
            md, ts = float(np.nanmean(d3[L])), NU.delta3_theory(L, beta)
            mv, tv = float(np.nanmean(s2[L])), NU.sigma2_theory(L, beta)
            print(f"  {L:>4d} {_fmt(md)} {_fmt(ts)} {100*(md/ts-1):7.1f}   "
                  f"{_fmt(mv)} {_fmt(tv)} {100*(mv/tv-1):7.1f}")
            rows.append({"ensemble": label, "beta": beta, "reps": reps,
                         "stat": "delta3", "L": L, "measured": md,
                         "theory": ts, "err_pct": 100*(md/ts-1),
                         "sd": float(np.nanstd(d3[L], ddof=1))})
            rows.append({"ensemble": label, "beta": beta, "reps": reps,
                         "stat": "sigma2", "L": L, "measured": mv,
                         "theory": tv, "err_pct": 100*(mv/tv-1),
                         "sd": float(np.nanstd(s2[L], ddof=1))})

    print("=" * 78)
    print("BLOCK A — Gaussian ensembles (unfolding is estimated, as in production)")
    print("=" * 78)
    run(f"GOE  N={a.n_goe}", lambda: NU.goe_levels(a.n_goe, rng), 1, a.reps, False)
    run(f"GUE  N={a.n_goe}",
        lambda: np.linalg.eigvalsh(
            (lambda A: (A + A.conj().T) / np.sqrt(2))(
                (rng.standard_normal((a.n_goe, a.n_goe))
                 + 1j*rng.standard_normal((a.n_goe, a.n_goe)))
                / np.sqrt(2.0*a.n_goe))), 2, a.reps, False)
    run(f"GSE  N={a.n_gse} (Kramers-collapsed)",
        lambda: NU.gse_levels(a.n_gse, rng), 4, max(3, a.reps // 2), False)
    run(f"Poisson n={a.n_poisson}",
        lambda: NU.poisson_levels(a.n_poisson, rng), 0, a.reps, False)

    print()
    print("=" * 78)
    print("BLOCK B — circular ensembles (density exactly uniform => exact unfolding)")
    print("         Discrepancies here are estimator or ASYMPTOTIC-FORMULA error only.")
    print("=" * 78)
    for beta, n in ((1, a.n_circ), (2, a.n_circ), (4, a.n_circ // 2)):
        run(f"C{'OUS'[{1:0,2:1,4:2}[beta]]}E  n={n}",
            lambda beta=beta, n=n: NU.circular_levels(n, beta, rng), beta,
            max(4, a.reps // 2), True)

    print("\nHOW TO READ THIS")
    print("  * Block B beta=2 and beta=4 agreeing to ~1-2% while beta=1 Delta_3 is")
    print("    ~+11% at L=5 and ~+26% at L=3 means the CODE is right and the beta=1")
    print("    ASYMPTOTE is wrong at small L. Never test against it below L~20.")
    print("  * Block A tracking Block B means the unfolding is not adding bias.")
    print("  * PASS criterion: |err| < 5% for L >= 10 in every ensemble.")
    _write_csv(rows, os.path.join(a.out, "ensemble_benchmark.csv"))


# --------------------------------------------------------------------------- #
# STAGE 2 — matched-shape null bands                                           #
# --------------------------------------------------------------------------- #
def stage_nulls(a):
    """The core artifact: what the pipeline returns on matrices that ARE random."""
    rows = []
    for txt in a.shapes:
        shape = _parse_shape(txt)
        kind = "wishart" if len(shape) == 2 else "goe"
        t0 = time.time()
        band = NU.null_band(shape if kind == "wishart" else shape[0],
                            n_reps=a.reps, kind=kind, seed=a.seed,
                            dtype=np.float32 if a.dtype == "float32" else np.float64,
                            bulk_frac=a.bulk_frac, unfold_method=a.unfold_method,
                            unfold_deg=a.unfold_deg,
                            brody=not a.no_brody, brody_bootstrap=a.brody_bootstrap)
        print(f"\n### shape {txt}   kind={kind}   reps={a.reps}   "
              f"bulk_frac={a.bulk_frac}   ({time.time()-t0:.0f}s)")
        print(f"  unfolding branch histogram: {dict(band['branches'])}"
              + ("    <-- UNSTABLE: results are not comparable across matrices"
                 if len(band["branches"]) > 1 else ""))
        print(f"  {'statistic':22s} {'null mean':>10s} {'null sd':>9s} "
              f"{'2.5%':>10s} {'97.5%':>10s} {'asymptote':>10s} {'in band?':>9s}")
        for k, v in band.items():
            if not isinstance(v, dict) or "mean" not in v:
                continue
            th = _asymptote(k)
            inb = ("-" if not np.isfinite(th)
                   else ("yes" if v["p2.5"] <= th <= v["p97.5"] else "NO"))
            print(f"  {k:22s} {_fmt(v['mean'],10)} {_fmt(v['sd'])} "
                  f"{_fmt(v['p2.5'],10)} {_fmt(v['p97.5'],10)} "
                  f"{_fmt(th,10)} {inb:>9s}")
            rows.append({"shape": txt, "n_levels": band.get("n_levels", {}).get("mean", ""),
                         "kind": kind, "reps": a.reps, "bulk_frac": a.bulk_frac,
                         "statistic": k, "null_mean": v["mean"], "null_sd": v["sd"],
                         "p2.5": v["p2.5"], "p97.5": v["p97.5"],
                         "asymptote": th, "asymptote_in_band": inb,
                         "branches": "|".join(band["branches"])})
        if a.porter_thomas:
            rows += _pt_band(shape, a)

    print("\nHOW TO READ THIS")
    print("  * 'in band? = NO' means the closed-form constant the suite currently")
    print("    compares against is OUTSIDE the null -- every real matrix will look")
    print("    like it deviates. Report z-scores against null_mean/null_sd instead.")
    print("  * null sd is the per-matrix noise floor. A deviation smaller than")
    print("    ~2 sd on ONE matrix is not a measurement.")
    _write_csv(rows, os.path.join(a.out, "null_bands.csv"))


def _asymptote(stat: str) -> float:
    if stat == "r_statistic_mean":
        return NU.R_THEORY[1]
    if stat == "unfold_mean_spacing":
        return 1.0
    if stat == "brody_beta":
        return 1.0
    if stat.startswith("delta3_L"):
        return NU.delta3_theory(float(stat.split("L")[1]), 1)
    if stat.startswith("sigma2_L"):
        return NU.sigma2_theory(float(stat.split("L")[1]), 1)
    return float("nan")


def _pt_band(shape, a):
    """Porter-Thomas null band on synthetic singular vectors of the same shape."""
    if len(shape) != 2:
        return []
    n, m = shape
    g = np.random.default_rng(a.seed + 999)
    reps = max(3, a.reps // 5)          # needs full SVD, so fewer replicas
    fr, pm = [], []
    for _ in range(reps):
        W = g.standard_normal((n, m)) / np.sqrt(max(n, m))
        _, _, Vh = np.linalg.svd(W, full_matrices=False)
        out = PT.porter_thomas_summary(Vh, n_vectors=a.pt_vectors,
                                       n_samples=a.pt_samples, rng=a.seed)
        fr.append(out["pt_frac_random"]); pm.append(out["pt_p_mean"])
    rows = []
    print(f"  Porter-Thomas ({reps} reps, {a.pt_vectors or 'all'} vectors):")
    for k, v in (("pt_frac_random", fr), ("pt_p_mean", pm)):
        v = np.array(v, float)
        print(f"  {k:22s} {_fmt(v.mean(),10)} {_fmt(v.std(ddof=1))} "
              f"{_fmt(np.percentile(v,2.5),10)} {_fmt(np.percentile(v,97.5),10)}"
              f" {_fmt(0.95 if k=='pt_frac_random' else 0.5,10)}")
        rows.append({"shape": f"{n}x{m}", "kind": "wishart", "reps": reps,
                     "statistic": k, "null_mean": float(v.mean()),
                     "null_sd": float(v.std(ddof=1)),
                     "p2.5": float(np.percentile(v, 2.5)),
                     "p97.5": float(np.percentile(v, 97.5)),
                     "asymptote": 0.95 if k == "pt_frac_random" else 0.5})
    return rows


# --------------------------------------------------------------------------- #
# STAGE 3 — power / planted deviations                                         #
# --------------------------------------------------------------------------- #
def stage_power(a):
    """What the suite can and cannot see. Put this table in the supplement."""
    n, m = _parse_shape(a.shape)
    g = np.random.default_rng(a.seed)
    W = g.standard_normal((n, m)) / np.sqrt(max(n, m))
    base = np.sort(np.linalg.svd(W, compute_uv=False))
    k = min(n, m)

    cases = OrderedDict()
    cases["null: iid Gaussian"] = base
    U, _ = np.linalg.qr(g.standard_normal((n, 20)))
    V, _ = np.linalg.qr(g.standard_normal((m, 20)))
    cases["+20 rank-1 spikes (amp 3)"] = np.sort(
        np.linalg.svd(W + 3.0 * U @ V.T, compute_uv=False))
    h1, h2 = n // 2, m // 2
    B = np.zeros((n, m))
    B[:h1, :h2] = g.standard_normal((h1, h2)) / np.sqrt(max(h1, h2))
    B[h1:, h2:] = g.standard_normal((n - h1, m - h2)) / np.sqrt(max(n - h1, m - h2))
    cases["2 independent blocks"] = np.sort(np.linalg.svd(B, compute_uv=False))
    for f in (0.05, 0.10, 0.20):
        nP = int(f * k)
        cases[f"{int(100*(1-f))}% GOE + {int(100*f)}% Poisson"] = np.sort(
            np.concatenate([base[:k - nP],
                            g.uniform(base.min(), base.max(), nP)]))
    msk = (g.random((n, m)) < 0.2)
    cases["80% sparse"] = np.sort(np.linalg.svd(W * msk / np.sqrt(0.2),
                                                compute_uv=False))
    cases["Student-t(3) entries"] = np.sort(np.linalg.svd(
        g.standard_t(3, (n, m)) / np.sqrt(3.0 * max(n, m)), compute_uv=False))

    band = NU.null_band((n, m), n_reps=a.reps, seed=a.seed + 1,
                        brody_bootstrap=40)
    stats = ["r_statistic_mean", "nn_KS_GOE", "brody_beta",
             "delta3_L10", "sigma2_L10"]
    print(f"\nPOWER TABLE   shape {n}x{m}   null band from {a.reps} replicas")
    print("  values are z-scores against the null band;  |z| > 2 = detected\n")
    print(f"  {'planted structure':30s} " + " ".join(f"{s.replace('_statistic_mean','').replace('nn_KS_','KS'):>11s}" for s in stats) + f" {'p_mc(KS)':>9s}")
    rows = []
    for label, lv in cases.items():
        st = NU.spacing_statistics(lv, seed=a.seed)
        zs = [NU.zscore(st.get(s, np.nan), band[s]) if s in band else np.nan
              for s in stats]
        p = NU.ks_pvalue_mc(lv, (n, m), n_null=a.n_null, seed=a.seed + 1)["p_mc"]
        print(f"  {label:30s} " + " ".join(f"{z:+11.1f}" if np.isfinite(z) else f"{'nan':>11s}" for z in zs) + f" {p:9.3f}")
        r = {"case": label, "p_mc_KS": p}
        r.update({f"z_{s}": z for s, z in zip(stats, zs)})
        r.update({s: st.get(s) for s in stats})
        rows.append(r)
    print("\nHOW TO READ THIS")
    print("  * Heavy-tailed entries and sparsity are typically INVISIBLE to every")
    print("    spacing statistic. That is universality, not a bug -- but it means")
    print("    'the bulk is still GOE' is NOT evidence the weights are Gaussian.")
    print("    Say 'the local correlations remain in the beta = 1 universality")
    print("    class', and let the tail-exponent block carry the other claim.")
    print("  * Sigma^2 is usually the only statistic that sees low-rank spikes.")
    _write_csv(rows, os.path.join(a.out, "power_table.csv"))


# --------------------------------------------------------------------------- #
# STAGE 4 — invariance and precision controls                                  #
# --------------------------------------------------------------------------- #
def stage_invariance(a):
    n, m = _parse_shape(a.shape)
    g = np.random.default_rng(a.seed)
    W = g.standard_normal((n, m)) / np.sqrt(max(n, m))
    s0 = np.sort(np.linalg.svd(W, compute_uv=False))
    gap = float(np.diff(s0).mean())
    rows, ok_all = [], True

    def chk(label, val, tol, unit=""):
        nonlocal ok_all
        ok = bool(val <= tol)
        ok_all &= ok
        print(f"  {label:44s} {val:11.3e} {unit:14s} {'PASS' if ok else 'FAIL'}")
        rows.append({"check": label, "value": val, "tol": tol,
                     "pass": int(ok)})

    print(f"\nINVARIANCE CONTROLS   shape {n}x{m}   mean NN gap = {gap:.3e}")
    chk("transpose: max|s(W) - s(W^T)|",
        float(np.max(np.abs(np.sort(np.linalg.svd(W.T, compute_uv=False)) - s0))),
        1e-10 * s0.max())
    D1 = np.diag(g.choice([-1.0, 1.0], n)); D2 = np.diag(g.choice([-1.0, 1.0], m))
    chk("random sign flips D1 W D2: max|ds|",
        float(np.max(np.abs(np.sort(np.linalg.svd(D1 @ W @ D2, compute_uv=False)) - s0))),
        1e-10 * s0.max())
    P, Q = g.permutation(n), g.permutation(m)
    chk("row/col permutation: max|ds|",
        float(np.max(np.abs(np.sort(np.linalg.svd(W[P][:, Q], compute_uv=False)) - s0))),
        1e-10 * s0.max())
    a1 = NU.spacing_statistics(s0, seed=0)
    a2 = NU.spacing_statistics(2.7 * s0, seed=0)
    chk("global rescale x2.7: |d delta3_L10|",
        abs(a1["delta3_L10"] - a2["delta3_L10"]), 1e-9)
    chk("global rescale x2.7: |d r_statistic|",
        abs(a1["r_statistic_mean"] - a2["r_statistic_mean"]), 1e-12)

    print("\nPRECISION")
    s32 = np.sort(np.linalg.svd(W.astype(np.float32), compute_uv=False).astype(np.float64))
    chk("fp32 SVD vs fp64 SVD, in units of a spacing",
        float(np.max(np.abs(s32 - s0)) / gap), 0.01, "spacings")

    def to_bf16(x):
        u = x.astype(np.float32).view(np.uint32)
        u = (u + 0x8000) & 0xFFFF0000
        return u.view(np.float32).astype(np.float64)
    sb = np.sort(np.linalg.svd(to_bf16(W), compute_uv=False))
    shift = float(np.max(np.abs(sb - s0)) / gap)
    print(f"  {'bf16 ROUNDING of entries, in units of a spacing':44s} "
          f"{shift:11.3e} {'spacings':14s} {'INFO':>4s}")
    rows.append({"check": "bf16 entry rounding (spacings)", "value": shift,
                 "tol": "", "pass": ""})
    print("  -> a bf16 checkpoint and its fp32 cast are DIFFERENT matrices, not")
    print("     the same matrix measured twice. Pick one dtype and never mix.")

    print(f"\n  OVERALL: {'PASS' if ok_all else 'FAIL'}")
    _write_csv(rows, os.path.join(a.out, "invariance.csv"))


# --------------------------------------------------------------------------- #
# STAGE 5 — score a real pipeline CSV against the bands                        #
# --------------------------------------------------------------------------- #
def stage_score(a):
    import csv
    with open(a.csv) as fh:
        data = list(csv.DictReader(fh))
    with open(a.bands) as fh:
        braw = list(csv.DictReader(fh))
    bands = {}
    for b in braw:
        bands.setdefault(b["shape"], {})[b["statistic"]] = b

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    stats = [k for k in ("r_statistic_mean", "nn_KS_GOE", "brody_beta",
                         "delta3_L5", "delta3_L10", "delta3_L50",
                         "sigma2_L5", "sigma2_L10", "sigma2_L20",
                         "pt_frac_random", "pt_p_mean")
             if any(k in v for v in bands.values())]
    # GUARD (same class of error as rmt.controls.control_suite): a band built
    # on a different fraction of the spectrum than the pipeline measured is not
    # comparable to it, and the z-scores would be silently meaningless.
    csv_fracs = {r.get("bulk_center_frac", "") for r in data} - {"", "nan"}
    band_fracs = {b.get("bulk_frac", "") for b in braw} - {"", "nan"}
    if csv_fracs and band_fracs:
        cf = {round(f(x), 6) for x in csv_fracs}
        bf = {round(f(x), 6) for x in band_fracs}
        if not (cf & bf):
            raise SystemExit(
                f"REFUSING TO SCORE: the pipeline CSV trimmed to bulk_center_frac"
                f"={sorted(cf)} but the bands were built at bulk_frac={sorted(bf)}."
                f"\n  Re-run stage 'nulls' with --bulk-frac {sorted(cf)[0]}, or"
                f" re-run the pipeline with --spacing_bulk_center_frac"
                f" {sorted(bf)[0]}. The two must see the same fraction of the"
                f" spectrum or every z-score below is meaningless.")

    # Warn on a mixed unfolding branch: sigma2 entries from different branches
    # are not on the same footing (and the gauss branch is only calibrated to
    # L <= win/3 -- see rmt.spacing.sigma2_reliable_lmax).
    branches = {r.get("unfold_method_used", "") for r in data} - {""}
    if len(branches) > 1:
        print(f"\n  !! MIXED UNFOLDING BRANCHES {sorted(branches)}: the Sigma^2"
              f" column is not homogeneous. Re-run with --unfold_method cheb.")

    rows, missing = [], set()
    print(f"\nZ-SCORES vs matched-shape null   ({len(data)} matrices)\n")
    print(f"  {'matrix':38s} {'shape':12s} " + " ".join(f"{s[:10]:>10s}" for s in stats))
    for r in data:
        key = f"{int(f(r.get('n',0)))}x{int(f(r.get('m',0)))}"
        if key not in bands:
            missing.add(key)
            continue
        out = {"name": r.get("name", ""), "shape": key,
               "unfold_method_used": r.get("unfold_method_used", "")}
        zs = []
        for s in stats:
            b = bands[key].get(s)
            z = (float("nan") if b is None
                 else NU.zscore(f(r.get(s)),
                                {"mean": f(b["null_mean"]), "sd": f(b["null_sd"])}))
            zs.append(z)
            out[f"z_{s}"] = z
            out[s] = f(r.get(s))
        rows.append(out)
        print(f"  {out['name'][:38]:38s} {key:12s} "
              + " ".join(f"{z:+10.1f}" if np.isfinite(z) else f"{'nan':>10s}" for z in zs))
    if missing:
        print(f"\n  !! no null band for shape(s): {sorted(missing)}"
              f"\n     re-run stage 'nulls' with --shapes " + " ".join(sorted(missing)))
    if rows:
        print("\n  Bonferroni threshold for "
              f"{len(rows)} matrices x {len(stats)} statistics: "
              f"|z| > {abs(__import__('scipy.stats', fromlist=['x']).norm.ppf(0.025/(len(rows)*len(stats)))):.2f}")
    print("\nHOW TO READ THIS")
    print("  * |z| is deviation in units of the pipeline's OWN null spread.")
    print("  * Report z, not 'measured / theory - 1'.")
    print("  * Check unfold_method_used is constant; a mixed column means the")
    print("    Sigma^2 entries are not on the same footing.")
    _write_csv(rows, os.path.join(a.out, "zscores.csv"))


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# STAGE 6 — bulk-fraction stability sweep (priority item 5)                     #
# --------------------------------------------------------------------------- #
def stage_bulkfrac(a):
    """Is the conclusion a statement about the bulk, or about the edge?

    The MP trim is a no-op on square matrices, so the spectral edges -- whose
    statistics are Airy/Bessel, not GOE -- stay in the sample.  A centred cut by
    rank removes them at any aspect ratio.  This stage runs the same statistics
    at several centred fractions; anything that MOVES across the sweep is an
    edge effect and must not be reported as a bulk result.
    """
    shape = _parse_shape(a.shape)
    fracs = [float(x) for x in a.fracs]
    rows = []
    for frac in fracs:
        t0 = time.time()
        band = NU.null_band(shape, n_reps=a.reps, seed=a.seed, bulk_frac=frac,
                            brody=not a.no_brody, brody_bootstrap=40)
        print(f"\nbulk_frac={frac}  reps={a.reps}  branches={dict(band['branches'])}"
              f"  ({time.time()-t0:.0f}s)")
        row = {"shape": a.shape, "bulk_frac": frac, "reps": a.reps,
               "branches": str(dict(band["branches"]))}
        print(f"  {'statistic':<22}{'mean':>10}{'sd':>10}{'rel sd':>9}")
        for k, v in band.items():
            if not isinstance(v, dict) or "mean" not in v:
                continue
            rel = v["sd"] / abs(v["mean"]) if v["mean"] else float("nan")
            print(f"  {k:<22}{_fmt(v['mean'],10)}{_fmt(v['sd'],10)}{_fmt(rel,9,3)}")
            row[f"{k}_mean"] = v["mean"]
            row[f"{k}_sd"] = v["sd"]
        rows.append(row)

    print("\n" + "=" * 78)
    print("STABILITY ACROSS THE SWEEP  (shift measured in units of the null sd)")
    print("=" * 78)
    print(f"  {'statistic':<22}{'min':>10}{'max':>10}{'shift/sd':>10}  verdict")
    keys = [k[:-5] for k in rows[0] if k.endswith("_mean")]
    for k in keys:
        vals = np.array([r.get(f"{k}_mean", np.nan) for r in rows], float)
        sds = np.array([r.get(f"{k}_sd", np.nan) for r in rows], float)
        if not np.all(np.isfinite(vals)) or not np.all(sds > 0):
            continue
        shift = (vals.max() - vals.min()) / sds.mean()
        verdict = "STABLE" if shift < 1.0 else "EDGE-SENSITIVE <-- do not report"
        print(f"  {k:<22}{_fmt(vals.min(),10)}{_fmt(vals.max(),10)}"
              f"{_fmt(shift,10,2)}  {verdict}")
    _write_csv(rows, os.path.join(a.out, "bulk_frac_sweep.csv"))


# --------------------------------------------------------------------------- #
# STAGE 7 — destructive controls on a real matrix (priority item 8)             #
# --------------------------------------------------------------------------- #
def stage_controls(a):
    """Run the §8 controls on a real weight matrix (or a planted synthetic one).

    Reports z-scores against the matched-shape null for the real matrix and for
    each destructive control, then the verdict from the decision ladder.  This
    is the table that distinguishes "the weights carry learned structure" from
    "the weights have heavy tails" or "the weights are heteroscedastic".
    """
    if a.npy:
        W = np.load(a.npy)
        label = os.path.basename(a.npy)
    else:
        shape = _parse_shape(a.shape)
        g = np.random.default_rng(a.seed)
        W = g.standard_normal(shape)
        if a.planted == "heteroscedastic":
            W *= np.exp(g.normal(0, 0.8, (shape[0], 1)))
            W *= np.exp(g.normal(0, 0.8, (1, shape[1])))
        elif a.planted == "heavytail":
            W = g.standard_t(3, size=shape)
        elif a.planted == "blocks":
            n, m = shape
            W = np.zeros(shape)
            W[:n // 2, :m // 2] = g.standard_normal((n // 2, m // 2))
            W[n // 2:, m // 2:] = g.standard_normal((n - n // 2, m - m // 2))
        label = f"synthetic:{a.planted}"

    vp = CT.variance_profile(W)
    print(f"\nmatrix: {label}   shape={W.shape}")
    print(f"  row sd CV {vp['row_sd_cv']:.4f} (iid expects {vp['row_sd_cv_iid_expected']:.4f})"
          f"   col sd CV {vp['col_sd_cv']:.4f} (iid expects {vp['col_sd_cv_iid_expected']:.4f})")
    print(f"  excess kurtosis {vp['excess_kurtosis']:.3f} (Gaussian = 0)")
    if vp["row_sd_cv"] > 5 * vp["row_sd_cv_iid_expected"]:
        print("  --> HETEROSCEDASTIC: the iid band is NOT the right null here;"
              " quote the row/col shuffle instead.")

    t0 = time.time()
    band = NU.null_band(W.shape, n_reps=a.reps, seed=a.seed,
                        bulk_frac=a.bulk_frac, brody=False)
    print(f"  null band: {a.reps} reps, branches={dict(band['branches'])}"
          f" ({time.time()-t0:.0f}s)")

    suite = CT.control_suite(W, band=band, rng=a.seed, seed=a.seed,
                             bulk_center_frac=a.bulk_frac, brody=False)
    stats = [k for k in suite["real"] if k.startswith("z_")]
    print("\n  " + "control".ljust(18) + "".join(s[2:].rjust(14) for s in stats))
    rows = []
    for name, row in suite.items():
        print("  " + name.ljust(18)
              + "".join(_fmt(row.get(s, float('nan')), 14, 2) for s in stats))
        rows.append({"matrix": label, "control": name,
                     **{k: row.get(k) for k in stats},
                     **{k: v for k, v in row.items() if not k.startswith("z_")}})

    print("\n  VERDICTS (|z| > 2 = detected)")
    for s in stats:
        print(f"    {s[2:]:<22}{CT.interpret(suite, s[2:])}")
    _write_csv(rows, os.path.join(a.out, "controls.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    def common(p):
        p.add_argument("--out", default="cal")
        p.add_argument("--seed", type=int, default=0)

    p1 = sub.add_parser("ensembles"); common(p1)
    p1.add_argument("--reps", type=int, default=6)
    p1.add_argument("--L", nargs="+", default=[3, 5, 10, 20, 50])
    p1.add_argument("--n-goe", type=int, default=1200)
    p1.add_argument("--n-gse", type=int, default=600)
    p1.add_argument("--n-poisson", type=int, default=6000)
    p1.add_argument("--n-circ", type=int, default=800)
    p1.set_defaults(func=stage_ensembles)

    p2 = sub.add_parser("nulls"); common(p2)
    p2.add_argument("--shapes", nargs="+", required=True,
                    help="e.g. 4096x4096 1024x4096 14336x4096 4096x14336")
    p2.add_argument("--reps", type=int, default=30)
    p2.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p2.add_argument("--bulk-frac", type=float, default=1.0)
    p2.add_argument("--unfold-deg", type=int, default=7,
                    help="Chebyshev degree for the global staircase fit. MUST match "
                         "the --unfold_deg used by the pipeline, or the band and the "
                         "observation are not the same estimator. Degree 7 under-fits "
                         "the MP density at ~4000 levels: measured Sigma^2(20) error "
                         "on 14336x4096 is +14.4%% at deg 7 and +2.0%% at deg 15.")
    p2.add_argument("--unfold-method", default="auto",
                    choices=["auto", "cheb", "gauss", "poly"])
    p2.add_argument("--no-brody", action="store_true")
    p2.add_argument("--brody-bootstrap", type=int, default=40)
    p2.add_argument("--porter-thomas", action="store_true")
    p2.add_argument("--pt-vectors", type=int, default=200)
    p2.add_argument("--pt-samples", type=int, default=3000)
    p2.set_defaults(func=stage_nulls)

    p3 = sub.add_parser("power"); common(p3)
    p3.add_argument("--shape", default="1024x1024")
    p3.add_argument("--reps", type=int, default=30)
    p3.add_argument("--n-null", type=int, default=200)
    p3.set_defaults(func=stage_power)

    p4 = sub.add_parser("invariance"); common(p4)
    p4.add_argument("--shape", default="512x512")
    p4.set_defaults(func=stage_invariance)

    p5 = sub.add_parser("score"); common(p5)
    p6 = sub.add_parser("bulkfrac", help="centred-fraction stability sweep")
    common(p6)
    p6.add_argument("--shape", default="1024x1024")
    p6.add_argument("--fracs", nargs="+", default=[0.6, 0.7, 0.8, 1.0])
    p6.add_argument("--reps", type=int, default=20)
    p6.add_argument("--no-brody", action="store_true")
    p6.set_defaults(func=stage_bulkfrac)

    p7 = sub.add_parser("controls", help="destructive controls on a real matrix")
    common(p7)
    p7.add_argument("--npy", default=None,
                    help="path to a .npy weight matrix; omit to use --planted")
    p7.add_argument("--shape", default="1024x1024")
    p7.add_argument("--planted", default="none",
                    choices=["none", "heteroscedastic", "heavytail", "blocks"])
    p7.add_argument("--reps", type=int, default=20)
    p7.add_argument("--bulk-frac", type=float, default=0.7)
    p7.set_defaults(func=stage_controls)

    p5.add_argument("--csv", required=True)
    p5.add_argument("--bands", required=True)
    p5.set_defaults(func=stage_score)

    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    a.func(a)


if __name__ == "__main__":
    main()
