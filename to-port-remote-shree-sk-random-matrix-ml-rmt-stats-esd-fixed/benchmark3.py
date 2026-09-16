"""Head-to-head benchmark: OLD repo vs files(3).zip (NEW) vs the corrected code (FIXED).

A drop-in extension of the repo's own ``benchmark.py``: identical ensembles,
identical seeds (``default_rng(1000 + seed)``), identical statistics, so the
OLD and NEW columns reproduce ``benchmark_results.csv``.  Two things are added:

1. A **FIXED** column from the corrected tree.
2. Scoring against the **exact finite-L law** (``rmt.reference``) as well as the
   L->infinity asymptote the original used as its ``theory`` column.

Why both references.  The asymptote is accurate to 0.1 % for Sigma^2 at every L,
so every Sigma^2 error is a genuine estimator error.  For Delta_3 at beta = 1 it
is -21 % at L = 3 and -10 % at L = 5, so most of the Delta_3 "error" was never
an estimator error at all.  Reporting one conflated number hides which is which.

Usage:  python benchmark3.py [n_seeds] [--out FILE]
"""
import argparse
import csv
import os
import sys
import types
import warnings

import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_REPO = os.environ.get("OLD_REPO", os.path.join(HERE, "remote-sk-random-matrix-ml-esd-fixed"))
NEW_REPO = os.environ.get("NEW_REPO", os.path.join(HERE, "tree_new"))
FIX_REPO = os.environ.get("FIX_REPO", os.path.join(HERE, "tree_fixed"))


def _load(path, name):
    """Execute a single module file standalone (the repo's own OLD-loading trick)."""
    m = types.ModuleType(name)
    m.__dict__["__file__"] = path
    exec(compile(open(path).read(), name, "exec"), m.__dict__)
    return m


OLD = _load(os.path.join(OLD_REPO, "rmt", "spacing.py"), "old_spacing")
NEW = _load(os.path.join(NEW_REPO, "rmt", "spacing.py"), "new_spacing")
FIX = _load(os.path.join(FIX_REPO, "rmt", "spacing.py"), "fix_spacing")
REF = _load(os.path.join(FIX_REPO, "rmt", "reference.py"), "reference")

sys.path.insert(0, FIX_REPO)
from rmt import mp as MP                                    # noqa: E402

LS = (5, 10, 20, 50)


# --------------------------------------------------------------------------- #
# ensembles -- byte-identical to the repo's benchmark.py                       #
# --------------------------------------------------------------------------- #
def goe_eigs(n, rng):
    A = rng.standard_normal((n, n))
    return np.linalg.eigvalsh((A + A.T) / np.sqrt(2 * n))


def poisson_levels(n, rng):
    return np.cumsum(rng.exponential(1.0, n))


def wishart_svals(n, m, rng):
    return np.linalg.svd(rng.standard_normal((n, m)) / np.sqrt(m), compute_uv=False)


def spiked_svals(n, m, rng, k=20, amp=4.0):
    W = rng.standard_normal((n, m)) / np.sqrt(m)
    for i in range(k):
        u = rng.standard_normal(n)
        v = rng.standard_normal(m)
        W += amp * (i + 1) ** -0.7 * np.outer(u / np.linalg.norm(u), v / np.linalg.norm(v))
    return np.linalg.svd(W, compute_uv=False)


def bulk_trim(s, n, m):
    sig = MP.estimate_sigma_gd_median(s=s, n=n, m=m)
    lo, hi = MP.mp_bounds(n, m, sig)
    b = s[(s >= lo) & (s <= hi)]
    return b if b.size >= 0.5 * s.size else s


CASES = [
    ("GOE eigenvalues (N=1500)", lambda r: goe_eigs(1500, r), "goe"),
    ("Poisson levels (N=6000)", lambda r: poisson_levels(6000, r), "poisson"),
    ("Wishart singular values, nu domain (3584x1024)",
     lambda r: wishart_svals(3584, 1024, r), "goe"),
    ("Wishart eigenvalue domain lambda=nu^2/N (3584x1024)",
     lambda r: wishart_svals(3584, 1024, r) ** 2 / 1024.0, "goe"),
    ("Wishart + 20 heavy spikes, nu domain (3584x1024)",
     lambda r: spiked_svals(3584, 1024, r), "goe"),
    ("SQUARE Wishart, nu domain (1024x1024) [Llama q/k/v/o shape]",
     lambda r: wishart_svals(1024, 1024, r), "goe"),
    ("SQUARE Wishart, eigenvalue domain lambda=nu^2/N (1024x1024)",
     lambda r: wishart_svals(1024, 1024, r) ** 2 / 1024.0, "goe"),
]


def summarise(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    return (float(v.mean()), float(v.std())) if v.size else (np.nan, np.nan)


def rel_err(mean, theory):
    if not np.isfinite(mean):
        return np.nan
    return abs(mean - theory) / abs(theory) if theory else abs(mean - theory)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", nargs="?", type=int, default=6)
    ap.add_argument("--out", default="benchmark_results_3way.csv")
    a = ap.parse_args(argv)

    rows = []
    for label, gen, ref in CASES:
        print(f"  {label}", flush=True)
        beta = 1 if ref == "goe" else 0
        acc, meta = {}, {}
        for seed in range(a.seeds):
            rng = np.random.default_rng(1000 + seed)
            lv = gen(rng)

            # NEW: what the delivered pipeline does -- MP bulk trim on the spiked case
            lv_new = bulk_trim(np.sort(lv), 3584, 1024) if "spikes" in label else lv
            # FIXED: adaptive outlier strip, a no-op unless levels are detached
            lv_fix, n_lo, n_hi = FIX.strip_edge_outliers(np.sort(lv))
            meta["stripped"] = n_lo + n_hi
            meta["branch"] = FIX.unfold_auto(lv_fix)[1]
            meta["transform"] = FIX.unfold_transform_auto(lv_fix)[2]

            for L in LS:
                acc.setdefault(("old", "Sigma^2", L), []).append(OLD.sigma2(lv, L))
                acc.setdefault(("old", "Delta_3", L), []).append(OLD.delta3(lv, L))
                acc.setdefault(("new", "Sigma^2", L), []).append(NEW.sigma2(lv_new, L))
                acc.setdefault(("new", "Delta_3", L), []).append(NEW.delta3(lv_new, L))
                acc.setdefault(("fix", "Sigma^2", L), []).append(FIX.sigma2(lv_fix, L))
                acc.setdefault(("fix", "Delta_3", L), []).append(FIX.delta3(lv_fix, L))
            for tag, mod, x in (("old", OLD, lv), ("new", NEW, lv_new), ("fix", FIX, lv_fix)):
                d = np.diff(np.sort(mod.unfold(x)))
                acc.setdefault((tag, "<s>", 0), []).append(float(np.mean(d)))
                acc.setdefault((tag, "nonmono%", 0), []).append(100.0 * float(np.mean(d <= 0)))

        for stat in ("Sigma^2", "Delta_3"):
            for L in LS:
                if stat == "Sigma^2":
                    t_as, t_ex = REF.sigma2_asymptote(L, beta), REF.sigma2_exact(L, beta)
                else:
                    t_as, t_ex = REF.delta3_asymptote(L, beta), REF.delta3_exact(L, beta)
                om, osd = summarise(acc[("old", stat, L)])
                nm, nsd = summarise(acc[("new", stat, L)])
                fm, fsd = summarise(acc[("fix", stat, L)])
                rows.append(dict(
                    case=label, statistic=stat, L=L,
                    theory_asym=t_as, theory_exact=t_ex,
                    old_mean=om, old_sd=osd, old_err_asym=rel_err(om, t_as),
                    new_mean=nm, new_sd=nsd, new_err_asym=rel_err(nm, t_as),
                    new_err_exact=rel_err(nm, t_ex),
                    fixed_mean=fm, fixed_sd=fsd, fixed_err_asym=rel_err(fm, t_as),
                    fixed_err_exact=rel_err(fm, t_ex),
                    branch=meta["branch"], transform=meta["transform"],
                    outliers_stripped=meta["stripped"]))
        for stat, t in (("<s>", 1.0), ("nonmono%", 0.0)):
            om, osd = summarise(acc[("old", stat, 0)])
            nm, nsd = summarise(acc[("new", stat, 0)])
            fm, fsd = summarise(acc[("fix", stat, 0)])
            rows.append(dict(
                case=label, statistic=stat, L=0, theory_asym=t, theory_exact=t,
                old_mean=om, old_sd=osd, old_err_asym=rel_err(om, t),
                new_mean=nm, new_sd=nsd, new_err_asym=rel_err(nm, t),
                new_err_exact=rel_err(nm, t),
                fixed_mean=fm, fixed_sd=fsd, fixed_err_asym=rel_err(fm, t),
                fixed_err_exact=rel_err(fm, t),
                branch=meta["branch"], transform=meta["transform"],
                outliers_stripped=meta["stripped"]))

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    sc = [r for r in rows if r["statistic"] in ("Sigma^2", "Delta_3")]
    g = lambda k: np.array([r[k] for r in sc], float)
    print(f"\nwrote {a.out}  ({len(rows)} rows, {a.seeds} seeds)\n")
    print(f"{'':26}{'median':>10}{'mean':>10}{'max':>10}")
    for lbl, k in (("OLD  vs asymptote", "old_err_asym"),
                   ("NEW  vs asymptote", "new_err_asym"),
                   ("NEW  vs exact law", "new_err_exact"),
                   ("FIXED vs asymptote", "fixed_err_asym"),
                   ("FIXED vs exact law", "fixed_err_exact")):
        v = g(k); v = v[np.isfinite(v)]
        print(f"{lbl:26}{np.median(v):10.4f}{np.mean(v):10.4f}{np.max(v):10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
