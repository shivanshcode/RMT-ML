"""Head-to-head benchmark: original RMT-ML spacing/Porter-Thomas vs the fixed code.

Everything is measured on ensembles whose exact answer is known analytically, so
"better" means "closer to the RMT prediction", not "different".

  GOE            -> Wigner-Dyson P(s), Sigma^2/Delta_3 GOE log-laws
  Poisson        -> P(s)=exp(-s), Sigma^2=L, Delta_3=L/15
  Wishart/MP     -> same bulk universality class as GOE (beta=1)
  Wishart+spikes -> GOE bulk + planted outliers (the trained-weight caricature)
  Haar vectors   -> Porter-Thomas null; p-values must be Uniform(0,1)

Usage:  python benchmark.py [n_seeds]
Writes benchmark_results.csv and prints a report.
"""
import sys, os, time, types, warnings, csv
import numpy as np

warnings.filterwarnings("ignore")

OLD_REPO = os.environ.get("OLD_REPO", "/home/claude/mycode/remote-sk-random-matrix-ml-esd-fixed")
NEW_REPO = os.environ.get("NEW_REPO", "/home/claude/testrepo")

sys.path.insert(0, NEW_REPO)
from rmt import spacing as NEW                      # noqa: E402
from rmt import porter_thomas as NEWPT              # noqa: E402
from rmt import scalars as NEWSC                    # noqa: E402
from rmt import mp as MP                            # noqa: E402

OLD = types.ModuleType("old_spacing")
exec(compile(open(os.path.join(OLD_REPO, "rmt", "spacing.py")).read(),
             "old_spacing", "exec"), OLD.__dict__)
OLDSC = types.ModuleType("old_scalars")
exec(compile(open(os.path.join(OLD_REPO, "rmt", "scalars.py")).read(),
             "old_scalars", "exec"), OLDSC.__dict__)

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
LS = (5, 10, 20, 50)
ROWS = []


# --------------------------------------------------------------------------- #
# ensembles
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
    """What the fixed pipeline does before the level statistics (spacing_bulk_only)."""
    sig = MP.estimate_sigma_gd_median(s=s, n=n, m=m)
    lo, hi = MP.mp_bounds(n, m, sig)
    b = s[(s >= lo) & (s <= hi)]
    return b if b.size >= 0.5 * s.size else s


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def summarise(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std())


def rel_err(mean, theory):
    return abs(mean - theory) / abs(theory) if np.isfinite(mean) and theory else np.nan


def record(case, stat, L, theory, old, new):
    om, osd = summarise(old)
    nm, nsd = summarise(new)
    ROWS.append(dict(case=case, statistic=stat, L=L, theory=theory,
                     old_mean=om, old_sd=osd, old_rel_err=rel_err(om, theory),
                     new_mean=nm, new_sd=nsd, new_rel_err=rel_err(nm, theory)))


def banner(t):
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


def table(case):
    rs = [r for r in ROWS if r["case"] == case]
    print(f"{'stat':>7} {'L':>4} {'theory':>9} | {'OLD':>19} {'err':>8} | "
          f"{'NEW':>19} {'err':>8} | winner")
    for r in rs:
        o = f"{r['old_mean']:8.3f}+/-{r['old_sd']:<7.3f}"
        n = f"{r['new_mean']:8.3f}+/-{r['new_sd']:<7.3f}"
        oe, ne = r["old_rel_err"], r["new_rel_err"]
        if not np.isfinite(oe) and not np.isfinite(ne):
            w = "-"
        elif not np.isfinite(ne):
            w = "OLD"
        elif not np.isfinite(oe):
            w = "NEW"
        elif abs(oe - ne) < 0.02:
            w = "tie"
        else:
            w = "NEW" if ne < oe else "OLD"
        print(f"{r['statistic']:>7} {r['L']:>4} {r['theory']:9.4f} | {o} {oe:7.1%} | "
              f"{n} {ne:7.1%} | {w}")


# --------------------------------------------------------------------------- #
# 1-4. level statistics on four ensembles
# --------------------------------------------------------------------------- #
CASES = [
    ("GOE eigenvalues (N=1500)",
     lambda r: goe_eigs(1500, r), "goe"),
    ("Poisson levels (N=6000)",
     lambda r: poisson_levels(6000, r), "poisson"),
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

for label, gen, ref in CASES:
    banner(label)
    acc = {}
    for seed in range(NS):
        rng = np.random.default_rng(1000 + seed)
        lv = gen(rng)
        # the fixed pipeline trims to the MP bulk before the level statistics
        lv_new = lv
        if "spikes" in label:
            lv_new = bulk_trim(np.sort(lv), 3584, 1024)
        branch = (NEW.unfold_auto(lv_new)[1] if True else "")
        acc.setdefault(("new", "branch", 0), []).append(branch)
        for L in LS:
            acc.setdefault(("old", "S2", L), []).append(OLD.sigma2(lv, L))
            acc.setdefault(("old", "D3", L), []).append(OLD.delta3(lv, L))
            acc.setdefault(("new", "S2", L), []).append(NEW.sigma2(lv_new, L))
            acc.setdefault(("new", "D3", L), []).append(NEW.delta3(lv_new, L))
        for tag, mod, x in (("old", OLD, lv), ("new", NEW, lv_new)):
            d = np.diff(np.sort(mod.unfold(x)))
            acc.setdefault((tag, "unfold_mean_s", 0), []).append(np.mean(d))
            acc.setdefault((tag, "nonmono_%", 0), []).append(100.0 * np.mean(d <= 0))
            ks = mod.nn_spacing_ks(x)
            acc.setdefault((tag, "KS_GOE", 0), []).append(ks["nn_KS_GOE"])
            acc.setdefault((tag, "KS_Pois", 0), []).append(ks["nn_KS_Poisson"])

    th_s2 = NEW.sigma2_goe_theory if ref == "goe" else (lambda L: float(L))
    th_d3 = NEW.delta3_goe_theory if ref == "goe" else (lambda L: L / 15.0)
    for L in LS:
        record(label, "Sigma^2", L, th_s2(L), acc[("old", "S2", L)], acc[("new", "S2", L)])
        record(label, "Delta_3", L, th_d3(L), acc[("old", "D3", L)], acc[("new", "D3", L)])
    record(label, "<s>", 0, 1.0, acc[("old", "unfold_mean_s", 0)], acc[("new", "unfold_mean_s", 0)])
    record(label, "nonmono%", 0, 0.0, acc[("old", "nonmono_%", 0)], acc[("new", "nonmono_%", 0)])
    ROWS.sort(key=lambda r: 0)      # keep insertion order
    table(label)
    br = acc.get(("new", "branch", 0), [])
    if br:
        print(f"  NEW unfolding branch chosen: "
              f"{max(set(br), key=br.count)} ({br.count(max(set(br), key=br.count))}/{len(br)} seeds)")
    ko, _ = summarise(acc[("old", "KS_GOE", 0)])
    kn, _ = summarise(acc[("new", "KS_GOE", 0)])
    po, _ = summarise(acc[("old", "KS_Pois", 0)])
    pn, _ = summarise(acc[("new", "KS_Pois", 0)])
    tgt = "GOE" if ref == "goe" else "Poisson"
    print(f"  P(s) KS distance   vs GOE: old {ko:.4f} new {kn:.4f}   "
          f"vs Poisson: old {po:.4f} new {pn:.4f}   (target law: {tgt})")


# --------------------------------------------------------------------------- #
# 5. Porter-Thomas calibration
# --------------------------------------------------------------------------- #
banner("Porter-Thomas: is the test calibrated?  (null p-values must be Uniform(0,1))")
print(f"{'N':>6} {'case':>26} | {'OLD verdict (D<0.1)':>22} | {'NEW p-value':>14} | correct?")
for N in (256, 1024, 4096):
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    Cbar, C, _ = NEWPT.pt_test_statistic(N, n_samples=3000, rng=0)

    cases = {
        "Haar (true PT null)": Q[:, :40].T,
        "Student-t(3) entries": np.array([rng.standard_t(3, N) for _ in range(40)]),
        "localized on 5% coords": np.array(
            [np.eye(N)[rng.choice(N, 1)[0]] * 0 + _loc for _loc in
             [np.concatenate([rng.standard_normal(max(2, N // 20)),
                              np.zeros(N - max(2, N // 20))]) for _ in range(40)]]),
    }
    for cname, V in cases.items():
        old = OLDSC.porter_thomas_ks(V)
        pv = np.array([NEWPT.pt_pvalue(V[i], C, Cbar) for i in range(V.shape[0])])
        expect_random = cname.startswith("Haar")
        old_ok = (old["pt_frac_random"] > 0.8) == expect_random
        new_ok = (np.mean(pv > 0.05) > 0.8) == expect_random
        ROWS.append(dict(case="porter_thomas", statistic=cname, L=N,
                         theory=1.0 if expect_random else 0.0,
                         old_mean=old["pt_frac_random"], old_sd=np.nan,
                         old_rel_err=0.0 if old_ok else 1.0,
                         new_mean=float(np.mean(pv > 0.05)), new_sd=np.nan,
                         new_rel_err=0.0 if new_ok else 1.0))
        print(f"{N:>6} {cname:>26} | frac_random={old['pt_frac_random']:.3f} "
              f"{'OK ' if old_ok else 'WRONG':>6} | mean p={pv.mean():.3f} "
              f"| {'OK' if new_ok else 'WRONG'}")

# uniformity of the null
banner("Porter-Thomas null uniformity (Haar vectors, N=512, 300 vectors)")
rng = np.random.default_rng(11)
Q, _ = np.linalg.qr(rng.standard_normal((512, 512)))
Cbar, C, _ = NEWPT.pt_test_statistic(512, n_samples=4000, rng=0)
pv = np.array([NEWPT.pt_pvalue(Q[:, i], C, Cbar) for i in range(300)])
import scipy.stats as st
print(f"  NEW: mean p = {pv.mean():.4f} (expect 0.500)   "
      f"frac p>0.05 = {np.mean(pv > 0.05):.3f} (expect 0.950)   "
      f"KS vs Uniform p = {st.kstest(pv, 'uniform').pvalue:.3f}")
oldD = OLDSC.porter_thomas_ks(Q[:, :300].T)
print(f"  OLD: no p-value produced; pt_ks_mean = {oldD['pt_ks_mean']:.4f}, "
      f"pt_frac_random = {oldD['pt_frac_random']:.3f} (threshold D<0.1 is "
      f"dimension-blind; 5% critical value at N=512 is {1.36/np.sqrt(512):.3f})")


# --------------------------------------------------------------------------- #
# 6. timing
# --------------------------------------------------------------------------- #
banner("Timing (4096 levels, single matrix)")
rng = np.random.default_rng(0)
s = wishart_svals(14336, 4096, rng)
for tag, mod in (("OLD", OLD), ("NEW", NEW)):
    t0 = time.perf_counter(); [mod.delta3(s, L) for L in LS]; d3 = time.perf_counter() - t0
    t0 = time.perf_counter(); [mod.sigma2(s, L) for L in LS]; s2 = time.perf_counter() - t0
    t0 = time.perf_counter(); mod.nn_spacing_ks(s); ks = time.perf_counter() - t0
    print(f"  {tag}: Delta_3 x4 {d3:6.2f}s   Sigma^2 x4 {s2:6.2f}s   P(s)+KS {ks:5.2f}s")

with open("benchmark_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
    w.writeheader()
    w.writerows(ROWS)
print("\nwrote benchmark_results.csv")
