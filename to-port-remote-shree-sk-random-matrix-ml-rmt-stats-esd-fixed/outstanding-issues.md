# Outstanding issues

Everything below is **still open**. Each item names the code location that needs
changing and the row of `benchmark_results_3way.csv` (or a named reproduction)
that demonstrates it is real rather than hypothetical.

Read `resolved-issues.md` for what is already fixed and `design.md` for why each
design choice was made.

Severity key: **P0** blocks a publishable claim · **P1** biases a reported
number · **P2** correctness/robustness · **P3** cosmetic or documentation.

---

## O1 — `strip_edge_outliers` uses a global median, not a local one **[P1]**

**Where:** `rmt/spacing.py`, `strip_edge_outliers()`, the `med` variable and both
`tol * med` / `peel_tol * med` comparisons.

**Problem.** Detachment is judged against the median spacing of the spectrum
*core*. But near a Marchenko–Pastur soft edge the density vanishes as a square
root, so the local mean spacing there is legitimately several times the core
median. A genuinely detached level at the low edge therefore does not look
detached, and a perfectly ordinary level at the edge can look detached.

**Evidence it is real.** On the repo's own spiked generator
(`spiked_svals(3584, 1024, seed=1000)`), the five lowest spacings are 3.96,
3.82, 3.74, 4.04, 3.61 × the *global* core median — but only 2.33, 2.25, 2.20,
2.38, 2.13 × the *local* median. The global-median view makes the soft edge look
like a run of outliers; the local view correctly says it is not. The MP-bound
trim removes one level from the low end here that `strip_edge_outliers` does
not. It happens not to matter for this generator (both give Σ²(20) ≈ 1.03), but
the criterion is measuring the wrong thing and will diverge on a spectrum with a
real low-edge outlier.

**Fix.** Compare each candidate spacing to `np.median` of the neighbouring ~`win`
spacings rather than to the core median. Keep the detached-count cap unchanged.

---

## O2 — Adaptive stripping duplicates, and is weaker than, the MP-edge trim **[P1]**

**Where:** `rmt/spacing.py`, `strip_edge_outliers()` vs `rmt/mp.py`,
`mp_bounds()` / `estimate_sigma_gd_median()`.

**Problem.** BBP outliers are *defined* by lying beyond the Marchenko–Pastur
edge, and `rmt/mp.py` already computes that edge from the matrix shape.
`strip_edge_outliers` re-derives the same cut from spacing statistics alone,
which is strictly less information. It exists because `sigma2(levels, L)` has no
access to the shape — but `per_matrix.py` does.

**Evidence it is real.** On `spiked_svals(3584, 1024)`, MP trim keeps 1019/1024
levels and gives Σ²(20) = 1.0296; the adaptive strip keeps 1020 and gives
1.0350. Both fine. But my *first* implementation, which only handled a detached
cluster and not a graded tail, gave Σ²(20) = **2.07** against exact 1.0491 —
a 97 % error that MP trimming would never have produced. The spacing-only route
is fragile in a way the shape-aware route is not.

**Fix.** In `per_matrix.py`, pass the MP bounds through to level selection and
use `bulk_levels(mode="mp")` when the shape is known; demote
`strip_edge_outliers` to the documented fallback for shape-free callers.

---

## O3 — `bulk_center_frac = 0.7` is still the shipped default and costs levels **[P1]**

**Where:** `rmt/config.py`, `RunConfig.spacing_bulk_mode = "center"` and
`spacing_bulk_center_frac = 0.7`.

**Problem.** The centred cut discards 30 % of the levels unconditionally. In the
singular-value domain — the domain `§7.6` of the review mandates — it buys
nothing, and fewer levels measurably *degrades* long-range Σ².

**Evidence it is real.** From `bulk_frac_sweep.csv` in files(3).zip, on a clean
512×512 null: Σ²(20) = 1.0197 at frac 1.0 (exact 1.0491), 0.9499 at 0.7, 0.8795
at 0.6 — monotone degradation as you trim, *away* from theory. The `EDGE-
SENSITIVE` flag the delivery put on `sigma2_L20` is a level-count artefact, not
an edge artefact. Separately, `benchmark_results_3way.csv` shows the FIXED
column reaching 0.0102 median with **no** centred cut at all.

**Fix.** Default `spacing_bulk_mode="auto"` (outlier-aware) or `"mp"`; keep
`"center"` available and require it to be justified per shape by the `bulkfrac`
sweep rather than assumed.

---

## O4 — Σ²(50) is systematically low at every level count tested **[P1]**

**Where:** `rmt/spacing.py`, `_sigma2_fixed()` window sampling range —
`lo, hi = xi[0] + L/2, xi[-1] - L/2`.

**Problem.** At L = 50 with ~1000 levels, the admissible window centres span only
`n − L` ≈ 950 units and the windows overlap heavily, so the effective number of
independent samples is ~`n/L` ≈ 20. Worse, excluding `L/2` at each end removes
exactly the region where the unfolding residual is largest, biasing the estimate.

**Evidence it is real.** Every L = 50 row in `benchmark_results_3way.csv` is low
and they are the largest remaining errors:

| case | Σ²(50) fixed | exact | err | sd |
|---|---|---|---|---|
| Wishart + spikes, ν | 1.1759 | 1.2348 | 4.77 % | 0.126 |
| Wishart ν (3584×1024) | 1.1878 | 1.2348 | 3.81 % | 0.112 |
| Wishart λ (3584×1024) | 1.1966 | 1.2348 | 3.10 % | 0.079 |

All three are the same sign. The replica sd is large (0.08–0.13), so each row
alone is ~0.4σ, but three independent cases all low is not noise.

**Fix.** Either report `sigma2_mc_error` alongside and refuse L > n/20, or switch
to the unbiased estimator that integrates over the full range with edge
weighting. Simplest correct action: cap L at `n/20` and NaN beyond it.

---

## O5 — GOE Σ²(10) and Σ²(20) run ~3 % high **[P2]**

**Where:** `rmt/spacing.py`, `_unfold_cheb()` degree, and the
`UNFOLD_TREND_GATE` selection in `unfold_transform_auto()`.

**Problem.** A degree-7 Chebyshev fit to a semicircle staircase leaves a small
residual density modulation that inflates the count variance.

**Evidence it is real.** `benchmark_results_3way.csv`, GOE N=1500: Σ²(10) =
0.9372 vs exact 0.9087 (+3.14 %, sd 0.025 over 6 seeds ⟹ ~1.2σ), Σ²(20) = 1.0836
vs 1.0491 (+3.29 %, sd 0.048). A separate degree scan (in `design.md` §D3) shows
degree 11 reduces GOE Σ²(20) from +4.9 % to +3.6 % and degree 15 to +2.9 % —
so the residual is fit capacity, not the estimator.

**Why it is still open.** Raising the degree globally was measured and rejected:
it regressed 9 of 56 rows. A per-ensemble degree needs a selection rule that does
not also over-fit Poisson. See `design.md` §D3 for the rejected attempt.

---

## O6 — Poisson Δ₃ is ~1 % low **[P3]**

**Where:** `rmt/spacing.py`, `looks_pre_unfolded()` — the `flat_tol = 1.5`
threshold and what the caller does when it fires.

**Evidence it is real.** `benchmark_results_3way.csv`: Poisson Δ₃(5) = 0.3308 vs
exact 0.3333 (0.75 %, NEW got 0.07 %); Δ₃(10) = 0.6601 vs 0.6667 (0.98 %, NEW got
0.11 %). Small, but a consistent regression against the delivered column.

**Cause.** When `looks_pre_unfolded` fires, levels pass through untouched — which
is right for Σ² but means the Δ₃ estimator sees the raw sequence with its
`mean(d) = 1.0044` rather than exactly 1. **Fix:** rescale to exact unit mean
spacing when the pre-unfolded path fires.

---

## O7 — No variance-profile control is wired into the pipeline **[P0]**

**Where:** `rmt/controls.py` — `variance_profile_gaussian()` exists but is not in
`CONTROLS`, not in `control_suite()`, and not in `interpret()`'s decision ladder.
Also `rmt_null_calibration.py`, `controls` stage.

**Problem.** This is review item 8.2, the single item that converts "we measured
something" into "we measured something randomness does not explain". The shuffle
controls cannot serve as this null: each preserves exactly one margin.

**Evidence it is real.** Measured on a matrix with row-sd CV 0.785 / col-sd CV
0.413: `row_shuffle` → 0.785 / 0.116, `col_shuffle` → 0.125 / 0.413,
`row_col_shuffle` → 0.127 / 0.116, `entry_shuffle` → 0.133 / 0.132.
`variance_profile_gaussian` → **0.787 / 0.427**, the only one holding both.
Pinned in `tests/test_reference_and_unfolding.py::
test_variance_profile_gaussian_preserves_both_margins`.

**Fix.** Add to `CONTROLS`, run it in `control_suite`, and make `interpret()`
return `variance-profile` when the real matrix deviates but this null does not.

---

## O8 — Bonferroni threshold still printed from `norm.ppf` **[P0]**

**Where:** `rmt_null_calibration.py`, the `score` stage, line ~469–471:
`scipy.stats.norm.ppf(0.025/(len(rows)*len(stats)))`.

**Problem.** `nulls.critical_value()` now exists and is correct, but the
calibrator still prints the normal quantile. Scoring an out-of-sample matrix
against a band of `n` replicas is `t_{n−1} · √(1+1/n)` distributed.

**Evidence it is real.** From files(3).zip artefacts: `null_bands.csv` was built
at `reps=20` and `zscores.csv` was scored against a printed threshold of 3.23;
the correct value at n=20, 40 tests is **3.88**. The `power_table.csv` null row
shows |z| = 4.10 at reps=12, flagged in the delivery as a "false positive" — the
correct threshold there is **4.48**, so it never was one.

**Fix.** Replace the call with `NU.critical_value(n_reps, n_tests)` and read
`n_reps` from the bands file.

---

## O9 — `complex_spacing_ratio` discriminates GinOE by the wrong statistic **[P2]**

**Where:** `rmt/spacing.py`, `complex_spacing_ratio()` — the `ensemble='auto'`
branch and its `frac_real` reporting.

**Problem.** The module reports `frac_real` because ⟨cos arg z⟩ did not separate
GinOE from GinUE. That conclusion is right, but the inference drawn from it is
too strong. GinOE genuinely *is* a distinct non-Hermitian universality class; the
difference is localised near the real axis and is O(N^{−1/2}) on a
spectrum-wide mean, hence invisible at N = 400, not absent.

**Evidence it is real.** Replicated at N = 400, 30 reps: GinOE − GinUE ⟨cos⟩ =
−0.0077 ± 0.0095 (0.8σ) — indistinguishable, as the delivery says. But
`frac_real` = 0.0407 against the exact Edelman–Kostlan–Shub prediction
√(2/πN) = 0.0399, i.e. `frac_real` measures the *count* of real eigenvalues, not
a spectral correlation.

**Fix.** Add a real-axis-strip CSR: restrict to eigenvalues within ~1 mean
spacing of the real axis and report ⟨cos arg z⟩ there, where the classes do
separate. Low priority — `do_complex_spacing` defaults to `False`.

---

## O10 — Porter–Thomas is untouched and unverified against the new path **[P2]**

**Where:** `rmt/porter_thomas.py`; benchmark section 5 of `benchmark.py`.

**Problem.** The PT module was declared correct by the review and I did not
re-derive it. My `benchmark3.py` **drops** the PT section entirely, so the
three-way CSV has no PT rows.

**Evidence.** `benchmark_results.csv` (delivered) rows `case=porter_thomas` show
NEW is already correct: Haar null `frac(p>0.05)` = 1.0/0.925/0.95 at N =
256/1024/4096 (expect 0.95), Student-t(3) = 0.225/0.0/0.0 (expect 0, OLD gave
0.45/0.675/0.875 — i.e. OLD fails to reject). Nothing is broken; it is simply
**not re-verified** under the new unfolding.

**Fix.** Restore section 5 of the repo's `benchmark.py` into `benchmark3.py`.
PT operates on singular *vectors* and does not touch the unfolding, so no
interaction is expected — but "expected" is not "measured".

---

## O11 — Σ² and Δ₃ use different RNG streams from the same seed **[P3]**

**Where:** `rmt/spacing.py`, `sigma2(..., rng=0)` and `delta3(..., rng=1)`
defaults; `per_matrix.py` passes `seed` and `seed + 1`.

**Problem.** Reproducibility depends on an undocumented offset. If a caller
passes the same seed to both, the window centres are correlated and the two
statistics are no longer independent estimates.

**Evidence.** Structural, not a benchmark row. Noted in review §7.6 ("log it;
`sigma2` uses `seed`, `delta3` uses `seed + 1`") but never enforced in code.

**Fix.** Derive both from `np.random.SeedSequence(seed).spawn(2)`.

---

## O12 — Conventions only you can decide **[P0 for the paper]**

Unchanged from review §8.5–8.6 and unaffected by any code here:

1. **RoPE permutation.** HF's Llama conversion permutes rows of `q_proj` and
   `k_proj`. Singular *values* are permutation-invariant (verified: the
   invariance check gives 9.99e-15), so the spacing block is safe — but do not
   interpret singular *vectors* head-by-head without undoing it.
2. **Define "the weight matrix".** `W`, or `W · diag(rmsnorm_gain)`? Both are
   defensible; the gains are non-uniform after training and change the singular
   values. State which and be consistent. `export_weights_npy.py --rmsnorm_gain`
   supports either.
3. **Do not compare Σ²(20) across level counts.** k/v_proj give 1024 levels,
   everything else 4096, and the per-matrix null sd grows sharply as levels
   drop. This is why bands are per shape.

---

## O13 — Bands are still 400–512 square **[P0]**

**Where:** operational, not code. `null_bands.csv` in files(3).zip has
`shape=400x400`, `reps=20`, `bulk_frac=0.7` for every row.

**Fix.** Re-run stage 2 at the real Llama shapes (4096×4096, 1024×4096,
14336×4096, 4096×14336) with ≥ 40 replicas before scoring anything. Everything
downstream — every z-score in `zscores.csv` — is currently against a 400×400
band and is not usable for a claim about a 4096-level matrix.
