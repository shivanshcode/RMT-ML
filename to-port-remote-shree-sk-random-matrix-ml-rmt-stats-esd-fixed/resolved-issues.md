# Resolved issues

Each item states what was wrong, where it was fixed, and **exactly which row of
`benchmark_results_3way.csv` or which test to look at** to confirm it.

Three provenance classes:

* **[T]** present in Thamm, Staats & Rosenow (2022) `src/rmt_utils.py`, the
  reference implementation this suite was ported from — inherited, not
  introduced here.
* **[R]** found by the referee report (`RMT_SUITE_REVIEW.md`) and fixed in
  `files(3).zip`.
* **[N]** found and fixed in this delivery.

Column names below refer to `benchmark_results_3way.csv`:
`old_*` = original repo, `new_*` = files(3).zip, `fixed_*` = this delivery.
`*_err_asym` scores against the L→∞ asymptote; `*_err_exact` against the exact
finite-L law.

---

## R1 — Δ₃ compared against an L→∞ asymptote that is wrong at small L **[R, fixed here]**

**Was.** `delta3_goe_theory(L)` is the Dyson–Mehta asymptote. For β = 1 it
converges far more slowly than β = 2, so every matrix "deviated" at small L.

**Now.** `rmt/reference.py` computes the exact finite-L law from the two-level
cluster function: Σ²(L) = L − 2∫₀ᴸ(L−r)Y₂(r)dr and Δ₃(L) =
(2/L⁴)∫₀ᴸ(L³−2L²r+r³)Σ²(r)dr (Mehta, *Random Matrices* 3rd ed., eqs. 16.1.4 /
16.1.7), with Y₂ for β = 1, 2, 4 and Poisson.

**Verify — this is the cleanest single check in the whole delivery.**
`benchmark_results_3way.csv`, row `case=GOE eigenvalues (N=1500)`,
`statistic=Delta_3`, `L=5`:

| column | value |
|---|---|
| `fixed_mean` | 0.1723 |
| `theory_asym` | 0.1561 → `fixed_err_asym` = **0.1039** |
| `theory_exact` | 0.1738 → `fixed_err_exact` = **0.0085** |

Same measurement, same code, two references: 10.4 % becomes 0.85 %. The
estimator was never wrong. Same row at `L=10`: 0.0288 → 0.0120.

**Independent confirmation.** On circular ensembles the eigenphase density is
exactly uniform, so ξ = nθ/2π is an *exact* unfolding and nothing but the
reference can be blamed. Measured COE Δ₃ against the asymptote: +27.2 % (L=3),
+10.9 % (L=5), +3.6 % (L=10). Against the exact law: **+0.4 %, −0.4 %, −0.6 %.**
Pinned in `tests/test_reference_and_unfolding.py::test_goe_delta3_exact_values`
and `::test_delta3_asymptote_is_badly_wrong_at_small_L_for_beta_one`.

**Note.** Σ²'s asymptote is accurate to <0.2 % at every L (pinned in
`::test_sigma2_asymptote_is_already_accurate`), so no Σ² error was ever a
reference error. That asymmetry is what lets the two error sources be separated.

---

## N1 — Global fit collapses on a hard-edge density; code fell back to the local kernel **[N]**

**Was.** In the λ = ν²/N domain a square Wishart has ρ(λ) ~ λ^(−1/2). No
finite-degree polynomial represents an inverse square root, so `unfold_auto`
failed its cross-check (ratio ≈ 4) and fell back to Gaussian broadening — whose
window-reach bias then cost ~11 % on Σ²(10). The fallback was treated as the
cure; it was the disease.

**Now.** `rmt/spacing.py::unfold_transform_auto` searches monotone coordinate
transforms (`identity`, `sqrt`, `cbrt`, `log`) and keeps the global fit. Unfolded
spacing statistics are invariant under a smooth monotone reparametrisation, so
the coordinate is a free *numerical* choice.

**Verify.** `benchmark_results_3way.csv`, `case=Wishart eigenvalue domain
lambda=nu^2/N (3584x1024)`. Check the `transform` column reads `sqrt`:

| stat | L | `old_err_asym` | `new_err_asym` | `fixed_err_exact` |
|---|---|---|---|---|
| Sigma^2 | 10 | 0.84 | 0.2028 | **0.0250** |
| Sigma^2 | 20 | 1.56 | 0.3886 | **0.0304** |
| Sigma^2 | 50 | 0.76 | **0.6270** | **0.0310** |

And `case=SQUARE Wishart, eigenvalue domain lambda=nu^2/N (1024x1024)`,
Σ²(50): OLD 61.66 → NEW 0.2524 → **FIXED 0.0217**.

Also `<s>` for the square-λ case: old 0.9775, new 0.9936, **fixed 0.9996**.

Pinned in `tests/test_reference_and_unfolding.py::
test_lambda_domain_picks_sqrt_and_stays_on_the_global_fit` and
`::test_nu_domain_is_not_gratuitously_transformed` (the ν domain must *not* be
gratuitously transformed — it stays `identity`).

---

## N2 — Spiked spectra wrecked the global fit; no shape-free outlier handling **[N]**

**Was.** Rank-one spikes sit outside the MP edge. The delivered pipeline removed
them via `mp_bounds`, which needs the matrix shape; `sigma2(levels, L)` has no
shape, so any shape-free caller got a destroyed fit.

**Now.** `rmt/spacing.py::strip_edge_outliers` handles *both* outlier shapes:
a detached cluster (single gap > 25 × median with a small detached count) and a
graded tail (peel inward while the edge spacing > 6 × median), capped at 3 % of
levels.

**Verify.** `case=Wishart + 20 heavy spikes, nu domain (3584x1024)`, column
`outliers_stripped` is non-zero while every clean case reads 0–1:

| stat | L | `old_err_asym` | `fixed_err_exact` |
|---|---|---|---|
| Sigma^2 | 10 | 0.84 | **0.0222** |
| Sigma^2 | 20 | 1.56 | **0.0118** |
| Sigma^2 | 50 | 2.11 | **0.0477** |

`<s>`: old 1.0177 → **fixed 1.0010**.

Both outlier shapes are pinned:
`::test_strips_equal_amplitude_spike_cluster` and
`::test_strips_graded_spike_tail_and_recovers_sigma2`, plus
`::test_does_not_strip_a_poisson_tail` and `::test_clean_spectra_lose_nothing`
as the negative controls. The Poisson negative control matters: a gap-size test
*alone* strips ~20 % of a Poisson spectrum.

---

## N3 — Fitting an already-unfolded spectrum destroys real fluctuation **[N]**

**Was.** A degree-7 staircase fit applied to a Poisson sequence (which arrives
with unit mean spacing) removes genuine long-wavelength count fluctuation — the
very thing Σ²(L) measures.

**Now.** `rmt/spacing.py::looks_pre_unfolded` — a density-flatness test. If mean
spacing ≈ 1 and the local mean spacing is flat across ten blocks, levels pass
through untouched. It cannot fire on a real spectrum: GOE varies ~3× across
blocks (semicircle), square Wishart ~5× (quarter circle).

**Verify.** `case=Poisson levels (N=6000)`, `statistic=Sigma^2`, `L=50`. Before
the guard the degree-7 fit gave 45.6 (−8.8 %); the delivered `new_mean` is
50.075 and `fixed_mean` matches to <1.5 %. Whole-case aggregate: NEW mean/max
0.0106 / 0.0480 → **FIXED 0.0079 / 0.0295**.

Pinned in `::test_pre_unfolded_detection_is_a_flatness_test`.

---

## N4 — `SIGMA2_RELIABLE_LMAX` floor certified a badly biased value **[N]**

**Was.** files(3).zip corrected `gauss: 15 → 5` and made it window-aware as
`max(3.0, win/3)`. The `max(3.0, …)` floor means a window of 5 still certifies
L = 3 — where the measured Σ² error is **−15.3 %**.

**Now.** `rmt/spacing.py::sigma2_reliable_lmax` returns `win/3` with no floor and
`0.0` below `GAUSS_WIN_MIN_USABLE = 12`.

**Verify.** Not a benchmark row (the benchmark never takes the gauss branch now).
Measured directly with the repo's own adaptive kernel:

| win_size | L=3 | L=10 | L=15 | L=20 |
|---|---|---|---|---|
| 5 | −15.3 % | −46.8 % | −49.4 % | −52.4 % |
| 15 | −2.6 % | −7.9 % | −15.6 % | −23.2 % |
| 30 | +1.7 % | −2.3 % | −1.7 % | −4.2 % |

Pinned in `::test_narrow_gauss_window_certifies_nothing`. The physics is the
standard unfolding artefact: a local kernel of window *w* absorbs the count
fluctuations Σ²(L) measures once L ≳ w/3 — Gómez, Molina, Relaño & Retamosa,
*Phys. Rev. E* **66**, 036209 (2002).

---

## N5 — z-scores thresholded with a normal quantile **[N, library fixed; caller still open — see O8]**

**Was.** A band built from `n` replicas gives an *estimated* mean and sd, so
scoring an out-of-sample matrix yields `t_{n−1} · √(1+1/n)`, not a standard
normal. Using `norm.ppf` is anti-conservative by 20 % at n = 20 and 39 % at n=12.

**Now.** `rmt/nulls.py::critical_value(n_reps, n_tests, alpha)`.

**Verify.** `tests/test_reference_and_unfolding.py::
test_critical_value_is_t_not_normal`: 3.88 at (20, 40), 4.48 at (12, 40), and
→ 3.23 as n → ∞. The delivery's own `power_table.csv` null row shows |z| = 4.10
at reps = 12 and calls it a false positive; against 4.48 it is not one.

---

## N6 — No variance-profile-matched null existed **[N, generator added; wiring still open — see O7]**

**Was.** Review §8.2 asked for a null preserving heteroscedasticity, prescribed
as "shuffle within rows, then within columns". Implemented as a composition that
destroys exactly what it should preserve.

**Now.** `rmt/controls.py::variance_profile_gaussian` draws `a_i b_j G_ij` from
the fitted row/column sd profile, with optional Sinkhorn iterations.

**Verify.** `::test_variance_profile_gaussian_preserves_both_margins`:

| control | CV(row sd) | CV(col sd) |
|---|---|---|
| original | 0.785 | 0.413 |
| row_shuffle | 0.785 | 0.116 |
| col_shuffle | 0.125 | 0.413 |
| row_col_shuffle | 0.127 | 0.116 |
| entry_shuffle | 0.133 | 0.132 |
| **variance_profile_gaussian** | **0.787** | **0.427** |

Theory: random matrices with a variance profile are the Wigner-type ensembles of
Ajanki, Erdős & Krüger, *Probab. Theory Relat. Fields* **169**, 667 (2017).

---

## T1 — Σ² stopping rule terminates on a self-fulfilling criterion **[T, fixed in files(3).zip]**

**Was.** Thamm's adaptive loop tracks the spread of the last `min_iters`
*running* variances. A cumulative mean changes by O(1/k) by construction, so the
spread falls below `tol` automatically regardless of convergence — ±1.4 % of pure
seed noise.

**Now.** `SIGMA2_N_WINDOWS = 200_000` fixed, `sigma2_mc_error()` reportable,
`n_windows=None` restores Thamm's loop.

**Verify.** Scatter over 6 seeds on one fixed spectrum: L=5 1.5 % → 0.35 %
(4.3×), L=20 2.2 % → 0.76 % (2.9×). Residual falls as 1/√N (sd 0.0089 → 0.0031 →
0.0018 → 0.0008 at 2×10⁴ → 4×10⁶ windows), confirming it is window-limited, not
spectrum-limited. `tests/test_sigma2_determinism.py` (13 tests).

---

## T2 — No Δ₃ implementation at all **[T, added upstream; verified here]**

Thamm does not implement Δ₃. The closed form here replaces a 400-point trapezoid
and is exact.

**Verify.** `benchmark_results_3way.csv`, `statistic=Delta_3`, β = 2 rows are
within ~0.5 % of `theory_exact` at every L, and my COE/CUE cross-check gives CUE
Δ₃ within 0.6 % at L = 3–20. Since β = 2's asymptote and exact law agree to
<0.1 %, agreement there validates the *estimator* independently of R1.

---

## T3 — `nn_spacing` sorts before differencing **[T, diagnosed in files(3).zip]**

**Was.** `np.sort(xi)` then `np.diff` makes every spacing non-negative by
construction, so the `s[s >= 0]` filter is unreachable and a non-monotone (i.e.
broken) unfolding was silently *repaired* rather than flagged. Dropping entries
also changes *n*, which changes the KS critical value, so the reported p-value no
longer refers to the test that was run.

**Now.** `_unfolded_raw` + `clean_spacings` judge monotonicity pre-sort; dips
beyond `UNFOLD_NONMONO_TOL` NaN the row and are counted in
`nn_frac_nonpositive` / `nn_n_spacings`.

**Verify.** `nonmono%` rows in `benchmark_results_3way.csv` are 0.0 for old, new
and fixed on every case — i.e. no case in this benchmark exercises it. The guard
is verified by `tests/test_spacing.py` instead, not by the benchmark.

---

## T4 — `nn_KS_GOE_p` is not a p-value **[T-adjacent, fixed in files(3).zip]**

**Was.** `scipy.stats.kstest` assumes iid draws from a fully specified reference.
Neither holds: unfolded spacings from a rigid spectrum are anti-correlated, and
the unfolding is fitted to the very data being tested (a Lilliefors effect —
Lilliefors, *JASA* **62**, 399 (1967)). Null median p ≈ 0.5–0.75, `frac(p<0.05)`
= 0.00, i.e. near-zero power.

**Now.** `nulls.ks_pvalue_mc` — parametric bootstrap against synthetic matrices
of the same shape through the same unfolding.

**Verify.** `benchmark_results.csv` (delivered) `case=porter_thomas` rows show
the analogous calibration for the PT test. For the KS test the evidence is in
`tests/test_nulls.py::test_ks_pvalue_mc_uniform_under_null` and
`::test_scipy_kstest_pvalue_is_not_uniform`. **This also cancels the Wigner-
surmise bias for free**: `wigner_goe_cdf` is the 2×2 surmise, not the exact
Gaudin β=1 law, and pooling >31 000 spacings rejects a perfect GOE spectrum from
the approximation alone — but the bias is present in observed *and* null
replicas, so it cancels identically.

---

## T5 — Strict `all(d > 0)` unfolding gate → non-random NaNs **[T, fixed in files(3).zip]**

**Was.** A single dip among ~10³ spacings flipped the matrix to the Gaussian
branch, and `per_matrix` then NaN'd `sigma2_L20` for exactly those — missingness
correlated with a spectral property (MNAR), which biases any layerwise average.
4/8 branch flips on statistically identical matrices.

**Now.** `UNFOLD_NONMONO_TOL = 1e-3`; and N1 above removes the branch flip at its
source rather than tolerating it.

**Verify.** `branch` column of `benchmark_results_3way.csv` reads `cheb` for
every case including both λ-domain rows. Previously the λ rows took `gauss`.

---

## Aggregate

`benchmark_results_3way.csv`, 6 seeds, the 56 Σ²/Δ₃ rows:

| scoring | median | mean | max |
|---|---|---|---|
| OLD vs asymptote | 0.0421 | 2.3348 | 61.66 |
| NEW vs asymptote (delivered) | 0.0274 | 0.0613 | 0.6270 |
| NEW vs exact law | 0.0123 | 0.0459 | 0.6270 |
| **FIXED vs exact law** | **0.0102** | **0.0140** | **0.0477** |

All seven cases improve on mean *and* max. Four rows of 56 regress against the
delivered column, all by ≤ 0.9 percentage points and all inside replica scatter;
they are listed as O5 and O6 in `outstanding-issues.md`.

Regression status: **263 passed, 4 skipped** (`pytest tests/ -m "not torch"`),
up from 240 + 4. `python -m rmt --selftest`: 11/11 PASS.
