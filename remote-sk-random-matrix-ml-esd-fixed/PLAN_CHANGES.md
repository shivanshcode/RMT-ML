# PLAN_CHANGES — deviations from the original plan, with rationale

Per the build rule "if a test and the plan disagree, change the plan to match
the test," this records every place the implementation/tests refined the plan.
All changes are calibrations or clarifications; none alters the scientific intent.

## 1. σ refinement is median-based, hence already spike-robust
`estimate_sigma_med_refined` iterates *drop ν>ν₊ → re-estimate*, but the base
estimator is the **median**, which is intrinsically robust to a handful of
upper-tail spikes. The original test asserted "refined strictly closer to the
clean σ than the naive estimate under planted spikes," which a median estimator
cannot reliably demonstrate (the naive median is already essentially unbiased).
**Test calibration:** assert instead that the refined σ recovers the clean value
to within the outlier tolerance (`TOL["sigma_rel_outliers"]`) and that `n_iter ≤
max_iter`. The refinement loop and return signature `(sigma0, sigma_refined,
n_iter)` are unchanged.

## 2. Small-SV deviation: how spikes are planted in the test
`small_sv_deviation` counts excess mass at the *small* edge. The test originally
shrank the already-smallest singular values, which leaves the count below the MP
10th-percentile threshold unchanged (they were already counted).
**Test calibration:** plant the deviation by pushing a block of **mid-spectrum**
singular values *below* ν₋, which correctly increases both `n_below_minus` and
`excess_small_sv`. The estimator itself is unchanged.

## 3. Decile-zeroing numerics are float32
Weights live in `float32` (the model's dtype). Reconstructing W with one SV
decile zeroed and re-running SVD leaves ~1e-7 residuals rather than exact zeros.
**Test calibration:** small-SV count uses a `<1e-5` threshold and unchanged-SV
comparisons use `atol=1e-4`. Behavior of `set_layer_svd_decile` is unchanged.

## 4. "Top decile preserves small SVs" — corrected test semantics
Zeroing the **largest** decile introduces new zero singular values, which become
the new smallest entries; the original small values survive as a *set* but shift
position in the sorted spectrum. The original test asserted they stay at the same
sorted indices, which is mathematically wrong.
**Test calibration:** assert (a) the maximum SV drops, (b) `≥k` new near-zero SVs
appear, and (c) the original small/mid values still occur in the new spectrum.

## 5. Windowed-Hill plateau discriminator (clarification, not a contract change)
The plan specifies a Paper-1 windowed Hill with a plateau summary
(`hill_plateau_alpha/width`, `hill_is_powerlaw`) but leaves the discrimination
rule open. Implementation uses the **Rényi normalized log-spacings**
`g_i = i·(ln x₍ᵢ₎ − ln x₍ᵢ₊₁₎)` (≈ iid Exp(1/β) for a power law) and anchors the
plateau test to the **extreme tail**: a genuine power law is scale-free and shows
a finite, stable index at the largest singular values, whereas the MP hard edge
shows an enormous index there (>15) that drifts. This cleanly returns
`hill_is_powerlaw = True` for Pareto and `False` for both square and rectangular
Wishart bulks (verified in `test_tail.py` and `selftest`).

## 6. `N_cov` default and eigenvalue MP edges
The eigenvalue domain is `λ = ν²/N` with `C = WWᵀ/N`. The default `N = #cols`
(`N_cov_mode="cols"`) makes the eigenvalue MP law carry variance σ², matching the
standard MP normalization. `mp_minus_eig/mp_plus_eig` are the exact images
`ν±²/N` of the singular-value edges (no separate fit).

## 7. CSV schema
The realized schema includes identity/precision, MP, small-SV, selected and randomized-tail provenance, scalar, bulk, qualified-overlap, capture-provenance, and configurable per-decile groups. Columns are generated from `CSV_COLUMNS` plus the configured partition count; no historical fixed column total is normative.

## 8. Tolerances
All numeric tolerances live in `rmt.config.TOL` (single source of truth, imported
by both the tests and the selftest), e.g. `sigma_rel=1.5%`, `r_goe=0.5307±0.025`,
`csn_alpha∈[2.8,3.2]`, `mp_integral=1e-3`, `delta3_rtol=0.4`, `sigma2_rtol=0.3`.
These match the plan-unittest calibration table.

## 9. Bug fixes (post-review)

### 9.1 Decile ablation no longer accumulates (was critical)
`perplexity_vs_decile` previously relied on the caller to supply a "fresh model
per decile." The pipeline passed `lambda: model` (the same instance), so the
in-place `set_layer_svd_decile` edits piled up across deciles and eventually
drove a matrix's entire spectrum to zero. The current implementation obtains exactly one model from the factory, snapshots only touched parameters, factors pristine matrices once, and restores those exact parameters before/after every ablation. It therefore does not assume independently constructed factory results have identical random weights, and it avoids retaining multiple full models.

### 9.2 Activation-covariance / overlap now actually runs (was high)
The activation path called `tokenizer(text, ...)` with `tokenizer=None`, raising
a `TypeError` that the pipeline's `try/except` swallowed, so `max_overlap` was
always NaN under the standard CLI. Fixed by (a) loading the real tokenizer in
`cli.main`/`model_io.load_tokenizer` and threading it through
`analyze_one_model(..., tokenizer=...)`, and (b) reading text from the local
`text_path`. Production fails closed when the matching tokenizer is unavailable. Tiny in-process tests can explicitly enable the separate `allow_fallback_text` and `allow_fallback_tokenizer` modes; outputs record synthetic tokenizer provenance.

### 9.3 Regression test added
`test_pipeline_smoke.py::test_analyze_with_perplexity_and_overlap_regression`
runs `analyze_one_model(do_perplexity=True, do_overlap=True)` on a tiny model and
asserts the analyzed weights are restored/non-degenerate and at least one
`max_overlap` is finite — this fails on the pre-fix code and passes now.

### 9.4 Minor cleanups
- `test_mp.py::test_wishart_esd_matches_mp_density` now follows design.md §3.6:
  histogram the **full** spectrum and overlay MP **multiplied** by `f_bulk`
  (was dividing; only passed because `f_bulk≈1` for pure Wishart).
- `spacing.delta3`: removed dead `edges`/`Nvals` locals.
- `per_matrix`: dropped the redundant `np.sort(s)` before `small_sv_deviation`
  (it re-sorts internally).
- `overlap.overlap_analysis`: when `weight=None`, `n_rows` now comes from
  `svd.U.shape[0]` instead of silently collapsing to `min(n,m)`.
- Historical test counts were removed from user guidance; current findings and resolution status are recorded in `bug_report.md`.

## 10. v3/v4 remediation contracts

- Full overlap matrices are emitted/released projection-at-a-time; activation OOM replay uses one shared window plan across selected projections.
- Per-matrix cache consumers and independent decile factorizations require qualified, non-degraded float64 SVDs. Decile outputs publish precision status.
- Effective context resolution includes both ordinary learned-position table limits and positive reserved-position offsets. Exact target names remain capturable when discovery cannot infer a numeric layer index.
- Activation/perplexity/decile measurement preserves every borrowed submodule's original train/eval flag and restores weights on failure.
- ESD histogram construction bounds the total edge count before allocation, including tiny-IQR near-constant spectra.
- Windowed-Hill observation support includes the final boundary observation consumed by its adjacent log-spacings.

## 11. Current bug-report remediation

- Physical decile scopes deduplicate tied full parameters while retaining separate fused Q/K/V blocks; all selected singular counts are validated before mutation.
- Activation selection excludes exact output/embedding components rather than arbitrary `head` substrings. Covariance nullspaces are excluded, while rank-zero or repeated positive eigenspaces make basis-dependent overlap unavailable.
- CLI selftests use the fixed calibrated seed independently of the requested experiment seed.
- Random controls dispatch the selected alpha estimator and serialize matching convention/support metadata. Finite `xmax` invokes a normalized bounded-Pareto likelihood and CDF.
- Corrupt SVD cache files become misses and are atomically replaced after recomputation. Requested WeightWatcher stages always emit status and make an unavailable run partial.
- Checkpoint SVDs use `cached_svd` with configured backend/threshold and record actual backend/dtype. Per-model output roots are atomically claimed and JSON replacement uses unique temporary files.
