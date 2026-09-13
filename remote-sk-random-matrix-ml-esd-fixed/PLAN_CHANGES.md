# Changes from the original plan

The build rule gives tests priority when a test and the plan differ. This record describes each related change. The changes calibrate or clarify the plan. They do not change its scientific purpose.

## 1. Median refinement of `σ`

`estimate_sigma_med_refined` repeats two operations. It removes values where `ν>ν₊`, and then it estimates `σ` again. The base estimator uses the median, which is resistant to a small number of upper-tail spikes.

The old test required the refined value to be closer than the first value to clean `σ`. A median estimate does not reliably satisfy that requirement because its first value is nearly unbiased.

The test requires recovery within `TOL["sigma_rel_outliers"]`. It also requires `n_iter ≤ max_iter`. The loop and return value `(sigma0, sigma_refined, n_iter)` did not change.

## 2. Small-value deviation in the test

`small_sv_deviation` measures excess mass at the small edge. The old test decreased the smallest values. Those values were already less than the tenth MP percentile, so the count did not change.

The test moves middle values to less than `ν₋`. Thus, both `n_below_minus` and `excess_small_sv` increase. The estimator did not change.

## 3. Float32 values in decile lesions

Model weights use `float32`. After reconstruction and a new SVD, a removed decile can contain residuals near `1e-7`. The values are not exact zeros.

The small-value test uses `<1e-5`. Comparisons of unchanged values use `atol=1e-4`. `set_layer_svd_decile` did not change.

## 4. Effect of a top-decile lesion

Removal of the largest decile adds new zero singular values. After sorting, they become the smallest values. Original small values stay in the set but move to new indices. The old test incorrectly required the same sorted indices.

The corrected test has three requirements. The maximum value must decrease. At least `k` new near-zero values must occur, which is the `≥k` requirement. Original small and middle values must remain in the new spectrum.

## 5. Windowed Hill plateau rule

The plan requires the Paper 1 windowed Hill estimate and a plateau summary. The summary fields are `hill_plateau_alpha/width` and `hill_is_powerlaw`. The plan did not specify a discrimination rule.

The implementation uses normalized Renyi log spacings. The formula is `g_i = i·(ln x₍ᵢ₎ − ln x₍ᵢ₊₁₎)`. For a power law, these values are approximately independent `Exp(1/β)` values.

The plateau test starts at the extreme tail. A true power law is independent of scale and has a finite stable index there. The MP hard edge has an index more than 15 there, and the index changes.

Tests in `test_tail.py` and `selftest` return `hill_is_powerlaw = True` for Pareto data. They return `False` for square and rectangular Wishart bulks.

## 6. Default `N_cov` and MP edges

The eigenvalue domain is `λ = ν²/N`, with `C = WWᵀ/N`. The default `N_cov_mode="cols"` uses `N = #cols`. This gives variance `σ²` in the standard MP eigenvalue law.

`mp_minus_eig` and `mp_plus_eig`, also written `mp_minus_eig/mp_plus_eig`, are exactly `ν±²/N`. The code does not fit them separately.

## 7. CSV schema

The schema includes identity, precision, MP, small-value, tail, scalar, bulk, overlap, capture, and decile data. Tail data includes selected and random-control provenance. Overlap data includes qualification data.

`CSV_COLUMNS` and the selected partition count make the columns. No old fixed column count is normative.

## 8. Tolerances

`rmt.config.TOL` is the one source for numerical tolerances. Tests and `selftest` import it. Examples include `sigma_rel=1.5%`, `r_goe=0.5307±0.025`, and `csn_alpha∈[2.8,3.2]`. Other values include `mp_integral=1e-3`, `delta3_rtol=0.4`, and `sigma2_rtol=0.3`. These values agree with the calibration table in `plan-unittest.md`.

## 9. Repairs after review

### 9.1 Independent decile ablation

Previously, `perplexity_vs_decile` required a new model for each decile. The pipeline gave it `lambda: model`, which returned the same model. Thus, lesions accumulated and finally made the full matrix spectrum zero.

The current code gets one model from the factory. It saves only parameters that it changes. It factors each original matrix one time. Before and after each lesion, it restores exact parameter values.

The code does not require independently made models to have equal random weights. It also does not keep many full models in memory.

### 9.2 Activation covariance and overlap

Previously, the activation path called `tokenizer(text, ...)` when `tokenizer=None`. This caused a `TypeError`, which the pipeline `try/except` caught. Thus, `max_overlap` was always NaN with the standard CLI.

`cli.main` uses `model_io.load_tokenizer` to load the tokenizer. It gives the tokenizer to `analyze_one_model(..., tokenizer=...)`. The analysis reads text from local `text_path`.

Production stops if the matching tokenizer is unavailable. Small in-process tests can enable `allow_fallback_text` and `allow_fallback_tokenizer` separately. Output records synthetic tokenizer provenance.

### 9.3 Regression test

`test_pipeline_smoke.py::test_analyze_with_perplexity_and_overlap_regression` uses a small model. It calls `analyze_one_model(do_perplexity=True, do_overlap=True)`. The test requires restored, nondegenerate weights and one finite `max_overlap`. It fails on the old code and succeeds on the repaired code.

### 9.4 Other corrections

- `test_mp.py::test_wishart_esd_matches_mp_density` uses the rule in `design.md` section 3.6. It plots the full spectrum and multiplies MP by `f_bulk`. The old division succeeded only because `f_bulk≈1` for pure Wishart data.
- `spacing.delta3` no longer contains unused `edges` and `Nvals` variables.
- `per_matrix` no longer calls `np.sort(s)` before `small_sv_deviation`, because that function sorts its input.
- If `weight=None`, `overlap.overlap_analysis` gets `n_rows` from `svd.U.shape[0]`, not `min(n,m)`.
- User instructions do not give fixed test counts. `bug_report.md` and `bug_reportv2.md` give current finding status.

## 10. v3 and v4 contracts

Activation output gives full overlap matrices one projection at a time. OOM replay uses one shared window plan for selected projections.

Matrix analysis and independent decile factors require qualified, non-degraded float64 SVD results. Decile output records precision status.

Effective context includes learned-position table limits and positive reserved-position offsets. Exact target names can be captured without a numeric layer index.

Activation, perplexity, and decile measurements preserve each original submodule mode. They restore weights after a failure.

ESD histograms limit the edge count before allocation. The limit applies to spectra with a very small IQR. Windowed Hill support includes the final observation used by adjacent log spacings.

## 11. Current bug-report repairs

Physical decile scopes remove tied full parameters and keep fused Q, K, and V blocks separate. Before mutation, the code examines all selected singular counts.

Activation selection excludes exact output and embedding components. It does not exclude an arbitrary name that contains `head`. Overlap excludes covariance nullspaces. Rank-zero or repeated positive eigenspaces make basis-dependent overlap unavailable.

CLI self-tests use the fixed calibration seed, not the experiment seed. Random controls use the selected alpha estimator and matching convention metadata. A finite `xmax` uses normalized bounded-Pareto calculations.

A corrupt SVD cache file becomes a miss. The code atomically replaces it after recalculation. A requested WeightWatcher stage always writes status. An unavailable stage makes the execution partial.

Checkpoint SVD operations use `cached_svd` with the selected backend and threshold. Output records the actual backend and dtype. Atomic operations claim per-model roots. JSON replacement uses unique temporary files.

The static audit added more controls. Decile preflight uses metadata and bounded physical parameters. The pipeline applies strict mode after optional stages. Cache data must have correct order and actual dtypes.

The pipeline qualifies singular subspaces and aligns discovery by component. Activation replay artifacts and failures are transactional. Disabled overlap stays disabled. An analyzed lesion scope must match exactly.

Spacing receives and records the seed. Random Hill output records support. Degenerate modified-MP fits and unsupported library configuration cause explicit errors. Checkpoint iterables become materialized lists. Optional power-law adapters return structured status. Read `bug_report.md` for these repairs.

## 12. Second audit repairs

Strict model analysis requires every requested layer. Checkpoint runs reject duplicate probes and compare exact matrix coverage with the first checkpoint.

Library tags cannot contain paths. Real-only analysis rejects complex values before conversion or mutation. Overlap uses the actual SVD precision.

MP output records numerical-spectrum availability. Density, cumulative probability, and energy fractions use normalized units. Hill search examines overlapping bands, and interval statistics require a finite positive length.

Read `bug_reportv2.md` for the second audit status and local test evidence.
