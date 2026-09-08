# Static analysis — bug report v3

## Scope and verification

Fresh review of the **current working-tree implementation**, including its uncommitted changes. `bug_report.md`, `bug_reportv2.md`, and `to_change_env.md` were not used as review inputs. Findings below are grounded in current code, not assumed to persist from an earlier report. No implementation files were changed.

Reviewed the runner, training/data/model paths, scientific dispatch and numerical modules, activation capture, lesions, asset staging, launch script, and tests. Line numbers refer to the pre-fix review snapshot; paths are relative to this directory.

Validation performed:

- `python -m pytest -q -p no:cacheprovider`: **79 passed**.
- `bash -n run_hpc.slurm`: passed.
- Small CPU-only numerical and fault-injection probes reproduced the findings identified as reproduced below. They supplemented the static review; they did not train production models.
- Local interpreter: Python 3.13.2. No CUDA/HPC, real-checkpoint, or connected asset-download validation was performed. `transformers` and `datasets` were unavailable locally.

**P1** = incorrect scientific results or corrupt/successfully misreported execution. **P2** = important conditional correctness, configuration, or reporting problem. These are findings for the next implementation pass, not fixes already made.

## Findings at a glance

| ID | Priority | Finding |
|---|---|---|
| COA-01 | P1 | FARMS-pooled observations are treated as one matrix's level spectrum |
| COA-02 | P1 | The default Golden path silently bypasses FARMS |
| COA-03 | P1 | Accepted method combinations mix fitted and observed spectral domains |
| COA-04 | P1 | Tracy–Widom finite-size dimensions come from pooled sample count |
| COA-05 | P1 | FP32/BF16 training applies non-finite gradients and credits successful updates |
| COA-06 | P1 | Evaluation/lesions can report invalid or saturated perplexities as results |
| COA-07 | P2 | Presets overwrite explicitly supplied CLI aliases |
| COA-08 | P2 | Epoch-only training can loop forever on a loader with no valid targets |
| COA-09 | P2 | Default Lanczos spike margin overwhelms transformer-scale eigenvalues |
| COA-10 | P2 | Scaling trajectories are split by exact realized-ratio floats |
| COA-11 | P2 | Unfolding failures unnecessarily suppress the unfolding-free gap ratio |

---

## COA-01 — FARMS pooling creates artificial level statistics

**Locations:** `run_experiments.py:346–356, 400–450`; `rmt/farms_aspect_ratio.py:315–362`.

**Trigger:** An MP method that uses a prepared FARMS spectrum, such as `analytic_mp`, `kde_bulk_fit`, or `farms_unbiased`, with multiple sampling windows and spacing/number-variance diagnostics enabled.

**Cause:** `analyze_model()` assigns the pooled ESD to `eigenvalues`, then assigns that same array to `spacing_eigenvalues`. It sorts levels across all windows and computes NNSD, Brody, gap ratio, number variance, and rigidity on their union. Pooling densities is valid for FARMS ESD/tail work, but pooling levels changes their correlations. Overlapping or identical-content windows can introduce artificial repeated levels.

**Reproduced:** Let `A = default_rng(22).normal(size=(32,32))`, and `W = vstack((A,A))`. Use 32-row FARMS windows, two row starts, and one column start. The full matrix has only 32 singular values, but the runner analyzes 64 pooled levels and returns 63 spacings, including **32 zeros**. Its reported gap ratio is **0.0**, while the full matrix's canonical spectrum gives approximately **0.5011497**. The duplicated blocks do not make the actual matrix's reduced covariance spectrum degenerate.

This also contradicts `design.md:33–35`, which explicitly says pooled FARMS levels must not enter one-matrix spacing diagnostics.

**Fix direction:** Maintain separate ESD/tail and single-operator level-statistics spectra. Select a bulk in the full matrix's canonical domain, or compute statistics independently per window and aggregate statistics with explicitly different semantics. Do not simply discard duplicate pooled eigenvalues; that does not restore the original correlations.

**Regression:** Use the repeated-block example. Changing the number of FARMS windows must not change full-matrix spacing diagnostics or create more than `min(W.shape)-1` full-matrix spacings.

## COA-02 — Golden configuration does not execute its advertised FARMS stage

**Locations:** `run_experiments.py:346–356, 383–392`; `rmt/factory.py:322–345`; `run_hpc.slurm:196–219`.

**Trigger:** The default `compute_optimal_rmt` configuration: `aspect_ratio_mode=farms_normalized` together with `mp_fit_method=lanczos_stieltjes`.

**Cause:** The runner's `raw_methods` branch bypasses `prepare_spectrum()` for Lanczos and Thamm fits. The raw full-matrix eigenvalues then feed the tail solver and ESD artifacts as well as the raw-domain MP analysis. All FARMS controls are consequently ineffective in the Golden path, even though the saved method configuration and launcher request FARMS.

**Reproduced:** On a fixed Gaussian `(64,128)` matrix, run `analyze_model()` with Lanczos steps 24/probes 1 and FARMS window size 16. Switching the aspect mode from `raw` to `farms_normalized` gives:

- **Zero calls to `prepare_spectrum()` in either run**.
- `spectrum_mode == "raw"` and 64 ESD observations in both.
- Identical tail exponent, approximately **6.5960189765**.

This is not a request to run Lanczos on an unrelated pooled array. Lanczos needs the original operator; the missing component is the independently labeled FARMS ESD/tail analysis.

**Fix direction:** Separate operator-domain MP/spike/spacing results from FARMS ESD/tail results, with independent provenance. Alternatively reject incompatible requests rather than recording a FARMS configuration that does no FARMS work. Update the Golden output contract accordingly.

**Regression:** Exercise the actual runner, not only parser/default equality. Changing a meaningful FARMS window setting must change the Golden FARMS ESD/tail input while leaving the original-operator Lanczos input unchanged.

## COA-03 — Fitted support, observations, and serialized units can refer to different spectra

**Locations:** `rmt/factory.py:346–372, 476–505`; `run_experiments.py:353–381, 400–417, 515–521, 539–555, 630–632`.

**Triggers and causes:**

1. With `mp_fit_method=farms_unbiased` and `aspect_ratio_mode=raw` or `shape_normalized`, `prepare_spectrum()` supplies non-FARMS observations. `dispatch_mp_fit()` nevertheless creates a separate FARMS spectrum internally. The runner then uses its support to mask/count/plot the original prepared observations. `mp_spectrum_domain` is incorrectly assigned the observation mode, so the plot's domain-equality check cannot detect the mismatch.
2. With a prepared FARMS/shape-normalized spectrum and `spike_detector=lanczos_poles`, the detector ignores the passed eigenvalues/aspect ratio and analyzes the original weight operator. The row combines this raw-domain threshold/count with a different ESD without an explicit detector-domain field.
3. Even a consistently FARMS-prepared run writes `spectrum_denominator=svd.normalization`, i.e. the **source matrix** denominator. Canonical FARMS divides by the window's larger dimension; raw FARMS divides by nothing; trace normalization has a different denominator per window. Those are not represented by the current field.

**Reproduced:** For a Gaussian `(64,128)` matrix with raw preparation and a 16-by-16 FARMS MP fit, the prepared aspect ratio is **0.5**, but the returned fit aspect ratio is **1.0**. Its serialized bulk fraction describes the sampled windows, while counting the raw observations against the same support produces a different fraction. All options are accepted without warning.

**Impact:** Outlier counts, spacing bulk selection, plotted edges, and exported units can be scientifically incomparable despite apparently matching metadata.

**Fix direction:** Give every spectrum/fit/detector a domain descriptor containing operator/window geometry, normalization, and observation identity. Validate compatibility before comparisons; otherwise retain separately named outputs or reject the combination. This should be coordinated with COA-01/02 rather than restoring one shared array for every diagnostic.

**Regression:** Test actual results for the fit × aspect-mode × detector cross-product, not merely construction of every configuration. Assert denominator/geometry provenance and that counted observations are the ones used for the reported fit.

## COA-04 — Tracy–Widom uses the pooled observation count as a matrix dimension

**Location:** `rmt/factory.py:458–474`.

**Trigger:** Tracy–Widom detection on pooled FARMS observations.

**Cause:** `effective_small = values.size`, followed by `effective_large = round(effective_small/q)`, treats the number of concatenated eigenvalues as the dimension of one Wishart matrix. Neither adding windows nor repeating measurements increases the dimension of each sampled operator. This incorrectly shrinks the finite-size edge correction as the pool grows.

**Reproduced:** With square 32-by-32 windows and variance 1:

- Correct single-window threshold: approximately **4.18104317**.
- Passing two windows' 64 observations: **4.12257739**.
- Repeating that identical pool ten times: **4.03009421**.

Only the repetition count changed, not the underlying matrix geometry or observed eigenvalue values.

**Fix direction:** Pass actual source/window dimensions explicitly. Apply TW to the selected single operator or per window. If the desired decision is an aggregate over multiple windows, define its multiple-testing/count semantics separately; do not manufacture a larger matrix from the pooled count.

**Regression:** Duplicating the same observations must not change the per-operator threshold. Cover rectangular windows and verify the exact dimension pair used by `tracy_widom_upper_threshold()`.

## COA-05 — Non-finite gradients are applied and counted as successful FP32/BF16 updates

**Location:** `pipelines/trainer.py:319–338`.

**Trigger:** A finite loss with an infinite/NaN backward gradient while the scaler is disabled. This includes CPU/FP32 and the default CUDA/BF16 mode.

**Cause:** Only the forward loss is checked for finiteness. Gradient clipping does not request an error for a non-finite norm. With no active scaler, `scaler.step()` delegates to AdamW anyway, and `updated` is unconditionally true. The trainer advances the scheduler and successful-target budget after corrupting parameters.

**Reproduced:** A one-parameter model with loss `w.square()` at `w=1`, and a parameter hook returning an infinite gradient, completes a one-target run with:

```text
train_loss=1.0; gradient_norm=inf; optimizer_update=1
processed_train_tokens=1; parameter=nan
```

If this happens on the final update, there need not be another forward pass to detect the corruption. A later SVD failure is too late and does not repair the incorrect accounting/checkpoint.

**Fix direction:** Reject or explicitly skip non-finite gradients before any optimizer mutation. Apply the policy consistently to scaled and unscaled modes, and advance scheduler/budget counters only after a valid update. Consider checking resulting parameters before calling a final run successful.

**Regression:** Inject both Inf and NaN gradients under FP32/BF16-equivalent unscaled operation. Verify unchanged weights/optimizer state or an explicit failure, and no credited successful update/targets.

## COA-06 — Invalid and clipped perplexities can become successful lesion results

**Locations:** `pipelines/trainer.py:151–164`; `pipelines/spectral_lesioning.py:412–442, 450–483`; `run_experiments.py:899–956`.

**Causes:**

- Evaluation never checks whether an observed loss or the accumulated mean is finite.
- `exp(min(mean_loss,80))` silently replaces distinct perplexities with the same finite constant. In particular, **positive-infinite loss becomes a finite perplexity**, defeating the benchmark's finite-baseline check.
- Both benchmark functions validate the baseline but not each lesioned evaluation. A NaN lesion result is appended to the result list rather than causing a failed/unavailable intervention.
- The runner can subsequently mark the cell complete; CSV/JSON serialization is not a substitute for a failed-evaluation policy.

**Reproduced:** Evaluating constant losses `81`, `100`, and `inf` returns the same perplexity, approximately **5.5406223844e34**. A benchmark callback returning `[1.0, nan]` produces a result with `lesioned=nan` and `delta=nan` without raising.

**Impact:** Catastrophic lesions may appear to have zero additional damage because both measurements saturated, or failed interventions may be serialized as completed experiments.

**Fix direction:** Validate each loss and the token count, retain loss/log-perplexity as the primary stable measurement, and report overflow explicitly rather than silently capping. Validate every lesioned result and consistently fail or mark that intervention unavailable while preserving restoration of pristine weights.

**Regression:** Cover NaN, Inf, finite losses above 80, zero valid targets, and an invalid lesion callback. Assert honest status and exact restoration on every failure path.

## COA-07 — Explicit alias options are overwritten by paper/Golden presets

**Locations:** `run_experiments.py:957–966, 995–1000, 1002–1034`; aliases in `pipelines/cli_config.py:166–176, 192–199`.

**Cause:** Explicit-option tracking stores the raw option spelling, but `preset()` checks only the canonical destination's flag spelling. Thus an explicitly provided alias is indistinguishable from an omitted option during preset resolution. Argparse's accepted abbreviations have the same structural risk.

**Reproduced:**

```text
--experiment-mode reproduce_paper2 --unfolding-degree 3
parsed degree: 3; resolved degree: 15

--overlap-mode frobenius_projection
parsed metric: frobenius_projection; resolved metric: staats_dual_end
```

The default experiment mode is Golden, so the second example does not need an explicit mode flag. Existing alias tests use the spectral-only parser and miss production preset resolution.

**Fix direction:** Track explicitly supplied argparse destinations, resolving aliases to the same destination, or canonicalize all accepted spellings before applying presets. Disable abbreviations if their precedence cannot be reliably tracked.

**Regression:** Exercise `parse_args()` plus `_resolve_runtime_configuration()` for every alias under every applicable preset, including `--flag=value` and paired boolean overrides.

## COA-08 — Epoch-only training has an unbounded no-progress loop

**Location:** `pipelines/trainer.py:236–260, 263–269, 313–314, 372–376`.

**Trigger:** Library use without `max_train_tokens`, on a nonempty finite loader whose batches contain no valid next-token targets, e.g. all labels ignored or fully masked sequences.

**Cause:** Completion is expressed only in successful update count. The loop is `while True`; `epoch` is incremented but never compared with `config.epochs`. Zero-target batches skip before any step/skip counter changes, and the complete-pass no-progress guard is enabled only for token-budget runs.

**Reproduced:** A length-one loader containing labels `[[-100,-100]]`, with `epochs=1` and no token budget, is entered for a fourth pass with `global_step=0`. The probe deliberately raised on that fourth entry to stop the otherwise unbounded loop.

**Fix direction:** Enforce an explicit epoch/pass boundary for epoch-only mode, and fail on complete passes with no usable targets/updates in either mode. Distinguish bounded skipped-update recovery from batches that can never supply targets.

**Regression:** All-ignored and all-masked finite loaders must terminate promptly with a clear error/status. Include mixed valid/invalid batches so epoch semantics remain defined.

## COA-09 — The default Lanczos margin is uncalibrated for transformer weight scale

**Locations:** `rmt/lanczos_stieltjes.py:916–920, 1029–1038`; defaults in `rmt/factory.py:108–109`; initialization in `models/transformer.py:33, 118–129`.

**Type:** Default integration/calibration issue, not an assertion that an explicitly chosen absolute threshold formula is algebraically wrong.

**Cause:** The factor adapter divides by matrix dimension but does not normalize variance. It then uses the absolute margin `c*N^-delta`, with default `c=1`. Actual initial weight standard deviations are around 0.02 or smaller; covariance eigenvalues scale as the square of that. Consequently the margin can dwarf even a strongly separated relative spike.

**Reproduced:** Generate a Gaussian `(128,256)` factor with seed 0 and multiply its first row by 6. With steps 50/probes 1:

| Factor scaling | Estimated upper edge | Threshold | Largest eigenvalue | Detected spikes |
|---|---:|---:|---:|---:|
| 1 | 2.81701 | 3.11431 | 37.33120 | 1 |
| 0.02 | 0.00112680 | 0.29842858 | 0.01493248 | 0 |

The relative spike is identical and remains over ten times the bulk edge. The default Golden run can therefore yield systematic zero-spike counts because of units, rather than absent relative outliers.

**Fix direction:** Calibrate the production adapter/preset in a normalized covariance domain and map results back, or expose an explicitly scale-relative margin while retaining the absolute reference convention as a named option. Serialize the effective scale and units. Do not silently change the pure reference API without documenting the convention.

**Regression:** Calibrate at actual transformer initialization/residual scales as well as unit-variance Wishart scale, and test rescaling under the chosen production convention.

## COA-10 — Exact realized-ratio grouping destroys scaling trajectories

**Location:** `run_experiments.py:719–749`.

**Cause:** `_plot_scaling()` groups/iterates by the exact floating-point `realized_allocation_ratio`. Architecture discretization and token rounding mean that the same requested intervention has a different realized ratio at each compute budget. The subsequent `isclose()` filter cannot establish a stable intervention series, and may also draw duplicates for nearly equal keys.

**Reproduced:** `build_manifest(vocab_size=50257, compute_budgets=[1e15,1e16,1e17])` gives these realized ratios for requested `kappa=0.25`:

```text
0.061074078, 0.064493805, 0.063348348
```

Those become separate single-point curves, not a three-budget undertrained trajectory. The same issue occurs in the nominal optimal series. This is a plotting/group-identity issue; the realized ratios should still be reported honestly.

**Fix direction:** Use a stable design identifier/requested kappa for series membership. Annotate the realized ratio at each point, and explicitly mark collapsed or severely distorted allocations. Do not falsely claim exact realized-ratio matching across budgets.

**Regression:** Inspect plotted series data for a multi-budget manifest with discretized architectures. Each noncollapsed requested intervention should have its intended sequence of budget points, without duplicate curves.

## COA-11 — Gap ratio disappears when the unrelated unfolding fit fails

**Location:** `run_experiments.py:420–450`.

**Cause:** `r_statistic(bulk_levels)` is nested under successful `dispatch_unfolding()`, plus the smoother-specific minimum bulk count. The adjacent-gap ratio needs no unfolding and only a few usable levels. A high-degree polynomial fold or insufficient Gaussian-window support currently turns an otherwise valid gap-ratio result into NaN.

**Confirmed:** A sorted 64-level exponential sample with seed 0 causes degree-15 Chebyshev unfolding to raise the expected fold error, but its raw gap ratio is finite, approximately **0.3668471**. Static tracing shows the runner skips the gap-ratio call in precisely that error path.

**Fix direction:** Compute the gap ratio independently on the selected single-operator bulk, with its own minimum-count and availability checks. Keep unfolding failures scoped to diagnostics that actually need unfolding.

**Regression:** Force or construct an unfolding failure on a valid distinct bulk and assert that NNSD/Brody are unavailable while the raw gap ratio still equals `r_statistic(bulk_levels)`.

## Suggested repair order

1. Address COA-01 through COA-04 together using explicit per-diagnostic spectral-domain metadata. Preserve the distinction between FARMS ESD pooling and an actual covariance operator.
2. Make training/evaluation/lesion numerical failure handling fail closed (COA-05/06/08).
3. Repair production option precedence and Lanczos default calibration (COA-07/09).
4. Fix reporting/independent diagnostics (COA-10/11).

The passing existing suite is not evidence that these orchestration paths are covered. Add runner-level and fault-injection regressions rather than only more individual solver/parser tests.
