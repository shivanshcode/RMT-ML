# Static bug report v2: compute_optimal_analysis

## Scope and evidence

This review found 12 defects at commit `d1e0698797b031b8fdf1a383cd769a21d710f0bb`. The current working tree fixes all 12 defects. Paths and line numbers in the findings refer to the reviewed commit.

Static analysis reads code without running the application. This review traced the runner, training, models, numerical methods, activation capture, lesions, asset staging, and launcher. It also examined test coverage and the earlier `bug_report.md`.

The finding descriptions preserve the original static evidence. The repair added focused CPU regression tests. Backend-specific behavior still requires tests on the selected backend.

All 37 Python files passed syntax parsing. Both project launchers passed `bash -n`. Targeted Ruff analysis found no syntax, undefined-name, or selected control-flow errors.

A broader Ruff pass flagged immutable default objects and immediately consumed test callbacks. Inspection did not establish defects from those warnings. This report excludes style warnings and does not repeat earlier findings without a remaining source-level cause.

The repair did not run model downloads, GPU calculations, compilation, or cluster jobs. Those items remain deployment tests.

This is a best-effort repair, not proof that no other bugs exist.

## Terms and priority

SVD factors a matrix into singular values and vectors. FP16, FP32, and FP64 denote 16-, 32-, and 64-bit floating-point numbers. MP describes expected spectra of specified random matrices.

An activation is a signal produced by a model layer. Covariance measures how activation values vary together. A lesion removes selected singular components from model weights.

Lanczos estimates a spectrum through repeated matrix-vector products. FARMS pools spectra from sampled fixed-shape matrix windows. KS measures the largest difference between two cumulative distributions.

P1 means a broad risk of unintended model or data damage. P2 means incorrect scientific output or a broken supported execution path. P3 means a narrower library or numerical boundary defect.

All findings have status FIXED. P2 findings are COA-V2-001 through COA-V2-008. P3 findings are COA-V2-009 through COA-V2-012.

## COA-V2-001: Spacing preparation bypasses the MP availability guard

The relevant code is `run_experiments.py:409-445,487-518`, `rmt/factory.py:299-326,427-465`, and `rmt/mp.py:223-239`.

`dispatch_mp_fit` qualifies numerical rank and returns unavailable results for insufficient spectra. The runner then calls `fit_marchenko_pastur` directly for spacing when the selected method is analytic, KDE, or FARMS. This second call ignores the qualified spectrum and its unavailable status.

The call also runs before the runner computes `needs_spacing`. Thus, disabling all three spacing controls does not disable this extra fit. An otherwise usable matrix can abort analysis in an analysis stage that the user disabled.

A source-derived case uses a 64-by-64 diagonal weight with three positive singular values and 61 zeros. Supply matching factors with exact zeros and select raw analytic MP. The dispatcher returns unavailable, but the spacing fit raises `at least four positive eigenvalues are required for scale fitting`.

If a solver returns tiny positive values for the null modes, the second fit instead treats those values as a noise spectrum. This also bypasses the rank repair described by COA-006 in the earlier report. The exact solver output requires a backend test.

Repair and acceptance steps:

1. Resolve whether spacing is requested before preparing its fit.
2. Use the qualified raw spectrum and an explicit availability result for the separate spacing fit.
3. Add exact-null and roundoff-null cases with spacing enabled and disabled.
4. Make sure that insufficient spacing data does not discard other matrix measurements.

## COA-V2-002: FP16 attention overflows before applying its scale

The relevant code is `models/modules.py:99-137` and `pipelines/trainer.py:239-242,354-358`.

Attention first computes the query-key product in the active dtype. It then divides that result by the square root of the head dimension. The later FP32 conversion for softmax cannot recover values that already became infinity.

For a head dimension of 64, query and key entries of 40 give an unscaled dot product of 102400. FP16 cannot represent that result. The intended scaled score is 12800, which FP16 can represent.

If this score reaches an allowed attention position, softmax receives infinity instead of the finite scaled score. The forward calculation can produce a nonfinite loss before gradient scaling can help. This affects the supported `--amp-dtype float16` path.

Repair and acceptance steps:

1. Compute attention scores and normalization through a numerically safe implementation.
2. Test finite FP16 queries and keys whose unscaled product exceeds the FP16 range.
3. Compare outputs and gradients with an FP32 reference.
4. Preserve causal masking, padding behavior, dropout, and grouped-query attention.

## COA-V2-003: Energy-match status changes when weight units change

The relevant code is `pipelines/spectral_lesioning.py:120-152,188-245`.

The energy selector computes a positive target before reconstruction. The status calculation then divides its absolute error by `max(target_energy, np.finfo(float).eps)`. That absolute floor replaces small physical targets with an unrelated denominator.

Consider `diag([4, 3, 2, 1])`, a top lesion, energy mode, and fraction 0.05. The target energy is 1.5, but the nearest permitted nonempty top lesion removes 16. The true relative error is about 9.67, so this is not a five-percent match.

Multiply the weight by `1e-10`. Selection and the true relative error stay the same, but the recorded error becomes about 0.000653. The code now reports `matched_5pct` and `target_reached=True`.

The runner includes this false status in `all_energy_matches_reached`. This defect can make an unmatched intervention appear suitable for a matched-energy comparison. The example needs no overflow or underflow.

Repair and acceptance steps:

1. Compute relative error in normalized energy units or divide by the already qualified positive target.
2. Preserve explicit zero-energy and infeasible-target behavior.
3. Test that selected indices, relative error, and match status stay unchanged under finite weight rescaling.

## COA-V2-004: The rank-ordered tail fit understates KS distance

The relevant code is `rmt/tail.py:311-360` and `run_experiments.py:697-707`.

`rank_ordered_mle` compares only one side of each empirical distribution jump with the fitted survival function. A KS distance requires both sides. The returned `ks_D` therefore understates the actual distance in some samples.

For `[1, 2, 4]` with `tail_fraction=1.0`, the fitted density exponent is `1 + log(3) / log(4)`. The current formula gives about 0.0893164. The two-sided KS distance is 1/3 because the fitted cumulative distribution is zero at the smallest observation.

The paper1 preset selects this estimator. The runner publishes its value as `tail_ks` without a different statistical label. Thus, output can show a better tail fit than the stated statistic supports.

Repair and acceptance steps:

1. Compare both empirical jump limits with the fitted cumulative distribution.
2. Preserve the selected exponent, cutoff, and density-exponent convention.
3. Test the three-value example against an independent KS calculation.
4. Add repeated-cutoff and longer-tail cases.

## COA-V2-005: Lanczos stopping uses absolute floors in spectral units

The relevant code is `rmt/lanczos_stieltjes.py:384-386,424-464,566-572,861-864,1011-1015,1055-1062`.

The factor adapter computes a tolerance from the covariance trace, then floors it at FP64 epsilon. The modified-tail search applies the same absolute floor. These tolerances have spectral units, but epsilon does not.

For a sufficiently small nonzero factor, recurrence entries are much smaller than this floor. The adaptive test can accept its first eligible window against the initial zero statistics. The backward tail search also accepts entries that differ substantially relative to the spectrum.

A source-derived case compares a fixed nonsingular 64-by-128 factor with the same factor multiplied by `1e-10`. Use default adaptive behavior, the same probe seed, and no explicit tolerance. The smaller factor enters the absolute-floor branch although it describes the same relative spectrum.

The edge-spread denominator also used an absolute epsilon floor. The repair uses spectral scale and the smallest positive FP64 value. A regression compares returned edges and stopping counts after rescaling.

Repair and acceptance steps:

1. Resolve automatic tolerances in normalized operator units.
2. Use scale-relative denominators for convergence diagnostics.
3. Preserve the documented units of explicitly supplied tolerances.
4. Compare normalized edges, stopping counts, and convergence status across finite rescalings.

## COA-V2-006: Asset verification ignores added files

The relevant code is `scripts/download_assets.py:285-334,337-368` and `run_hpc.slurm:131`.

`_verify_manifest` examines only files listed in the manifest. It never compares that list with the current asset tree. Adding an unrecorded file leaves every examined hash unchanged and still permits successful verification.

For example, start with a manifest that does not contain `data/tokenizers/gpt2/added_tokens.json`. Add that file without changing the recorded files, then request `--verify-only`. The verifier has no branch that detects this addition.

Tokenizer loaders can consume additional files in a snapshot directory. A passing verification result therefore does not establish that the copied asset tree matches the recorded tree. The partial-staging verifier already enforces membership, but the deployment verifier does not.

Repair and acceptance steps:

1. Compare recorded and current file membership before reporting successful verification.
2. Define explicit exclusions for the manifest and any permitted transient files.
3. Test added, removed, changed, duplicated, and out-of-root records in temporary asset trees.

## COA-V2-007: Failed restaging destroys the last valid asset set

The relevant code is `scripts/download_assets.py:22-25,117-127,132-138,167-203,404-455`.

Restaging replaces live asset files before it writes the new manifest. The tokenizer path also deletes the previous snapshot before replacing it. No rollback restores the prior files if a later operation fails.

Start with a valid asset tree, then restage all assets with a different synthetic seed. If revision lookup or a later download fails, the synthetic files already differ from the old manifest. The previously valid offline tree now fails its own integrity test.

A failed manifest write can also truncate the previous manifest because `_write_json` opens it directly for writing. This defect concerns preservation of existing data during a failed update. It does not require two concurrent staging processes.

Repair and acceptance steps:

1. Build replacement assets and their manifest in a separate release tree.
2. Publish the new release only after every selected asset passes its integrity tests.
3. Preserve the previous release on download, serialization, replacement, and manifest-write failures.
4. Prevent concurrent staging processes from publishing mixed releases.

## COA-V2-008: Failed hook registration leaves hooks attached

The relevant code is `pipelines/activation_extractor.py:231-248,359-380`.

A hook is a callback that runs during a module call. `ActivationExtractor.__enter__` registers hooks one module at a time without a cleanup guard. If selection or registration raises after the first hook, context entry fails before `__exit__` can run.

A source-derived case uses two linear modules and a filter that selects the first, then raises on the second. The first module retains its callback. `compute_activation_covariances` restores training flags but does not remove these partially registered hooks.

Later model calls can continue to allocate and update hidden covariance buffers. This also changes the state seen by a later capture attempt. The failure path differs from an exception inside an already entered context, which the existing cleanup handles.

Repair and acceptance steps:

1. Remove all registered hooks if context entry fails.
2. Preserve the original exception after cleanup.
3. Test a failing filter and a failing registration after one successful registration.
4. Make sure that later model calls do not update abandoned accumulators.

## COA-V2-009: Architecture search can return an invalid RoPE head size

The relevant code is `models/chinchilla_scaling.py:240-289` and `models/transformer.py:38-44`.

RoPE encodes token positions through rotations. The model requires an even dimension for each RoPE head. Architecture search requires width divisibility by the head count, but it does not require an even quotient.

A source-derived call uses `suggest_architecture(8484, vocab_size=512, width_multiple=12, max_layers=1, max_parameters=9000)`. The exact candidate has width 12, four heads, and 8484 estimated parameters. Its head dimension is three, so the returned architecture fails model construction.

The default width multiple of 64 avoids this case. The defect affects the accepted library search bounds. It is separate from the earlier repair for embedding-dominated parameter caps.

Repair and acceptance steps:

1. Filter head choices through every applicable model-dimension constraint.
2. Test the width-12 case and other nondefault width multiples.
3. Make sure that every returned candidate constructs a valid `TransformerConfig`.

## COA-V2-010: Alternate real-only entry points still discard imaginary data

The relevant code is `pipelines/spectral_lesioning.py:39-62,157-185`, `rmt/farms_aspect_ratio.py:25-31`, and `rmt/lanczos_stieltjes.py:36-59,67-89,1049-1052`.

The shared SVD container and tensor bridge reject complex weights. Several alternate entry points cast directly to FP64 instead. This cast discards the imaginary component before the repaired guards can inspect it.

For example, the count-mode lesion path accepts a tensor equal to `1j * eye(4)`. Its SVD helper analyzes an all-zero real matrix and returns a zero lesion result. `farms_spectrum` likewise turns the same NumPy input into a zero spectrum.

The production dispatcher rejects complex weights before these calls. Direct library callers remain exposed. This is a remaining entry-point gap in the earlier COA-014 repair, not a claim that the repaired container still accepts complex input.

Repair and acceptance steps:

1. Reject unsupported complex input before any real cast.
2. Apply the same policy to dense operator, factor, FARMS, and lesion entry points.
3. Test purely imaginary and mixed complex inputs through each public path.

## COA-V2-011: MP density and energy fractions still lose scale invariance

The relevant code is `rmt/mp.py:119-147,937-969,1146-1173`.

The quantile solver now uses normalized units, but the density still multiplies dimensional support differences. Its denominator also multiplies dimensional variance and eigenvalue values. Both products can underflow or overflow before their ratio cancels the scale.

At aspect ratio one, evaluate the density at `x=variance=1e-200`. The radicand and denominator both underflow to zero, although the expected density is finite. At `x=variance=1e200`, both products overflow, although the expected density is again finite.

The cumulative-distribution path integrates this same density. `small_sv_deviation` also squares unnormalized singular values to compute an energy fraction. Fixing quantile brackets and the scalar module did not repair these separate calculations.

Repair and acceptance steps:

1. Evaluate density and cumulative probability in normalized units before rescaling density output.
2. Normalize singular magnitudes before computing dimensionless energy fractions.
3. Test density scaling, cumulative probabilities, and lower-tail fractions over large and small finite scales.
4. Make sure that representable reference results remain finite.

## COA-V2-012: The constant-tail transform is undefined at a regular zero argument

The relevant code is `rmt/lanczos_stieltjes.py:669-704`.

A Stieltjes transform summarizes a spectrum as a function of position. The constant-tail formula divides by `z` without handling its removable zero singularity. For a spectrum bounded away from zero, the true transform at zero is finite.

Use `extended_stieltjes_transform(0j, [2.0], [], tail_alpha=2.0, tail_beta=1.0)`. The support is `[1, 9]`, so zero lies outside it. The formula produces zero divided by zero instead of the finite value 1/3.

Density plots use a positive imaginary offset and avoid the exact-zero case. Direct transform callers do not have that protection. Nearby values also need tests for cancellation in the same formula.

Repair and acceptance steps:

1. Handle the finite zero-argument limit or use an equivalent stable expression.
2. Preserve genuine divergence when zero belongs to the relevant spectral support or point mass.
3. Test zero and nearby complex arguments against direct inverse calculations.

## Risks that need separate evidence

The following items are not included in the 12 confirmed source-level defects. Their practical effect depends on workload, numerical method policy, or hardware. Do not close them through assumptions:

- Test covariance memory at production model sizes. The compute runner captures all selected covariances and retains training state during analysis.
- Test whether TF32 products satisfy the intended covariance precision contract. Disabling autocast does not itself disable TF32 multiplication.
- Define lesion behavior when a selection boundary splits equal singular values. Different valid singular bases can produce different interventions.
- Decide whether nonconverged Lanczos estimates remain available. The runner records convergence separately but still consumes their support and pole estimates.
- Exercise status finalization for dataset and plotting failures. These stages sit outside the per-cell exception handler.

## Repair validation

Keep this project isolated from `remote-sk-random-matrix-ml-esd-fixed`. Both projects expose an incompatible package named `rmt`.

The focused command `python -m pytest -q -p no:cacheprovider tests/test_bug_report_v2.py` passed 12 tests. The full command `python -m pytest -q -p no:cacheprovider` passed 92 tests. Both commands ran from this project root.

The repair tests cover FP16 attention, energy matching, two-sided KS, Lanczos scaling, asset membership, and failed restaging. They also cover hook cleanup, RoPE dimensions, complex input, MP scaling, and the zero transform.

`bash -n run_hpc.slurm` passed. CUDA, BF16, compilation, real downloads, and cluster execution remain deployment tests.
