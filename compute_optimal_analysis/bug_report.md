# Static bug report — compute_optimal_analysis

## Review and handoff

- **Baseline:** commit `0bd4a13` (`before another pass`). All 16 findings are now **FIXED** in the working tree. The descriptions below are retained as historical reproducers.
- **Scope:** maintained `models/`, `pipelines/`, `rmt/`, `run_experiments.py`, asset downloader, dependency declarations, launcher, and test coverage. The sibling project's incompatible `rmt` package was reviewed separately.
- **Method:** source/control-flow review, Ruff inspection, existing tests, and small offline CPU reproducers. This is a best-effort audit, not a guarantee that every possible bug has been found. Cosmetic lint warnings are not listed as bugs.
- **Pre-repair test baseline:** `python -m pytest -q -p no:cacheprovider` -> **80 passed**. `bash -n run_hpc.slurm` passed; the focused reproducers exposed gaps in that baseline suite.
- **Validation environment:** Python 3.13.2, NumPy 2.4.4, SciPy 1.17.1, CPU Torch 2.13.0, Matplotlib 3.10.8, pytest 8.3.3. This is **not** the pinned standalone environment. CUDA, compilation, real HF downloads, and SLURM execution were not available/validated.
- Paths/line numbers are relative to this directory and refer to the baseline. Reproduce in a separate process launched from this project root.

**Priority:** P1 = potential invalid training/model state; P2 = incorrect scientific output or broken supported path; P3 = narrower library/numerical edge case. CPU observations below are reproducible examples, not universal backend-specific constants.

## Finding index

| ID | Priority | Finding | Status |
|---|---|---|---|
| COA-001 | P1 | Safe gradient-norm check is followed by overflowing FP32 clipping | FIXED |
| COA-002 | P2 | Fractional token-progress warmup can exceed the peak learning rate | FIXED |
| COA-003 | P2 | Bounded-Pareto log-likelihood uses an overwritten variable | FIXED |
| COA-004 | P2 | Constant-tail poles and returned Stieltjes transform can describe different operators | FIXED |
| COA-005 | P2 | FARMS MP admissibility is checked against the wrong spectrum | FIXED |
| COA-006 | P2 | SVD roundoff turns null modes into an apparently valid MP bulk | FIXED |
| COA-007 | P2 | Float32 SVD precision is lost before singular-subspace qualification | FIXED |
| COA-008 | P2 | Direct covariance-overlap APIs accept rotation-ambiguous eigenspaces | FIXED |
| COA-009 | P2 | Presets enable lesions after tranche validation has already run | FIXED |
| COA-010 | P2 | Architecture search can reject feasible parameter caps | FIXED |
| COA-011 | P2 | Caller-supplied lesion cache silently reuses factors of changed weights | FIXED |
| COA-012 | P2 | MP quantiles fail under simple changes of spectral units | FIXED |
| COA-013 | P2 | Modified-MP optimization has scale-dependent infeasible bounds | FIXED |
| COA-014 | P3 | Real casts silently analyze the wrong matrix for complex input | FIXED |
| COA-015 | P3 | Scale-invariant scalar metrics overflow/underflow before normalization | FIXED |
| COA-016 | P2 | Porter–Thomas output is emitted for nonidentifiable weight bases | FIXED |

## Repair validation

The repair pass uses FP64-qualified clipping, bounded token warmup, one constant-tail operator, prepared-domain MP gates, factorization-precision rank/basis checks, scale-normalized MP/scalar calculations, content-bound lesion caches, post-preset validation, and explicit real-only input checks. Porter–Thomas output is availability-qualified.

Validation after repair: `python -m pytest -q -p no:cacheprovider` -> **80 passed**; `bash -n run_hpc.slurm` passed. Focused reproductions for bounded Pareto likelihood, constant-tail pole residues, low-rank cached SVDs, MP quantile scaling, modified-MP scaling, complex rejection, and capped architecture search also passed. CUDA and SLURM remain deployment gates.

## Findings

### COA-001 — Gradient clipping can zero finite gradients and still count a successful update

**Locations:** `pipelines/trainer.py:354–398`, especially `clip_grad_norm_` at 370.

The trainer correctly computes `raw_gradient_norm` in float64, but then calls `torch.nn.utils.clip_grad_norm_` on the original FP32 gradients. That function recomputes its norm in the gradient dtype. Finite gradients with entries around `1e20` pass the double-precision check but overflow this second norm to infinity. Clipping then multiplies the gradients by zero. The optimizer/scaler path nevertheless increments `global_step` and `processed_train_tokens`, and advances the scheduler.

**Confirmed reproducer:** a module with FP32 parameter `p = zeros(2)` and scalar loss `(p * 1e20).sum() + 1`, trained for one CPU step with zero weight decay and no warmup. Both parameter entries remain zero; history records `gradient_norm=inf`, `optimizer_update=1`, and one successful training target. Ordinary finite clipping would produce a nonzero update.

**Fix:** use the already computed safe norm to derive and apply the clipping coefficient, rather than recomputing an unsafe norm. Validate the applied gradients and recorded norm before declaring success. Preserve scaler-overflow behavior.

**Regression:** cover ordinary gradients, finite FP32 gradients whose sum of squares overflows FP32, actual nonfinite gradients, and successful-target/scheduler accounting.

### COA-002 — Token-based fractional warmup overshoots the configured peak LR

**Locations:** `pipelines/trainer.py:77–93`, `268–281`.

Warmup returns `(step + 1) / warmup_steps` whenever `step < warmup_steps`. That was bounded for integer update indices, but token-budget scheduling passes a **fractional** step. For example:

```python
cosine_warmup_multiplier(1.5, 10, 2, 0.1)  # returns 1.25
```

Partial/variable-size batches can therefore apply more than `learning_rate`, then abruptly drop to 1 at the warmup boundary. The function also no longer meets its stated continuous-schedule contract.

**Fix:** define warmup consistently for continuous token progress, with a peak bound and a continuous transition to decay. Do not change skipped-update accounting.

**Regression:** sample fractional progress throughout warmup and across its boundary; assert the multiplier never exceeds 1 and the training history records the intended applied LR.

### COA-003 — Wrong bounded-tail log-likelihood despite a correct fitted exponent

**Location:** `rmt/tail.py:45–109`, particularly 86 and 96–101.

`denominator` initially means `sum(log(tail / xmin))`. In the bounded-Pareto Fisher-information branch, it is reassigned to `-expm1(-beta * log(xmax / xmin))`. The final likelihood still multiplies `alpha * denominator`, now using the normalizer instead of the log-sum. This affects bounded fits when `abs(beta * log(xmax/xmin)) >= 1e-3`; the near-zero expansion branch does not overwrite it.

**Confirmed reproducer:**

```python
u = (np.arange(1000) + 0.5) / 1000
x = (1 - u * (1 - 5.0**-2))**-0.5
fit = fixed_cutoff_mle(x, xmin=1, xmax=5)
```

The exponent is about `3.00000349`, but `log_likelihood` is about **+731.09** instead of the direct normalized-density sum **-564.85**. CSN delegates to the same helper.

**Fix:** keep separate variables for the sufficient statistic and normalization/Fisher-information terms.

**Regression:** compare the returned likelihood with the sum of the fitted bounded log-density, including both Fisher-information branches, nonunit `xmin`, and the unbounded case.

### COA-004 — Default constant-tail pole extraction disagrees with its own transform

**Locations:** `rmt/lanczos_stieltjes.py:542–635`, `718–778`, `889–920`, `951–959`, `990–998`.

With `pole_method="constant_tail"` and no explicit `tail_window`, `finite_section_poles` cuts the original Cholesky recurrence using its own default fixed suffix length. The returned `stieltjes()` instead uses `reference_modified_cholesky`'s independently selected/backward-expanded tail and prefix. These can be different operators. This is not just the distinction between finite Ritz poles and an asymptotic density: it occurs in the **constant-tail** mode itself, with one probe.

**Confirmed reproducer:**

```python
w = np.random.default_rng(0).normal(size=(64, 128))
w[0] *= 3
r = detect_spikes_from_factor(
    w, steps=20, n_probes=1, adaptive=True,
    pole_method="constant_tail", threshold_c=0.1,
)
```

The first pole is about `8.7305294` with residue `0.000442658`. But `1e-8 * imag(r.stieltjes(pole + 1e-8j))` is about `2.77e-13`, not that residue; the other reported pole also fails. Explicit `tail_window=5` happened to align the constructions and passed this check, explaining why fixed-window tests miss the default-path bug.

**Fix:** use one resolved constant-tail recurrence/prefix for support, finite-section poles, residues, and transform evaluation. Keep the separately named finite-Ritz method distinct.

**Regression:** verify pole/resolvent consistency for omitted and explicit tail windows, adaptive and nonadaptive runs, and single/multiple probes. Use a one-probe case to avoid confusing representative-pole semantics with averaged measures.

### COA-005 — MP dispatcher validates the raw source rather than the selected FARMS sample

**Locations:** `rmt/factory.py:333–342`, `371–431`; `rmt/mp.py:218–233`, `526–532`.

The analytic/KDE minimum-positive-count gate is applied to the **raw** matrix SVD before preparing FARMS. The actual fit later consumes `prepared.eigenvalues`, whose count can be larger or smaller.

**Confirmed examples:**

- An `8x8` Gaussian matrix with `farms_window_size=2`, `farms_row_windows=1`, `farms_column_windows=1`, and analytic MP passes the raw gate, then crashes with `at least four finite nonnegative eigenvalues are required` on the two-value prepared sample.
- `diag([3, 2, 1])`, `window_size=2`, and two row/column windows produces **six positive pooled observations**, but analytic dispatch reports unavailable because the raw matrix has only three positives. The unavailable result also reports raw rather than requested-domain geometry.

**Fix:** resolve the fit domain first, then apply the method's availability criteria to that exact sample and geometry. Return a structured unavailable result for undersized prepared samples instead of leaking fit exceptions. Apply the same policy to the FARMS-specific branch.

**Regression:** cover pooled samples both larger and smaller than the raw sample, all-zero sampled windows, and supplied versus internally generated `PreparedSpectrum`.

### COA-006 — Exact positivity checks mistake numerical null modes for a fitted bulk

**Locations:** `rmt/factory.py:333–342`; `rmt/mp.py:218–264`; `pipelines/activation_extractor.py:307–322`.

MP availability and `rank_deficiency` use exact `> 0` / `== 0`. A full-vector SVD can return roundoff-sized positive values for mathematical zeros, whereas a singular-values-only SVD returns exact zeros. The same matrix then changes from unavailable to a valid-looking near-zero MP fit merely by supplying cached factors.

**Confirmed reproducer:** `w = diag([3, 2, 1] + [0]*61)`, raw analytic MP. Without supplied factors, dispatch reports insufficient positive eigenvalues and rank deficiency 61. With `compute_tensor_svd(torch.tensor(w), backend="cpu")`, it reports 64 positive fit values, rank deficiency 0, and variance around `2.69e-33`. This is the factor path used by `analyze_model`.

**Fix:** qualify the numerical rank and resolvable bulk using factorization precision and spectral scale. Retain raw tiny singular values as data, but do not turn unresolved null modes into a trustworthy noise fit. Avoid a blanket absolute cutoff that would destroy genuine small-singular-value measurements.

**Regression:** cached/uncached and values-only/full-factor paths must agree on availability and rank for low-rank matrices; also preserve valid ill-conditioned full-rank examples above their precision limit.

### COA-007 — Float32 factorization provenance disappears before overlap qualification

**Locations:** `pipelines/activation_extractor.py:281–323`; `rmt/svd_result.py:40–43`; `rmt/overlap.py:51–59`.

`--analysis-dtype float32` performs an FP32 SVD, but `SVDResult` immediately casts every factor to float64 and stores no factorization dtype. `_weight_vectors_identifiable` then uses float64 epsilon. Numerically unresolved FP32 singular clusters can thus be treated as individually identifiable, even though the CSV's requested dtype alone cannot repair the calculation.

**Confirmed reproducer:** generate two seeded `16x16` orthogonal bases, construct singular values `[5, 5, linspace(4, 1, 14)]`, cast the weight to FP32, and request an FP32 SVD. The leading returned values were `5.00000143` and `5.00000095`; overlap status was `available`, despite their gap being below FP32 resolution (`eps32 * 16 * smax ≈ 9.54e-6`).

**Fix:** retain actual factorization precision in the pure container and use it for null/cluster qualification and related numerical-rank decisions. Casting storage to double must not upgrade the precision contract.

**Regression:** use an isolated repeated/near-repeated pair, not only an entirely repeated spectrum; test FP32 and FP64 separately.

### COA-008 — Direct maximum-overlap/coincidence APIs do not reject repeated activation modes

**Locations:** `rmt/overlap.py:316–356`, `367–413`.

The two direct covariance-based APIs check rank, indefiniteness, and weight singular degeneracy, but not repeated **positive covariance eigenvalues**. Their maxima and rank coincidences depend on the arbitrary eigenbasis returned by `eigh`.

**Confirmed example:** unique singular values `diag([3, 2, 1, 0.5])` against `C = eye(4)`. Supplying two equally valid eigensolver results—identity and a normalized Hadamard basis—changes per-vector squared maximum overlap from 1 to 0.25, and maximum cosine with the top activation eigenvector from 1 to 0.5. Both results say `available`.

**Fix:** return unavailable for individual-vector claims in unresolved covariance clusters, or explicitly replace them with cluster-projector statistics. Preserve the cluster-safe subspace handling already implemented in `dual_end_alignment`.

**Regression:** rotate only a repeated positive covariance eigenspace while holding the matrix fixed; individual-vector availability or invariant cluster statistics must not depend on that rotation.

### COA-009 — Invalid lesion tranches can survive preflight and fail only after training

**Locations:** `run_experiments.py:1170–1207`, `1068–1076`.

Tranche validation is conditional on `args.run_spectral_lesioning` **before** paper/Golden presets are applied. The parser default is false, but the default Golden preset and paper1 preset subsequently enable lesions.

**Confirmed reproducer:** parse `['--device', 'cpu', '--lesion-tranches', 'invalid']`, then call `_resolve_runtime_configuration`. It returns successfully with lesions enabled and the invalid tranche unchanged. Execution performs training and spectral analysis before the second check raises.

**Fix:** apply presets before all dependent validation, then validate the resolved configuration once before output acquisition/data/model work.

**Regression:** invalid and empty tranches must fail preflight for implicit preset-enabled lesions as well as explicit `--run-spectral-lesioning`; preserve explicit `--no-run-spectral-lesioning` behavior.

### COA-010 — Architecture heuristic can exclude every feasible capped architecture

**Location:** `models/chinchilla_scaling.py:234–278`.

Candidate widths are restricted to a narrow neighborhood of `sqrt(target / (16 * layers))`, which ignores the vocabulary embedding term. For embedding-dominated models, that neighborhood can lie entirely above a hard parameter cap even when a valid smaller-width architecture exists.

**Confirmed reproducer:** `suggest_architecture(150e6, vocab_size=1_000_000, max_parameters=150e6)` raises "no valid architecture". Yet width 64, two layers, and four heads is a valid candidate with **64,131,392** parameters, below the cap. The CLI exposes vocabulary and parameter-cap controls. Separately, `max_layers=1` passes validation but the even-layer loop is empty.

**Fix:** include embedding cost in width estimation and always consider feasible lower-bound widths under the cap; either support one layer or reject that search bound explicitly. Do not relax the cap silently.

**Regression:** embedding-dominated vocabularies, valid tight caps, genuinely infeasible caps, and the smallest accepted layer bound.

### COA-011 — Reusing a lesion factor cache after a weight update silently reconstructs old weights

**Location:** `pipelines/spectral_lesioning.py:336–387`, especially 368–375.

The public `factor_cache` is keyed only by parameter name. No model identity, content/version, shape, or precision contract is checked. A cache reused across models or checkpoints silently replaces retained singular components with those of an earlier matrix.

**Confirmed reproducer:** cache a top-25% lesion of `diag([4,3,2,1])`; exit the context, double the live weight, and invoke the same context with the same cache. The lesion contains `diag([0,3,2,1])` instead of `diag([0,6,4,2])`. The context restores the current weight afterward, but evaluation inside it measures the wrong intervention.

The production benchmark currently creates a fresh cache internally, so the demonstrated exposure is the public reusable-cache path.

**Fix:** bind entries to a verified pristine weight/model and analysis contract, recomputing or rejecting stale entries. Alternatively make cache ownership explicitly internal to a single pristine benchmark.

**Regression:** changed weights, another model with identical parameter names, and repeated independent lesions of unchanged weights.

### COA-012 — MP quantile root finding is not scale invariant

**Locations:** `rmt/mp.py:177–192`, `932–941`, and quantile consumers such as `small_sv_deviation`.

The quantile bracket uses `eps * max(1, upper)` and absolute `xtol=1e-11`, even when the entire eigenvalue support is much smaller. The bracket can leave or reverse the support, and the root tolerance can exceed its width.

**Confirmed examples for `mp_median(100,100,sigma)/sigma`:** about `8.079455` at sigma 1; `8.178388` at `1e-5`; a bracketing exception at `1e-8`; and NaN at `1e-10`. This ratio must be constant for the same aspect ratio.

**Fix:** solve quantiles in a unit-variance dimensionless domain and rescale afterward, or use genuinely scale-relative brackets/tolerances. Never accept roots outside support.

**Regression:** multiply singular spectra and sigma by several powers of ten; median ratios and lower-tail counts should remain invariant and finite.

### COA-013 — Modified-MP fit bounds depend on arbitrary physical units

**Location:** `rmt/mp.py:349–420`, especially 380–402.

The fit optimizes dimensional amplitude and support width in logarithmic coordinates with fixed bounds `[-40,40]`, while initial amplitude scales approximately as inverse squared spectral scale. A valid rescaled spectrum can put the initial point outside those bounds. Fixed absolute floors further change the small-scale objective.

**Confirmed reproducer:** use singular values of a seeded `100x100` Gaussian matrix. `fit_modified_mp_singular(s)` succeeds; `fit_modified_mp_singular(s * 1e-8)` raises `Initial guess is outside of provided bounds`. The Thamm dispatcher does not turn this into a structured unavailable result, so it can abort a cell.

**Fix:** normalize the spectrum before density construction and optimization, then transform support and density amplitude back into original units. Validate optimizer success/fit quality and propagate failures explicitly.

**Regression:** fit scaled copies of the same nondegenerate spectrum and assert equivalent rescaled support/density, including the CLI Thamm adapter.

### COA-014 — Complex matrices are reduced to their real parts instead of rejected

**Locations:** `rmt/svd_result.py:10–18`, `40–43`; similar casts in `rmt/factory.py:73–76` and weight-taking numerical helpers.

`compute_svd(1j * np.eye(4)).s` returns four zeros instead of four ones, with only a cast warning. Complex ensembles are available in this package, making accidental passage to general matrix helpers plausible. The returned valid-looking SVD describes a different matrix.

**Fix:** if these APIs are deliberately real-only, reject complex input before casting. Otherwise preserve complex factors and use conjugate-transpose algebra consistently. Do not silently erase the imaginary component.

**Regression:** complex matrices with purely imaginary and mixed entries must either produce correct singular values/reconstruction or a clear unsupported-input exception.

### COA-015 — Scale-invariant scalars square before normalizing

**Locations:** `rmt/scalars.py:42–49`, `91–101`, `133–145`, `350–358`, `377–395`.

Finite supported float64 values can overflow/underflow during `s**2`, before normalization cancels the scale. For `[3,2]`, stable rank is `1.444444` and entropy is `0.617242`. Scaling by `1e200` makes stable rank raise and entropy return `-0.0`; scaling by `1e-200` yields NaN rank and zero entropy. The entropy result falsely resembles a rank-one or zero matrix. Other energy-fraction/contribution helpers share the same pattern.

**Fix:** normalize by the largest magnitude before squaring for dimensionless metrics; use per-row scaling for row entropy and preserve the explicit all-zero convention.

**Regression:** stable rank, entropy, energy fractions, and summed per-decile contributions must be invariant across large/small finite rescalings.

### COA-016 — Porter–Thomas calibration bypasses singular-basis availability checks

**Location:** `run_experiments.py:588–596`; `rmt/scalars.py:281–339`.

The runner sends all right singular vectors into pooled Porter–Thomas calibration solely based on vector count. Null/repeated singular subspaces are rejected by overlap qualification, but the same ambiguous vectors are still published as PT randomness evidence. For an identity matrix, identity and arbitrary orthogonal factors are equally valid SVD bases, with very different localization/PT behavior.

**Fix:** qualify the weight eigenspaces before making matrix-level PT claims, marking unresolved bases unavailable or using a deliberately invariant cluster diagnostic. The explicit-vector scalar APIs may still measure the vectors the caller supplies; the missing guard is in matrix-level aggregation.

**Regression:** inject two valid SVD bases of the same repeated-spectrum matrix into `analyze_model`; PT availability or a replacement invariant diagnostic must not depend on the solver's arbitrary basis.

## Original repair order and acceptance (completed)

1. Fix COA-001/002 before trusting new training budgets or learning curves.
2. Fix likelihood/Lanczos/domain/rank/precision findings before generating scientific comparisons.
3. Add a focused regression for each repaired ID; the current passing suite is not sufficient evidence of remediation.
4. Keep the sibling `rmt` package isolated. Its related fixes are not API-compatible drop-ins.
5. Re-run this project's suite and shell syntax check. Validate the pinned or inventoried cluster environment, CUDA SVD drivers, BF16, compilation, and real offline assets separately before an HPC launch.
6. Update individual status entries only after the corresponding reproducer and regression pass; retain unresolved findings rather than replacing this report with a blanket success claim.
