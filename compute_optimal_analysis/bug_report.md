# Static-analysis bug report — compute_optimal_analysis

## Scope and verification

Reviewed the **current working tree**, including the pre-existing uncommitted repairs and `tests/test_bugfix_v4.py`, against base commit `c359c9c`. This is a fresh report of remaining issues, not a replay of deleted historical reports. The findings are retained below as an audit trail; implementation fixes are now included in the working tree.

Scope: `run_experiments.py`, `models/`, `pipelines/`, the numerical `rmt/` modules, asset staging, the SLURM launcher, and related tests/contracts. Paths and line numbers below are relative to this directory and refer to the reviewed tree.

Supplemental verification, in a separate process from the sibling project:

- After remediation, `python -m pytest -q -p no:cacheprovider`: **99 passed**.
- Ruff undefined-name/export/local-variable checks (`F821,F822,F823`): passed.
- `bash -n run_hpc.slurm`: passed.
- Small CPU-only probes reproduced the numerical and control-flow failures described below.
- Environment: Python 3.13.2, NumPy 2.4.4, SciPy 1.17.1, Torch 2.13.0+cpu. This is **not** the pinned production environment. CUDA, compilation, real training jobs, downloads, and cluster execution were not validated.

Severity: **P1** = high-impact incorrect results or failure of an advertised execution path; **P2** = conditional correctness/reliability defect. Confidence/evidence is stated per finding. A passing current suite does not cover these cases.

## Resolution status

- [x] **CO-01 — Fixed**
- [x] **CO-02 — Fixed**
- [x] **CO-03 — Fixed**
- [x] **CO-04 — Fixed**
- [x] **CO-05 — Fixed**
- [x] **CO-06 — Fixed**
- [x] **CO-07 — Fixed**
- [x] **CO-08 — Fixed**
- [x] **CO-09 — Fixed**
- [x] **CO-10 — Fixed**

## Prioritized findings

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| CO-01 | P1 | KDE MP fit is dominated by the square-matrix hard-edge singularity | CPU reproduction |
| CO-02 | P1 | Non-finite-gradient guard prevents GradScaler overflow recovery | Static trace + enabled CPU-scaler reproduction |
| CO-03 | P2 | Token-budget training uses an epoch-capped learning-rate horizon | CPU reproduction |
| CO-04 | P2 | Exactly three bulk levels crash gap-ratio analysis | CPU reproduction |
| CO-05 | P2 | Standalone Lanczos detector failures abort otherwise valid analysis | CPU reproduction |
| CO-06 | P2 | Unselected collapsed allocation cells block selected-cell execution | CPU reproduction |
| CO-07 | P2 | `xmax` filtering still fits an unbounded power law | Deterministic numerical reproduction |
| CO-08 | P2 | Rank-ordered tail estimation is not scale invariant | CPU reproduction |
| CO-09 | P2 | Zero/degenerate activation covariance produces spurious alignment | CPU reproduction + linear-algebra analysis |
| CO-10 | P2 | Fresh-output ownership check is not atomic | Static concurrency trace |

## CO-01 — FIXED — Square-matrix KDE MP fits can inflate noise variance by an order of magnitude

**Locations:** `rmt/mp.py:535-568`, especially the grid at line 547 and density objective at lines 553-558; selected by the paper-3 preset in `run_experiments.py:1105-1109`.

**Trigger/root cause:** For aspect ratio `q=1`, the MP eigenvalue density diverges as `lambda -> 0+`. The KDE grid starts at machine epsilon whenever the empirical minimum minus bandwidth is negative. The empirical triangular KDE is finite there, but the objective compares it with the **unsmoothed divergent MP density** and assigns that point a positive weight. This point overwhelms the fit; increasing variance reduces the divergent density, driving the solution toward the upper optimization bound. Optimizer success does not make this a scientifically valid fit.

**Reproduction:**

```python
import numpy as np
from rmt.mp import fit_marchenko_pastur_kde
W = np.random.default_rng(3).normal(size=(512, 512))
lam = np.linalg.svd(W, compute_uv=False)**2 / 512
fit = fit_marchenko_pastur_kde(lam, 1.0)  # default trim_upper=0.1
print(fit.variance, fit.lambda_plus)
```

Observed variance approximately **15.83**, hence upper edge approximately **63.32**. The generating variance is 1 and the asymptotic upper edge is 4. With no trimming, the same probe returned variance approximately 19.75. This is not ordinary finite-sample error.

**Impact:** Square attention projections in the advertised paper-3 workflow get inflated MP edges, suppressed outlier counts, and invalid soft-rank/bulk diagnostics, without an unavailable-fit status.

**Fix direction:** Fit integrated probability masses/CDFs, or compare empirical and theoretical densities after applying the same kernel and boundary treatment. Account for retained-sample normalization when trimming. Do not merely replace epsilon by another arbitrary absolute cutoff. Flag boundary-seeking or otherwise unqualified fits.

**Regression tests:** Recover unit and rescaled variance for square and rectangular Gaussian factors; include a planted spike. Verify scale invariance, sensible upper edges, and that default square fits do not terminate near the variance search bound.

## CO-02 — FIXED — Recoverable scaled-gradient overflow terminates training before the scaler can recover

**Locations:** `pipelines/trainer.py:211-217,336-361`.

**Trigger/root cause:** FP16 enables GradScaler. After `unscale_`, the code clips gradients and raises on a non-finite norm **before** `scaler.step()` / `scaler.update()`. The normal scaled-gradient-overflow path therefore cannot skip the unsafe optimizer update, lower the scale, and retry. The later skipped-step accounting is unreachable for the ordinary overflow case it is supposed to handle.

**Evidence:** An otherwise ordinary CPU trainer with its scaler replaced by `torch.amp.GradScaler("cpu")`, scalar loss `weight * 1e34`, and initial scale 65536 raises `FloatingPointError` at the gradient-norm guard. The unscaled loss is finite; the scale remains 65536 and `skipped_steps` remains zero. The same control-flow ordering applies to the production CUDA FP16 scaler; CUDA itself was not tested.

**Impact:** Valid `--amp-dtype float16` jobs can abort during normal loss-scale calibration rather than recovering. Token-budget completion and skipped-update statistics are then misleadingly unavailable.

**Fix direction:** Distinguish recoverable scaler-detected overflow from an invalid unscaled loss or genuinely non-finite gradients with scaling disabled. Allow the scaler's skip/backoff path without clipping unsafe gradients or advancing successful-update counters/scheduler. Retain the fail-closed non-scaler guard already covered by `test_nonfinite_gradients_fail_before_optimizer_mutation`.

**Regression tests:** Inject one scaler-detected overflow followed by finite gradients; assert unchanged parameters and scheduler on the skipped attempt, reduced scale, correct skipped/attempted counters, and eventual exact successful-token completion. Also retain the CPU/BF16 invalid-gradient rejection tests.

## CO-03 — FIXED — Recycling the loader does not extend the learning-rate schedule

**Locations:** `pipelines/trainer.py:250-265,273-280,350-361,388-398`; runner step estimates at `run_experiments.py:877-894`.

**Trigger/root cause:** With a token budget, epochs no longer bound execution, but the scheduler is still sized by `min(len(loader) * epochs, max_steps)`. A short finite loader is recycled beyond this horizon. Almost all subsequent updates can run at the minimum LR even when an explicit larger `max_steps` was supplied.

**Reproduction:** One batch containing one next-token target; `TrainConfig(epochs=1, max_steps=5, max_train_tokens=5, warmup_steps=0, learning_rate=3e-4, device="cpu", amp_dtype="float32")`. Record LR in an optimizer step pre-hook.

Observed five successful updates with LRs `[3e-4, 3e-5, 3e-5, 3e-5, 3e-5]`: a **one-step** decay horizon, not five. Variable-size loader batches can also make the runner's full-batch step estimate underestimate actual successful updates.

**Impact:** Compute-allocation experiments can receive different, unintentionally truncated optimization schedules. Token totals can be correct while the training intervention is not.

**Fix direction:** Define the scheduler horizon from the authoritative successful-target budget, preferably scheduling directly by successful target progress. If retaining a step-based schedule, do not cap a token-budget run's explicit horizon by one finite data pass; handle partial batches and skipped updates consistently.

**Regression tests:** Budget requiring multiple data passes, an incomplete last batch, and one skipped update. Assert LR progression tracks the declared training budget rather than loader exhaustion.

## CO-04 — FIXED — Three-level gap ratio is accepted by the caller but rejected by the callee

**Locations:** `run_experiments.py:456-459`; `rmt/spacing.py:321-324`.

**Trigger/root cause:** The runner calls `r_statistic` when the fitted bulk has at least three distinct/nonconstant levels, but `r_statistic` requires at least four. Three levels are mathematically sufficient for one adjacent-gap ratio. This call is outside the unfolding exception handler.

**Reproduction:** Analyze a matrix drawn by `np.random.default_rng(7).normal(size=(4, 4))`, using raw spectrum, analytic MP fit, and BBP spikes. The fit leaves three bulk levels and analysis raises `ValueError: at least 4 finite levels are required`. A `(4, 8)` Gaussian factor with seed 2 also reproduced it.

**Impact:** One small matrix or strongly depleted bulk aborts the entire cell instead of returning the remaining diagnostics.

**Fix direction:** Make the caller/callee contracts agree; allowing three levels in `r_statistic` is mathematically valid. Below the supported count, emit an explicitly unavailable gap ratio without discarding other analysis.

**Regression tests:** Three levels with an analytically known ratio, fewer than three levels, and an end-to-end matrix analysis whose MP bulk contains exactly three levels.

## CO-05 — FIXED — Lanczos assumption failures are contained only when Lanczos is also the MP fitter

**Locations:** `rmt/factory.py:321-345,483-501`; `run_experiments.py:372-410`.

**Trigger/root cause:** `dispatch_mp_fit` converts Lanczos numerical/assumption errors to an unavailable `MPFitResult`. The independently selectable `dispatch_spike_detector` calls the same detector without equivalent handling. A finite matrix with too few distinct covariance eigenvalues can terminate Lanczos before the three iterations required for tail estimation.

**Reproduction:** Analyze `np.eye(64)` with raw aspect mode. With MP fit/detector both Lanczos, a row is returned with unavailable fit/detector status. With `mp_fit_method="analytic_mp"` and `spike_detector="lanczos_poles"`, the same matrix raises `ValueError: at least three Lanczos iterations are required`.

**Impact:** A supported method combination can abort after training even though its analytic MP/scalar results are usable. Detector failure handling depends incorrectly on which independent fitter was selected.

**Fix direction:** Give standalone detector assumption failures an unavailable result with reason/convergence metadata, consistently with the MP adapter. Keep invalid user configuration distinguishable from a scientifically unavailable fit; do not silently substitute a different detector.

**Regression tests:** Identity, low-rank, and ordinary random matrices under both Lanczos/Lanczos and analytic/Lanczos combinations. Assert consistent unavailable status and retention of unaffected metrics.

## CO-06 — FIXED — Collapses outside the requested cell subset prevent execution

**Locations:** `run_experiments.py:1139-1144,1168-1173`.

**Trigger/root cause:** Collapse rejection runs over the full manifest before `--cells` is parsed. A selected cell is rejected merely because it duplicates an unselected cell, even though no duplicate intervention is being executed.

**Reproduction:** Call the runner with `--execute --device cpu --parameter-cap 3500000 --max-train-tokens 10 --cells 0` and a fresh temporary output path. It rejects collapsed cells `[0, 1]` before checking/loading the dataset, despite only cell 0 being selected.

**Impact:** Single-cell calibration and subset execution unexpectedly require `--allow-collapsed-allocations` for interventions that are not part of the run.

**Fix direction:** Validate/resolve selected indices immediately after manifest construction. Apply execution-level collapse rejection to duplicate groups within that selection, while retaining full-manifest collapse metadata for provenance.

**Regression tests:** Select one member of a collapsed pair, both members, and a cell outside the pair. Reject only executed duplicate designs unless explicitly allowed; still reject invalid indices early.

## CO-07 — FIXED — Upper-cutoff tail fits use the wrong likelihood and CDF

**Locations:** `rmt/tail.py:60-64,88-104` (`fit_powerlaw_csn`), `246-278` (`fixed_cutoff_mle`); related filtering/comparison in `508-579`.

**Trigger/root cause:** Supplying `xmax` discards observations above the bound, but the estimator still uses the unbounded Pareto MLE and CDF. Once observations are selected on `x <= xmax`, the retained distribution is conditional on that upper bound. Its normalization depends on alpha, so the unbounded closed-form MLE is no longer valid. The reported KS and likelihood are likewise for the wrong distribution.

**Reproduction:** Exact mid-quantiles of an alpha-3 density restricted to `[1, 2]`:

```python
u = (np.arange(10000) + 0.5) / 10000
x = (1 - u * (1 - 2**-2))**(-0.5)
fixed_cutoff_mle(x, xmin=1, xmax=2)["alpha"]
fit_powerlaw_csn(x, min_tail=1000, xmax=2)["alpha"]
```

Both return approximately **4.718**, not 3.

**Impact:** Public upper-bounded tail analyses report systematically too-steep exponents. This option is not currently exposed by the production CLI, so the direct library callers are the immediate affected surface.

**Fix direction:** Fit the properly normalized density on `[xmin, xmax]` and its conditional CDF, including consistent likelihood/uncertainty calculations. Alternatively reject unsupported bounded fitting explicitly; do not silently label an unbounded fit as a valid bounded analysis. Distinguish a hard observation bound from an exponentially truncated tail model.

**Regression tests:** Known bounded-Pareto quantiles/samples, a finite cutoff near xmin, scale transformations, and convergence to the existing unbounded estimator as the upper bound grows. The sibling `rmt/tail.py` has the same issue; fix independently without merging the incompatible packages.

## CO-08 — FIXED — Absolute `allclose` tolerance suppresses valid low-scale rank tails

**Location:** `rmt/tail.py:313-315`.

**Trigger/root cause:** `rank_ordered_mle` uses `np.allclose(tail, tail[0])` to detect a constant sample. Default `atol=1e-8` declares any sufficiently small positive spectrum constant even when its relative dynamic range is large.

**Reproduction:** For `x = np.arange(1., 1001.)**(-0.5)`, `rank_ordered_mle(x, tail_fraction=1)["alpha"]` is approximately 3. For `1e-10 * x`, it is NaN. Multiplication by a constant cannot change this power-law exponent.

**Impact:** The paper-1 rank-tail path and library users can lose valid diagnostics solely because weight/eigenvalue units changed.

**Fix direction:** Detect actual or relative/log-domain degeneracy without an absolute unit-dependent tolerance. Keep truly repeated levels unavailable.

**Regression tests:** Rescale the same nondegenerate tail over several orders of magnitude; require equal alpha and appropriately scaled xmin. Retain a constant-tail rejection case.

## CO-09 — FIXED — An absent activation signal can appear perfectly aligned with bottom singular vectors

**Locations:** `rmt/overlap.py:16-27,207-231`; consumption at `run_experiments.py:491-515`.

**Trigger/root cause:** Activation eigendecomposition accepts zero/degenerate covariance. `dual_end_alignment` then selects a fixed number of eigenvectors as a dominant subspace without checking positive numerical rank or a spectral gap at the selection boundary. Eigenvectors within a repeated eigenspace are arbitrary; selecting part of that space is not an identifiable data-derived subspace.

**Reproduction:**

```python
svd = compute_svd(np.diag([4., 3., 2., 1.]))
r = dual_end_alignment(svd, np.zeros((4, 4)))
```

Observed `top_alignment=0`, `bulk_alignment=0`, and **`bottom_alignment=1`**, although the activation covariance contains no signal whatsoever. A different valid zero-eigenspace basis changes these answers.

**Impact:** Dead/constant activations, undersampled covariance, or a tied leading eigenspace can produce a false dual-end finding. This directly affects the interpretation of the project's central activation-alignment metrics.

**Fix direction:** Check covariance numerical rank using a scale-aware tolerance. Report rank-zero alignment as unavailable. Do not split an unresolved eigenvalue cluster at the dominant-subspace cutoff: use cluster/projector-aware analysis or explicitly mark the basis-dependent statistic unqualified. Preserve the distinct meanings of the selectable overlap metrics.

**Regression tests:** Constant activations, zero covariance, low-rank covariance, and orthogonal rotations within a repeated leading eigenspace. Qualified reported subspace scores must be invariant to such rotations, or explicitly unavailable.

## CO-10 — FIXED — Concurrent runners can both claim the same fresh output directory

**Locations:** `run_experiments.py:1127-1131`; shared temporary paths in `_write_json` (`93-100`) and `_write_csv` (`111-120`).

**Evidence:** Static control-flow/concurrency analysis; no concurrent destructive test was run.

**Trigger/root cause:** Checking nonemptiness and then calling `mkdir(..., exist_ok=True)` is a check-then-act race. Two processes can both observe the path as absent/empty and proceed. There is no exclusive owner marker or lock. Per-file atomic replacement does not make the complete run exclusive, and both processes even use the same `.tmp` filenames.

**Impact:** Concurrent CLI submissions using one output path can mix configurations/metrics, overwrite each other, or fail on temporary-file replacement. The normal per-job SLURM default reduces this risk, but direct CLI runs and shared explicit output paths remain affected.

**Fix direction:** Atomically acquire output ownership before any artifact write, using exclusive directory creation or an exclusive owner/lock file that also supports intentionally pre-created empty directories. Reject a competing owner. Use uniquely named temporary files as additional protection, not as a substitute for ownership.

**Regression tests:** Two processes synchronized at output acquisition: exactly one may own/write the run; the other must fail before altering its artifacts. Keep the sequential nonempty-directory rejection test.

## Completed remediation notes

1. CO-01 through CO-10 are fixed and covered by the current suite, including `tests/test_bug_report_current.py`.
2. The sibling package was repaired independently; the incompatible `rmt` APIs and normalization conventions were not merged.
3. Pinned-environment, CUDA FP16/BF16, compilation, covariance-device, SVD-driver, asset, and cluster execution remain operator validation gates; CPU verification is not GPU certification.
