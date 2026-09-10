# Static-analysis bug report — compute_optimal_analysis

## Scope and handoff

Reviewed the maintained Python implementation, CLI/configuration flow, numerical methods, asset staging, SLURM launcher, and relevant test contracts at commit `c93d344`. **No implementation files were changed.** Paths and line numbers below are relative to this directory and refer to that revision.

Validation: all 37 Python files parsed successfully; Ruff's undefined-name/export/local checks found no errors; `bash -n run_hpc.slurm` succeeded. Small, isolated offline NumPy/PyTorch probes confirmed the findings explicitly marked **observed**. These are not full-suite, CUDA, cluster, or production-training validation; the local environment is not the pinned standalone environment. Other findings follow directly from control flow or the stated statistical contract. This is a list of findings, not a guarantee that no other bugs exist.

Severity: **P1** = significant scientific-result corruption or experimental-design failure; **P2** = functional/numerical failure or misleading output under the stated trigger; **P3** = narrower reporting/numerical defect. There are **23 findings**.

Keep the sibling project's incompatible `rmt` package out of this process. Preserve intentional conventions, including canonical covariance normalization, explicitly diagnostic rank-order unfolding, density versus survival exponents, and reversible parameter identity/dtype preservation.

## Findings

### COA-001 — P1 — Autocast overrides the requested covariance computation precision

**Locations:** `pipelines/activation_extractor.py:57–90, 307–342`.

The forward hooks run inside the model's autocast context. Casting activations to float32 does not protect `centered.T @ centered`: autocast casts this multiplication back to BF16/FP16. The result is then stored in a float32 buffer, concealing the lower-precision computation. This affects the default CUDA/BF16 plus float32-covariance configuration and its small-eigenvalue/overlap diagnostics.

**Observed:** a CPU BF16-autocast analogue changed a requested-float32 covariance by approximately `0.00235` versus the same inputs outside autocast.

**Repair/test:** explicitly disable autocast around moment calculations, including when the accumulator is called from an external autocast context. Test the arithmetic against a reference, not just the buffer dtype; add a CUDA regression when available.

### COA-002 — P1 — Activation rank uses float64 tolerance for float32 data and accepts indefinite covariance

**Locations:** `rmt/overlap.py:15–26, 197–268`; covariance output in `pipelines/activation_extractor.py:224–239`.

The eigensolver promotes covariance to float64, then `dual_end_alignment` uses float64 epsilon to identify signal modes. Promotion does not remove float32 accumulation error. Spurious positive eigenvalues are counted as signal; negative eigenvalues are silently ignored, even for a genuinely indefinite input. The inferred rank also changes the dominant-subspace size.

**Observed:** a 16-dimensional float32 outer product, mathematically rank one, was reported as rank eight; its most negative eigenvalue was about `-5.16e-8`.

**Repair/test:** carry accumulation precision/error and observation-count provenance into qualification, distinguish rounding noise from significant indefiniteness, and bound identifiable rank. Test float32 low-rank, zero, and deliberately indefinite covariances.

### COA-003 — P1 — Collapsed designs across different requested budgets escape rejection

**Locations:** `run_experiments.py:319–334, 1183–1201`.

The collapse signature includes the *requested* `compute_budget`. After parameter/token caps, different requested budgets can produce exactly the same realized architecture, parameter count, token count, and compute. Those duplicate interventions are nevertheless labeled non-collapsed and execute without the explicit override.

**Observed:** `build_manifest(vocab_size=32, compute_budgets=[1e9, 2e9], allocation_ratios=[1.0], parameter_cap=200000, token_cap=100)` produced two `(133440 parameters, 100 tokens)` designs, both with `allocation_collapsed=False`.

**Repair/test:** compare actual executable designs independently of requested budget, using the eventual integer token target. Preserve requested-budget metadata separately and keep rejection limited to selected cells.

### COA-004 — P2 — Sub-token budgets silently become one-token runs and exceed hard caps

**Locations:** `run_experiments.py:290–296, 930–946, 1126–1129`.

Positive fractional caps/budgets are accepted, but execution uses `max(1, floor(realized_tokens))`. A cap of `0.5`, or a compute budget that affords less than one target for the minimum architecture, therefore trains one target. The reported realized compute exceeds the requested cap instead of rejecting an unrealizable design.

**Repair/test:** resolve and validate integer target counts during manifest construction/preflight. Reject designs that cannot afford one target; do not silently increase them. Test fractional caps, tiny compute budgets, and agreement between manifest and trainer targets.

### COA-005 — P2 — Invalid configuration can survive until after expensive training

**Locations:** `run_experiments.py:966, 1009–1049, 1113–1169, 1202–1205`; `rmt/factory.py:119–212, 355–359`; `pipelines/trainer.py:40–58`.

Cross-method compatibility and lesion names are not validated before training. For example, `farms_unbiased` with `aspect_ratio_mode=raw` is accepted by the configuration class but rejected only during post-training analysis. Invalid lesion names are checked after training and spectral analysis. Several numeric checks use only `<= 0`, allowing NaN caps/learning settings; NaN caps can silently disable the intended restriction.

**Observed:** runtime resolution accepted both `--parameter-cap nan` and `--max-train-tokens nan`.

**Repair/test:** centralize finite-value, cross-method, selected-geometry, training, and tranche validation before output acquisition/data loading/model construction. Test rejection without calling the trainer. Do not require a successful GPU allocation merely to validate method compatibility.

### COA-006 — P2 — Training work counters include targets/positions never executed

**Locations:** `pipelines/trainer.py:301–340`; exported compute proxy in `run_experiments.py:974–995`.

`attempted_target_tokens` increments before final-budget masking, so a batch with hundreds of eligible labels and only one remaining target counts all labels as attempted. Also, `forwarded_input_positions` increments before the `batch_tokens < 1` early continue, so an all-ignored batch counts input positions even though no forward occurred. This corrupts attempted-work telemetry and the forwarded-position compute proxy.

**Repair/test:** count actual post-masking targets per attempted forward/update and actual positions only when a forward is executed. If eligible/fetched counts are useful, expose separately named counters. Test partial final batches, all-ignored batches, and skipped optimizer updates.

### COA-007 — P2 — Token-budget training can omit validation of the final model

**Locations:** `pipelines/trainer.py:241–273, 405–436`; `run_experiments.py:930–941, 966–968`.

Validation is triggered by a logging multiple or `global_step == total_steps`. In token-budget mode, `total_steps` is merely an estimate, while variable/partial loader batches can require more successful updates. The final update need not hit either condition. For example, alternating batches with four and two targets need four updates for a 12-target budget, despite the supplied three-step estimate; step three is validated, but step four need not be.

**Repair/test:** validate at actual budget completion and preserve that final metric even when it is off the logging cadence. Test recycled variable-sized loaders with logging intervals larger than the run.

### COA-008 — P2 — A finite-gradient norm overflow bypasses clipping and does not make GradScaler skip

**Locations:** `pipelines/trainer.py:346–384`.

The FP32 norm calculation can overflow although every gradient entry is finite. The enabled-scaler branch assumes an infinite norm means GradScaler recorded an overflow during `unscale_`, skips clipping, and calls `scaler.step`. GradScaler checks gradient entries, not this separately calculated norm, so it can execute the unbounded update and advance successful-token/LR counters.

**Observed:** a finite gradient vector with entries `1e20` had infinite FP32 norm, while an enabled CPU GradScaler still executed the optimizer step.

**Repair/test:** use overflow-safe norm computation and distinguish norm overflow from non-finite gradient entries. Never rely on an overflow flag that the scaler did not record. Test the finite-entry/infinite-norm case separately from genuine AMP gradient overflow.

### COA-009 — P3 — Logged learning rate describes the next update, not the recorded update

**Locations:** `pipelines/trainer.py:375–405`.

The scheduler advances before the training record reads the optimizer's learning rate. The loss/gradient/step fields describe the update just performed, but `learning_rate` describes the following update. Warmup/decay plots and checkpoint diagnostics are therefore shifted.

**Repair/test:** capture the applied rate before the optimizer step, or clearly expose separate applied/next-rate fields. Spy on optimizer rates and compare each successful-update log to the rate actually used.

### COA-010 — P2 — Degenerate nonzero spectra abort analysis instead of returning unavailable fits

**Locations:** `rmt/factory.py:306–391`; `rmt/mp.py:219–233, 360–367, 526–532`; `run_experiments.py:401–405, 474–478`.

Only the zero-matrix and Lanczos branches consistently translate an unavailable fit into a structured result. Nonzero rank-one/low-rank spectra reach analytic or KDE minimum-positive-count exceptions; short/degenerate Thamm inputs can similarly raise. The runner does not isolate these expected numerical-unavailability cases, so an otherwise usable trained cell can be discarded, even when other scalar/tail diagnostics are available.

**Observed:** dispatching an analytic/raw fit for `diag(1, 0, ..., 0)` raised `ValueError: at least four positive eigenvalues are required for scale fitting`.

**Repair/test:** qualify data-dependent fit availability for every dispatched method and let independent diagnostics proceed. Keep invalid configuration errors distinct. Test zero, rank-one, insufficient-positive-count, and constant spectra.

### COA-011 — P2 — FARMS wrappers overwrite and lose the underlying MP fit's unavailable status

**Locations:** `rmt/factory.py:344–369`; `rmt/mp.py:635–649`.

The FARMS wrappers replace, rather than merge, `fitted.diagnostics`. This drops `available=False` and rank-deficiency information from the underlying fit. A nonzero source can have entirely zero sampled windows, so the source-level zero check does not prevent this. The runner then treats a zero-variance fit as available and invokes a detector that rejects variance zero.

**Observed:** an `8x16` matrix with only its last entry nonzero and one leading `4x4` FARMS window produced variance zero but an effective available flag of true; the TW detector then raised.

**Repair/test:** preserve fit availability/status and numerical diagnostics while adding FARMS provenance. Cover both the factory and standalone FARMS-fit API, including zero sampled windows from a nonzero source.

### COA-012 — P2 — Absolute Lanczos tolerances break scale equivariance

**Locations:** `rmt/lanczos_stieltjes.py:331–344, 410–416, 482–508, 1030–1038`.

The production factor path has a scale-relative spike margin, but recurrence breakdown and Cholesky pivots still use absolute floors (`eps * dimension`, `1e-14`, and an epsilon floor for convergence). Rescaling a perfectly well-conditioned factor can turn a valid fit into an unavailable one solely because its entries are small.

**Observed:** a seeded Gaussian `32x64` factor succeeded with 20 nonadaptive steps; multiplying it by `1e-7` raised a positive-definiteness error despite unchanged condition number.

**Repair/test:** normalize internally or use operator-relative tolerances, then map edges, ridge, and poles back consistently. Test invariance over several decades, not just order-one matrices.

### COA-013 — P2 — Cached SVD normalization can disagree with operator-fit units

**Locations:** `rmt/factory.py:321–336, 371–377`; `rmt/mp.py:450–466, 814–858`; `rmt/svd_result.py:28–36, 89–95`.

`SVDResult` supports a custom covariance normalization. The Lanczos dispatch passes its eigenvalues into a fit whose operator is always divided by `max(weight.shape)`; Thamm similarly hardcodes the canonical denominator. There is no compatibility check. Cached observations and returned support can therefore describe different units.

**Observed:** using normalizations 64 and 1 for the same `32x64` factor returned the identical Lanczos upper edge (`~2.279`), while reported upper outliers changed from 2 to all 32.

**Repair/test:** either reject incompatible cached normalization explicitly or rescale all operator-fit outputs/diagnostics consistently. Test normalization changes with cached factors and verify invariant outlier membership.

### COA-014 — P2 — Weight-side degenerate singular subspaces produce arbitrary overlap claims

**Locations:** `rmt/overlap.py:197–267`, especially tranche selection and `svd.V[:, indices]`.

Activation-side repeated clusters are handled, but repeated/zero singular values of the weight are not. An SVD may rotate freely within such a subspace. Splitting it into top/bulk/bottom tranches makes the scores solver/basis-dependent while reporting them as available. Null singular vectors have the same issue.

**Observed:** two valid SVDs reconstructing the identical `10x10` identity matrix, with the same nondegenerate diagonal activation covariance, produced top alignment `1.0` versus `~0.3753`.

**Repair/test:** qualify weight eigenspaces too; use whole-cluster invariant comparisons or mark affected tranche/vector claims unavailable. Test rotations within repeated positive and null singular clusters. The sibling project has the same underlying issue, but different overlap conventions.

### COA-015 — P2 — Public legacy overlap APIs bypass activation qualification

**Locations:** `rmt/overlap.py:272–347`.

`overlap_analysis` and `eigenvector_eigenvalue_coincidence` operate on every covariance eigenvector without excluding null directions, rejecting significant negative eigenvalues, or checking unresolved positive eigenspaces. Thus a zero covariance can produce apparently perfect overlap from an arbitrary eigensolver basis, despite there being no activation signal. The safer checks in `dual_end_alignment` are not shared by these public APIs.

**Repair/test:** share qualification and explicit availability handling across entry points while preserving each metric's documented convention. Test zero, indefinite, repeated-positive, and rank-deficient covariances with nondegenerate weights.

### COA-016 — P2 — Spike-count provenance can falsely label a raw-operator count as pooled FARMS data

**Location:** `run_experiments.py:631–647`.

`spike_count_semantics` selects `pooled_observations_at_per_operator_threshold` whenever `prepared.farms` exists, regardless of `mp_is_raw`. With FARMS ESD preprocessing plus `thamm_modified_singular` and a TW/BBP detector, detection actually uses the original single-operator eigenvalues and geometry. The count's exported interpretation contradicts its actual domain.

**Repair/test:** derive count semantics from the spectrum actually passed to the detector, not the independent ESD preprocessing. Cover mixed raw-operator/FARMS method combinations in CSV-schema tests.

### COA-017 — P2 — KDE fit claims a complete-sample ECDF while discarding zero observations

**Locations:** `rmt/mp.py:527–559, 584–603`.

The fitter removes zeros before constructing ECDF ranks, then divides by the positive count without the zero-count offset. Nevertheless, diagnostics call the objective `complete_sample_ecdf_cvm`. Adding any number of zero eigenvalues leaves the fitted objective/variance unchanged, while the final KS is computed against the complete sample. This silently switches fitting to a conditional-positive distribution on rank-deficient inputs and hides that distinction.

**Repair/test:** implement the stated complete-sample ranks, or explicitly expose a separately named conditional-positive fitting contract and its omitted mass. Preserve zero/rank metadata. Test spectra with substantial zero mass, not only full-rank Wishart samples.

### COA-018 — P2 — The internal tail model-comparison helper ignores bounded-support normalization

**Locations:** `rmt/tail.py:525–592`.

`powerlaw_pkg_fit(..., xmax=...)` filters observations at `xmax`, but calls the CSN fitter without `xmax`, uses an unbounded Pareto log density, and integrates the exponentially truncated model to infinity. Consequently the comparison is not the likelihood of either model conditioned on the supplied finite observation interval. Both exponents and likelihood ratios can be biased.

**Repair/test:** propagate finite support through cutoff fitting and both likelihood normalizers, or reject unsupported finite bounds rather than silently reinterpret them. Test against direct bounded-density integration and samples for which the upper cutoff is consequential.

### COA-019 — P2 — Reported tail-comparison p-value uses a nonnested test for nested models

**Locations:** `rmt/tail.py:548–600`.

The pure power law is the zero-cutoff boundary case of the exponentially truncated power law. The helper applies the ordinary nonnested Vuong normal approximation to per-observation likelihood differences and reports the resulting value as `LR_p`. That approximation is not a calibrated significance test for this nested, boundary-parameter comparison; its variance degenerates under the pure-law null. Merely fixing the normalizers in COA-018 does not fix this.

**Repair/test:** use a statistically justified nested/boundary test or parametric bootstrap, and document the likelihood-ratio sign/normalization. Calibrate false-positive rates on pure-law simulations and power on truncated alternatives.

### COA-020 — P3 — Bounded-Pareto standard errors overflow for valid steep tails

**Locations:** `rmt/tail.py:61–72`.

The information calculation forms `exp(beta * log(xmax/xmin))` and squares `expm1` of the same positive argument. For a remote upper bound/steep tail these overflow, yielding `inf/inf` and a NaN standard error even though the fit and limiting information are finite.

**Observed:** `_pareto_fit(linspace(1, 1.001, 100), 1, 10)` returned alpha `~2001.67`, NaN standard error, and overflow warnings; the negligible-truncation limit gives an error near `200.07`.

**Repair/test:** express the correction in decaying exponentials and use stable small-argument limits. Test steep tails and weak/strong truncation.

### COA-021 — P2 — Partial asset staging re-certifies additions to untouched asset families

**Locations:** `scripts/download_assets.py:276–297, 324–337, 382–412`.

Untouched-family verification checks only previously recorded files. The new manifest then inventories every current file under `data/`. An unrecorded addition to an untouched tokenizer/dataset family is accepted and certified under its old source-revision metadata. Added tokenizer override files can change behavior even though every previously recorded file still matches.

**Observed:** adding `data/tokenizers/gpt2/added_tokens.json` after creating a manifest did not fail synthetic-only untouched verification; the subsequent inventory included the new file.

**Repair/test:** verify untouched family membership as well as contents, and do not certify unrelated files as part of a partial restage. Test additions, deletions, and modifications in untouched families.

### COA-022 — P2 — Tokenizer restaging can retain files deleted by a newer revision

**Locations:** `scripts/download_assets.py:89–115, 395–407`.

The downloader reuses the existing `data/tokenizers/gpt2` directory. Downloading a snapshot into an existing local directory does not make that directory an exact mirror by deleting obsolete files. A removed `added_tokens.json` or tokenizer configuration can remain active, while the newly generated manifest labels the mixed directory with the new commit SHA.

**Repair/test:** stage into a fresh directory and replace the tokenizer tree only after successful verification, or explicitly reconcile against the resolved snapshot's file list. Mock two revisions where the second deletes an overriding tokenizer file; the final tree must contain only the second revision's permitted files.

### COA-023 — P2 — SLURM result-root acquisition is check-then-create, not exclusive

**Location:** `run_hpc.slurm:80–91`.

The launcher checks `[[ -e "$OUTPUT_ROOT" ]]` and then runs `mkdir -p`. Two jobs using the same explicit root can both pass the check and both succeed at creation. Child runners protect individual track directories, but jobs choosing different tracks can still silently share a supposedly job-owned root; same-track jobs do expensive setup before one finally loses ownership.

**Repair/test:** claim the job root atomically, using exclusive directory creation or an exclusive owner record, and define the empty-precreated-directory policy consistently with the README. Race two lightweight launcher stubs against one root; exactly one must acquire it regardless of track.

## Suggested repair order

1. Covariance precision/qualification and design/budget preflight: COA-001–005.
2. Training accounting/update safety: COA-006–009.
3. Numerical availability, units, eigenspace qualification, and provenance: COA-010–017.
4. Tail model-comparison statistics and stable errors: COA-018–020.
5. Asset and launcher ownership: COA-021–023.

Add executable regression tests for the triggers above. Existing source-text/dispatch tests do not establish numerical precision, budget completion, asset-tree identity, or end-to-end scientific-method consistency. Immutable default dataclass instances and immediately evaluated loop lambdas flagged by broader lint rules were not counted as bugs.
