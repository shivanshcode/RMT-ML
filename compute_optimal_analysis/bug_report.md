# Static bug report — compute_optimal_analysis

## Scope and evidence

Reviewed the Python implementation and tests, CLI, asset downloader, SLURM launcher, requirements, and documented runtime/scientific contracts. Paths and line numbers below are relative to this directory and refer to the source at review time.

This is a source-level review, **not an HPC execution or numerical certification**. All 37 Python files parse with Python 3.10 grammar using the local Python 3.13.2 AST parser; `bash -n run_hpc.slurm` succeeds. No training, downloads, package installation, GPU jobs, or pytest suite were run. Reproduction cases below are source-derived regression cases to implement, not claimed test results. Static analysis cannot establish that no additional bugs exist.

**Severity:** High = invalid scientific output, substantial lost computation, or an important execution failure; Medium = configuration-dependent incorrect behavior; Low = narrower robustness/diagnostic defect. Separate scientific caveats follow the concrete findings.

## Training, execution, and data

### CO-01 — High: the experiment seed is applied after model initialization

- **Locations:** `run_experiments.py:603-605,650-651`; `pipelines/trainer.py:159-166,55-61`.
- `CausalTransformer(config)` consumes the global Torch RNG before `LanguageModelTrainer` calls `seed_everything`. The loader has its own seed, but the initial weights do not have the advertised per-cell seed. A cell run alone can initialize differently from the same cell after earlier cells; separate identical invocations also need not agree.
- **Fix/check:** seed before constructing the model, and record the resolved cell seed. Compare initial state dictionaries for repeated runs and for `--cells 1` versus cell 1 in a multi-cell run.

### CO-02 — High: reported realized tokens and FLOPs are not measured training totals

- **Locations:** `run_experiments.py:201-218,633-651`; `pipelines/trainer.py:235-283`.
- Step count is `ceil(realized_tokens / (batch_size * (sequence_length - 1)))`, assuming every batch is full. The loader does not drop its last partial batch. Short datasets and repeated partial batches can undershoot substantially; the last full step can overshoot a small `--max-train-tokens` cap. Masked targets and skipped AMP updates are also not reflected in the manifest. The pre-training requested/capped float remains labeled `realized_tokens` in the final CSV.
- **Fix/check:** count valid shifted targets actually processed, distinguish attempted steps from optimizer updates, enforce the chosen cap semantics, and write measured totals after training. Test a one-sequence dataset with batch size 8 and a token cap smaller than one batch.

### CO-03 — High: parameter caps are not caps, and realized experiments are not necessarily IsoFLOP

- **Locations:** `run_experiments.py:195-218`; `models/chinchilla_scaling.py:229-271`.
- `parameter_cap` only caps the **search target**. The nearest architecture can exceed it, and tokens are not adjusted for the selected architecture's actual parameter count. Thus even uncapped allocations no longer necessarily share the requested compute budget.
- With vocabulary 50257, the minimum searched architecture (width 64, two layers) has **3,347,840 parameters**. At the default budget `1e16`, the overtrained requested model has about **2,282,177 parameters**: the minimum architecture already uses **46.7% more parameters**, hence proportionally more training FLOPs at unchanged token count. The manifest exposes an estimated compute difference, but the experiment and scaling plot still group by requested budget.
- **Fix/check:** search subject to an actual hard upper bound, reject infeasible caps, and either recompute tokens from actual `model.num_parameters()` to conserve FLOPs or explicitly classify the experiment as non-IsoFLOP. Plot measured compute. Test caps below the minimum and all default allocation ratios.

### CO-04 — High: `--validation-batches` does not limit training-time validation

- **Locations:** `run_experiments.py:651,684-692`; `pipelines/trainer.py:271-279`.
- The CLI limit is passed only to lesion evaluation. Every training log interval evaluates the **entire** validation loader. For the default WikiText-103 split, this can make validation dominate the run despite a requested limit of 20 batches, repeated every 50 optimizer steps.
- **Fix/check:** add a validation limit to `TrainConfig`/`fit` and forward it to `evaluate_language_model`. Count validation forwards with a loader longer than the configured limit.

### CO-05 — High: results/checkpoints are persisted too late for long HPC runs

- **Locations:** `pipelines/trainer.py:283-303`; `run_experiments.py:654-739,829-855`.
- Training history is held in memory. Even with checkpoints enabled, saving occurs only **after** activation extraction, all spectral diagnostics, and lesions. CSV/JSONL output is deferred until every selected cell finishes. An analysis error, later-cell failure, or wall-time kill can discard all preceding training and metrics; no periodic checkpoint or resume path exists.
- **Fix/check:** stream/flush training records, atomically persist each completed cell, save checkpoints before diagnostics and periodically during training, and record failure status. For exact resumability also save RNG, scaler, and data-order state. Inject a later-cell/lesion error and verify earlier output survives.

### CO-06 — High: the advertised one-SVD path repeatedly performs full CPU decompositions

- **Locations:** `run_experiments.py:272-283`; `rmt/factory.py:257-346,394-438`; `rmt/mp.py:195-207,755-796`; `pipelines/spectral_lesioning.py:348-389`.
- The accelerated `svd` is not passed to spectrum preparation or fit/detector dispatch. `prepare_spectrum` computes a full NumPy SVD even before taking the FARMS branch. The default Lanczos fit computes another SVD for MP summary statistics; spike dispatch computes an unused SVD before rerunning Lanczos. FARMS windows and lesion tranches add further independent decompositions.
- This is not merely an unused helper: `--svd-backend cuda` still leaves repeated dense CPU SVDs and duplicate Lanczos runs on the production path, undermining runtime/memory expectations.
- **Fix/check:** pass the original factors/spectrum and one prepared result through dispatch, share Lanczos results, and reuse pristine factors across lesion tranches. FARMS window SVDs are intrinsically separate, but must not force redundant full-matrix SVDs. Add decomposition-call-count tests for all tracks.

### CO-07 — High: different spectrum domains are combined without converting edges/scales

- **Locations:** `run_experiments.py:278-295,383-428,465-494`; `rmt/factory.py:257-346,394-420`; `rmt/mp.py:555-598`.
- The default runner plots/tail-fits FARMS-window eigenvalues, while `lanczos_stieltjes` fits the **original** covariance. For a rectangular matrix, the full-matrix aspect ratio and support differ from the square FARMS windows. `_plot_esd` overlays full-matrix edges on the pooled histogram. The row's `aspect_ratio` comes from the prepared spectrum, but MP outlier counts/fit may describe the original matrix.
- `farms_normalization=raw` or `trace` can make the units differ too. `farms_unbiased` MP fitting always uses canonical normalization, independently of the selected preparation. Thamm fitting also bypasses preparation. Conversely, analytic/KDE fits to `shape_normalized` or raw/trace FARMS values produce a scaled variance that TW/BBP dispatch applies directly to **untransformed** full-matrix eigenvalues.
- **Fix/check:** represent full, pooled, and rescaled spectra as separate labeled results, with separate aspect ratios/denominators; transform thresholds consistently or reject incompatible combinations. Test rectangular factors and all normalizations, not just whether individual dispatchers return finite numbers.

### CO-08 — High: the accelerator bridges silently force float32 spectral analysis

- **Locations:** `pipelines/activation_extractor.py:245-256`; `pipelines/spectral_lesioning.py:55-59`; `run_experiments.py:277`; `rmt/svd_result.py:40-68`.
- Both CPU and CUDA bridge paths downcast to float32, even for float64 input. Converting returned factors to float64 in `SVDResult` cannot restore small singular values lost during factorization. Other diagnostics independently recompute in float64, so bottom-vector alignment/lesions/conditioning can use a materially different numerical spectrum from MP/tail calculations.
- This particularly affects the small-singular-value research target. Explicit approximate `gesvda` is an additional precision tradeoff, not a substitute for validating the tail.
- **Fix/check:** expose and record analysis precision separately from training AMP; preserve float64 inputs and offer float64 SVD/lesions for small-SV studies. Test a rotated ill-conditioned factor spanning singular values 1 to `1e-8`, not only random well-conditioned reconstruction.

### CO-09 — High: a supported subset of lesion tranches crashes plotting

- **Locations:** `run_experiments.py:546-559,699-717,851-852`; `pipelines/cli_config.py:393`.
- `--lesion-tranches top` is accepted as a valid subset, but `_plot_lesions` always iterates top, bulk, bottom and uses `next(...)` without a default. Missing tranches raise `StopIteration` after training/analysis. The runtime record never receives completion metadata.
- **Fix/check:** plot only requested/present tranches, handle missing cell/tranche pairs explicitly, and validate before training. Cover every nonempty subset.

### CO-10 — Medium: diagnostic disable flags do not disable the corresponding work

- **Locations:** `run_experiments.py:654-670,324-350,846-850`.
- Activation covariance capture runs even when `compute_activation_overlap` is false. With spacing-distribution disabled but number variance or rigidity enabled, Brody fitting and spacing artifacts are still computed; the spacing plot is emitted unconditionally when artifacts exist. These switches cannot currently be relied upon to reduce GPU/CPU work.
- **Fix/check:** gate covariance capture, Brody/NNSD calculation, and plot creation independently. Test disabled paths with spies that raise if called.

### CO-11 — Medium: presets silently override explicit negative switches and do not fully select paper methods

- **Locations:** `run_experiments.py:748-780`; `pipelines/cli_config.py:255-405`.
- `reproduce_paper1` and `compute_optimal_rmt` unconditionally turn lesions and overlap back on, including after explicit `--no-run-spectral-lesioning` / `--no-compute-activation-overlap`. Paper 2/3 similarly override some negative flags. The preset booleans are documented, but the parser does not distinguish defaults from explicit user choices or reject the conflict.
- Conversely, selecting only `--experiment-mode reproduce_paper2` retains Golden FARMS/Lanczos/spline defaults and does not enable Delta3; only the fully expanded SLURM/README command selects the intended paper protocol. The mode name alone is not a scientific dispatcher.
- **Fix/check:** define actual mode presets, apply them before explicit overrides, or reject incompatible overrides and document that full method flags are mandatory. Test resolved execution configuration, not parsing alone.

### CO-12 — High: the verified asset manifest need not cover the dataset being executed

- **Locations:** `run_hpc.slurm:39-55`; `scripts/download_assets.py:295-321`; `run_experiments.py:141-163,806-820`.
- `DATASET_PATH` can point to any existing NPY/NPZ while `--verify-only` validates an unrelated nonempty manifest. Nothing requires a record matching the selected array. Standalone execution does not recompute its digest; `_runtime_environment` copies a recorded checksum if found, including a stale one.
- **Fix/check:** require the selected file to match a verified manifest record, recompute/compare its digest at preflight, and carry that verification result into runtime metadata. Test an external or modified dataset beside an otherwise valid manifest.

### CO-13 — Medium: partial asset staging destroys provenance for existing assets

- **Locations:** `scripts/download_assets.py:276-292,341-392`.
- Every invocation replaces `asset_manifest.json` with only the current invocation's `assets`/`source_revisions`, while `_file_manifest` includes **all existing data files**. Running `--assets synthetic` after staging WikiText keeps the WikiText files/checksums but drops their source revisions and asset metadata. Re-staging one asset family has the same inconsistency.
- **Fix/check:** preserve and validate untouched asset records or use isolated manifests/directories per asset family. Test sequential WikiText and synthetic staging.

### CO-14 — Medium: token loading silently accepts and changes invalid data

- **Locations:** `pipelines/dataset.py:62-83,98-116,119-131`.
- A loader documented to require a one-dimensional integer array casts and flattens arbitrary arrays without checking original dtype/shape. Fractional IDs are truncated; for example `-0.5` becomes zero and evades the negative-ID check. An NPZ without `tokens` silently selects the alphabetically first array, which may be metadata, while an empty NPZ raises an incidental `IndexError`.
- **Fix/check:** require an explicit token key, integer dtype, and expected shape before conversion; validate range and reject empty archives. Cover floats, negative fractions, matrices, and multi-array NPZ files.

### CO-15 — Medium: activation covariance includes padding/invalid token positions

- **Locations:** `pipelines/activation_extractor.py:47-66,165-175,289-308`.
- Hooks flatten every token activation. Passing `attention_mask` to the model does not prevent masked positions from entering sums/Gram matrices. Covariance and overlap therefore depend on padding length/content for padded callers. The default fixed-length dataset happens to use all-one masks.
- **Fix/check:** propagate a valid-position mask into hook accumulation, including any subsampling. Compare the same valid sequences with different padding.

### CO-16 — Medium: fully masked attention rows can expose future information under left padding

- **Locations:** `models/modules.py:110-122`; `models/transformer.py:163-174`.
- An entirely masked query row becomes a row of the same finite minimum value, whose softmax is **uniform**, not zero. A padded query can therefore attend to all values, including future real tokens. With left padding, the shifted loss still scores the first real target from the preceding padded query because it masks only the target position.
- **Fix/check:** safely handle all-masked attention rows and exclude predictions whose predecessor is padding. Add a left-padded causal-invariance test and a fully masked-row test. This does not affect the default unpadded training loader.

### CO-17 — Medium: the parameter estimator disagrees with supported Transformer variants

- **Locations:** `models/chinchilla_scaling.py:191-226`; `models/modules.py:150-156`; `models/transformer.py:61-63`.
- The estimator assumes one parameter per normalization dimension regardless of `norm_type`; LayerNorm actually has weight **and bias**, even when projection `bias=False`. It also uses `round(ratio * width)` without the model's minimum hidden width of one. This makes library allocation estimates wrong for supported non-default configurations.
- **Fix/check:** mirror normalization and hidden-width rules, validate GQA divisibility, and compare estimates with `num_parameters()` over all supported variants.

### CO-18 — Medium: AMP-skipped optimizer steps are counted as completed updates

- **Locations:** `pipelines/trainer.py:251-270`.
- For `amp_dtype=float16`, GradScaler may skip `optimizer.step()` after gradient overflow even if the forward loss was finite. The scheduler and `global_step` advance unconditionally. Training can exhaust its step budget without the intended number of updates and advance the LR schedule on failed updates.
- **Fix/check:** distinguish attempted batches from successful optimizer updates and schedule appropriately. Log overflow/skipped-step counters; test a forced overflow. The default BF16 path does not use enabled gradient scaling.

## Lesion and numerical-engine correctness

### CO-19 — Medium: count-matched bulk lesions can remove fewer values than top/bottom lesions

- **Locations:** `pipelines/spectral_lesioning.py:94-116`.
- When MP candidates are insufficient, the fallback interior set can still contain fewer than the requested count. The function silently reduces the count instead of reporting an infeasible matched comparison. At rank 20 and fraction 1, top/bottom remove 20 values but the fallback bulk removes only 8.
- **Fix/check:** enforce equal counts or explicitly return an infeasible/unmatched status. Do not label unequal interventions count-matched. Cover fractions near one and depleted MP bulks.

### CO-20 — Medium: energy mode silently reports unmatched interventions as comparable

- **Locations:** `pipelines/spectral_lesioning.py:119-145,348-389`; `run_experiments.py:722-731`.
- Bulk selection stops at all available candidates when their energy is below the target. Top selection can overshoot by an arbitrarily large singular-energy atom. `reference_energy` is not bounded by matrix or tranche energy. The output has no target-reached/error status, and the production CSV retains only the mean removed fraction, hiding per-matrix mismatches.
- **Fix/check:** report target and actual energy per matrix, flag/reject infeasible matches, and define a tolerance or partial-singular-value control if exact matching is intended. Test a single dominant singular value and a bulk whose total energy is below the target.

### CO-21 — Medium: accepted Lanczos step counts cannot run the detector

- **Locations:** `rmt/factory.py:185-190`; `rmt/lanczos_stieltjes.py:330-334,544-547,868-887`.
- The CLI/config accepts two steps, but `reference_modified_cholesky` always requires at least three realized iterations. Steps greater than the matrix dimension are accepted until deep execution fails. Early Krylov breakdown on identity/low-rank factors likewise reaches the tail estimator with too few entries.
- **Fix/check:** require an appropriate detector minimum, validate/clamp against dimension before expensive work, and explicitly handle exact breakdown/unsupported one-cut estimation. Cover steps 2, steps exceeding rank dimension, and identity factors.

### CO-22 — High: Lanczos ridge shifts fitted support but not reference Ritz poles

- **Locations:** `rmt/lanczos_stieltjes.py:474-501,879-925,936-942`; `rmt/mp.py:755-810`.
- A positive ridge is added during Cholesky factorization, so support/threshold estimation describes a shifted operator. `reference_ritz` still extracts poles from the **unshifted** Lanczos matrix. The MP adapter also compares unshifted eigenvalues with the shifted support. `constant_tail` and `reference_ritz` therefore use inconsistent ridge semantics.
- **Fix/check:** either add the ridge to the operator and consistently transform all outputs back, or use it only for numerical stabilization with explicit correction. Test a spiked factor at zero and nonzero ridge.

### CO-23 — Medium: modal spike-count aggregation can return poles below its threshold

- **Locations:** `rmt/lanczos_stieltjes.py:953-981`.
- With the default zero residue floor, the modal count is selected across probes, but locations come from the longest recurrence **regardless of that probe's count**. The code then takes its largest `spike_count` Ritz values without reapplying the threshold. If that probe detected fewer spikes than the mode, subthreshold values are returned as spikes.
- **Fix/check:** select a representative consistent with the modal count and threshold, or report count/location uncertainty separately. Construct probes whose counts disagree and assert every returned pole exceeds `threshold`.

### CO-24 — Medium: Lanczos results are relabeled as an MP fit while fit/failure diagnostics are lost

- **Locations:** `rmt/mp.py:789-813`; `run_experiments.py:383-425`; `rmt/lanczos_stieltjes.py:991-1014`.
- The adapter computes `ks_distance` against an analytic MP CDF fitted only to the upper edge, then replaces the support with the Lanczos lower/upper edges. That KS value is not goodness-of-fit to the reported Lanczos density. It also makes `n_upper_outliers` the count above the finite-size threshold while `bulk_fraction` excludes everything above the bulk edge, so these summaries need not partition the same sample.
- The runner drops `converged`, per-probe counts/edges, and fit optimizer success diagnostics. Unconverged or unsuccessful fits can be written as ordinary scientific metrics without a status.
- **Fix/check:** label analytic-projection KS separately, distinguish edge departures from thresholded spikes, and persist convergence/optimizer metadata with an explicit failure policy.

### CO-25 — Medium: the random FARMS option fails with its default window size on square matrices

- **Locations:** `rmt/farms_aspect_ratio.py:131-156,250-258`; `rmt/factory.py:243-254`.
- With no explicit window size, a square matrix at target ratio 1 is sampled using its entire shape. There is one possible window start, but `sampling=random` defaults to `5 * 5 = 25` distinct starts and raises. The CLI exposes random sampling but not `n_submatrices`; all square attention weights encounter this combination.
- **Fix/check:** cap the requested unique count, require an explicit smaller window, or reject this combination during preflight with a clear remedy. Test `farms_spectrum(eye(64), FARMSConfig(sampling='random'))`.

### CO-26 — Medium: degenerate matrices can abort the entire post-training analysis

- **Locations:** `rmt/mp.py:210-224,333-388`; `rmt/lanczos_stieltjes.py:487-499,544-547`; `run_experiments.py:272-289,471-480`.
- Zero/very-low-rank spectra fail positive-eigenvalue minimums or Cholesky/tail estimation; there is no per-matrix diagnostic-failure boundary in `analyze_model`. Plotting also calls `positive.min()/max()` without handling an empty positive spectrum and makes repeated bin edges for a constant spectrum.
- **Fix/check:** return structured unavailable diagnostics for unsupported/degenerate spectra, retain other matrix results, and make plots handle empty/constant data. Test zero, rank-one, and equal-singular-value matrices.

### CO-27 — Medium: MP outlier summaries silently exclude zero eigenvalues

- **Locations:** `rmt/mp.py:101-103,220-249,505`.
- MP fitting drops all zeros before computing counts, KS, and `bulk_fraction`. For a rectangular rank-deficient factor, genuine reduced-spectrum zeros below the positive MP lower edge vanish from the lower-outlier count and denominator. These are not just the structural extra zeros of a tall full covariance: the function is given the reduced spectrum.
- **Fix/check:** distinguish the positive sample used to estimate scale from the complete reduced spectrum used for counts/ESD metadata. Report rank deficiency explicitly.

### CO-28 — Medium: several reported KS distances are midpoint-CDF discrepancies, not KS statistics

- **Locations:** `rmt/mp.py:235-237,458-460`; `rmt/tail.py:87-89,250-252`.
- True two-sided one-sample KS must compare the model with both `i/n` and `(i-1)/n`. Using `(i-0.5)/n` underestimates KS (by `1/(2n)` for a continuous ordered sample under the usual construction). For CSN candidate tails, changing sample size changes that bias and can change the selected cutoff, not just the printed number.
- **Fix/check:** use a shared two-sided KS implementation and test it against SciPy on fixed sorted samples. Review lower-edge and PT-specific CDF discrepancy labels separately.

### CO-29 — Medium: CSN candidate cutoffs can split a group of equal observations

- **Location:** `rmt/tail.py:74-88`.
- Candidates are indices and `tail = data[index:]`; when `data[index]` is tied with earlier values, the selected tail can omit observations equal to its own `xmin`. This violates `tail = data[data >= xmin]` and affects alpha, tail count, and cutoff selection on repeated/quantized or pooled spectra.
- **Fix/check:** enumerate distinct cutoff values or their first indices. Assert `n_tail == count(data >= xmin)` on arrays with ties.

### CO-30 — Medium: Hill-selected results retain empty CSN metadata

- **Locations:** `rmt/tail.py:429-476`; `run_experiments.py:417-425,481-488`.
- Selecting Hill returns a finite `selected_alpha` but merges `_empty_fit()` for `xmin`, `n_tail`, and KS. The CSV then describes a finite tail estimate with zero observations and no cutoff, and the tail overlay cannot be drawn. Hill window metadata is similarly not a CSN fit.
- **Fix/check:** populate Hill's actual `k` and threshold/order statistic; mark inapplicable CSN fields explicitly rather than presenting zero observations. Test consistency of exponent and support metadata for every estimator.

### CO-31 — Medium: nonpooled Porter–Thomas calibration uses the wrong sample size when pooling is requested

- **Location:** `rmt/scalars.py:220-273`.
- `porter_thomas_monte_carlo(..., pooling_window=p)` calibrates KS distances using `p * dimension` random entries, but each observed distance is still computed from a single vector of `dimension` entries. For `p > 1`, null p-values are miscalibrated. The separate `_pooled` function does not have this particular size mismatch.
- **Fix/check:** reject pooling in the single-vector API or actually pool observed vectors identically to the null. Test false-positive calibration, not only p-value bounds and seed equality.

### CO-32 — Medium: QR orthogonalization invents subspace directions for dependent columns

- **Locations:** `rmt/overlap.py:64-105`.
- Reduced QR returns the requested number of columns even when the input is rank-deficient. Passing duplicate, nonzero columns causes principal-angle/projector metrics to include arbitrary orthogonal-completion directions, so equivalent mathematical spans can receive different scores.
- **Fix/check:** use rank-revealing orthogonalization with a tolerance and retain only the actual span. Compare a basis with the same basis containing duplicate columns.

### CO-33 — Medium: level cleaning deletes degeneracies before spacing statistics

- **Locations:** `rmt/spacing.py:29-34,128-153,288-298`.
- `np.unique` removes repeated levels. Real degeneracies/zero spacings are thus erased, changing sample size and adjacent-gap statistics toward stronger apparent repulsion. Nearest-neighbor fitting further discards zero gaps. Spectra affected by rank collapse or quantization are especially relevant here.
- **Fix/check:** preserve multiplicities for statistics; if a smoother needs distinct abscissae, fit a multiplicity-aware staircase and map back. Alternatively reject and report degeneracies explicitly. Test `[0,0,1,2,3]` and a rank-deficient covariance spectrum.

### CO-34 — Medium: Brody standard-error calculation is not valid in two exposed cases

- **Locations:** `rmt/spacing.py:233-268`.
- The inverse-curvature likelihood formula is also applied to the empirical-CDF least-squares objective, which is not a log likelihood and whose residuals are correlated. Near beta 0 or 1, the finite-difference stencil is asymmetric but the code uses a symmetric second-derivative formula. Both can emit misleading uncertainty.
- **Fix/check:** use bootstrap/calibrated uncertainty for CDF fitting and boundary-aware/profile likelihood or bootstrap for endpoint MLEs. Return unavailable uncertainty when it is not justified.

### CO-35 — Low: Delta3 integration is numerically dependent on the absolute spectral origin

- **Location:** `rmt/spacing.py:384-403`.
- Integrals use differences of large absolute squares/cubes and solve for a line using absolute coordinates. A large constant shift of otherwise identical unfolded levels can cause cancellation or a nearly singular Gram matrix, although Delta3 must be translation-invariant.
- **Fix/check:** perform every window calculation in local coordinates `[0,L]`. Compare regular levels before and after an offset such as `1e9`.

## Tests, packaging, and HPC entry point

### CO-36 — High for preflight: the overlap dispatcher test has a deterministically wrong assertion

- **Locations:** `tests/test_cli_dispatch.py:206-215`; `rmt/overlap.py:143-146`; `tests/test_pure_rmt_overlap.py:14-18`.
- The dispatcher test expects every metric to return 1 for an aligned three-column basis. The documented `staats_dual_end` implementation is the **mean of all pairwise squared overlaps**: `mean(eye(3)) = 1/3`. Another test explicitly expects the analogous four-column value to be `1/4`.
- **Fix/check:** resolve the metric contract and make assertions metric-specific. Under the present contract, change the test expectation, not the implementation to satisfy the conflicting test. The README's unqualified pytest preflight cannot currently be assumed green.
- Broader gaps: no end-to-end runner/trainer test exercises measured budgets, initialization seeds, partial batches, validation limits, subset plotting, or checkpoint-on-failure; combination tests only construct configurations. Add regressions for the findings above.

### CO-37 — High when sharing the environment: both projects use the incompatible top-level name `rmt`

- **Locations:** this `rmt/__init__.py`; `run_experiments.py:27-63`; `pipelines/__init__.py:3-11`; sibling `remote-sk-random-matrix-ml-esd-fixed/pyproject.toml`.
- The ESD project can be installed as distribution/package `rmt`; this project uses the same import name with different exports and container conventions. Running combined pytest collection or using the wrong current directory/PYTHONPATH can resolve the wrong package, causing import errors or incorrect helper implementations. Loading both under the same name in one interpreter cannot work reliably.
- **Fix/check:** ideally namespace/package this project uniquely; until then use separate Python processes, the correct working directory, and an explicit `rmt.__file__` assertion. See `to_change_env.md`.
- Related dependency-boundary issue: importing `pipelines.cli_config` eagerly executes `pipelines/__init__.py`, importing Torch/training/data code. Importing `models.chinchilla_scaling` likewise imports the Transformer through `models/__init__.py`. These nominally lightweight configuration/allocation imports are not framework-free in practice.

### CO-38 — Medium/conditional: the launcher is not configured for the demonstrated cluster deployment

- **Locations:** `run_hpc.slurm:2-25,35-55`; `requirements.txt`; `other_requirements.md`.
- It assumes `cuda/12.1`, `python/3.10`, and a local `.venv`, while the saved ESD run used `/home/shivansh/.conda/envs/rmt_ml_env/bin/python`. It omits the demonstrated `gpulong` partition. Those assumptions are portability mismatches, not evidence that the cluster lacks these modules.
- `SLURM_SUBMIT_DIR` is assumed to be this project; submitting the script by path from the repository root makes its relative script/data paths wrong. `logs/` must exist before submission, since its creation inside the script is too late for SLURM to open stdout/stderr; the README does document that prerequisite.
- **Observed local-checkout issue:** `git ls-files --eol` reports tracked LF but working-tree CRLF for both SLURM files. Copying this Windows checkout verbatim can make Linux/SLURM reject or misinterpret the script despite `bash -n` accepting its grammar.
- **Fix/check:** apply the environment checklist in `to_change_env.md`, use a verified project root/interpreter, create output directories before `sbatch`, and transfer LF scripts. Do not blindly reinstall these pins into the working ESD environment.

### CO-39 — Medium: the runner overrides the requested minimum tail size

- **Locations:** `run_experiments.py:284-289`; `pipelines/cli_config.py:196`; `rmt/factory.py:349-362`.
- `analyze_model` replaces `tail_minimum` with `max(8, min(config.tail_minimum, eigenvalues.size // 3))`. For a 64-value spectrum, a requested minimum of 50 or 1000 becomes 21; a requested minimum below 8 is raised to 8. A run can therefore publish a tail fit that does not meet the operator's minimum evidence threshold even though the serialized configuration retains the requested value.
- **Fix/check:** honor the configured minimum and return an unavailable fit when the sample is too small, or expose and record an explicit adaptive-minimum policy. Exercise the runner, not just `dispatch_tail_solver`, with an oversized requested minimum.

## Scientific limitations requiring explicit decisions, not automatically code bugs

1. **Empirical versus analytic optimum:** `build_manifest` fixes `D/N=20` (`run_experiments.py:189-194`) even when a different `ScalingLaw` is supplied. That empirical rule is documented but is not generally the minimizer of the supplied unequal-exponent loss law. Label it empirical or expose the analytic optimum instead of treating the two as identical.
2. **Lanczos threshold has units:** `threshold_c * N**(-delta)` is an absolute additive gap (`rmt/lanczos_stieltjes.py:924-926`). With default `c=1` and trained weight variances far below 1, it can dwarf the spectrum and detect no meaningful spikes. This follows the implemented formula; it requires scale-aware calibration or documented normalization, not an assertion that unit-noise synthetic tests validate every trained layer.
3. **Recycled tokens are not fresh data:** the runner repeatedly shuffles the finite training corpus to reach large D. Chinchilla fits based on fresh data do not automatically apply to repeated-token exposure. Record unique corpus size and epochs separately.
4. **This is not official WikiText validation:** the downloader tokenizes only the training split, and the runner holds out its last 10%; the staged official validation/test JSONL files are unused. This avoids direct train/validation reuse but must not be labeled official benchmark perplexity.
5. **FARMS observations are dependent:** overlapping windows produce correlated spectra. A tail exponent can be a useful descriptive statistic, but naive iid standard errors/bootstrap interpretations based on pooled eigenvalue count are not automatically calibrated.
6. **`xmax` is exclusion, not a truncated-law fit:** `rmt/tail.py` filters values above `xmax` but retains an unbounded Pareto likelihood/CDF. Do not interpret that as fitting a distribution normalized on `[xmin,xmax]`. Likewise `powerlaw_pkg_fit` uses a nonnested-style normal/Vuong p-value for pure versus nested truncated power laws; boundary/nesting-aware calibration is needed for inferential claims.
7. **Precision and lesion meaning need controls:** BF16/TF32 training/covariance and approximate SVD choices need calibration. A full SVD reconstruction itself changes floating-point weights, so a reconstruct-without-removal control is important before attributing very small bottom-lesion effects solely to removed directions.

## Suggested repair order

1. Fix CO-01–09, CO-12, CO-22, and the contradictory test before spending another large training allocation.
2. Make every completed cell recoverable; add end-to-end small offline tests for caps, masks, flag dispatch, and failures.
3. Resolve spectrum-domain/uncertainty semantics and preserve diagnostics before drawing cross-regime scientific conclusions.
4. Apply `to_change_env.md`, then validate on an allocated GPU with the exact intended environment. Static syntax checks are not evidence of CUDA/compiler/driver compatibility.
