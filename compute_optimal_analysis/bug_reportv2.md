# Static bug report v2 — compute_optimal_analysis

## Scope and validation

Reviewed the **current working tree**, including the supplied bug fixes and environment migration, against `bug_report.md` and `to_change_env.md`. `HEAD` is `c987fc4`, but the fixes are uncommitted; this report describes the files on disk, **not just HEAD**. Locations below are relative to this directory. No implementation files were changed for this review.

This is a source-level review supplemented by small, offline CPU checks of selected numerical helpers and a stub-model analysis path. No training, real-model loading, downloads, package installation, full pytest suite, CUDA execution, or HPC submission was performed. The local check environment was Windows, Python 3.13.2, NumPy 2.4.4, SciPy 1.17.1, and Torch 2.13.0+cpu—not the cluster environment. All 37 Python files parse with Python 3.10 grammar. `bash -n run_hpc.slurm` succeeds, and this launcher now has LF line endings.

**Severity:** High = misleading important scientific results, significant budget/data loss, or an important execution failure; Medium = configuration-dependent correctness/resource failure; Low = narrower documentation/robustness defect. “New” means newly found or introduced by the reviewed changes, as specified. Suggested regression tests are **not claimed to have been run** unless explicitly identified as observed below.

## Confirmed open findings

### CO-V2-01 — High: the factor Lanczos adapter now replaces the Lanczos support with an analytic MP law

**New regression. Locations:** `rmt/lanczos_stieltjes.py:1013-1057`; `rmt/mp.py:802-831`.

After running the one-cut Lanczos detector, `detect_spikes_from_factor` overwrites both support edges with `bulk_variance * (1 ± sqrt(q))**2`, estimated from the trace after subtracting the already-selected poles. This is an MP assumption, not a correction generally valid for an arbitrary one-cut covariance. The Golden runner uses this adapter.

The returned object becomes internally inconsistent: its Stieltjes recurrence, `tail_alpha`, `tail_beta`, probe edges/counts, and convergence flag still describe the original Lanczos fit. If the new threshold is lower, the filter-only update cannot recover poles discarded by the old threshold. If it is higher, removed poles are not put back into the trace estimate and probe counts are not updated.

**Observed:** for `W = diag(sqrt(64 * linspace(1, 2, 64)))`, with 20 steps, one probe, adaptive stopping disabled, seed 0, direct Lanczos on `covariance_linear_operator(W)` reports approximately `[1.0330, 2.0187]`. The factor adapter reports `[0, 6]`, while its own transform still encodes `[1.0330, 2.0187]`.

**Fix/test:** remove the unconditional substitution. If a trace/MP projection is desired, expose it as a separate, explicitly named estimator with independent metadata. Test factor/operator equivalence on non-MP one-cut spectra, as well as Wishart matrices; verify that support, transform, threshold, poles, and probe diagnostics describe the same result.

### CO-V2-02 — High: partial batches and skipped updates still terminate training before the planned token budget

**Partially fixed CO-02/CO-18. Locations:** `run_experiments.py:741-758`; `pipelines/trainer.py:231-251,268-305,333-336`.

The new counter records successful-update target tokens, but the runner still calculates `max_steps` assuming every batch has `batch_size * (sequence_length - 1)` valid targets. Epochs are then sized to that step estimate. Small/partial batches reach the step ceiling before the token budget; AMP-skipped batches consume the finite epoch allowance without replacement. These cells are nevertheless recorded as complete.

**Source-derived example:** one training sequence of length 8, batch size 8, and 100 planned targets produces `max_steps=2`, `epochs=2`; at most **14** targets can contribute to updates. This arithmetic was checked locally without training.

**Fix/test:** make the resolved token budget an actual stopping condition, with sufficient data recycling and an explicit overflow/retry/failure limit. If a separate update/epoch ceiling intentionally takes precedence, mark the cell incomplete and record why. Test one-sequence datasets, repeated partial batches, masked batches, and forced skipped updates.

### CO-V2-03 — High: “measured training FLOPs” still count loss targets rather than executed work

**Partially fixed CO-02; new consequence of token-cap masking. Locations:** `pipelines/trainer.py:268-305`; `run_experiments.py:784-790,684-685`.

To honor a final token cap, the trainer masks excess labels but still executes the **entire batch/context** through the model and backward pass. It increments `processed_train_tokens` only on successful updates, then the runner labels `6 * parameters * processed_train_tokens` as realized/measured compute. Work on masked positions and AMP-overflow attempts disappears from this estimate.

For a one-target remainder with the default batch/context, a full 8×256 forward/backward is performed while the compute estimate increases by only `6N`. Successful-update targets, attempted targets, forwarded positions, and physical FLOPs are different quantities.

**Fix/test:** persist those quantities separately; label `6ND` as an approximation with a declared counting convention. Enforce physical compute caps by reducing work, not only changing labels. Verify accounting for a one-target remainder and a skipped update.

### CO-V2-04 — High: the IsoFLOP repair can collapse all allocation regimes into the same experiment while retaining different labels

**New regression/remaining CO-03 issue. Locations:** `run_experiments.py:218-264,440-458,658-685`; `models/chinchilla_scaling.py:135-160`.

Tokens are now recomputed from the discrete architecture to conserve compute, but `kappa`, `regime`, `tokens_per_parameter`, and predicted loss remain those of the **ideal requested** allocation. Hard caps or architecture discretization can select the same architecture in several regimes; recomputed token budgets then also become identical. The spectral/scaling plots still present these as different allocation interventions.

**Observed:** `build_manifest(vocab_size=50257, compute_budgets=[1e16], parameter_cap=3400000)` gives all three cells **3,347,840 parameters and 497,833,428.917 planned tokens**, but labels them undertrained/optimal/overtrained with kappas 0.25/1/4. Differences between these cells would be seed differences, not the intended N/D intervention.

**Fix/test:** distinguish requested from realized allocation ratios and predicted losses, detect collapsed cells, and warn/reject designs that cannot realize distinct regimes. Plot realized ratios or clearly label capped calibration runs as not testing the requested regime comparison.

### CO-V2-05 — High: spectrum-domain consistency is only partially repaired

**Partially fixed CO-07. Locations:** `rmt/factory.py:272-346,402-454`; `rmt/mp.py:572-615`; `run_experiments.py:324-348,364-369,468-474,578-580`.

The default FARMS/Lanczos plot no longer overlays obviously different domains, but other accepted combinations still mix units:

- Analytic/KDE fitting uses prepared `shape_normalized` or raw/trace FARMS values, then TW/BBP detection receives that variance for the **untransformed full matrix**.
- `farms_unbiased` fitting always uses canonical windows, independently of preparation normalization. The row labels both as the same prepared mode and can overlay incompatible edges.
- Thamm fits the full raw matrix but is labeled with `prepared.mode`; FARMS or shape-normalized preparation therefore gets incorrect edge overlays and soft-rank normalization.
- Lanczos on raw preparation is labeled `full_covariance` versus `raw`, suppressing its edge overlay even though those particular numerical domains coincide.

**Observed:** for a 16×64 factor, changing only raw preparation to shape normalization multiplies the BBP detector threshold by **16/9**, although the detector examines the same raw eigenvalues.

**Fix/test:** carry an explicit spectrum identity, units, denominator, aspect ratio, and any scale transform through every fit/detector/plot. Convert variance back when possible; reject incompatible combinations otherwise. Test rectangular matrices and every preparation/normalization against every fit/detector, not merely finite dispatcher returns.

### CO-V2-06 — Medium: the zero-matrix shortcut crashes the default runner; low-rank/short-recurrence failures still abort a cell

**Partially fixed CO-21/CO-26; new shortcut integration bug. Locations:** `rmt/factory.py:306-310`; `run_experiments.py:326-339`; `rmt/mp.py:220-235`; `rmt/lanczos_stieltjes.py:836-837,869-888,545-548`.

`dispatch_mp_fit` now returns an unavailable analytic result for a zero matrix. The default runner assumes the configured Lanczos fit actually ran and unconditionally indexes `diagnostics['threshold']` and `['lambda_plus']`. FARMS-fit zero results similarly lack the `spectral_max` accessed by the runner. Rank-one spectra still fail MP positive-count requirements or Lanczos Cholesky/tail estimation, with no per-matrix boundary in `analyze_model`.

**Observed:** a stub exposing one 64×64 zero spectral weight makes default `analyze_model` raise `KeyError: 'threshold'`. The explicit minimum of three Lanczos steps is repaired, but identity/early Krylov breakdown and steps exceeding a selected matrix dimension are still not handled before the post-training analysis.

**Fix/test:** branch on actual result availability/method, return structured unavailable diagnostics, and preserve other matrix/cell results. Preflight matrix-dependent settings. Cover zero, exact rank-one, identity, small GQA projections, and excessive step counts through the runner.

### CO-V2-07 — Medium: positive ridge is removed from reported edges/poles but not from the Stieltjes transform

**Partially fixed CO-22. Locations:** `rmt/lanczos_stieltjes.py:243-268,474-501,909-940,974-1009`.

Edges and constant-tail poles are now shifted back by `ridge`. However, the stored Cholesky recurrence and tail parameters still describe the shifted operator, and `result.stieltjes(z)` evaluates them at `z`, rather than at `z + ridge`. `density()` consequently remains in shifted units. `LanczosSpikeResult` does not retain ridge metadata needed to perform that correction.

**Observed:** the narrow-spectrum example from CO-V2-01 with ridge 1 reports support approximately `[1.0227, 2.0128]`; its transform's branch support is `[2.0227, 3.0128]`.

**Fix/test:** preserve and apply the shift consistently to transforms/densities as well as edges/poles. Test ridge-zero/nonzero density and resolvent semantics, separately from the factor adapter substitution in CO-V2-01.

### CO-V2-08 — Medium: several production tracks still repeat CPU SVDs despite the one-SVD contract

**Partially fixed CO-06. Locations:** `rmt/factory.py:327-346,409`; `rmt/mp.py:444,592-612`; `run_experiments.py:316-346`; `rmt/farms_aspect_ratio.py:329-337`.

The default Lanczos fit/detector now share results, but TW/BBP dispatch still computes a new full NumPy SVD. The Lanczos-detector-only branch also computes that unused spectrum before Lanczos. Thamm ignores the supplied factors, and FARMS fitting resamples/redecomposes windows already computed by preparation. A full-size square FARMS window additionally decomposes the very matrix whose factors are already available.

Thus paper tracks using `--svd-backend cuda` still incur independent dense CPU decompositions. Lesion tranche factor reuse is improved and should be retained.

**Fix/test:** thread cached raw spectra and prepared windows into all adapters; do not compute an eigenvalue array before a matrix-free detector that does not use it. Add decomposition-count tests for the four actual runner tracks.

### CO-V2-09 — High for small-SV studies: production analysis still cannot request float64 SVD independently of model dtype

**Partially fixed CO-08. Locations:** `pipelines/activation_extractor.py:259-269`; `pipelines/spectral_lesioning.py:57-62`; `run_experiments.py:316-324`; `pipelines/cli_config.py:320-347`.

The bridges now preserve **already-float64** weights, but newly trained model parameters remain float32 and there is no analysis-precision control. Both CPU and CUDA paths therefore factor them in float32; selecting FP32 training or CPU SVD does not solve this. Converting factors to float64 inside `SVDResult` cannot recover small singular values lost during factorization. FARMS and other independent NumPy paths still factor copies in float64.

**Fix/test:** expose and record an analysis dtype independent of training/execution dtype; allow float32 model weights to be promoted **before** SVD and lesion factorization. Test rotated ill-conditioned float32 matrices against a float64 decomposition of those exact stored weights, and include a reconstruct-without-removal lesion control.

### CO-V2-10 — Medium: count-matched “bulk” lesions now silently include top/bottom directions

**Regression from the CO-19 repair. Locations:** `pipelines/spectral_lesioning.py:81-94,111-115`.

When the MP bulk has too few candidates, `_count_selection` replaces it with **all indices**. This fixes unequal counts by changing the intervention itself: the result is still labeled bulk even if it removes outliers or bottom modes. The earlier fallback inside `_mp_bulk_candidates` is also not recorded.

**Observed:** rank 20 with `fraction=1` reports a bulk lesion removing indices 0–19, exactly the entire spectrum.

**Fix/test:** reject/mark infeasible matched bulk lesions, or give a fallback intervention a distinct name and serialize the actual candidate policy. Assert that every removed bulk index is in the declared bulk, including depleted-bulk and large-fraction cases.

### CO-V2-11 — Medium: energy matching remains scientifically unmatched, and the new status is discarded

**Partially fixed CO-20. Locations:** `pipelines/spectral_lesioning.py:127-145,185-211,399-406`; `run_experiments.py:857-874`.

`target_reached` means only `removed_energy >= target`, so a dominant top singular value can overshoot dramatically while being labeled reached. Insufficient bulk energy still yields an under-target result. The runner discards per-matrix targets, actual energies, indices, and `target_reached`, preserving only the mean removed fraction. Production CSVs therefore still hide whether an energy-matched comparison was possible.

**Fix/test:** retain per-matrix intervention records, target/actual error, and a justified tolerance/match status. Define overshoot/infeasibility policy explicitly. Test a single dominant singular value and a bulk whose total energy is below the target.

### CO-V2-12 — Medium: spacing degeneracies are still deleted in the runner, and retained zero gaps use the wrong mean-one normalization

**Partially fixed CO-33. Locations:** `run_experiments.py:392-397`; `rmt/spacing.py:117-125,152-158,202-206`.

The low-level cleaner now preserves repeated levels, but the production runner and Brody fitter still discard zero gaps. The standalone spacing helper keeps them while dividing by the mean of **positive** gaps, so its complete spacing sample is not mean one. These two paths no longer describe the same empirical spacing distribution.

Also, `maximum.accumulate` can flatten a nonmonotone polynomial/spline over genuinely distinct levels; those artificial zero gaps are then silently discarded by the runner rather than identifying a failed unfolding.

**Observed:** `nearest_neighbor_spacings([0,0,1,2,3], unfolded=True)` has mean **0.75**, not 1.

**Fix/test:** distinguish physical degeneracies from a failed/folded unfolding; preserve the empirical zero mass and use a declared mean-one convention over the complete spacing sample. Reject or explicitly label continuous Brody fits conditional on positive gaps. Exercise repeated levels and nonmonotone fitted staircases end to end.

### CO-V2-13 — Medium: the Brody boundary standard-error fix does not actually exclude asymmetric stencils

**Partially fixed CO-34. Locations:** `rmt/spacing.py:240-253`.

`left < beta < right` is true for a bounded optimizer result extremely close to 0 or 1; it does not establish that `beta-left == right-beta`. The code still applies a symmetric second-difference formula with denominator `(right-beta)**2`. Near the upper boundary this can produce an arbitrarily tiny reported uncertainty.

**Observed:** `fit_brody(ones(100))` returns beta approximately `0.99999992` and standard error approximately **1.14e-6**. The two stencil widths are about `1e-4` and `7.8e-8`.

**Fix/test:** require a genuinely interior symmetric stencil or use a boundary-aware/profile/bootstrap interval. The repaired CDF-NLS unavailability behavior should remain. Test both endpoint optima.

### CO-V2-14 — Medium: zero eigenvalues remain excluded from KDE summaries and analytic MP KS

**Partially fixed CO-27. Locations:** `rmt/mp.py:220-247,521-565`.

Analytic MP outlier counts now use the complete spectrum, but its KS still uses only the positive observations with a positive-only denominator. KDE fitting filters zeros at entry and passes that filtered sample to the summary fit, so even the repaired lower-outlier counts/bulk fraction lose rank deficiency on that track.

**Observed:** KDE on ten zeros plus `linspace(0.2,2,20)` at q=0.25 reports one lower outlier and bulk fraction 0.95, despite a positive lower edge and ten additional true reduced-spectrum zeros.

**Fix/test:** separate scale-fitting samples from complete ESD samples throughout all adapters; label a conditional-positive KS explicitly if needed. Test exact zeros through analytic, KDE, FARMS, and Lanczos adapters and verify denominator/count consistency.

### CO-V2-15 — Medium: fit-status persistence is incomplete, and upper-outlier counts still have inconsistent definitions

**Partially fixed CO-24. Locations:** `rmt/mp.py:815-831`; `run_experiments.py:483-496,532-545`.

The new CSV stores a convergence boolean, but drops probe counts/edges, tail diagnostics, and result availability. A zero/unavailable result defaults to `mp_fit_converged=True` and status `available` in the row construction. Unconverged fits are accepted into subsequent conclusions without an explicit failure policy.

The Lanczos adapter still calls only separated poles `n_upper_outliers`, whereas `bulk_fraction` excludes every value above the edge. These values do not form the same partition, unlike the analytic fit's summaries.

**Fix/test:** persist the full fit/detector diagnostics and an explicit availability/status field; distinguish above-edge observations from thresholded spikes. Decide whether failed convergence aborts, marks unavailable, or is an explicit permissive mode. Test a nonconverged result and eigenvalues between the edge and threshold.

### CO-V2-16 — Medium: paper mode names still do not select their scientific method presets

**Partially fixed CO-11. Locations:** `run_experiments.py:916-931`; `pipelines/cli_config.py:157-178,258-263`.

Explicit negative boolean options are now respected, and paper 2 enables Delta3, but mode resolution still changes only booleans. Selecting `--experiment-mode reproduce_paper2` alone retains FARMS/Lanczos/spline methods rather than the Thamm/raw/Chebyshev protocol. Paper 1 and paper 3 likewise retain Golden method defaults unless the user expands the full documented command.

**Fix/test:** apply complete method presets before explicit user overrides, or make mode names metadata-only and require/validate explicit methods. Test resolved runtime configuration for each bare mode, not only parsing the already-expanded launcher commands.

### CO-V2-17 — Medium: plot execution and Brody labels do not honor the resolved diagnostic settings

**Partially fixed CO-10; additional method inconsistency. Locations:** `run_experiments.py:589-612,1013-1016`.

Spacing and overlap plots are still emitted whenever any artifacts exist, even when their diagnostics were explicitly disabled. The spacing plot also refits pooled spacings using default **MLE**, even if the recorded per-matrix method was `--brody-fit-method cdf_nls`; the legend gives no indication of that method change.

**Fix/test:** gate each plot independently and pass the chosen fit method to aggregate plot fitting, clearly distinguishing pooled estimates from per-matrix estimates. Use spies for disabled paths and compare CDF-NLS plot/CSV metadata.

### CO-V2-18 — High: direct runner invocations can mix new manifests with stale results

**Remaining output-safety issue; launcher mitigation is not sufficient. Locations:** `run_experiments.py:105-107,939-952,983,1001-1004,1013-1020`.

The new launcher refuses an existing job output root, but the supported Python CLI simply reuses its output directory. It immediately overwrites manifests and truncates training JSONL. Empty/disabled lesion output is not rewritten because `_write_csv` returns early; old lesion CSVs, plots, checkpoints, and cells can remain alongside new settings. A manifest-only invocation can replace the provenance of an earlier executed run without touching its metrics.

**Fix/test:** enforce fresh run directories in the runner itself, or implement explicit overwrite/resume modes with ownership checks and complete stale-artifact handling. Run once with lesions/multiple cells, then once without lesions/with fewer cells, and ensure files cannot be mistaken for one coherent run.

### CO-V2-19 — Medium: partial asset staging can now silently re-certify modified old assets with stale provenance

**Partially fixed CO-13. Locations:** `scripts/download_assets.py:276-292,353-359,384-397`.

Old asset/source metadata is preserved, but old file records are not verified before `_file_manifest` recalculates checksums for **every** file. If a previously staged WikiText array is corrupted or manually replaced, running `--assets synthetic` updates its checksum while retaining the original WikiText source revision/token metadata. Later verification succeeds against this newly sealed but false provenance.

**Fix/test:** verify untouched asset families against their existing records before any restaging; require explicit adoption/replacement metadata for changed files. Write manifests transactionally. Test staging A, modifying A, then staging unrelated B.

### CO-V2-20 — Medium: asset manifests are not portable from Windows staging to Linux execution

**New finding, migration-dependent. Locations:** `scripts/download_assets.py:287,309-310`; `run_experiments.py:138-144`.

`str(path.relative_to(root))` writes Windows backslashes on this platform. On Linux, `Path('data\\tokenized\\file.npy')` treats the backslashes as filename characters, so both the full verifier and selected-dataset lookup fail after transferring otherwise valid assets.

**Fix/test:** serialize platform-independent POSIX relative paths and validate/normalize supported older manifests. Test a Windows-produced manifest on POSIX; do not assume that a parseable/LF launcher makes staged data metadata portable.

### CO-V2-21 — Medium: the library's windowed-Hill cutoff/count metadata is fabricated from plateau width

**Partially fixed CO-30. Locations:** `rmt/tail.py:183-218,469-473`.

`hill_plateau_width` counts local-window estimates, not the number of tail observations or the lower cutoff of one fitted Pareto sample. `select_tail_estimator(..., 'hill_windowed')` now assigns that width to `n_tail` and uses `clean[width]` as `xmin`, even though each local estimate starts near rank 5 and spans `window` additional log spacings. These are not the observations supporting the reported exponent.

**Fix/test:** return the actual rank/window support and keep inapplicable single-tail fit fields unavailable. Do not repair missing metadata by inventing an unrelated cutoff. This is a library path; the current CLI does not expose `hill_windowed` as a tail solver.

### CO-V2-22 — Medium/conditional: the incompatible top-level `rmt` namespaces remain

**Unfixed CO-37, now mitigated in the compute launcher. Locations:** `rmt/__init__.py`; `run_experiments.py:41-64`; sibling `pyproject.toml:6,21-22`.

The launcher correctly isolates imports and checks `rmt.__file__`, but combined test collection, console/SDK use, or co-importing both projects still resolves only one package named `rmt` per interpreter. The APIs are incompatible. Reusing the Conda environment is not itself the bug; sharing the import name/process is.

**Fix/test:** use distinct package namespaces, or enforce/document separate-process entry points for every supported integration. Keep the launcher's import guard; test each project independently from its own root.

### CO-V2-23 — Low: migration documentation still asserts a NumPy incompatibility that the sibling fix removed

**Environment-change follow-up. Locations:** `README.md:22`; `other_requirements.md:98`; sibling `rmt/spacing.py:155-158`.

The active documentation says NumPy 1.26.4 necessarily breaks remote-sk Delta3 because it requires `np.trapezoid`. The current sibling implementation now falls back to `np.trapz`, so that specific incompatibility is no longer established. The old report/plan may remain historical, but active environment guidance should not require a separate stack for a defect already repaired.

**Fix/test:** update this factual claim while retaining the correct instruction to inventory and validate the live environment before changing packages. This is **not** a recommendation to downgrade the cluster stack.

## Repair status and limitations

Source inspection shows substantial repairs to initialization seeding, actual parameter-cap enforcement, validation batch limits, incremental/atomic cell persistence, pre-analysis checkpoints, subset lesion plotting, explicit negative boolean overrides, selected-dataset checksum verification, token dtype/shape checks, padding masks/left-padding loss, parameter estimation, AMP scheduler advancement, random FARMS window counts, CSN ties/two-sided KS, rank-revealing overlap bases, and local-coordinate Delta3. Do not undo these fixes while addressing the findings above. Their presence is not a full regression-suite certification.

Still-open design/scientific limitations from the first report:

- Checkpoints now include more RNG/scaler state, but there is still no resume entry point or saved DataLoader generator/permutation/position. They are recovery snapshots, not demonstrated exact resumable training.
- `C≈6ND` is an approximation; empirical D/N=20 allocation, repeated-token exposure, non-official held-out WikiText splits, FARMS sample dependence, and absolute Lanczos threshold calibration still need explicit scientific labeling.
- `xmax` filtering is not a bounded-Pareto likelihood, and nested truncated-tail inference still needs justified calibration.
- Live Conda package versions, CUDA/driver/toolchain compatibility, BF16, compilation, and memory headroom remain unverified. The migrated compute launcher now uses the recorded interpreter, partition, isolated paths, fresh job roots, and LF; those are improvements, not numerical certification.

## Recommended repair order

1. Fix CO-V2-01–06 and CO-V2-18 before another expensive allocation.
2. Add tiny **runner-level** regressions for budgets, collapsed regimes, degenerate matrices, spectrum combinations, and output reuse.
3. Repair precision, intervention matching, unfolding/uncertainty, and diagnostic provenance before interpreting paper/regime comparisons.
4. Run the complete suite from this directory in a separate process, then validate one small job on the actual cluster stack. The existing mostly helper-level tests do not exercise the major integration failures above.
