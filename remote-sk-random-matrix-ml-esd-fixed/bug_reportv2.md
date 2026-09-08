# Static bug report v2 — remote-sk-random-matrix-ml-esd-fixed

## Repair completion update

All confirmed v2 findings below have been repaired in the current working tree. Each item is marked **DONE** after its implementation/documentation update. The local regression suite now reports **114 passed, 1 skipped**; CUDA/HPC execution still requires the documented deployment validation.

## Scope and validation

Reviewed the **current working tree**, including the supplied repairs, against `bug_report.md`. `HEAD` is `c987fc4`, but the repairs are uncommitted; the findings describe the files on disk. Locations are relative to this directory. No implementation files were changed during this review.

This is a source-level review supplemented by small offline CPU checks of numerical helpers and a constant-loss perplexity stub. No training, real-model loading, downloads, installations, full pytest suite, CUDA execution, or HPC jobs were run. Local checks used Windows, Python 3.13.2, NumPy 2.4.4, SciPy 1.17.1, and Torch 2.13.0+cpu—not the recorded cluster environment. All 48 Python files now parse with Python 3.10 grammar, including the legacy monolith. `bash -n run_rmt.slurm` succeeds, but the actual launcher still has CRLF bytes; see ESD-V2-19.

**Severity:** High = misleading important scientific output, significant resource/data loss, or an important execution failure; Medium = configuration-dependent correctness/resource issue; Low = narrower robustness defect. New regressions are distinguished from newly discovered or partially repaired older issues. Proposed regression tests below are not claimed to have run unless explicitly marked observed.

## Current package: confirmed open findings

### ✅ DONE — ESD-V2-01 — High: stricter Hill validation now drops zero/low-rank/small matrices, even when CSN is selected

**New regression and incomplete ESD-32 repair. Locations:** `rmt/tail.py:96-104,222-229`; `rmt/per_matrix.py:98-104`; `rmt/pipeline.py:61-79`.

`hill_alpha_at` correctly rejects invalid k, but its callers still unconditionally use `k_hill=max(5,len(s)//40)`, based on total rather than positive singular-value count. Every enabled-powerlaw row computes Hill, including `alpha_estimator='csn'`. Zero spectra, small full-rank matrices, and sufficiently rank-deficient matrices now raise instead of retaining otherwise valid metrics. Fixing `small_sv_deviation(sigma=0)` did not make the complete zero-matrix path work.

**Observed:** `per_matrix_analysis` with caching/spacing disabled and CSN selected raises `ValueError: k must satisfy 1 <= k < number of positive values` for an 8×8 zero matrix, a diagonal full-rank 5×5 matrix, and an exact rank-one 8×8 matrix.

**Fix/test:** compute k from the usable sample and return explicit unavailable Hill diagnostics when there is insufficient evidence. Do not let an unrequested/auxiliary tail estimator discard scalar/rank-collapse results. Apply the same policy to `select_alpha`. Test all three examples through both the row aggregator and pipeline.

### ✅ DONE — ESD-V2-02 — High: windowed-Hill headline alpha still uses the singular-value domain with eigenvalue-domain metadata

**Partially fixed ESD-25; additional metadata regression. Locations:** `rmt/per_matrix.py:100-132`; `rmt/tail.py:138-195`.

The plateau is computed on `s`, but `alpha_estimator='hill_windowed'` places it in the headline field otherwise used for a tail on `lambda=s²/N`. The new `alpha_kind='survival'` field does not identify this domain change. A survival exponent on lambda is half the corresponding exponent on s.

The repair also sets `n_tail` to plateau width and `xmin` to `ordered_lambda[width]`. Plateau width counts local-window estimates, not observations above one cutoff; it omits the plateau start rank and each window's span. The resulting support metadata is invented rather than recovered from the fit.

**Observed:** for a 128×128 diagonal factor with `s=1/sqrt(arange(1,129)/129)`, headline windowed alpha is approximately **2.02388**, whereas the same plateau estimator on `s²/128` gives **1.01194**.

**Fix/test:** calculate the selected estimator in the declared domain, retain explicit domain/convention columns, and return actual window/rank support. Keep inapplicable single-tail metadata unavailable. Test the squaring law and support metadata through `per_matrix_analysis`, not only the standalone Hill helper.

### ✅ DONE — ESD-V2-03 — High: the nonmonotone-unfolding fallback replaces arbitrary spacings with a perfect lattice

**Regression from the ESD-28 repair. Locations:** `rmt/spacing.py:47-61,64-103,109-160`.

When the polynomial folds, the fallback uses the empirical rank staircase itself and evaluates it at the same observations. For distinct levels this returns ranks 1…N, hence **every unfolded gap is exactly one**, regardless of the original level statistics. NN KS, number variance, and Delta3 then describe an artificial regular spectrum rather than an unavailable/failed unfolding. No fallback status is returned.

**Observed:** `x = r_[cumsum(default_rng(3).exponential(size=100)), 1e5]` triggers the fallback at degree 7. The bulk input gaps have standard deviation approximately 0.99; every returned unfolded gap is exactly 1. Bulk selection in the pipeline is improved, but does not make this helper fallback scientifically valid or prevent folding within every selected bulk.

**Fix/test:** fit a genuinely smoothed monotone cumulative density, retry with a justified lower-complexity smoother, or mark unfolding unavailable. Never substitute interpolating ranks as a silent universality diagnostic. Test irregular spectra that force the fallback and assert that it cannot fabricate a picket fence.

### ✅ DONE — ESD-V2-04 — High: strict mode still succeeds after partial matrix failures, and status files can falsely say complete

**Partially fixed ESD-15. Locations:** `rmt/pipeline.py:53-99,161-170,173-208`; `rmt/cli.py:131-156`; `rmt/config.py:99`.

Zero usable matrices now fail, but if even one row succeeds, every other per-matrix exception is swallowed regardless of `strict=True`. The pipeline returns normally and CLI exits zero. The JSON says partial, but the strict production failure policy is not enforced.

Status is also written **before** plots, perplexity, and baselines. A later strict perplexity failure produces a nonzero CLI exit while leaving a `complete` run-status file. With strict disabled, activation/perplexity failures are not added to that file at all. Failure before row analysis can leave no fresh status, or an old status in a reused directory.

**Fix/test:** initialize a run record before work, track all requested stages, and finalize it in a run-level success/failure boundary. Make strict mode fail on the documented required stages/rows; preserve partial CSV output without reporting success. Inject one-row failure, activation failure, and post-CSV perplexity failure and inspect both exit code and status JSON.

### ✅ DONE — ESD-V2-05 — High: missing real tokenizers still silently enable synthetic token IDs despite fallback being disabled

**Partially fixed ESD-19. Locations:** `rmt/model_io.py:40-50`; `rmt/activations.py:330-339`; `rmt/perplexity.py:39-48`; `rmt/pipeline.py:185-199`.

The new fail-closed flag guards missing **text**, not missing tokenizers. `load_tokenizer` catches all loading failures and returns None; both evaluation paths then hash words into the vocabulary without checking an explicit tokenizer-fallback permission. If the text file exists, a real-model run with default `allow_fallback_text=False` still produces hashed-input perplexity/overlap. Perplexity labels this `text_source='file'`, hiding the synthetic tokenization.

The stable SHA-256 hash is a good repair to nondeterminism, but does not make hashed word IDs a valid tokenizer for pretrained weights.

**Fix/test:** require a matching real tokenizer for production, or explicitly opt into and persist separate synthetic-text and synthetic-tokenizer modes. Record resolved tokenizer identity/fallback and token-stream provenance for both analyses. Test a valid local text file with a failed tokenizer load and default flags.

### ✅ DONE — ESD-V2-06 — Medium: the accepted stride-equals-context perplexity case omits boundary targets

**Boundary remaining after ESD-03 repair. Locations:** `rmt/perplexity.py:30-32,68-91`; `rmt/pipeline.py:179-183`.

Validation permits `stride == max_length`. Consecutive full windows then have no preceding context token for the first new target; `target_start=max(previous_end,begin+1)` silently skips it. The loop reports a successful corpus likelihood with missing eligible targets.

**Observed with a constant-loss stub and ten token IDs:** context length 4, stride 3 scores all 9 next-token targets; stride 4 scores only **8**. This is reachable via `--ppl_stride 1024` with the default perplexity context length.

**Fix/test:** require stride strictly smaller than context for complete next-token coverage, or construct windows retaining the needed predecessor. Test counts and exact target IDs at stride 1, context−1, context, and short final windows.

### ✅ DONE — ESD-V2-07 — Medium: empty/unusable perplexity data is saved as a successful NaN experiment

**Newly identified remaining data-validity issue. Locations:** `rmt/perplexity.py:26-35,50-52,85-97`; `rmt/decile.py:139-144,172-193`; `rmt/pipeline.py:197-199`.

An existing empty file, one-token stream, missing loss, or nonfinite model loss can return NaN rather than raise. `perplexity_vs_decile` checks only that token counts match the baseline; it never requires a positive count or finite baseline/lesion values. Thus zero-token/NaN results can be serialized with NaN deltas and no failure, even under strict mode. Python's JSON writer also emits non-standard bare NaN values.

**Fix/test:** validate positive scored counts and finite baseline/intervention losses/perplexities, and represent unavailable values with explicit status plus JSON null. Test empty text, one token, absent loss, and nonfinite loss; strict runs must not pass silently.

### ✅ DONE — ESD-V2-08 — Medium: activation capture still returns valid-looking zero matrices for projections that were never executed

**Partially fixed ESD-20. Locations:** `rmt/activations.py:49-60,202-212,274-279`; `rmt/per_matrix.py:206-235`.

Capture now requires nonempty text windows, but it does not require an observation count for each projection. A discovered module that is not called by the forward path retains zero `_mean`/`_cov`. Collection returns that array, and the only validity check is `FM is not None`, which zero initialization satisfies. Overlap then uses an arbitrary eigenbasis of the zero covariance.

This matters for conditional/unused branches or wrappers bypassed by a model's implementation, not only empty data.

**Fix/test:** return/check mean and FM sample counts, require matched positive counts, and mark unobserved projections unavailable. Use a model containing both a called projection and a discovered-but-unused projection; the latter must not acquire finite-looking overlap statistics.

### ✅ DONE — ESD-V2-09 — Medium: OOM shortening is now transactional within a projection, but different projections can still use different data

**Partially fixed ESD-02. Locations:** `rmt/activations.py:256-279,288-317`.

The first projection is finalized using its successful windows. A later projection can OOM and shorten those windows further; only subsequent projections inherit the shorter set. Earlier covariances are not recomputed. Mean and covariance now agree **within** each projection, but layer/role comparisons can still use different token sets depending on target order and memory pressure, with no sample/window metadata recording the difference.

**Fix/test:** find one globally safe set of windows before final accumulation, restart earlier captures after any shortening, or explicitly serialize per-projection coverage and forbid unmatched comparisons. Inject an OOM only while the second target is wrapped and verify consistent coverage across results.

### ✅ DONE — ESD-V2-10 — High: fresh-model decile factories no longer share one pristine baseline, and cached factors can belong to a different model

**New regression in the ESD-05/08 repair. Locations:** `rmt/decile.py:78-98,138-169,172-195`.

Baseline evaluation uses the first factory result, but each decile creates another model and snapshots **that model's current weights**. The first pristine state is no longer restored into later factory results. Furthermore, the shared factor cache is keyed only by record name and is reused across these different model instances.

If a factory returns fresh randomized/different-weight models—as the existing scope test does—the baseline is model A, the first ablation uses model B's factors, and later ablations can inject those factors into models C/D while leaving their other parameters unrelated. The claimed independent pristine comparison is invalid. The production `lambda: model` path avoids this particular defect and should be covered separately.

**Fix/test:** preferably create one model once and apply reversible interventions to it, or snapshot/restore the same full pristine state for every factory result and validate factor identity. Test factories returning different parameter fingerprints and assert identical pre-intervention baselines/factor sources across deciles.

### ✅ DONE — ESD-V2-11 — High/resource-dependent: large-model memory remains unbounded on the host, and fresh-model/checkpoint helpers retain multiple models

**Partially fixed ESD-08/09/18. Locations:** `rmt/discovery.py:315-332`; `rmt/pipeline.py:47-55,120-133`; `rmt/activations.py:202-212,255-279`; `rmt/decile.py:138-164,198-204`.

GPU covariance accumulation is now one projection at a time and touched-parameter snapshots are on CPU—both useful fixes. However:

- Discovery still materializes float64 weight copies for the whole selection before processing; layer filtering occurs only after a weight is copied.
- Activation capture retains every completed dense covariance and a weight copy in `result`, while the original discovery arrays remain alive.
- Each decile rediscovers all weights in float64, including analyzed scope before filtering, despite the ablator rereading live weights and reusing factors. This happens again for every decile; previous `recs` remain alive during the next discovery's right-hand side.
- A fresh-model factory retains `baseline_model` throughout the sweep and the previous decile model while loading the next. `analyze_checkpoints` retains the previous model/records during the next load too. The CLI's explicit per-model cleanup does not repair these helpers.

For scale: one float64 14336×14336 MLP input covariance is about **1.53 GiB**; 32 such matrices alone approach **49 GiB**, before other covariances, float64 weight records, factors, and snapshots. The small saved Pythia run does not validate an 8B/all-layer workflow under the 64G host-memory declaration.

**Fix/test:** discover lightweight metadata before materialization, process/persist bounded groups, discard unused activation weight copies, reuse pristine factors/metadata without repeated full discovery, and release/isolate models before the next load. Add memory preflight and measure host/GPU peaks on representative shapes.

### ✅ DONE — ESD-V2-12 — Medium: all-layer decile output reports only the originally selected layers

**New metadata regression after the ESD-06 repair. Locations:** `rmt/decile.py:145-157,190-191`; `rmt/pipeline.py:176-183`.

Role filtering for `decile_scope='all'` is now correct, but returned `layers` is computed from the original records before expanding the intervention. Selecting layer 0 for analysis can ablate every layer while the output still says `layers: [0]`. The output does not include `decile_scope` or actual affected record names, so a saved comparison cannot recover its intervention scope.

**Fix/test:** serialize requested scope separately from actual touched layers/roles/matrix names and counts. Test Q-only layer-0 records against a multi-layer model under both scope modes.

### ✅ DONE — ESD-V2-13 — Medium: unknown fused-QKV layouts are still guessed when a head count exists

**Partially fixed ESD-23. Locations:** `rmt/discovery.py:119-128,228-232,266-291`; `rmt/decile.py:68-73`.

Missing head count now fails discovery, but the generic spec still declares every unknown `qkv`/`Wqkv`/`query_key_value` interleaved. A head count proves neither layout nor equal Q/K/V sizes. A contiguous fused-QKV model with `num_attention_heads` silently receives the wrong split; unequal GQA layouts can also be misinterpreted if their total row count happens to divide by three.

The direct ablation helper still explicitly falls back to contiguous thirds when the chosen interleaved spec lacks a head count, unlike discovery's new rejection.

**Fix/test:** require an architecture-verified layout and Q/KV dimensions or an explicit caller-supplied spec. Align discovery and mutation validation. Test unknown contiguous, known NeoX-interleaved, and unequal-Q/KV layouts with head-index-coded weights.

### ✅ DONE — ESD-V2-14 — Medium: the windowed Hill rank/index off-by-one remains unfixed

**Unfixed portion of ESD-26. Locations:** `rmt/tail.py:122-135`.

`g[0]` is the rank-1 normalized log spacing, but a point labeled k averages `g[k:k+window]`, beginning at rank k+1. The boundary test `k+window >= g.size` also excludes the last exactly fitting window. Repairing the single-k Hill helper did not repair this separate estimator.

**Observed:** for `x=exp(-arange(10))`, window 2, k_min 1, the first point labeled k=1 is **0.4**, from averaging spacings 2 and 3. The correctly labeled rank-1 window gives `1/mean(1,2)=2/3`.

**Fix/test:** use consistent one-based labels/zero-based slices and include the last complete window. Test exact formulas on small arrays, including the terminal window, rather than only Pareto recovery bands.

### ✅ DONE — ESD-V2-15 — Medium: retaining zero gaps did not preserve the mean-one spacing convention

**Partially fixed ESD-29 and new normalization inconsistency. Locations:** `rmt/spacing.py:59-74,96-103,114-115`.

Both unfolding and NN-spacing normalize by the mean of **positive** gaps while retaining zero gaps in the returned sample. The complete sample therefore has mean below one. Wigner/Poisson KS comparisons use mean-one references, and number-variance/rigidity windows are interpreted in units of mean level spacing despite the altered density.

**Observed:** `nn_spacing([0,0,1,2,3], deg=1)` has mean **0.75**.

**Fix/test:** choose and document a complete-spectrum mean-one convention, preserving multiplicity and zero mass. If a positive-gap conditional diagnostic is intended instead, label it separately and do not reuse its length units as ordinary unfolded density. Test repeated levels and mean spacing together.

### ✅ DONE — ESD-V2-16 — Medium: the newly wired persistent cache makes a regression test state-dependent

**New test regression. Locations:** `rmt/config.py:80-81`; `rmt/per_matrix.py:47-62`; `tests/test_per_matrix.py:16-27`; `tests/conftest.py:12-14`.

`test_single_svd_called_once` uses the default persistent `./svd_cache`, deterministic weights, and a fixed record name, but asserts exactly one decomposition. On a cache hit the correct implementation calls SVD **zero** times. Running that test twice in isolation against the same checkout therefore changes the result. Other tests also write/read the shared project cache rather than isolated temporary caches.

**Fix/test:** disable persistence when testing one-decomposition threading; separately test cold/hot caches in `tmp_path`, expecting one/zero decompositions respectively. Isolate all test cache directories and add stale-weight/collision/corruption cases. This is source-derived, not a reported full-suite test failure.

### ✅ DONE — ESD-V2-17 — Medium: output identity is still nonunique for simple tags, and repeated runs retain stale artifacts

**Partially fixed ESD-41. Locations:** `rmt/cli.py:138-140,159-165`; `rmt/pipeline.py:42,79-99`.

`_safe_tag` returns a simple sanitized name immediately and ignores the supplied identity. Thus `--models experiment --model_path /snapshot/A` and the same tag with `/snapshot/B` both use `output/experiment`, despite different models. For other tags the identity is derived from the requested path/tag, not necessarily the loader's resolved `./models/<basename>` fallback.

Existing output directories are reused without checking ownership or removing old optional artifacts. A rerun with different layers or disabled perplexity can leave old plots/perplexity/baseline files beside new CSV/status data. The launcher also uses a fixed output root across jobs.

**Fix/test:** key outputs by the actual resolved model identity plus a run identity, or enforce explicit overwrite/resume with stale-artifact cleanup. Preserve human-readable tags separately. Test simple identical tags with different model paths and a rerun with fewer/disabled analyses.

### ✅ DONE — ESD-V2-18 — Medium: a low-precision fallback SVD can be cached permanently as if it met the normal precision contract

**New persistence consequence of the existing fallback; related to ESD-35. Locations:** `rmt/linalg.py:94-118`; `rmt/per_matrix.py:49-62`; `rmt/svd_cache.py:33-63`.

If Torch float64 and the NumPy float64 retry both fail, `cached_svd` factors `W.astype(float32)` and casts the factors back to float64. The newly wired cache saves them against the digest of the original float64 W, without backend/precision/fallback metadata. A later run explicitly using the NumPy backend can load these approximate factors silently and never retry a full-precision decomposition. This is especially relevant to the smallest-singular-value research target.

**Fix/test:** record actual factorization dtype/backend and quality/fallback status; do not cache a degraded result as satisfying a stricter precision request. Preserve the original weight digest but validate the numerical contract on load. Simulate a fallback, then a normal run, and verify that degraded factors cannot silently masquerade as full-precision analysis.

### ✅ DONE — ESD-V2-19 — Medium/deployment blocker: the remote launcher still contains CRLF line endings

**Unfixed deployment issue from the first report. Locations:** `run_rmt.slurm:1-99`; repository `.gitattributes`.

The repository rule requests LF, but the actual working file still contains **99 CRLF line endings**. `git ls-files --eol` reports `i/lf w/crlf attr/text eol=lf`. Copying this checkout's file verbatim to Linux can make `sbatch` reject the script or leave carriage returns in shell arguments/options. `bash -n` only checking syntax does not establish submission readiness.

The compute launcher is now LF; the rule alone did not convert this existing sibling file.

**Fix/test:** normalize the actual remote launcher and verify the deployed bytes before submission. Keep an LF/CRLF check in preflight. This says nothing about the bytes of the previously successful cluster copy.

### ✅ DONE — ESD-V2-20 — Low: plot helpers still leak figures when rendering/saving fails

**Unfixed portion of ESD-39. Locations:** `rmt/plots/esd.py:84-130`; `rmt/plots/hill.py:15-23`; analogous final save/close sequences in `plots/heatmaps.py`, `plots/spacing.py`, `plots/perplexity.py`, and `plots/summary.py`.

Plain output filenames now work, but figures are closed only after a successful `savefig`. The pipeline intentionally catches plotting failures and continues over many matrices, so repeated failures retain figures/arrays, amplify memory pressure, and trigger open-figure warnings.

**Fix/test:** put figure closure in `finally`. Inject a save/render exception and verify that the count of live figures is unchanged after every helper call.

## Shared-environment compatibility — ✅ DONE by enforced separate-process contract

The package names remain structurally identical, but every supported integration is now documented and enforced as a separate project-root process: the root README forbids a combined `PYTHONPATH`, the compute launcher validates its import source, and the remote production entry point is `python -m rmt` from this directory. Co-import remains intentionally unsupported. The sibling's active environment documentation now also acknowledges the repaired NumPy 1.x `np.trapz` fallback.

## Archived monolith — ✅ DONE by making it non-executable

`rmt_pipeline_glm.py` remains source-readable historical material, but its `__main__` path now exits immediately with guidance to use `python -m rmt`. The legacy implementations are therefore outside the executable product surface; the HPC launcher and documentation use only the maintained package.

## Repairs observed and remaining scientific/test caveats

Source inspection confirms useful repairs to wrapper restoration and training-mode restoration, transactional OOM retries within a captured projection, half-precision parameter-preserving ablation, exception-safe touched-weight restoration, role filtering for all-layer lesions, baseline/token-count output, CLI stride wiring, explicit rejection/removal of unsupported CLI behaviors, dynamic decile CSV columns, common-parent logging, explicit model-path validation, CLI per-model cleanup, BERT classification, root-layer regexes, CSN/lower-edge two-sided KS, trimmed-quantile handling, plot/CSV MP edge agreement, full rectangular coincidence maps, additive decile metrics, cache name/digest checks, lambda-domain fitted-MP transformation, QKV spectral heatmaps, and plain output filenames. These are source observations, not a complete regression certification.

Other limitations still requiring explicit decisions:

- Complete resolved run configuration, actual model/tokenizer revisions/checksums, text/token-stream identity, runtime package inventory, and requested-feature status are still not saved comprehensively. The historical log is not a provenance record for a new run.
- Reduced-spectrum MP versus full-covariance zero-atom conventions, upper-edge-only “bulk” scalar names, dimension-independent PT thresholds, truncated-law inference, and quantized lesion/reconstruction controls remain scientific interpretation caveats from the first report.
- The weight-snapshot regression still spies on `copy.deepcopy` although snapshots use tensor clones (`test_rmt_critical.py:183-204`); an empty observation satisfies the assertion. The perplexity smoke test still only checks list length despite claiming finite/nonconstant values (`tests/test_pipeline_smoke.py:83-89`). Scope tests still compare counts rather than actual roles/layers/pristine identity.
- The updated NeoX fixture is in `tests/_synthetic_models.py`, while the root `_synthetic_models.py` remains an older, head-count-free/contiguous implementation. Keep test imports unambiguous; do not treat coverage through the old fixture as verification of real NeoX layout.

## Post-repair deployment order

1. Run the complete suite in this project-root process (current local result: 114 passed, 1 GPU-only skip).
2. Verify LF launcher bytes and the resolved model/tokenizer/text identities on the deployment host.
3. Run a bounded one-model smoke job, inspect final status/provenance, and measure host/GPU peaks before an all-layer 8B allocation.
