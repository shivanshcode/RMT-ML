# Static bug report — remote-sk-random-matrix-ml-esd-fixed

## Scope, validation, and the recorded HPC run

Reviewed the current `rmt/` package, tests, packaging, plots, launcher, requirements, saved job logs, and the older `rmt_pipeline_glm.py`. Locations below are relative to this directory.

This is a **static review**, not a rerun of your cluster job. Of 48 Python files, 47 parse with Python 3.10 grammar using the local Python 3.13.2 AST parser. The legacy monolith fails at its notebook magic on line 1. `bash -n run_rmt.slurm` succeeds. No pytest suite, numerical experiments, installations, downloads, or HPC jobs were executed. Example triggers/regression checks below are source-derived, not claimed test outcomes. Static review cannot guarantee discovery of every possible defect.

The saved evidence is important:

- `RMT_All_Models_376402.log:1-4` records `/home/shivansh/.conda/envs/rmt_ml_env/bin/python`, a successful selftest gate, and the completion message.
- `RMT_All_Models_376402.error:2-3` records a **missing WikiText file**, fallback perplexity text, and a written metrics CSV with **18 rows**.
- This establishes a successful exercised path, not correctness of every metric or every optional branch. The CSV/perplexity outputs themselves are not present in this checkout. Exact installed dependency versions, driver, CUDA toolkit, and GPU memory are not recorded.
- `run_rmt.slurm` runs `python -m rmt`, **not** the legacy monolith. Legacy defects below must not be attributed to that successful job.

**Severity:** High = incorrect important scientific output, model corruption, substantial lost computation, or an execution blocker; Medium = configuration-dependent incorrect behavior; Low = narrower robustness/documentation problem.

## Current package: activation capture and ablation

### ESD-01 — High: activation wrapping is not exception-safe

- **Locations:** `rmt/activations.py:193-227`; `rmt/pipeline.py:143-156`.
- Linears are replaced before text loading/forwards, but restored only at the successful end. Tokenization errors, exhausted OOM retries, failed covariance copies, or even allocation failures halfway through replacement leave some/all `FeatureLayer`s installed. The pipeline catches the error and proceeds, so later perplexity/discovery/baseline work runs on a modified model with potentially huge active covariance buffers.
- The function also switches the supplied model to eval mode and never restores its original training state, even on success.
- **Fix/check:** wrap the complete replacement/capture lifecycle in `try/finally`, restore modules and mode after any exit, and make partial replacement transactional. Inject failures during replacement, tokenization, both passes, and collection.

### ESD-02 — High: OOM retries double-count partial forwards and can use different data for mean/covariance

- **Locations:** `rmt/activations.py:65-99,211-221,230-243`.
- Earlier layers update accumulators before a later layer throws OOM. `_safe_forward` retries a shortened input without rolling those updates back, so earlier layers receive both failed-prefix samples and the retry. Mean and FM passes retry independently, and shortened successful windows are not saved for replay. The frozen mean can therefore describe different samples from the covariance pass.
- **Fix/check:** preselect safe windows before collecting statistics, or stage/commit accumulators only after successful complete forwards and replay exactly the same windows for both passes. Test an OOM raised after an early captured layer has executed.

### ESD-03 — High: sliding-window perplexity counts overlapping targets repeatedly

- **Locations:** `rmt/perplexity.py:52-68`; `rmt/pipeline.py:159-166`.
- Every window passes `labels=ids` and scores all `length-1` targets. With default `max_length=1024`, stride 512, interior targets are counted twice, sometimes with different available context, while edge targets receive different weighting. The output is not the advertised strided corpus likelihood.
- **Fix/check:** track the previous scored end, mask already-scored targets with `-100`, and count valid **shifted** labels exactly. Validate stride/context bounds. Use a deterministic fake loss/tokenizer and verify every eligible target is scored once.

### ESD-04 — High: ablation promotes only the weight and breaks half-precision models

- **Locations:** `rmt/decile.py:76-87`; `rmt/model_io.py:9-38`; `rmt/config.py:46`; `rmt/cli.py:28`.
- FP16/BF16 weights are replaced with new float32 Parameters while input activations and bias remain half precision. Ordinary `F.linear`/Conv1D execution without autocast can then fail with dtype mismatches. Restoration through `load_state_dict` copies values into the new float32 parameter and does not restore dtype, parameter identity, ties, or external optimizer references.
- The supplied job explicitly uses FP32, so it does not exercise this defect. The library loader's own default is still FP16, unlike the CLI's FP32 default.
- **Fix/check:** make the whole ablation evaluation consistently FP32, or preserve execution dtype with a clearly labeled precision limitation; do not silently replace only selected Parameters. Test an actual forward after FP16 and BF16 ablation and verify dtype/identity restoration.

### ESD-05 — High: failed decile evaluation leaves the model ablated

- **Locations:** `rmt/decile.py:111-137`; `rmt/pipeline.py:159-189`.
- Pristine weights are restored only after the loop, not in `finally`. An SVD, assignment, or perplexity failure leaves the currently supplied model modified; the production factory is `lambda: model`, and the pipeline catches the failure and continues. A subsequent WeightWatcher baseline can therefore analyze an unintentionally ablated model.
- The final restoration also suppresses restoration errors, hiding an invalid state.
- **Fix/check:** use a reversible context for each decile and an unconditional restoration boundary. Report restoration failures prominently. Test evaluation failure halfway through the sweep against exact original weights.

### ESD-06 — Medium: `decile_scope='all'` ignores the requested matrix types

- **Locations:** `rmt/decile.py:100-127`; `EXECUTIVE_SUMMARY.md`, “Decile ablation”.
- The documented scope is all layers of the **analyzed types**, but the implementation rediscovers every supported matrix and never filters by the supplied records' roles. Passing only Q records still ablates K/V/O/MLP weights. `_rebind_records` also ignores an explicitly supplied custom spec by resolving its own.
- **Fix/check:** preserve the selected roles/spec when expanding scope across layers. Test Q-only records and assert every touched record is Q, rather than only checking that the all-scope count is larger.

### ESD-07 — Medium: ablation output cannot measure pristine perplexity impact

- **Locations:** `rmt/decile.py:109-138`; `rmt/plots/perplexity.py:6-15`.
- Only lesioned perplexities are evaluated/saved. There is no pristine baseline, delta, or actual token-count metadata. All selected roles are lesioned together rather than producing independently attributable role-wise effects. The raw plot is correctly labeled perplexity, but these outputs alone cannot reproduce a baseline-relative, role-wise lesion-impact comparison.
- **Fix/check:** evaluate and save the pristine baseline on exactly the same tokens, add deltas and the precise affected role/layer scope, and expose independent role-wise sweeps if paper reproduction is intended.

### ESD-08 — High/resource-dependent: ablation bypasses the SVD backend/cache and repeatedly materializes all weights

- **Locations:** `rmt/decile.py:21-28,111-145`; `rmt/discovery.py:298-330`; `rmt/svd_cache.py`; `rmt/config.py:80-81`.
- Each decile rediscovers float64 copies of all weights (also in analyzed scope, before filtering), then reconstructs by a new **NumPy CPU SVD** from the live module. Ten deciles repeat expensive decompositions. `backend`, `gpu_svd_min_dim`, `use_svd_cache`, and `svd_cache_dir` never control this path.
- The pristine snapshot clones every `.weight` entry on its existing device, including embeddings/head that are not ablated; tied state-dict names can also be cloned separately. This adds roughly another model's weight footprint on GPU. The float64 discovery arrays are unnecessary because the ablator rereads module weights.
- **Fix/check:** snapshot only touched unique Parameters, use CPU/disk snapshots when appropriate, reuse pristine SVD factors across deciles, and discover lightweight metadata without duplicate arrays. Respect configured backends/cache and measure peak RAM/VRAM on the target model.

### ESD-09 — High/resource-dependent: activation covariance allocates all selected dense buffers simultaneously

- **Locations:** `rmt/activations.py:44-59,114-141,180-190`; `rmt/discovery.py:298-330`; `rmt/pipeline.py:43-53`.
- Every selected linear receives a float64 `d_in x d_in` buffer on the model device before a forward; all float64 weight records are already materialized on the host. For an MLP input width 14336, a single covariance buffer is about **1.53 GiB**, before temporaries and model weights. Selecting all layers can OOM during wrapper construction; shortening text cannot reduce that fixed allocation.
- **Fix/check:** capture a bounded group of layers/modules at a time, share identical input covariances where valid, support CPU accumulation, and preflight quadratic buffer memory. Combine with ESD-01 so allocation failure does not corrupt the model. The small saved Pythia run does not validate large-model memory safety.

### ESD-10 — Medium: GPT-2 weights are supported, but GPT-2 activation overlap is not

- **Locations:** `rmt/discovery.py:109-117,197-214`; `rmt/activations.py:123-125`.
- Discovery supports transposed HF `Conv1D` projections, but activation capture wraps only `nn.Linear`. GPT-2 therefore produces no projection covariance/overlap despite `do_overlap=True`, without a specific unsupported-module error.
- **Fix/check:** use input hooks that preserve original module forwards for both Linear and Conv1D, or implement a correctly oriented Conv1D adapter. Test nonempty GPT-2 overlap results, not only discovery shapes.

## Configuration, loading, failure reporting, and HPC

### ESD-11 — Medium: multiple accepted CLI/config switches have no implementation

- **Locations:** `rmt/config.py:55-96`; `rmt/cli.py:45-89`; `rmt/pipeline.py:95-128,143-166`; `rmt/activations.py:193-208`.
- The automatically generated parser exposes options simply because fields exist. Source searches show:
  - `ppl_stride`, `ppl_dataset`: never forwarded/read by evaluation.
  - `fm_dataset`, `fm_token_weighted`: forwarded as arguments, but capture ignores `dataset_name`/`token_weighted` and always loads `text_path` with token-weighted accumulation.
  - `use_svd_cache`, `svd_cache_dir`: not wired to analysis/ablation.
  - `do_finetune_recovery`, `ft_method`, `ft_steps`, `ft_task`: no recovery implementation.
  - `pythia_steps`, `epoch_probe_fracs`, `epoch_checkpoint_every_frac`: not used by the CLI to load checkpoints or dispatch epoch tracking. The separate library helper does not make those CLI switches effective.
- **Fix/check:** implement each promised behavior or remove/reject the flags. Add behavioral tests, including a spy proving the requested perplexity stride reaches evaluation.

### ESD-12 — Medium: arbitrary decile counts are written through a hard-coded ten-decile CSV schema

- **Locations:** `rmt/per_matrix.py:197-198,210-233`; `rmt/pipeline.py:294-299`; `rmt/config.py:76`.
- `n_deciles` controls computed fields, but CSV columns always contain groups 1–10 and `extrasaction='ignore'` silently discards later groups. Values greater than ten lose data; smaller values retain meaningless empty columns. Invalid zero/negative values are not rejected early.
- **Fix/check:** build the schema from configuration or enforce exactly ten. Test actual CSV contents for 4, 10, and 12 groups, not only returned dictionaries.

### ESD-13 — High for affected installations: NumPy 2-only code is allowed with NumPy 1.x dependencies

- **Locations:** `rmt/spacing.py:133`; `pyproject.toml:10`; `requirements.txt:7`; `rmt/selftest.py:20-69`.
- `np.trapezoid` was added in NumPy 2.0, but package metadata/requirements have no minimum. NumPy 1.26 satisfies the declared dependencies yet raises `AttributeError` in Delta3. Since default per-matrix spacing invokes Delta3, rows can be dropped by the broad exception handler.
- The selftest never calls Delta3, so its gate can succeed in precisely such an environment. **Do not downgrade the working ESD environment to the sibling project's NumPy 1.26.4 pin.**
- **Fix/check:** declare an appropriate NumPy minimum or use a compatible integration function, and add API/minimum-version coverage to the gate. The logged 18 rows are consistent with this exercised API being available, but do not establish the exact installed NumPy version.

### ESD-14 — Medium: only the first logger is configured

- **Locations:** `rmt/config.py:136-151`; `rmt/cli.py:20`; `rmt/selftest.py:17`; `rmt/pipeline.py:25`.
- One global `_LOGGER_CONFIGURED` flag is shared across different named loggers, but the handler and INFO level are attached to only the first logger. In CLI execution that is normally `rmt.cli`; sibling `rmt.selftest`/`rmt.pipeline` loggers do not inherit from it. INFO diagnostics disappear and warnings can use the unformatted last-resort handler. This matches the sparse saved log, but is independently evident from the configuration.
- **Fix/check:** configure the common `rmt` parent once and allow child propagation, or configure each named logger idempotently. Test capture of both selftest and pipeline INFO records.

### ESD-15 — High: empty/failed analyses can still exit successfully

- **Locations:** `rmt/pipeline.py:58-74,159-189`; `rmt/cli.py:124-131`.
- Any exception in an individual metric drops the entire row. No discovered matrices, invalid layer selections, or all rows failing still produce header-only CSV/summary output and return success. Requested overlap/perplexity failures are swallowed as warnings, with no structured run-status artifact or CLI failure policy.
- **Fix/check:** record failed matrix names/stages and requested-feature status, reject zero usable matrices, and provide a strict mode for production. Do not silently discard valid earlier metrics because one optional diagnostic failed. Test missing layers, all-row failure, and requested perplexity failure.

### ESD-16 — High/conditional: the SLURM script can report success after a failed main command

- **Locations:** `run_rmt.slurm:25,35-49,63-93`.
- There is no strict shell mode. `cd`, Conda activation, and interpreter probing can fail without aborting. The selftest has an explicit guard, but the main analysis command does not; the final successful `echo` can leave the batch exit status zero after Python fails.
- The fallback interpreter is `$CONDA_BASE/envs/rmt/bin/python`, inconsistent with the activated `rmt_ml_env` and the actual recorded `/home/shivansh/.conda/envs/rmt_ml_env/bin/python`.
- **Fix/check:** use strict mode and validated paths, explicitly select the known environment, and propagate the main command's exit status. Do not hide all module errors without verifying CUDA afterward.

### ESD-17 — Medium: model-loading fallback can load a different snapshot than requested

- **Locations:** `rmt/model_io.py:53-59`; `rmt/cli.py:124-129`.
- A missing explicit `model_path` falls back to `./models/<basename(name_or_path)>`, even when the operator intended the explicit path. Also one global `model_path` is reused for every `--models` tag, so multiple differently named output directories can contain repeated analysis of the same snapshot.
- **Fix/check:** fail for an invalid explicit path, restrict a single explicit path to a single model or support a tag-to-path mapping, and record the resolved snapshot/revision in every output. Test a misspelled path beside a valid basename fallback directory.

### ESD-18 — Medium/resource-dependent: multi-model loading retains the previous GPU model

- **Locations:** `rmt/cli.py:124-130`; `rmt/pipeline.py:108-120`.
- In `model = load_model(...)`, the previous `model` remains referenced while the next loader constructs and moves the next model to GPU. Thus two models can briefly coexist and OOM even when each fits individually. Checkpoint loading has the analogous assignment pattern, plus retained discovery arrays.
- **Fix/check:** explicitly release old models/records and any references before loading the next model, or isolate each model in a subprocess. Test multi-model peak memory.

### ESD-19 — Medium: fallback tokenization is nondeterministic and provenance is incomplete

- **Locations:** `rmt/activations.py:252-266,275-281`; `rmt/perplexity.py:23-37,74-79`; `rmt/model_io.py:41-50`; `rmt/pipeline.py:167-181`.
- The claimed deterministic tokenizer uses Python `hash(word)`, which changes between processes unless `PYTHONHASHSEED` is fixed; `cfg.seed` does not control it. A missing real tokenizer silently turns real-model evaluation into modulo-hashed word IDs.
- Only perplexity text-file existence is tagged. A present text file with a missing tokenizer is still marked `text_source='file'`, concealing synthetic token IDs. Activation overlap has no equivalent fallback-text/tokenizer provenance. The saved run demonstrably used fallback text for perplexity.
- **Fix/check:** require real assets for production, reserve fallback mode for explicit tests, use a stable hash when needed, and save text/tokenizer identities, checksums, token counts, and fallback status for both analyses.

### ESD-20 — Medium: empty activation data becomes a valid-looking zero covariance

- **Locations:** `rmt/activations.py:193-227,265-272,180-190`.
- Empty/very-short text, zero batches, or invalid stride/window settings can yield no usable batches. Capture still returns zero-initialized covariance matrices and overlap is then computed against an arbitrary eigenbasis of the zero matrix. Numeric finiteness is not evidence that observations existed.
- **Fix/check:** validate all counts/lengths/strides, require positive sample counts for every returned covariance, and represent missing data as unavailable diagnostics rather than zero covariance.

### ESD-21 — High when co-imported: the two repository projects collide on package name `rmt`

- **Locations:** `pyproject.toml:6,21-25`; `rmt/__init__.py`; sibling `compute_optimal_analysis/rmt/__init__.py`.
- Both projects expose different packages named `rmt`, with different APIs and conventions. Combined test collection, wrong PYTHONPATH order, or reuse of a process that already imported the other package can cause import errors or wrong helper resolution.
- **Fix/check:** use distinct package namespaces; until then run from the correct project directory in separate Python processes and assert `rmt.__file__`. The sibling `to_change_env.md` explains safe environment reuse without mixing packages.

## Current package: discovery and scientific metrics

### ESD-22 — Medium: BERT attention output is misclassified as an MLP down projection

- **Locations:** `rmt/discovery.py:98-106,177-184`.
- BERT's `attention.output.dense` matches both the specific O pattern and the broad D pattern `output.dense`. Classification checks D **before** O, so O becomes D. Role summaries and type-scoped interventions are wrong.
- **Fix/check:** prefer the most specific full pattern or context-aware classification. Assert that `bert.encoder.layer.0.attention.output.dense` is O and the block's separate `output.dense` is D.

### ESD-23 — Medium: unsupported fused-QKV layouts are guessed rather than rejected

- **Locations:** `rmt/discovery.py:119-129,220-291`; `rmt/decile.py:62-65`.
- The corrected registered GPT-NeoX path handles known head-interleaving, but the generic spec assumes interleaving for unknown `qkv`/`Wqkv` modules. Many such models use contiguous or unequal GQA blocks. If head count is missing, discovery warns and guesses contiguous thirds; ablation makes the same fallback without warning. A wrong guess silently relabels/ablates Q/K/V.
- **Fix/check:** require verified layout and Q/K/V sizes, make unsupported layouts explicit, and reject uncertainty in production. Test known contiguous, NeoX-interleaved, and unequal-Q/KV layouts. The current Pythia split with known heads is **not** the old contiguous-thirds bug.

### ESD-24 — Medium: layer regexes fail on valid root-level module paths

- **Locations:** `rmt/discovery.py:72-129,188-194,317-320`; `tests/test_discovery.py:91-115`.
- Patterns require a preceding dot, so root paths such as `layers.0.self_attn.q_proj` or `h.0.attn.c_attn` get layer index -1. Applying a legitimate layer filter then silently excludes them. The root GPT-2-like fixture only checks shapes, not layer indices, so it misses this.
- **Fix/check:** match start-of-string or dot before the layer component, and test root and nested forms with filters.

### ESD-25 — Medium: headline alpha changes exponent convention without changing its metadata/plot label

- **Locations:** `rmt/per_matrix.py:89-105`; `rmt/tail.py:214-239`; `rmt/plots/summary.py:19`; `rmt/pipeline.py:302-313`.
- Hill selectors replace only `row['alpha']` with a **survival** exponent. `xmin`, KS, and `n_tail` remain from the CSN density fit, and the plot still labels the result `α (CSN, λ)`. CSV/summary files do not persist the selected estimator/config. The same headline field is therefore not comparable across selector choices.
- **Fix/check:** retain separate explicit density/survival columns and estimator/cutoff metadata; derive plot labels and summaries from the selected method. Test both metadata and numeric values.

### ESD-26 — Medium: `hill_alpha_at` silently evaluates a different k

- **Locations:** `rmt/tail.py:72-99,105-131`.
- `hill_alpha_at(k)` searches a curve capped near `n//2` and returns the nearest available point. Valid `k` beyond that cap, or k=1, is silently replaced. Invalid k can also return a seemingly valid estimate instead of an error. The windowed helper labels `k` but starts its Rényi slice at zero-based index `k`, i.e. rank k+1, and excludes the last exactly fitting window.
- **Fix/check:** compute the single-k statistic directly for `1 <= k < n`, validate other inputs, and align window rank labels. Check exact formulas on a small known sorted array.

### ESD-27 — Medium: power-law/lower-edge KS calculations omit one side of the empirical jump

- **Locations:** `rmt/tail.py:58-65`; `rmt/mp.py:240-247`.
- CSN uses midpoint empirical ranks rather than both one-sided empirical CDF values; lower-edge KS uses only the upper side. Neither is the exact two-sided one-sample KS statistic. CSN's sample-size-dependent bias can change selected xmin across candidates.
- **Fix/check:** use the two-sided calculation already exemplified by `rmt/spacing.py:_ks_against`, and validate against SciPy on fixed samples.

### ESD-28 — High for affected spectra: polynomial unfolding can fold the spectrum and fabricate level statistics

- **Locations:** `rmt/spacing.py:32-53,76-83,89-135`; `rmt/per_matrix.py:150-158`; `rmt/plots/spacing.py:12-18`.
- An unconstrained raw-coordinate degree-7 polynomial need not be monotone, particularly for heavy-tailed spectra with large outliers. Negative gaps enter `nn_spacing` and its normalization. KS later drops negative spacings without renormalizing the survivors, while the plot retains them. Delta3/number variance sort the folded coordinates, silently changing level order.
- The caller labels these as bulk statistics but passes the **entire** covariance spectrum, including separated outliers, rather than a selected bulk. Extreme outliers can dominate the staircase fit.
- **Fix/check:** select and label the bulk, use scaled/monotone unfolding with checks, preserve ordering, and use the identical valid spacing sample for metrics and plots. Test an otherwise regular bulk with a few very large outliers.

### ESD-29 — Medium: gap-ratio statistics discard zero gaps and join nonadjacent gaps

- **Location:** `rmt/spacing.py:18-26`.
- Removing `d == 0` before forming adjacent ratios erases real degeneracies and creates new neighbors across them. The resulting r-statistic overstates repulsion for repeated eigenvalues/quantized or rank-collapsed spectra.
- **Fix/check:** define and report zero-gap handling explicitly; preserve adjacency, treating only truly undefined 0/0 pairs specially. Add repeated-level regression cases.

### ESD-30 — Medium: trimmed sigma estimation still matches the untrimmed MP median

- **Locations:** `rmt/mp.py:143-160,167-193`; `rmt/pipeline.py:229-245`.
- Both explicit `discard_largest` and iterative refinement take the median of a truncated sample but divide by the original full-law median. Even pure MP data deliberately trimmed by a fraction then produces a downward-biased scale. Refinement can compound the effect when actual bulk values are discarded.
- **Fix/check:** account for retained quantile mass and the assumed contamination model, rather than treating all truncation as a full MP sample. Validate on known MP quantiles with a controlled upper fraction removed, as well as planted spikes.

### ESD-31 — Medium: ESD plots and CSV outlier metrics use different fitted edges

- **Locations:** `rmt/per_matrix.py:64-85`; `rmt/pipeline.py:229-245`.
- CSV edges, outlier counts, and small-SV metrics use `sigma_med`, but plots prefer `sigma_med_refined`. The plot does not serialize its selected edges or identify them as a different fit. Visual and tabular conclusions can disagree even before the refinement issue above.
- **Fix/check:** select one consistent fit or persist/label both complete edge/count sets and use explicit legend metadata.

### ESD-32 — Medium: zero matrices can disappear rather than being reported as rank collapsed

- **Locations:** `rmt/per_matrix.py:64-85`; `rmt/mp.py:89-96,208-244`; `rmt/pipeline.py:58-66`.
- A zero spectrum yields sigma zero and coincident MP bounds. `small_sv_deviation` eventually calls `mp_median(..., sigma=0)` with a degenerate bracket and raises. The pipeline drops the row, removing precisely the kind of rank-collapse evidence the tool should retain.
- **Fix/check:** explicitly handle zero/rank-deficient spectra, preserving valid scalar metrics and unavailable fit status. Test a zero discoverable linear layer.

### ESD-33 — Medium: rectangular coincidence maps discard activation eigenvectors

- **Locations:** `rmt/overlap.py:82-106`.
- For wide W (`n < m`), `k=min(n,m)` and the coincidence map uses only `evecs[:, :k]` although all m activation directions live in input space. A right singular vector whose strongest match is outside the top k is forced to choose an incorrect match. `rho_top_singular_vs_evals` also ignores the rest of the activation spectrum. The separate `overlap_analysis` uses all columns, so the summaries disagree in scope.
- **Fix/check:** compute the full `k x m` map and explicitly define any top-k-restricted diagnostic. Test a wide factor with a strongest overlap in activation column m-1.

### ESD-34 — Medium: per-decile values are standalone tranche metrics, not global contributions

- **Location:** `rmt/scalars.py:147-166`.
- Each decile's entropy and stable rank is normalized by that decile's own energy/maximum, although the docstring calls stable rank a contribution and the summary calls this a breakdown. These values cannot be summed to recover global metrics; even a negligible-energy decile can have maximal local entropy/rank.
- **Fix/check:** either rename/document them as within-tranche metrics or calculate additive contributions using global normalization. Test sums of reported contributions, rather than the current test which only sums independently computed raw energies.

### ESD-35 — Medium: cache filenames are not unique or tied to weight identity

- **Locations:** `rmt/svd_cache.py:12-35`.
- Names such as `a.b` and `a_b` map to the same filename. Cache keys omit model/checkpoint/weight digest, and load does not verify even the saved original `name` or factor dimensions. A library caller can retrieve another matrix's factors. This cache is currently unwired to the production pipeline, so this is a library/future-integration defect rather than evidence that the saved run used stale cache entries.
- **Fix/check:** key by collision-resistant model/checkpoint/weight identity and validate metadata/factors; use atomic writes. Add collision and stale-weight tests.

## Current package: plotting and test coverage

### ESD-36 — Medium: modified-MP fitting is wrong in the eigenvalue plot domain

- **Locations:** `rmt/plots/esd.py:83-105`; `rmt/mp_fit.py:48-88`.
- For `domain='lambda'`, `vals` is `s**2/N`, but `mp_mode='fit'/'both'` passes those eigenvalues to a model explicitly defined in the **singular-value domain**. It neither fits original singular values nor applies the density Jacobian. The default production plot uses the nu/theory branch and does not exercise this bug.
- **Fix/check:** fit in nu then transform the fitted edges/density with the correct Jacobian, or define an eigenvalue-domain model. Verify integral and domain-change consistency.
- Related fit robustness: bounds permit `nu_max <= nu_min`; a bad initial/degenerate spectrum can produce a flat zero model with apparently converged optimizer status. Validate fitted ordering and residual quality.

### ESD-37 — Medium for extreme tails: the histogram bin cap reintroduces bulk smearing

- **Locations:** `rmt/plots/esd.py:17-32`.
- Bulk FD width is converted to a bin count capped at 400, then bins are spread over the **entire** range including extreme outliers. Once that cap is reached, width grows with the largest outlier rather than remaining the bulk FD width. This partially reintroduces the very plotting artifact the comment says was fixed.
- **Fix/check:** separate bulk and tail panels/ranges, use appropriate nonuniform bins, or state the effective bin-width cap. Test a narrow bulk plus an extreme outlier and inspect actual edge differences.

### ESD-38 — Medium: the advertised QKV heatmap is a singular-spectrum line plot

- **Locations:** `rmt/plots/heatmaps.py:7-17`; `rmt/pipeline.py:258-278`.
- `plot_qkv_heatmap` calls `ax.plot` on three singular-value arrays. It neither renders a heatmap nor receives weight matrices. Enabling `--do_qkv_heatmap` therefore does not produce the advertised Q/K/V matrix visualization.
- **Fix/check:** rename the flag/output to spectral comparison or actually supply bounded weight blocks and render the requested heatmaps. A file-existence test alone cannot verify plot semantics.

### ESD-39 — Low: public plot helpers fail for a plain output filename

- **Locations:** `rmt/plots/esd.py:110`; `rmt/plots/heatmaps.py:15,27`; analogous `hill.py`, `spacing.py`, `perplexity.py`, `summary.py`.
- `os.makedirs(os.path.dirname('plot.png'))` receives an empty string and raises. Production usually supplies a directory, but valid-looking library calls fail. Exceptions after figure creation can also leave figures open because close is not in `finally`.
- **Fix/check:** create the parent only if nonempty (or use `Path(...).parent`) and always close figures. Test both plain and nested paths.

### ESD-40 — Medium: tests do not verify several claimed scientific invariants

- **Locations:** `tests/_synthetic_models.py:10-13,41-62,115-152`; `tests/test_decile.py:48-58,85-118`; `tests/test_mp.py:77-91`; `test_rmt_critical.py:160-201,208-259`; `tests/test_pipeline_smoke.py:60-89`.
- Pythia fixtures omit head count and implement contiguous `chunk(3)`, so tests take the warned fallback rather than exercising actual head-interleaving. Add head-index-coded weights and a NeoX-like forward.
- The FP16 ablation regression checks reconstructed numbers but never performs a forward or verifies dtype/Parameter restoration; it misses ESD-04.
- The weight-only snapshot regression spies on `copy.deepcopy`, but the implementation uses tensor clones; an empty `seen` dictionary satisfies the assertion without observing the snapshot.
- The MP density test calls a **mean** of binwise absolute integral errors “L1”; a true integrated L1 error is the sum, so the threshold is weakened by the bin count. Its histogram is also conditionally renormalized to the bulk while theory is multiplied by the bulk fraction.
- The pipeline regression describes finite/nonconstant perplexity but only asserts list length. Scope tests only compare counts, missing wrong roles. Add exception, token-accounting, role, minimum-version, and real-layout tests.
- Historical test-count/pass claims in `EXECUTIVE_SUMMARY.md` are not a test result for this checkout/environment.

### ESD-41 — Medium: model-output tag sanitization is nonunique and accepts parent-directory tags

- **Locations:** `rmt/cli.py:124-135`; `rmt/pipeline.py:68-74`.
- Distinct model identifiers such as `a/b` and `a_b` map to the same output tag and directory, so the later model overwrites the earlier model's results. The special local path `..` also survives `_safe_tag` unchanged and makes `os.path.join(output_dir, tag)` resolve outside the requested output directory.
- **Fix/check:** use a readable stem plus a digest of the resolved model identity, reject special dot-directory stems, and prevent accidental reuse of another model's output directory. Test colliding identifiers and local dot paths.

## Older reference monolith — not the current HPC entry point

All locations in this section are in **`rmt_pipeline_glm.py`**. Except LEG-01, these are latent defects in the body that would matter if the notebook magic were removed or the body were executed in a notebook. Keep it clearly archived, or repair it independently; do not confuse it with the improved package.

| ID / severity | Location | Trigger, consequence, and required correction |
|---|---|---|
| **LEG-01 High** | **1** | Literal `%%writefile rmt_pipeline_glm.py` is IPython cell magic, not Python syntax. Direct execution/import fails immediately. Remove the magic from a runnable script or store this as a notebook/text reference. Confirm with AST/compile checks. |
| **LEG-02 High** | 508-523, 902-909 | Pythia discovery and ablation split/write contiguous thirds, which is wrong for GPT-NeoX head-interleaved QKV. Use the current package's verified head-layout logic. |
| **LEG-03 High** | 621-634, 1068-1084, 1550-1553, 1596-1607 | Covariance keys are module names, but lookups use parameter names containing `.weight` and sometimes fused tags. Ordinary and fused overlap/coincidence blocks are therefore skipped. Normalize keys consistently. |
| **LEG-04 High** | 558-580 | Mean and FM passes share `self.computation`. After b mean batches and b FM batches, covariance is scaled by roughly one half; Gram matrices also lack token normalization and means weight batches equally. Use independent sample counters and consistent sample-weighted covariance. |
| **LEG-05 High** | 584-610, 675-720 | Wrappers are never restored or disabled; later forwards keep accumulating expensive covariance. OOMs skip partial forwards after earlier accumulators were updated, again corrupting statistics. Use reversible, transactional capture. |
| **LEG-06 Medium** | 1515-1536, 1720-1727, 1741-1742 | `--no_overlap` does not skip covariance: default-true `do_qkv_heatmap` also triggers capture, and its `store_true` parser has no disabling form. Decouple cheap QKV plots from activation capture and honor explicit opt-outs. |
| **LEG-07 High** | 1033, 1175-1190, 1271-1290 | “WW-style” CSN is fit on singular values, not covariance eigenvalues; Hill is a survival exponent on singular values. Both are compared to the same WW density-exponent boundaries 2 and 6. Convert domains/conventions or label distinct quantities; do not reuse those boundaries unchanged. |
| **LEG-08 Medium** | 1036-1041 | The reported Hill value at `sqrt(N)` uses `searchsorted(...)-1`, selecting the preceding k when the requested k exists. Index the exact point. |
| **LEG-09 Medium** | 1087-1089 | `mp_softrank` is actually energy below the edge divided by total energy, and appears only if overlap is available. Rename it bulk energy fraction; compute true selected soft-rank semantics independently of covariance availability. |
| **LEG-10 Medium** | 778-815 | The random-overlap band ignores `sigma_level` and uses a signed Gaussian CDF for an absolute maximum. Use the absolute-Gaussian CDF and a clearly defined requested family-tail probability. |
| **LEG-11 Medium** | 846-860 | After masking already-scored prefixes, perplexity always subtracts one from valid target count. Later overlapping windows already have their first position masked, so this undercounts by one; count valid shifted labels directly. OOM-skipped windows also advance `prev_end`, silently changing evaluated coverage. |
| **LEG-12 Medium** | 373-390 | Zero rows are removed from the average despite the stated convention that they contribute zero. Average over all rows, and distinguish this absolute-entry entropy from the package's squared-entry entropy. |
| **LEG-13 Medium** | 1297-1323 | The outlier plot indexes rows by integer layer, then checks `if sh not in sub.index` where sh is Q/K/etc. Every series is skipped. Removing that guard alone still overlays all roles at the same bar coordinates; use actual grouped/stacked offsets. |
| **LEG-14 Medium** | 1243-1261 | Negative or zero perplexity deltas are clipped to positive `1e-3` for a log bar chart, hiding improvements and fabricating positive damage. Use a signed/symlog/linear presentation. |
| **LEG-15 High for offline reuse** | 654-669, 826-831, 1451-1461, 1649-1653 | Model/tokenizer/dataset loaders do not enforce local-only assets and use `trust_remote_code=True`. This is a connected Colab workflow, not a substitute for the offline package launcher. Do not run it as the air-gapped workflow; explicitly stage/pin assets and audit any remote code before opting in. |
| **LEG-16 High/resource-dependent** | 1493-1497, 1620-1637, 914-951 | The original model and activation buffers remain resident while the lesion factory loads another full model on the same GPU. Repeated NumPy SVDs also occur across metrics, plots, and each lesion. Release/partition resources and reuse pristine factors. |
| **LEG-17 Medium** | 1488-1489, 899-911 | GPU loading always chooses BF16, including the advertised T4 target, and reconstruction is stored back to the original low precision. Hardware support is not checked and small-SV effects can be dominated by quantization. Choose a verified device/precision policy. |
| **LEG-18 Medium** | 285-298, 337-366 | Trimming still matches an untrimmed MP median; power-law fitting uses one-sided KS and divides by a possibly zero log sum for constant spectra. Use corrected quantile/KS logic and degenerate-spectrum handling. |
| **LEG-19 Medium** | 1744-1773 | The main loop catches every model failure and continues to “All models done” with a successful process exit. Return failure/partial-failure status and a structured error manifest. |
| **LEG-20 Low** | 1370-1375 | Raw MathText strings contain doubled backslashes before `cdot`/`lambda`, unlike the single command slash MathText expects. The optional summary plot can fail at render time. Use valid MathText strings and a render test. |

## Additional scientific and deployment caveats

- **Full covariance versus reduced ESD:** `rmt/mp.py:102-137` returns only `min(n,m)` eigenvalues and a unit-mass nonzero MP law. For tall W, the full `WW^T/N` also contains `n-m` structural zeros and continuous mass `m/n`. The formulas are useful for the reduced/nonzero law, but documentation must not equate it with the full covariance ESD without the zero atom and mass convention.
- **One-sided “bulk” scalars:** `ipr_summary` and `bulk_mass_frac` use only the upper edge (`rmt/scalars.py:72-86,124-130`). Genuine lower outliers are included. If the intended statistic means inside the two-sided MP support, pass and apply the lower edge too; otherwise explicitly label “below upper edge”.
- **PT classification is heuristic:** `rmt/scalars.py:89-114` calls any KS distance below 0.1 random, independent of vector dimension. That is not a calibrated significance level; use finite-dimensional null calibration before interpreting it as a statistical acceptance rate.
- **`xmax` semantics:** CSN discards values above xmax but still fits an unbounded Pareto. This is not a likelihood normalized on a bounded interval. Also preserve likelihood-ratio ordering: `LR_trunc` here favors the truncated law for positive values; the sibling internal comparison uses the opposite ordering.
- **Model registry is not universal:** aliases such as Mixtral/Qwen-MoE can use expert projection names not covered by the chosen dense-model patterns; unknown/GQA fused layouts need explicit support. Check discovered role counts against the actual architecture rather than trusting a nonempty list.
- **Offline is not reproducible by itself:** no immutable model/text revision, source digest, complete resolved configuration, package inventory, or runtime environment is saved with current metrics. Optional WeightWatcher import/analysis exceptions are also suppressed inside its wrapper without an explicit result status. Add provenance and feature-status artifacts.
- **Local LF/CRLF issue:** both SLURM files are LF in Git but CRLF in this Windows working tree (`git ls-files --eol`). Normalize before copying to Linux. This is not evidence about the line endings of your previously executed cluster copy.
- **Documentation drift:** `README.md` still says fused QKV is always contiguous thirds, contradicting the correct registered NeoX implementation. `FOLDER_GUIDE.md` describes `docs/`, model outputs, and archives not included in this checkout. Update these references so absent results and obsolete conventions are not mistaken for current evidence.

## Suggested repair order

1. Fix exception-safe restoration, OOM accounting, correct strided perplexity, half-precision ablation execution, and reliable exit/status reporting.
2. Wire or reject inactive flags; fix NumPy minimum/API compatibility without disturbing the already working environment.
3. Repair role/layout handling and numerical/plot metadata consistency; add the targeted regression checks above.
4. Keep the legacy monolith clearly non-production unless separately repaired.
5. Reuse the cluster environment only after inventorying it. The concrete migration checklist for the sibling project is `../compute_optimal_analysis/to_change_env.md`.
