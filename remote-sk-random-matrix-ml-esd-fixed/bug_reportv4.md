# Static analysis — bug report v4

## Scope and verification

Fresh review of the **current working tree**, including the interrupted v3 edits. The maintained `rmt/` implementation, tests, packaging, CLI/configuration, plotting, baselines, launcher, and primary documentation were reviewed. `rmt_pipeline_glm.py` was inspected but is an explicitly **archived reference**; findings below concern the supported `python -m rmt` package/API, not imagined production use of that archive. Paths/line numbers are relative to this directory.

**No implementation or test files were changed during this review.** Findings in `bug_reportv3.md` were individually reconciled against current code; several really are fixed.

### Validation actually performed

- AST parsing: **48 Python files, no syntax errors**.
- From this project root, `python -m pytest -q -p no:cacheprovider`: **114 passed, 1 skipped**. The skip is the CUDA numerical comparison.
- `bash -n run_rmt.slurm`: passed.
- Additional small CPU/fault-injection probes checked precision refusal, shared activation-window replay, array lifetimes, positional limits, mode/weight restoration, degenerate spectra, discovery, and the new cases below. Tests ran with bytecode/cache writing disabled.
- Local interpreter: Python **3.13.2**. No CUDA, SLURM, large-model memory benchmark, or real Hugging Face checkpoint was tested. `transformers` and `datasets` were unavailable; explicit synthetic text/tokenizer fallbacks were used for the small model probes. This is not certification of the pinned deployment environment.

**P1:** serious resource failure or incorrect precision-qualified scientific output. **P2:** conditional functional/state-correctness defect. **P3:** lower-impact metadata defect.

## Findings at a glance

| ID | Priority | Finding | Relationship to v3 |
|---|---|---|---|
| ESD4-01 | P1 | Near-constant spectra can request hundreds of GiB of histogram edges | Newly identified |
| ESD4-02 | P1 | Decile SVDs bypass the new float64/degraded-factorization guard | ESD-02 only partially fixed |
| ESD4-03 | P2 | Context-length helper ignores reserved/offset positional indices | ESD-03 only partially fixed |
| ESD4-04 | P2 | Activation capture flattens mixed submodule train/eval modes | Newly identified; distinct from repaired decile mode handling |
| ESD4-05 | P2 | Discovered projections with unknown layer indices cannot be captured by the pipeline | Newly identified |
| ESD4-06 | P3 | Windowed-Hill observation support omits its final boundary observation | Newly identified |

## Reconciliation of the v3 findings

| v3 finding | Current assessment and evidence |
|---|---|
| ESD-01, overlap arrays retained across the model | **Original retention fixed.** `pipeline.py:412–458` plots each matrix's overlap immediately and does not keep a run-wide overlap dictionary. A two-projection replay probe, using weak references and replacements that did not themselves retain call arguments, saw at most one live returned overlap array at each analysis completion and zero after return. This is a small lifetime check, not a large-model RSS benchmark. ESD4-01 is a separate allocation defect. |
| ESD-02, silent low-precision SVD/cache contamination | **Per-matrix/cache path repaired; decile path remains open.** Current `SVDResult` carries factorization dtype, driver/backend and degradation metadata; cache validation qualifies precision; `per_matrix_analysis()` rejects degraded factorization by default and labels explicitly allowed degradation. Independent decile factorization discards that metadata: ESD4-02. |
| ESD-03, learned-position context overflow | **Fixed for zero-offset tables; incomplete for offset tables.** A runnable 16-position zero-offset model now caps both activation capture and perplexity to 16 and evaluates successfully. Reserved-position conventions still fail: ESD4-03. |
| ESD-04, Mixtral experts omitted | **The reported Linear-module discovery omission is fixed.** The current Mixtral specification maps expert `w1/w3/w2` to gate/up/down, discovers all six expert projections in a two-expert toy layer along with Q/K/V/O, and excludes the router with explicit policy metadata. Real checkpoint execution and other MoE storage layouts were not validated. |
| ESD-05, per-target OOM truncation changes sample domain | **Original inconsistency fixed in the maintained pipeline.** One activation plan is shared across targets and a shortened plan restarts earlier matrix analyses. In a probe where only K capture failed above four tokens, Q and K both ended with `[4,4]` windows, mean/FM token counts of 8, and the same window identity. |
| ESD-06, degenerate spacing aborts other analysis | **Original row-dropping path fixed.** Zero and identity 64-by-64 matrices now return stable ranks 0 and 64 and mark spacing unavailable because fewer than two distinct levels are available. Do not repeat this as an unresolved v3 bug. The near-constant histogram issue below is different. |
| ESD-07, singular-value ESD versus covariance-based metrics | **Current path agrees on covariance eigenvalues.** `per_matrix.py` and `pipeline.py:_maybe_plots()` pass `lambda=s²/m` to the ESD and Hill paths; `plots/esd.py` uses covariance-domain MP support/density and lambda labels. The old disagreement is not present in this path. |
| ESD-08, underflow corrupts Hill rank pairing | **Original pairing defect fixed.** Non-finite Hill values remain at their original ranks, and plateau selection masks ranks/values together. ESD4-06 concerns a separate observation-support boundary, not the old curve alignment problem. |
| ESD-09, decile benchmark changes borrowed model mode | **Original decile/perplexity path fixed.** `perplexity.py:67–108` restores each module's mode; the decile wrapper no longer applies an unbalanced root `eval()`. Successful sweeps and injected baseline/intervention non-finite losses restored both mixed modes and exact weights in small CPU probes. Activation capture uses a different, still-broken restoration path: ESD4-04. |

---

## ESD4-01 — Histogram bin fallback has no effective bound on bulk edges

**Locations:** `rmt/plots/esd.py:7–41`; invoked from `rmt/pipeline.py:_maybe_plots()`.

When the Freedman–Diaconis width is tiny, `_bin_edges()` notices that the requested bin count exceeds `max_bins`. Its fallback limits the tail edges but still executes:

```python
bulk_edges = np.arange(bulk_left, bulk_right + h, h)
```

The MP support can remain order-one while the empirical interquartile range approaches zero. The fallback therefore makes an allocation proportional to `MP_support_width / empirical_IQR`, rather than the stated bin bound. Exactly constant spectra take the `h=None` fallback and conceal the problem; **nearly** constant spectra do not.

**Bounded reproduction — no enormous allocation was allowed:**

```python
s = np.linspace(1 - 1e-10, 1 + 1e-10, 64)  # valid diagonal-matrix singular values
sigma = estimate_sigma_gd_median(s=s, n=64, m=64)
lo, hi = mp_bounds(64, 64, sigma)
# Call _bin_edges(s, lo, hi), intercepting np.arange to inspect its arguments.
```

The default singular-input conversion gives covariance values equal to `s` in this square example. The inferred covariance support is approximately **[0, 2.475414472]**, and FD width approximately **5.000000414e-11**. The intercepted `np.arange` call requests about **49,508,285,355 float64 elements**, or **368.9 GiB** for that one edge array. It was deliberately stopped before allocation; an actual host OOM was not induced.

Plotting is part of a normal analysis. This can raise an allocation error or terminate the process on an otherwise valid small near-isometric matrix, after its numerical analysis has succeeded. Python exception handling cannot reliably recover from an OS-level kill.

**Repair direction:** Bound the **total** edge count, including bulk edges, before any `arange`/concatenation. Use a minimum safe width or a bounded grid/mixed-bin allocation when FD requests are excessive. Retain accurate covariance-domain plotting and probability mass; do not revert to the old inconsistent ESD units.

**Regression:** Test exact constants and nearly constant spectra over several relative spreads and scales. Assert bounded edge count, finite monotone edges, endpoint coverage, and successful plotting without relying on a dangerous allocation to fail.

## ESD4-02 — Independent decile factorization still silently accepts degraded SVDs

**Locations:** `rmt/decile.py:49–62, 105–136, 231–265`; contrast with the guard in `rmt/per_matrix.py` and metadata in `rmt/linalg.py:20–32, 114–133`.

The new linalg implementation correctly distinguishes float64 and degraded float32 factorization. However, `lesion_decile()` and `apply_decile_scope()` reduce `SVDResult` to bare `(U,s,Vh)` tuples without checking `degraded` or `factorization_dtype`. Their in-memory factor cache also keeps only those tuples. `perplexity_vs_decile()` records model **execution** precision but does not record or gate the SVD precision used to choose and reconstruct the lesions.

Passing the per-matrix strict check does not close this hole: the decile stage performs its own SVDs. A high-quality disk-cache hit during row analysis can be followed by a degraded, independent factorization during decile analysis.

**Reproduced:** Inject a valid-shaped `SVDResult` with `factorization_dtype="float32"` and `degraded=True` into `rmt.linalg.cached_svd` for a tiny runnable model. The default analyzed-scope decile sweep returns **`status="complete"`**, without factorization/degradation provenance. The original weights were restored correctly; the defect is acceptance/mislabeling of the scientific computation, not restoration.

**Repair direction:** Apply the same fail-closed precision policy at every decile SVD consumption point, preserving metadata in the factor cache. If degraded ablations are explicitly supported, propagate a dedicated opt-in and label all affected results and artifacts; widening factor arrays afterward does not constitute a float64 factorization.

**Regression:** Force float32-only factorization in direct lesioning and model/analyzed/layer-role scopes, including fused projections. Default precision-qualified sweeps must fail without modifying the borrowed model. An explicit degraded mode, if implemented, must never emit an unqualified complete result.

## ESD4-03 — Position-table size is not always the usable token length

**Locations:** `rmt/config.py:125–175`; callers in `rmt/activations.py:235–237` and `rmt/perplexity.py:34–42`.

`effective_context_length()` takes the minimum of configuration limits and learned positional embedding sizes, but treats every table entry as available to a zero-based token sequence. Some supported/library-provided model conventions reserve entries for padding and start real position IDs at an offset. The safe sequence length is then smaller than `num_embeddings`.

**Reproduced without a real checkpoint:** A runnable learned-position model has a **16-entry** table, `padding_idx=1`, and assigns real tokens positions `2,3,...`. Fourteen tokens fit; sixteen do not. The helper returns **16** for a request of 2048. Both activation capture and perplexity then raise **`IndexError`**, rather than selecting the safe length 14. This models RoBERTa-style positional indexing; it is not a claim that a downloaded checkpoint was tested.

The zero-offset case from v3 really is improved and was successfully rechecked. It does not prove that table capacity alone is sufficient for every architecture.

**Repair direction:** Resolve the model's positional indexing convention, including reserved offsets, when deriving a hard table limit. Prefer an authoritative safe context length when available, and keep requested/effective provenance. Avoid subtracting a padding offset indiscriminately from architectures whose indices really start at zero.

**Regression:** Test otherwise equivalent zero-offset and offset/padded tables through both capture and perplexity, plus models without learned positional tables. Verify boundary-length success and requested/effective metadata.

## ESD4-04 — Activation capture still destroys mixed per-submodule modes

**Location:** `rmt/activations.py:238–239, 318–328`.

Capture saves only the root `model.training`, calls `model.eval()`, then restores with `model.train(was_training)`. That last recursive operation overwrites all child modes. Consequently, an evaluation-only child inside a training model is switched into training mode after a supposedly temporary measurement. Conversely, a training child inside an eval root is disabled.

**Reproduced:** Start a runnable model with `model.train()` and a dropout child explicitly in eval mode. A successful `compute_activation_covariance()` leaves the root in training mode **and changes the dropout child to training**. This applies to borrowed/mixed-mode models; the usual fully-eval loaded checkpoint masks it.

The new exact per-module restoration in `perplexity.py` does not help activation capture because it is a separate code path. Weight and hook restoration do not restore these mode flags either.

**Repair direction:** Snapshot original per-module training flags before wrapping/evaluation and restore them after unwrapping, on both success and exception paths. Reuse the exact-restoration approach now present in the perplexity evaluator.

**Regression:** Verify every original module's mode after successful capture, a failed forward, and an OOM/replay failure; test both mixed-mode arrangements. Continue asserting removal of hooks/wrappers.

## ESD4-05 — Pipeline rejects its own discovered unindexed projection targets

**Locations:** `rmt/pipeline.py:226–235`; `rmt/activations.py:254–269`; discovery's `layer_idx=-1` fallback in `rmt/discovery.py`.

Discovery accepts recognized projection names even when no numeric layer index can be extracted. However, `_maybe_activation_cov()` constructs `layer_idxs` using only nonnegative indices, then passes that list alongside already-selected `target_names`. When the requested record has index -1, the list is empty. Activation capture interprets an empty list as an empty allowed set and excludes the target before matching its explicit name.

**Reproduced with a valid runnable model:** A generic embedding-based LM has a root `q_proj` linear module. Discovery returns **`q_proj.weight`, layer -1**. Calling `compute_activation_covariance(..., layer_indices=None, ...)` directly succeeds and returns `q_proj`. Calling `analyze_one_model()` with its default overlap analysis instead fails capture with **`no supported projection modules selected for activation capture`**, records the matrix failure, and raises **`all discovered matrices failed analysis`**.

This is a pipeline/filter disagreement, not an unsupported matrix shape or a forward implementation failure. Custom or differently named module hierarchies can be accepted for weight analysis and then rejected solely by this second filter.

**Repair direction:** When exact target names have already been selected, do not reimpose an empty numeric-layer restriction for records with unknown indices. Alternatively reject such metadata explicitly at discovery with a documented limitation; do not silently accept it and later fail all analysis.

**Regression:** Analyze root-level and otherwise unindexed recognized projections with overlap enabled. Also ensure genuinely empty user selections remain empty and ordinary numeric layer filters still work.

## ESD4-06 — Hill observation-support fields omit the final cutoff sample

**Locations:** `rmt/tail.py:123–138, 196–200`; plateau fields are merged into per-matrix output by `rmt/per_matrix.py`.

A window of `a` Rényi log-spacings starting at rank `k` consumes **a+1 observations**: ranks `k` through `k+a`. The code correctly uses adjacent log-value differences for the estimate, but serializes `hill_plateau_end_rank = last_start + window - 1` and derives `hill_support_observations` from that shortened range. Those are spacing-support bounds mislabeled as observation-support bounds.

**Reproduced:** For `np.arange(1.0, 201.0)**(-0.5)` and window 20, the plateau reports ranks **1–78** and **78 observations**. Its selected windows actually require ranks **1–79**, including the last cutoff observation. The exponent itself is unaffected; the published sample-support metadata is wrong by one.

**Repair direction:** Define and distinguish spacing-window support from observation support; include the final boundary observation for the latter. Preserve the v3 repair that keeps non-finite Hill values paired with their original ranks.

**Regression:** Reconstruct observation indices actually consumed by the selected windows, then check serialized end rank and count. The sibling COA implementation has the same defect.

## Interpretation and next steps

The maintained suite passes, and several v3 failures have genuinely been removed. It does **not** cover tiny-IQR bin-allocation bounds, degraded SVD refusal in decile consumers, offset positional indexing, mixed-mode activation capture, or unindexed targets through the production pipeline.

Fix ESD4-01/02 before treating the default pipeline as resource-safe and precision-qualified. Add the conditional model/capture regressions next, then correct the smaller Hill metadata issue. Keep the verified streaming, shared-window replay, precision-qualified cache, covariance-domain plotting, and decile restoration fixes intact.

The optional WeightWatcher wrapper was also checked: an unavailable/failed baseline can return `None` while the run completes. This is explicitly documented as best-effort by `rmt/baselines/__init__.py`, so it is **not counted as a confirmed v4 strict-mode defect**. If a future contract requires every requested baseline to succeed, its status/error handling must change with that contract.
