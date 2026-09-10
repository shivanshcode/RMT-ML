# Static-analysis bug report — remote-sk-random-matrix-ml-esd-fixed

## Scope and verification

Reviewed the **current working tree**, including the pre-existing uncommitted repairs and `tests/test_bugfix_v4.py`, against base commit `c359c9c`. This report records remaining issues; it does not assume deleted historical bug reports are still applicable. The findings are retained below as an audit trail; implementation fixes are now included in the working tree.

Scope: supported `python -m rmt` entry point, discovery/model loading, matrix aggregation, activation capture, decile intervention/perplexity, numerical core, persistent caching, plots, optional baselines, checkpoint tracking, launcher, and related tests. The explicitly non-executable `rmt_pipeline_glm.py` archive is not treated as a production entry point. Paths/line numbers below are relative to this directory and refer to the reviewed tree.

Supplemental verification, isolated from the sibling package:

- After remediation, `python -m pytest -q -p no:cacheprovider`: **130 passed, 1 skipped**; the skipped test requires CUDA.
- Ruff checks `F821,F822,F823`: passed. `bash -n run_rmt.slurm`: passed.
- Small CPU-only probes reproduced the failures identified as reproduced below.
- Environment: Python 3.13.2, NumPy 2.4.4, SciPy 1.17.1, Torch 2.13.0+cpu. Transformers and CUDA were unavailable. No real snapshot download/load, accelerator run, or cluster submission was performed.
- The existing tests wrote into the tracked `svd_cache/` directory. Only those test-generated cache changes were restored to their previously clean state; pre-existing user changes were preserved.

Severity: **P1** = high-impact incorrect result/intervention; **P2** = conditional correctness/reliability defect; **P3** = lower-priority execution-contract defect. Static-only findings are explicitly labeled.

## Resolution status

- [x] **ESD-01 — Fixed**
- [x] **ESD-02 — Fixed**
- [x] **ESD-03 — Fixed**
- [x] **ESD-04 — Fixed**
- [x] **ESD-05 — Fixed**
- [x] **ESD-06 — Fixed**
- [x] **ESD-07 — Fixed**
- [x] **ESD-08 — Fixed**
- [x] **ESD-09 — Fixed**
- [x] **ESD-10 — Fixed**
- [x] **ESD-11 — Fixed**

## Prioritized findings

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| ESD-01 | P1 | Tied projection parameters are lesioned multiple times | CPU reproduction |
| ESD-02 | P2 | Activation name blacklist contradicts discovery and exact targeting | CPU reproduction |
| ESD-03 | P2 | Valid experiment seeds can fail the stochastic selftest gate | CPU reproduction |
| ESD-04 | P2 | Degenerate covariance eigenbases create arbitrary overlap results | CPU reproduction + linear-algebra analysis |
| ESD-05 | P2 | More deciles than singular values creates successful no-op interventions | CPU reproduction |
| ESD-06 | P2 | Random-control alpha does not follow the selected estimator/convention | CPU reproduction |
| ESD-07 | P2 | Corrupt optional cache files prevent recomputation and fail matrices | CPU reproduction |
| ESD-08 | P2 | `xmax` tail fitting uses an unbounded likelihood/CDF | Deterministic numerical reproduction |
| ESD-09 | P2 | Missing/failed requested baseline is silently reported as complete | Injected optional-stage reproduction |
| ESD-10 | P3 | Checkpoint tracking ignores the requested SVD backend | Static call-path analysis |
| ESD-11 | P2 | Fresh output-directory ownership is not atomic | Static concurrency trace |

## ESD-01 — FIXED — Tied projection weights receive multiple, cumulative lesions

**Locations:** `rmt/decile.py:97-154,208-218`; discovery at `rmt/discovery.py:337-367`.

**Trigger/root cause:** Different projection modules can share the same `Parameter`. Discovery returns one record per module name. `set_layer_svd_decile` processes every record, reading the current live weight and caching factors by **record name**. A second alias therefore factors an already-lesioned parameter and removes another tranche. The benchmark deduplicates snapshot storage by parameter identity, but not the actual interventions or factor-cache identity.

**Reproduction:** Create separate `q_proj` and `k_proj` linear modules, assign `k_proj.weight = q_proj.weight`, and initialize the shared float64 weight to `diag(4, 3, 2, 1)`. Discover both records and call:

```python
set_layer_svd_decile(model, records, decile=4,
                     n_deciles=4, factor_cache={})
```

Observed singular values: **`[2, 1, 0, 0]`**. Removing the largest quarter once should leave `[3, 2, 1, 0]`.

**Impact:** Ablation magnitude and perplexity deltas are wrong for tied projections. In a multi-decile sweep, cached factors for later aliases can also originate from the first decile's already-modified state. Correct final restoration does not make the intermediate experiments valid.

**Fix direction:** Identify unique physical parameter/block interventions and factor them from pristine weights. Deduplicate exact aliases while preserving distinct Q/K/V blocks sharing a fused parameter; simply deduplicating the whole fused parameter would incorrectly drop legitimate blocks. Record logical aliases and actual physical intervention scope.

**Regression tests:** Tied separate modules, ordinary untied modules, and fused Q/K/V block records. Verify each intended physical tranche is removed once, changing record order does not change the result, every decile uses pristine factors, and all bytes are restored after evaluation failure.

## ESD-02 — FIXED — Legitimate `multihead_*` projections are discovered but cannot be captured

**Locations:** `rmt/activations.py:121,138-145,255-267`; contrast `rmt/discovery.py:325-347` and the exact-target adapter in `rmt/pipeline.py:224-242`.

**Trigger/root cause:** Activation capture rejects any module whose full name contains `"head"`. Weight discovery does not use that broad exclusion. Both activation target collection and wrapper replacement apply the blacklist even when `target_names` explicitly names a supported projection.

**Reproduction:** A model with an ordinary `nn.Linear` at `multihead_attention.q_proj` is discovered as `multihead_attention.q_proj.weight` by the generic spec. Calling `compute_activation_covariance` with that exact target, valid synthetic input opt-ins, and a short window raises `ValueError: no supported projection modules selected for activation capture`.

**Impact:** Default overlap-enabled analysis fails in strict mode for otherwise supported generic projections solely because an ancestor's name contains `head`. In non-strict mode it loses requested activation diagnostics.

**Fix direction:** Share a precise projection-selection contract with discovery. Exclude actual output heads/embeddings by module role or exact path component, not arbitrary ancestor substrings. Honor explicit validated targets consistently in both capture selection and replacement.

**Regression tests:** Exact and discovery-driven capture for `multihead_attention.q_proj` and `multihead_attn.out_proj`; ensure a real `lm_head` remains excluded and unindexed exact targets continue working.

## ESD-03 — FIXED — Changing an analysis seed can prevent the model from being loaded

**Locations:** `rmt/cli.py:108-118`; `rmt/selftest.py:20-57`; tolerances in `rmt/config.py:17-24`.

**Trigger/root cause:** The mandatory analytic selftest uses the requested experiment seed. Its pass/fail decision includes single finite random-matrix samples with fixed tolerance bands. An ordinary statistical fluctuation is therefore treated as an implementation failure, and the CLI aborts before loading the requested model.

**Reproduction:** `rmt.selftest.run(seed=7)` returned False on the review environment: only `r_goe` failed; all other checks passed. Consequently `python -m rmt --seed 7 ...` reaches the gate and returns 1 before the real analysis. The default calibration seed passed. Additionally, `seed=0` is silently changed to 1234 by `seed or 1234`.

**Impact:** Valid experiment configurations are arbitrarily rejected. Seed sweeps can look like numerical/environment failures even though the implementation and assets are unchanged.

**Fix direction:** Decouple a fixed, calibrated selftest seed set from the experiment's randomness. If randomized gate testing is retained, use a statistically qualified multi-sample procedure and report its uncertainty rather than treating one fluctuation as a broken core. Do not loosen tolerances indiscriminately to hide errors.

**Regression tests:** Spy on gate invocation for experiment seeds 0, 7, and the default: calibration inputs should be stable, while the actual randomized analysis still receives the requested seed. Keep deterministic ground-truth failure tests so genuine core regressions still abort.

## ESD-04 — FIXED — Overlap with zero/degenerate activation eigenspaces is not identifiable

**Locations:** `rmt/overlap.py:32-41,77-109`; `rmt/per_matrix.py:239-266`.

**Trigger/root cause:** The overlap maximum uses every activation eigenvector, including zero-eigenvalue directions, and coincidence metrics assume individual eigenvectors are identifiable. Within a repeated eigenspace, an eigensolver may choose any orthonormal basis. Absolute cosine removes sign ambiguity but **not rotation ambiguity**. No covariance-rank/eigengap check qualifies these statistics.

**Reproduction:** Use `W = diag(4, 3, 2, 1)` and `C = zeros((4, 4))`. Both identity and a normalized 4x4 Hadamard matrix are valid covariance eigenvector bases. Passing these through the existing `eig` reuse argument to `overlap_analysis` yields respectively:

- `overlap = [1, 1, 1, 1]`;
- `overlap = [0.5, 0.5, 0.5, 0.5]`.

The weight and activation covariance are unchanged, and there is no activation signal.

**Impact:** Constant/dead activations and rank-deficient or repeated-eigenvalue covariance can produce arbitrary alignment/coincidence claims. This is relevant beyond zero matrices: covariance from fewer independent observations than input features necessarily has a nullspace, and all of it currently participates in the maximum.

**Fix direction:** Track covariance numerical rank and unresolved eigenvalue clusters. Exclude null directions from signal-overlap claims; use projector/cluster-level comparisons for repeated eigenspaces or explicitly mark basis-dependent metrics unavailable. Rank-zero capture should not become a finite successful overlap result. Keep raw paper-style metrics distinctly labeled if retained for reference.

**Regression tests:** Rotate only a covariance nullspace or repeated positive eigenspace while holding `C` fixed. Qualified results must be invariant or explicitly unavailable; nondegenerate reference cases should retain their current convention. Add an end-to-end constant-activation case.

## ESD-05 — FIXED — Oversized decile counts silently create no-op experiments

**Locations:** `rmt/decile.py:92-93,122-123,136-137,177-181`; `rmt/scalars.py:133-146`.

**Trigger/root cause:** Configuration only checks that `n_deciles` is positive. If it exceeds a matrix's reduced singular count, floor-rounded partition boundaries repeat and some ranges satisfy `lo == hi`. Reconstruction then removes no singular values, yet the benchmark reports those entries as ordinary completed decile interventions.

**Reproduction:** A 4x4 `diag(4, 3, 2, 1)` projection, `n_deciles=10`, `decile=1`. The singular spectrum/weight is unchanged; no exception or no-op status is produced. The first range is `(0, 0)`.

**Impact:** Small calibration models or large configured partition counts produce baseline-like perplexity deltas that can be misinterpreted as evidence that a singular-value tranche is unimportant. Models with unequal projection ranks can have inconsistent no-op subsets.

**Fix direction:** Validate a common meaningful partition count against all selected matrix ranks before taking snapshots or mutating anything. Alternatively provide explicit per-matrix empty-tranche metadata and do not label an entirely empty intervention as a successful lesion. Do not silently change the partition count independently per matrix.

**Regression tests:** Rank smaller than, equal to, and larger than `n_deciles`, including mixed ranks/GQA. Validate the full scope before any matrix mutation and verify coverage of every singular index.

## ESD-06 — FIXED — Randomized-control alpha is silently a different estimator from headline alpha

**Locations:** `rmt/per_matrix.py:138-154,163-170`; CSV schema at `rmt/per_matrix.py:300-307`.

**Trigger/root cause:** `alpha` follows `cfg.alpha_estimator` and publishes density/survival metadata. `alpha_rand`, however, always uses CSN on covariance eigenvalues, regardless of whether the headline estimator is Hill or windowed Hill. It has no independent estimator/kind metadata.

**Reproduction:** A seeded 64x64 Gaussian factor with `RunConfig(alpha_estimator="hill", do_randomize=True, do_overlap=False, do_spacing=False, use_svd_cache=False)` produced a row with `alpha_kind="survival"`, headline alpha approximately 11.24, and `alpha_rand` approximately 1.56 from the CSN **density** estimator. These two numbers are not a matched estimator comparison.

**Impact:** Users comparing a matrix with its control can attribute estimator/cutoff/convention differences to learned structure. Adding or subtracting one is not a valid repair for arbitrary non-power-law samples with different fitted support.

**Fix direction:** Either dispatch the same estimator/settings/support-selection policy for the control, or rename the field to an explicitly CSN-density control and serialize separate source/kind/cutoff metadata. Specify which interpretation downstream summaries should use.

**Regression tests:** Every alpha selector with randomization on; verify matched estimator/domain/kind and selected support, or explicitly distinct control labeling. Keep the default `all`/CSN behavior covered.

## ESD-07 — FIXED — A corrupt SVD cache entry causes matrix failure instead of a cache miss

**Locations:** `rmt/svd_cache.py:62-89`; `rmt/per_matrix.py:48-65`; failure propagation in `rmt/pipeline.py:117-145`.

**Trigger/root cause:** `load_svd` opens and reads NPZ contents without containing file-format, CRC, missing-member, or malformed-metadata errors. `per_matrix_analysis` only recomputes when the loader returns None, not when it raises. Thus optional cache corruption overrides the availability of a perfectly valid live weight.

**Reproduction:** Save a qualified entry with `save_svd`, truncate/replace its bytes in a temporary cache directory, and call `load_svd` with the correct weight digest and `required_dtype="float64"`. The reviewed environment raised `ValueError` rather than returning a miss. An NPZ missing `U`, `s`, or `Vh` also raises instead of being rejected cleanly.

**Impact:** A damaged/stale cache can drop usable matrices or abort a strict run until manually removed. Valid weights are never refactored even though caching is only an optimization.

**Fix direction:** Treat expected corruption/schema/read-validation failures as logged, invalid cache entries and recompute from live weights. Validate factor shape against expected geometry, finiteness, ordering, and precision provenance before accepting entries. Keep `allow_pickle=False`; never bypass safety or the float64 contract to recover a cache.

**Regression tests:** Truncated ZIP, missing arrays, malformed scalar metadata, and incompatible factor shapes. Each should recompute exactly once and replace/quarantine the invalid entry; genuine factorization failures must still surface.

## ESD-08 — FIXED — Bounded tail observations are fitted as an unbounded Pareto sample

**Locations:** `rmt/tail.py:20-29,49-64`.

**Trigger/root cause:** `xmax` filters the data, but alpha still uses `1 + n / sum(log(x/xmin))` and KS uses the unbounded Pareto CDF. A sample retained under an upper observation bound requires a conditional normalization depending on alpha. Simply removing upper observations does not preserve the unbounded MLE.

**Reproduction:**

```python
import numpy as np
from rmt.tail import fit_powerlaw_csn
u = (np.arange(10000) + 0.5) / 10000
x = (1 - u * (1 - 2**-2))**(-0.5)  # alpha=3 on [1, 2]
print(fit_powerlaw_csn(x, min_tail=1000, xmax=2)["alpha"])
```

Observed approximately **4.718**, instead of 3, for deterministic conditional quantiles.

**Impact:** The public bounded-fit option systematically overstates tail steepness and gives a KS distance for a different distribution. The main aggregator currently calls this routine without `xmax`; bounded library analyses are the immediate affected surface.

**Fix direction:** Use the normalized bounded density and CDF on `[xmin, xmax]` and optimize its likelihood, or explicitly reject unsupported bounded fits. Validate the bound. Distinguish a hard observation cutoff from the optional exponentially truncated power-law model.

**Regression tests:** Known bounded-Pareto data, rescaled bounds/data, and the large-upper-bound limit. The sibling project has the same mathematical defect in its independent tail implementations; do not cross-import its incompatible `rmt` package.

## ESD-09 — FIXED — An unavailable requested WeightWatcher stage is reported as complete

**Locations:** `rmt/baselines/weightwatcher.py:9-21`; `rmt/pipeline.py:39-55,151-165`.

**Trigger/root cause:** The adapter catches import and analysis exceptions and returns None. The caller only writes a baseline artifact for a non-None result; None adds no failure or stage status. The outer runner therefore finalizes `status="complete"` with no baseline artifact, even when `do_ww=True` and `strict=True`.

**Reproduction:** Inject `rmt.baselines.run_weightwatcher = lambda *args, **kwargs: None`, then run a tiny model with `do_ww=True`, strict defaults, other text stages disabled, and a temporary output directory. The run-status file says **complete**, and `<tag>_weightwatcher.json` is absent. This follows the same path as a missing package or a swallowed adapter exception.

**Impact:** A requested comparison can silently disappear, including actual execution errors rather than just a deliberately optional dependency. The status file cannot distinguish 'not requested' from 'requested but unavailable'.

**Fix direction:** Preserve best-effort baseline behavior if desired, but return/record a structured unavailable/failed status and reason. Make the strict-versus-best-effort policy explicit; do not imply every requested stage completed. Avoid swallowing analysis exceptions without provenance.

**Regression tests:** Missing baseline dependency, adapter execution failure, and successful baseline. Verify truthful stage status/artifact presence and the declared strict/non-strict exit policy.

## ESD-10 — FIXED — Checkpoint tracking bypasses the SVD dispatcher and backend configuration

**Locations:** `rmt/pipeline.py:177-195`; dispatcher contract in `rmt/linalg.py:47-71`.

**Evidence:** Static call-path analysis; no GPU benchmark was run.

**Trigger/root cause:** `analyze_checkpoints` accepts `cfg_flags` and constructs a `RunConfig`, but calls `np.linalg.svd` directly for every probe matrix. `backend="torch"` / GPU threshold settings have no effect. Unlike the main aggregator, it cannot use the configured dispatcher/fallback/provenance path.

**Impact:** Repeated transformer-sized checkpoint decompositions unexpectedly run on CPU despite an explicitly requested accelerator backend. This violates the documented centralized heavy-SVD contract and can make an otherwise feasible tracking job exceed its wall time.

**Fix direction:** Route checkpoint singular-value computation through a shared, precision-qualified backend-aware path; a values-only dispatcher extension is reasonable to avoid retaining unnecessary factors. Record actual backend/precision or reject unsupported options rather than ignoring them.

**Regression tests:** Spy/stub the shared dispatcher in checkpoint tracking, supply non-default backend/threshold, and verify propagation. Compare returned stable ranks against a small NumPy reference without requiring a GPU in the unit test.

## ESD-11 — FIXED — Two processes can both acquire the same supposedly fresh model output directory

**Locations:** `rmt/pipeline.py:32-37`; direct file writes at `rmt/pipeline.py:301-314,458-484`; deterministic per-model output path in `rmt/cli.py:146-148`.

**Evidence:** Static control-flow/concurrency analysis; no simultaneous destructive run was performed.

**Trigger/root cause:** The nonempty-directory check and `os.makedirs(..., exist_ok=True)` are separate operations. Two processes can both observe a path as absent/empty, both proceed, and overwrite the same status, CSV, summary, and plot files. The identity-hashed model tag identifies a model, not a unique run, so two concurrent runs of that model intentionally converge on the same path.

**Impact:** Concurrent submissions using the same output root can produce mixed/truncated artifacts or a success status from one process masking another's failure. The sequential rerun guard does not establish exclusive ownership.

**Fix direction:** Acquire ownership atomically before writing the running-status file, using exclusive directory creation or a lock/owner file that handles permitted pre-created empty directories. Reject a competing owner. Atomic individual artifact writes are useful additionally, but do not replace run-level ownership.

**Regression tests:** Synchronize two processes at ownership acquisition and assert exactly one starts analysis; the rejected process must not modify any winner artifacts. Retain sequential nonempty-output rejection coverage. Audit checkpoint output separately because it currently permits ordinary overwrites.

## Completed remediation and test-hygiene notes

1. ESD-01 through ESD-11 are fixed; focused regressions are in `tests/test_bug_report_current.py` and the existing precision/QKV/restoration/covariance/spacing coverage remains green.
2. Run this suite in a separate process from `../compute_optimal_analysis/`; shared mathematical fixes were implemented independently without mixing package conventions.
3. Cache-specific tests should use temporary cache roots and non-cache tests should disable persistence where practical, so production caches remain untouched.
4. Real tokenizer/model loading, CUDA fallback, shared-parameter interventions on deployed architectures, and cluster resource behavior remain target-environment validation gates.
