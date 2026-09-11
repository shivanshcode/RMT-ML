# Static bug report — offline RMT analysis

## Review and handoff

- **Baseline:** commit `0bd4a13` (`before another pass`). All 16 findings are now **FIXED** in the working tree. The descriptions below are retained as historical reproducers.
- **Scope:** maintained `rmt/` runtime, CLI/model loading, discovery/capture, numerical routines, lesions, cache, plots, checkpoint workflow, packaging, launcher, and test coverage. `rmt_pipeline_glm.py` is a disabled legacy archive, not the supported runner; it should not be revived to work around these findings.
- **Method:** source/control-flow review, Ruff inspection, existing tests, and focused offline CPU reproducers. This is a best-effort audit, not proof that no other bugs exist. Style-only lint warnings are omitted.
- **Pre-repair test baseline:** `python -m pytest -q -p no:cacheprovider` -> **114 passed, 1 skipped**. `bash -n run_rmt.slurm` passed; focused reproducers exposed gaps in that baseline suite.
- **Validation environment:** Python 3.13.2, NumPy 2.4.4, SciPy 1.17.1, CPU Torch 2.13.0, Matplotlib 3.10.8, pytest 8.3.3. Real transformers/datasets assets, CUDA/OOM recovery, and SLURM execution were not validated. Fault-injection checks are identified explicitly.
- The suite modified a tracked SVD cache entry; that test-generated change was restored. Only this report and the sibling report are intended audit changes.
- Paths/line numbers are relative to this directory at the baseline. Run reproducers/tests **from this project root in a separate process**; the sibling `rmt` API is incompatible.

**Priority:** P1 = unintended model mutation; P2 = scientific correctness, reproducibility, or supported-workflow failure; P3 = narrower library/numerical/test-isolation issue.

## Finding index

| ID | Priority | Finding | Status |
|---|---|---|---|
| ESD-001 | P1 | Decile metadata validation can fail after earlier weights were mutated | FIXED |
| ESD-002 | P2 | Decile sweep consumes one-shot record iterables multiple times | FIXED |
| ESD-003 | P2 | Metadata discovery loses the supplied QKV head-count override | FIXED |
| ESD-004 | P2 | Checkpoint analysis silently succeeds for missing probe layers/unknown options | FIXED |
| ESD-005 | P2 | Checkpoint output bypasses collision protection and atomic writing | FIXED |
| ESD-006 | P2 | Permissive degraded-SVD runs are finalized as fully complete | FIXED |
| ESD-007 | P2 | IPR/PT metrics use nonidentifiable singular-vector bases | FIXED |
| ESD-008 | P2 | MP lower-tail diagnostics change under spectral rescaling | FIXED |
| ESD-009 | P2 | Modified-MP fit is not invariant to spectral units | FIXED |
| ESD-010 | P2 | Random-control SVD bypasses the configured backend | FIXED |
| ESD-011 | P2 | Source precision and resolved run configuration are lost from artifacts | FIXED |
| ESD-012 | P3 | Complex weights are silently replaced by their real parts | FIXED |
| ESD-013 | P3 | Bounded histogram construction can overflow before applying its cap | FIXED |
| ESD-014 | P3 | Tests read/write the real persistent SVD cache | FIXED |
| ESD-015 | P3 | Scale-invariant scalar metrics overflow/underflow | FIXED |
| ESD-016 | P2 | Reusable decile cache is not bound to the pristine weight contents | FIXED |

## Repair validation

The repair pass prevalidates all decile metadata, materializes record iterables once, preserves QKV layout/head and source-dtype provenance, content-binds lesion factors, qualifies vector diagnostics, normalizes MP/scalar fitting units, dispatches random controls through the configured SVD backend, and makes checkpoint coverage/status/writes fail-closed and atomic. Persistent SVD caching is now opt-in.

Validation after repair: `python -m pytest -q -p no:cacheprovider` -> **114 passed, 1 skipped**; `bash -n run_rmt.slurm` passed. Focused reproductions for quantile and modified-MP scaling, bounded histograms, complex rejection, and scalar rescaling also passed. Real HF assets, CUDA fallback, and scheduler execution remain deployment validations.

## Findings

### ESD-001 — A metadata error in a later record leaves earlier parameters lesioned

**Locations:** `rmt/decile.py:104–138`, `140–174`; `rmt/discovery.py:260–268`.

Prevalidation checks that fused rows divide into thirds and that a head count exists, but does not check that the resulting Q/K/V width divides by that head count. The check is deferred to `_qkv_view` during reconstruction. The full-versus-fused scope conflict check is likewise inside the mutation loop. By then earlier physical parameters may already have been committed.

**Confirmed reproducer:** a model with an ordinary `4x4` parameter `a.weight`, followed by a `12x4` fused parameter `z.weight`, an explicitly interleaved spec, and `num_attention_heads=3`. Analyze `a.weight` followed by `z.weight[Q]`, then call `set_layer_svd_decile(..., decile=1, n_deciles=2)`. It raises "not divisible into 3 heads" **after changing `a.weight`**.

The high-level perplexity sweep has a restoration guard, but a direct caller of the public mutation API is left with an unexpectedly partially modified model. This also contradicts the metadata-prevalidation contract.

**Fix:** resolve/check every head-layout divisibility and full/fused scope conflict before the first write. Keep processing physical parameters one at a time. If the API is intended to guarantee rollback on later numerical failures too, implement bounded disk-backed rollback rather than retaining the entire model in RAM.

**Regression:** late invalid head layout and late conflicting scopes must fail without changing any parameter bytes; preserve alias deduplication and valid fused-block behavior.

### ESD-002 — Generator records are exhausted before the requested decile scope is resolved

**Location:** `rmt/decile.py:217–223`, `301–310`.

`records` is iterated to build `selected_roles`, again for `selected_layers`, and again by `_rebind_records`. A one-shot iterator is exhausted by the first comprehension.

**Confirmed reproducer:** pass `iter(discover_weight_metadata(tiny_model))` to `perplexity_vs_decile` with `decile_scope="analyzed"`. After a successful baseline evaluation it raises `decile scope selected no matrices`. Under scope `all`, role discovery can still succeed, but requested-layer provenance becomes empty.

**Fix:** materialize and validate `records` once at entry, before the expensive baseline pass. Reuse that list for every scope/provenance operation.

**Regression:** list and generator inputs must produce identical actual/requested roles, layers, matrix names, and interventions under both scopes.

### ESD-003 — A valid explicit head-count override cannot survive metadata-first discovery

**Locations:** `rmt/discovery.py:339–374`, `378–391`; `MatrixRecord` at 50–57.

`discover_weight_metadata(..., num_heads=...)` accepts an explicit override, but records contain no resolved head count/layout. `materialize_record` ignores the override and calls `get_num_heads(model)` again. Discovery can therefore succeed but every fused record can fail to materialize, or be split differently if the explicit count intentionally overrides model metadata.

**Confirmed reproducer:** a model with `Linear(8,24)` named `qkv`, no head count in its config, an explicit interleaved `ModelSpec`, and `num_heads=2` supplied to metadata discovery. Discovery returns three records; materializing one raises `num_heads is required for head-interleaved QKV`.

**Fix:** carry the verified layout/head count in metadata, or expose and consistently thread the same override through materialization and interventions. Validate full divisibility during discovery.

**Regression:** metadata-first and eager discovery must agree for explicit overrides, missing config heads, and overrides differing from config metadata.

### ESD-004 — Missing checkpoint probes and misspelled options silently produce successful-looking output

**Location:** `rmt/pipeline.py:201–249`.

Checkpoint analysis does not verify that each requested probe layer yields a matrix at every checkpoint. It appends NaN and `unavailable`, then returns a CSV path successfully even with default `strict=True`. Unlike `analyze_one_model`, it also filters unknown `cfg_flags` out before constructing `RunConfig`, silently ignoring caller mistakes.

**Confirmed reproducer:** request probe layer 99 from a one-layer synthetic model. The function returns a CSV containing `0,nan,unavailable,unavailable`, without a failure/status artifact or exception. A checkpoint architecture change can similarly make part of an otherwise plausible trajectory disappear silently.

**Fix:** validate keyword options without dropping unknown names; validate requested layer/role coverage for every checkpoint. In strict mode, fail on missing coverage. In permissive mode, serialize explicit partial-stage status and the missing selections rather than treating the CSV alone as success.

**Regression:** absent probes, a layer missing only in a later checkpoint, empty discovery, and an unknown configuration keyword.

### ESD-005 — Checkpoint writer can overwrite another run's results

**Locations:** `rmt/pipeline.py:217`, `251–266`.

`analyze_checkpoints` uses `os.makedirs(..., exist_ok=True)` and later `open(path, "w")`. It neither claims an output artifact/directory nor uses unique atomic replacement. Two calls with the same output root/tag can silently replace each other's expensive trajectories; a crash during writing leaves a truncated CSV. This workflow bypasses the collision protection implemented for per-model runs.

**Confirmed reproducer:** pre-create `<tag>_stable_rank_per_epoch.csv` with a sentinel; a normal checkpoint call replaces it without warning. No owner check occurs even if the directory belongs to another run.

**Fix:** give checkpoint runs a clearly defined ownership policy, reject accidental reuse, and write through an atomic temporary file. If appending to an already-owned analysis directory is supported, acquire an exclusive checkpoint-artifact claim rather than allowing arbitrary overwrite.

**Regression:** preexisting output, two competing writers, and a forced write failure must not destroy the previously committed artifact.

### ESD-006 — Degraded precision is omitted from run-level failure aggregation

**Locations:** `rmt/per_matrix.py:70–73`, `94–99`; `rmt/pipeline.py:38–40`, `114–120`, `193–198`.

With `strict=False`, a degraded float32 fallback is allowed to generate a row. The row truthfully says `precision_status="degraded"`, but the pipeline never adds this to `failures`. Consequently the final status can say `complete` with an empty failure list even when every matrix violated the requested SVD precision contract.

**Confirmed fault-injection check:** inject an SVD result marked `degraded=True`, `factorization_dtype="float32"`, disable unrelated optional stages, and analyze a tiny model permissively. All seven rows are degraded; `<tag>_run_status.json` says `complete`, `failures=[]`.

**Fix:** propagate row-level precision degradation into run-level partial/degraded status without forcing permissive runs to abort. Record the affected matrices and actual precision. Apply an equivalent policy to checkpoint tracking.

**Regression:** one degraded matrix among good matrices, all degraded matrices, and strict rejection. This check does not require a real CUDA OOM; retain a separate GPU fallback smoke test.

### ESD-007 — IPR and Porter–Thomas claims depend on arbitrary bases for repeated/null singular subspaces

**Locations:** `rmt/per_matrix.py:251–261`; `rmt/scalars.py:73–115`; compare `rmt/overlap.py:20–31`.

Overlap checks weight-basis identifiability, but IPR/PT aggregation runs independently and unconditionally on the same SVD vectors. With repeated singular values, an eigensolver may rotate the basis without changing the matrix. Null directions have the same problem. Reported localization/randomness can therefore be a solver artifact.

**Confirmed reproducer:** for `W=eye(16)`, inject two valid factorizations `(I, ones, I)` and `(Q, ones, Q.T)`, with seeded orthogonal `Q`. The first gives `ipr_top10_mean=1`, `ipr_bulk_mean=1`; the second gives approximately `0.232` and `0.164`. No unavailable status accompanies these matrix-level metrics. A rank-deficient diagonal matrix also reports ordinary IPR values for its arbitrary null basis.

**Fix:** share numerical subspace qualification across matrix-level vector diagnostics, returning explicit availability or cluster-invariant alternatives. Do not change the definition of the low-level IPR function on explicitly supplied vectors.

**Regression:** rotate repeated and null subspaces while keeping `W` fixed; aggregation must not turn that arbitrary choice into different scientific conclusions.

### ESD-008 — MP quantiles/lower-tail counts use absolute tolerances in dimensional units

**Locations:** `rmt/mp.py:89–97`, `143–152`, `256–276`.

The median root has an absolute `xtol=1e-10`; the lower-decile root uses support offsets of `1e-9`. At small spectral scales these are comparable to, or larger than, the whole support. A failed lower-decile root silently substitutes a **linear 10% support location**, which is not the MP 10th percentile. Median inaccuracies then affect lower-half KS selection as well.

**Confirmed reproducer:** singular values of a seed-4 `100x100` Gaussian matrix, with sigma 1. Rescale both singular values and sigma together. At scales 1 and `1e-12`, `excess_small_sv` changes from 1 to 5 and `ks_lower` from about `0.03586` to NaN. `mp_median(100,100,sigma)/sigma` changes from about `8.079455` to `2e-8`; it should be constant.

**Fix:** solve a dimensionless unit-scale quantile and rescale it. Remove the scientifically unrelated linear-support fallback; return explicit unavailability if a properly normalized solve genuinely fails.

**Regression:** several positive rescalings of the same Wishart sample must preserve normalized quantiles, counts, and lower-tail diagnostics.

### ESD-009 — Modified-MP fitting can return a huge false edge or reject an ordinary rescaled spectrum

**Locations:** `rmt/mp_fit.py:29–42`, `59–103`, particularly 75–77 and 91–98.

The degeneracy tolerance uses `max(max(abs(s)), 1)`, Gaussian broadening has an absolute width floor, and the optimizer starts from a fixed dimensional amplitude. Changing spectral units changes admissibility and optimization behavior even though the underlying curve is identical.

**Confirmed reproducer:** let `s` be singular values of a seed-4 `100x100` Gaussian matrix and use `n_grid=128`. At scale 1, fitted `nu_max / scale` is about 23.14. At `1e-8`, the function returns about **1.64e11** for that normalized edge. At `1e-10`, it incorrectly rejects the sample as degenerate. All inputs are finite and have the same relative spread.

This affects the public modified-MP fitter and `plot_esd(..., mp_mode="fit" or "both")`; the main runner's default theoretical overlay is not the demonstrated path.

**Fix:** construct the density and optimize in normalized coordinates, with scale-relative degeneracy checks and meaningful fit-quality/convergence validation; transform parameters back to original units.

**Regression:** scale-equivalent fitted support and density, near-constant genuinely degenerate inputs, and both plot-fit modes.

### ESD-010 — Random controls ignore the SVD backend and threshold

**Location:** `rmt/per_matrix.py:191–196`.

The real weight goes through `cached_svd`, but `do_randomize=True` calls `np.linalg.svd(Wr, compute_uv=False)` directly for an equally large control matrix. `backend="torch"` and `gpu_svd_min_dim` have no effect on that work. This contradicts the shared heavy-SVD dispatch contract and can put the dominant decomposition of a large GPU analysis on the CPU unexpectedly.

**Fix:** route random-control factorization through a backend-aware values-only/full-factor dispatcher with the same precision policy. Serialize actual random-control backend/dtype/degradation separately from the headline weight's metadata.

**Regression:** use a dispatcher spy with randomization enabled to verify that both decompositions honor the selected backend and threshold, and test random-control precision failures. A GPU benchmark is still needed for performance validation.

### ESD-011 — Output artifacts lose source precision and the resolved run configuration

**Locations:** `rmt/cli.py:125–130`, `135–149`; `rmt/model_io.py:20–34`; `rmt/discovery.py:229–242`; `rmt/per_matrix.py:42–43`, `91–99`; `rmt/pipeline.py:43–58`.

The loader accepts FP16/BF16/FP32, but the CLI explicitly removes `dtype` from forwarded flags. Discovery promotes weights to float64 and `MatrixRecord` retains no source dtype. Standard rows record only factorization precision, and the final status records tokenizer/text provenance, not the resolved configuration or model source precision. With perplexity disabled—the default—there is no corresponding execution-precision record in the run artifacts.

**Confirmed example:** analyze a half-precision tiny model. Rows say `precision_status="complete"`, `svd_factorization_dtype="float64"`, with no source dtype. Those fields are true about the SVD, but cannot distinguish an analysis of rounded FP16 weights from an FP32 source. Other choices such as unfolding degree and the complete requested configuration are likewise not preserved in one authoritative artifact.

**Fix:** write a resolved run/environment/model provenance manifest before work and retain source/storage dtype per matrix when needed. Distinguish source quantization, factorization precision, and lesion execution dtype. Do not forcibly promote only a lesioned parameter and break the model's forward dtype contract.

**Regression:** output artifacts for FP16, BF16, and FP32 sources must identify their source precision independently of an identical FP64 factorization request; validate both CLI and borrowed-model paths.

### ESD-012 — Complex input is silently analyzed as a different real matrix

**Locations:** `rmt/linalg.py:136–148`; similar casts in `rmt/mp.py:20–28` and `rmt/scalars.py:13–19`.

`cached_svd(1j * np.eye(4), backend="numpy").s` returns zeros rather than ones, with only a cast warning. The package exposes complex random ensembles, but its generic weight-taking helpers neither preserve complex values nor explicitly reject them.

**Fix:** reject unsupported complex input before real casting, or implement complex-preserving factors and conjugate-transpose operations throughout the affected APIs. The real-LM production path need not be broadened if real-only support is intentional.

**Regression:** purely imaginary and mixed complex matrices must either yield correct factors/scalars or a clear unsupported-input exception.

### ESD-013 — Histogram bin cap is applied after an overflowing integer conversion

**Location:** `rmt/plots/esd.py:37–45`.

`requested = int(ceil((right-left)/h))` is evaluated before limiting it to 400. A finite, extremely small IQR makes the ratio overflow to infinity; converting that to `int` raises instead of constructing a bounded grid.

**Confirmed reproducer:**

```python
_bin_edges(np.r_[np.linspace(1e-312, 2e-312, 100), 1.0], 0, 1)
# OverflowError: cannot convert float infinity to integer
```

**Fix:** saturate the bin-count calculation before division/integer conversion, or explicitly map a nonfinite ratio to `max_bins`. Validate the final edges as finite and suitable for the histogram.

**Regression:** zero IQR, subnormal positive IQR with a finite outlier, near-constant spectra, and ordinary spectra must all stay within the documented bound.

### ESD-014 — Tests pollute and depend on the production SVD cache

**Locations:** `tests/test_per_matrix.py:30–78`; `test_rmt_critical.py:104–130`; multiple pipeline smoke calls using default `RunConfig`.

Many tests leave `use_svd_cache=True` and `svd_cache_dir="./svd_cache"`. They read/write persistent files in the project instead of an isolated test directory. During this audit, the passing suite modified the tracked file `svd_cache/blk_0_lin_weight_12712bcb65e0af2b.npz`. Existing cache contents can also let tests bypass the factorization path they were intended to exercise, or require writes to a read-only source checkout.

**Fix:** disable persistent caching in non-cache tests, or explicitly provide a per-test temporary cache directory. Cache-specific tests should create all their own entries, including corruption/provenance fixtures.

**Regression:** run the suite twice from a clean checkout and verify no tracked or production-cache artifacts change; verify dedicated tests still exercise misses, hits, and precision rejection.

### ESD-015 — Finite rescalings corrupt supposedly scale-invariant scalar summaries

**Locations:** `rmt/scalars.py:22–43`, `46–60`, `128–135`, `160–175`.

Stable rank, spectral/row entropy, bulk-energy fractions, and per-decile contributions square raw values before dividing by total/max energy. Overflow or underflow occurs before scale cancellation.

**Confirmed example:** `[3,2]` has stable rank `1.444444` and entropy `0.617242`. Multiplication by `1e200` makes rank raise and entropy become `-0.0`; multiplication by `1e-200` yields NaN rank and zero entropy, despite finite nonzero input.

**Fix:** normalize singular magnitudes by their maximum before squaring, and normalize each row independently for row entropy. Preserve the all-zero convention and additive per-decile contract.

**Regression:** dimensionless summaries must remain unchanged across large and small finite rescalings; summed decile contributions must still recover global entropy/rank.

### ESD-016 — Float64-qualified cached decile factors can still belong to an older weight

**Locations:** `rmt/decile.py:54–80`, `114`, `159–165`.

The public factor cache is keyed by `(id(parameter), scope_tag)` and qualification only establishes the string precision marker. Parameter identity survives training/checkpoint loading, so the cache can silently substitute factors from an earlier pristine state.

**Confirmed reproducer:** cache a top-half decile of `diag([4,3,2,1])`; restore the model to **twice** that pristine weight; call `set_layer_svd_decile` again with the same parameter/cache. It produces `diag([0,0,2,1])` instead of `diag([0,0,4,2])`. The retained part of the model is wrong, not merely the amount removed.

The internal perplexity sweep currently creates a fresh cache for one pristine model, so this exposure is the public reusable-cache path.

**Fix:** bind cached entries to verified pristine contents/version and geometry as well as precision, or restrict the cache to a single owned pristine sweep and reject external stale reuse. Do not use parameter identity alone as a weight-content identity.

**Regression:** unchanged pristine sweeps reuse factors, while updated weights and reloaded checkpoints recompute or fail clearly before mutation; preserve Q/K/V scope separation.

## Original repair order and acceptance (completed)

1. Fix metadata prevalidation and stale-cache scope before running interventions on borrowed models.
2. Correct scientific qualification, numerical unit invariance, backend dispatch, and run/provenance status before publishing new measurements.
3. Add one focused regression per repaired ID; avoid relying only on the existing passing suite.
4. Preserve the independent conventions of this project: ascending 1-based deciles, explicit eigenvalue normalization, separate text/tokenizer fallback opt-ins, and strict versus permissive execution.
5. Keep tests/cache artifacts isolated; re-run the suite and launcher syntax check, then validate actual local HF models, CUDA precision/fallback behavior, and cluster resource requirements separately.
6. Update each finding's status with its repair/test result. Keep unresolved entries visible for the next coding run.
