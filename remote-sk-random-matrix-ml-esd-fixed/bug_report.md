# Static-analysis bug report — remote-sk-random-matrix-ml-esd-fixed

## Scope and handoff

Reviewed the maintained `rmt/` implementation, model discovery/capture/interventions, caching, CLI, plotting, launcher, packaging, and relevant tests at commit `c93d344`. **No implementation files were changed.** Paths and line numbers below are relative to this directory at that revision. The deliberately non-executable `rmt_pipeline_glm.py` archive was syntax-checked, not treated as a supported analysis entry point; generated caches and historical logs are not implementation sources.

Validation: all 48 Python files parsed successfully; Ruff's undefined-name/export/local checks found no errors; `bash -n run_rmt.slurm` succeeded. Small isolated offline probes confirmed findings marked **observed**, using temporary outputs and small in-process matrices/modules. Full suites, downloaded models, CUDA OOM behavior, and SLURM execution were not run. Static review cannot guarantee an exhaustive absence of other bugs.

Severity: **P1** = significant result corruption, false-success execution, or production-scale resource failure; **P2** = functional/numerical/provenance failure under the stated trigger; **P3** = narrower test/support defect. There are **17 findings**.

Run this project's tests in a separate process from `../compute_optimal_analysis/`; its `rmt` API is incompatible. Preserve ascending deciles, verified fused-QKV layouts, reversible Parameter identity/execution dtype, float64 factorization qualification, and explicitly labeled density/survival exponents. Do not reactivate the archived runner to solve these findings.

## Findings

### ESD-001 — P1 — Decile prevalidation materializes and retains the entire intervention scope

**Locations:** `rmt/decile.py:103–140, 225–242`.

Despite the disk-backed snapshots/factor cache, `set_layer_svd_decile` converts every selected parameter to a host float64 array and retains `pristine_mat`/`pristine_block` in `prepared` before processing any matrix. For fused QKV, separate records can additionally retain separate copies of the full fused parameter. This happens again on every decile, even when factors are already cached.

With the supported fp32 LLM input and `decile_scope=all`, host memory is therefore O(all selected parameters), not O(one projection). Billions of selected entries require tens of GiB just for these copies, defeating the stated 64-GiB/bounded-memory workflow.

**Repair/test:** prevalidate names, layouts, dimensions, aliases, and group counts using metadata only; materialize/reconstruct one physical scope at a time. Preserve disjoint Q/K/V updates and exact restoration. Instrument live materializations across many layers and warm-cache deciles; peak retained weight arrays must not grow with total layer count.

### ESD-002 — P1 — Strict runs return success after a requested WeightWatcher stage fails

**Locations:** `rmt/pipeline.py:137–179, 38–55`; `rmt/cli.py:134–164`.

The strict failure check occurs before WeightWatcher. Missing dependencies or execution failures in that later stage append to `failures`, but nothing subsequently enforces strictness. The wrapper writes `status=partial` and returns normally; the CLI only marks exceptions as failures, so it returns zero and the SLURM script prints completion.

**Observed:** with `strict=True`, `do_ww=True`, and the baseline adapter returning `unavailable`, `analyze_one_model` returned normally with a partial status file.

**Repair/test:** enforce strict requested-stage completion after all stages while retaining truthful baseline/status artifacts. Test dependency-unavailable and execution-failed baselines through both the library and CLI. Define non-strict partial-run exit policy explicitly rather than changing it accidentally.

### ESD-003 — P1 — Accepted cache ordering can detach singular values from their vectors

**Locations:** `rmt/svd_cache.py:78–101`; `rmt/per_matrix.py:74–77, 221–224, 272–288`.

Cache validation checks shape/finiteness/nonnegativity but not descending singular-value order. The aggregator independently sorts `s` without applying the same permutation to `U` and `Vh`. A cache containing a consistently permuted factorization is accepted, after which top/bottom IPR, overlap, and coincidence attach values to the wrong singular vectors.

**Observed:** cached factors reconstructing `diag(3,2,1)` in ascending singular order were accepted; the reported singular index best aligned with the top activation mode was 2 instead of 0.

**Repair/test:** reject unsorted cache entries as misses, or reorder all factors together before use. Enforce the invariant on directly supplied results too. Test a permuted but otherwise valid cached factorization and compare every vector-associated result with a fresh SVD.

### ESD-004 — P2 — Cache precision qualification trusts metadata despite contradictory array dtypes

**Locations:** `rmt/svd_cache.py:71–93`; `rmt/per_matrix.py:49–57`.

`required_dtype="float64"` checks only the stored string. Float32 `U`, `s`, and `Vh` carrying that string and a false degraded marker are accepted and later labeled as complete float64 factorization results. A malformed, self-contradictory entry can therefore satisfy the precision contract merely by declaring the desired metadata.

**Observed:** a cache of float32 identity factors labeled `factorization_dtype="float64"` passed a float64-required load.

**Repair/test:** validate real floating array dtypes and metadata consistency, and reject unsupported/malformed precision combinations. Test mismatched factors/metadata, missing qualification, and degraded entries. This will not prove the historical precision of arbitrarily forged float64 arrays, but it must reject directly observable contradictions.

### ESD-005 — P2 — Weight-side degenerate singular vectors are treated as identifiable

**Locations:** `rmt/overlap.py:20–43, 59–101, 119–165`; `rmt/per_matrix.py:272–288`.

The covariance qualifier rejects unresolved activation eigenspaces, but there is no corresponding check for repeated or zero weight singular values. SVD vectors inside such a cluster can rotate without changing the weight, changing per-vector maximum cosine, top/bottom overlap, and coincidence while all results remain `available`.

**Observed:** two valid SVDs of the same `10x10` identity matrix and the same distinct diagonal covariance gave first-vector overlap 1 versus approximately 0.613, both available.

**Repair/test:** qualify weight singular clusters and null spaces; use whole-subspace invariant statistics or mark nonidentifiable vector/rank claims unavailable. Test rotated representations of identical matrices. Do not replace this project's absolute-cosine convention with the sibling's squared-projection convention.

### ESD-006 — P2 — Discovery still excludes legitimate projections by ancestor substrings

**Locations:** `rmt/discovery.py:325–343, 398–404`; compare `rmt/activations.py:128–140`.

Discovery skips any full path containing `embed`, `norm`, `shared`, etc. Activation capture was changed to component-based exclusions, but discovery was not. A legitimate `pre_norm_attention.q_proj` or `embed_projection` can therefore disappear before capture or analysis, including under an explicit numeric layer selection.

**Observed:** a real `nn.Linear` named `embed_projection`, which matches the generic projection classifier, yielded no discovery records.

**Repair/test:** make discovery and capture use consistent component/type-aware exclusions without excluding unrelated ancestors or similarly named projections. Exercise both metadata and materializing discovery APIs, and retain tests proving real embeddings/normalizations/output heads remain excluded.

### ESD-007 — P2 — Activation-window replay leaves stale plots and failures from discarded passes

**Locations:** `rmt/pipeline.py:87–128, 137–143`.

A shortened shared window plan restarts the row/singular-value pass, but `failures` is outside the restart loop and overlap plots have already been written to final paths. If a formerly available overlap becomes unavailable under the final shorter windows, its old heatmap remains. A transient failure from a discarded pass also remains and can make an otherwise successful replay fail strictness.

**Repair/test:** make rows, failures, and artifacts transactional per window-plan revision. Publish only a stabilized pass or remove all artifacts belonging to abandoned revisions. Mock a later projection shortening the plan and an earlier overlap becoming unavailable; verify there are no stale heatmaps or obsolete failures and every published artifact matches the final token-window identity.

### ESD-008 — P2 — Direct per-matrix callers cannot disable overlap when supplying cached feature matrices

**Location:** `rmt/per_matrix.py:262–307`.

The overlap block tests only `fm_dict is not None`, not `cfg.do_overlap`. A caller reusing feature matrices for several analysis configurations still incurs covariance eigendecomposition/overlap work and receives populated metrics when overlap is explicitly disabled. Pipeline-level gating masks this bug in normal full runs, but the public aggregator itself does not honor its configuration.

**Observed:** `RunConfig(do_overlap=False)` with a matching feature matrix returned `overlap_status="available"` and finite overlap.

**Repair/test:** gate the block on both the flag and data availability. Spy on eigendecomposition/overlap routines to establish that disabled analyses are not invoked, and check disabled output metadata.

### ESD-009 — P2 — An analyzed-only lesion scope silently drops missing requested matrices

**Locations:** `rmt/decile.py:210–216, 291–296`.

`_rebind_records` intersects requested names with rediscovery and never checks that every requested name resolved. The caller rejects only an entirely empty result. With one existing and one missing/stale requested record, the experiment completes while intervening on only a subset of its explicit analyzed scope.

**Repair/test:** resolve every requested matrix before evaluation/mutation and fail clearly on missing or ambiguous scope members. Keep deliberate role-based expansion for `decile_scope=all` separate. Test partial as well as total name-resolution failure and ensure no parameter changes survive a rejection.

### ESD-010 — P2 — Experiment seed is not propagated to randomized spacing estimators

**Locations:** `rmt/spacing.py:121–150`; `rmt/per_matrix.py:239–251`; `rmt/config.py:103–105`.

`RunConfig.seed` reaches the randomized weight control, but number variance always uses RNG seed 0 and rigidity always uses seed 1. There is no RNG argument on these routines or propagation from the per-matrix caller. Thus the requested experiment seed does not govern these real randomized diagnostics, contrary to the run-level reproducibility contract; their actual fixed seeds are not serialized either.

**Repair/test:** accept an explicit RNG/seed for randomized interval estimators, thread the experiment seed through, and preserve fixed calibration seeds only in selftests. Verify same-seed repeatability and the actual sampled-window change for different experiment seeds; avoid asserting that rounded aggregate metrics must always differ.

### ESD-011 — P2 — Windowed-Hill random controls omit their actual fitted support

**Locations:** `rmt/per_matrix.py:165–194, 337–341`.

For a windowed-Hill headline, the random control computes a full plateau result but keeps only its alpha. Its cutoff/count fields remain NaN/zero, and there are no random-control plateau start/end/window/union-support columns. The original matrix's plateau fields cannot describe the independently selected random-control plateau. Consequently the documented random-control support provenance is missing for this selector.

**Repair/test:** serialize the control's own rank-window support and availability, including the final boundary observation. Retain unavailable single-cutoff fields rather than inventing a Pareto cutoff. Test schema/value correctness for every headline/control estimator pairing.

### ESD-012 — P2 — Modified-MP fitting returns a collapsed or invalid support as a successful fit

**Locations:** `rmt/mp_fit.py:59–88`; consumer `rmt/plots/esd.py:113–125`.

The optimizer bounds require only `nu_max >= 0`, not `nu_max > nu_min`. Degenerate spectra are not rejected, and no validity/convergence result accompanies the returned tuple. A collapsed support gives an identically zero density and can leave the optimizer at its initial values, yet is returned for plotting as a fit.

**Observed:** `fit_modified_mp(np.ones(40), n_grid=128)` returned `(a=0.5, nu_min=1.0, nu_max=1.0)`.

**Repair/test:** validate sufficient distinct observations, parameterize a strictly positive support width, and expose failure/unavailability rather than a bogus curve. Test constant/near-constant spectra and attempts to cross the lower support boundary.

### ESD-013 — P2 — Library configuration accepts unsupported scientific selectors and mislabels results

**Locations:** `rmt/config.py:107–120`; `rmt/pipeline.py:74–76`; `rmt/per_matrix.py:79–80, 140–159`; CLI-only validation in `rmt/cli.py:28–38`.

Choice validation exists in argparse but not in `RunConfig`. Library callers can request unimplemented sigma estimators, unknown covariance normalization modes, or unknown alpha selectors. Unknown normalization silently becomes `cols`. More seriously, an unknown alpha selector leaves CSN density alpha as the headline but labels its kind as survival because it is not equal to `csn`.

**Repair/test:** validate supported selectors and numerical ranges in the shared configuration itself, including library keyword paths, instead of relying on the CLI. Test invalid alpha/sigma/normalization selectors and ensure no analysis or artifact creation precedes rejection.

### ESD-014 — P2 — Scalar decile reporting allows more groups than singular values

**Locations:** `rmt/scalars.py:134–170`; `rmt/per_matrix.py:315–316`; `rmt/config.py:107–109`.

The documented restriction `n_deciles <= singular_count` is enforced by lesion mutation but not by scalar reporting. With perplexity disabled, a rank-four reduced spectrum and `n_deciles=10` produce several empty deciles with entropy/rank contribution zero and no unavailability indication. Those zeros appear to be measurements of actual tranches.

**Repair/test:** validate selected matrix singular counts before analysis, or explicitly revise and label an empty-group contract rather than reporting ordinary deciles. Keep reporting and intervention partition validation consistent. Test overpartitioning through both the scalar API and pipeline, not only the lesion setter.

### ESD-015 — P2 — Iterator arguments can silently empty checkpoint-tracking output

**Locations:** `rmt/pipeline.py:193–240`.

`checkpoint_fracs` is iterated to analyze models and then iterated again to write rows. A generator is exhausted after analysis, leaving a header-only CSV even though all checkpoint loads/SVDs ran. `probe_layers` is likewise iterated repeatedly for several dictionaries, discovery, aggregation, and headers; a generator can disappear during initial setup and silently remove the requested probes.

**Repair/test:** materialize these finite inputs once, or reject non-reiterable inputs explicitly before doing work. Test lists and generators with identical contents; their output rows/columns must agree, and empty/unresolved selections should not masquerade as completed tracking.

### ESD-016 — P2 — Requested powerlaw-package failures disappear into unqualified NaNs

**Locations:** `rmt/tail.py:240–259`; `rmt/per_matrix.py:155–163`; overall status in `rmt/pipeline.py:38–55`.

When `use_powerlaw_pkg=True`, the optional adapter returns `None` both for dependency/import failures and for exceptions during fitting/comparison. The aggregator converts this into NaN LR fields without any stage failure, availability reason, or warning; the enclosing run can be marked complete even in strict mode. This is different from an expected statistically unavailable fit due to too few observations, but the outputs do not distinguish them.

**Repair/test:** return structured complete/unavailable/failed status and propagate dependency/execution failures into requested-stage reporting and the defined strict policy. Keep insufficient-tail data separately labeled. Test missing dependency, comparison exception, insufficient data, and successful comparison.

### ESD-017 — P3 — Duplicate synthetic-model fixtures disagree about the verified Pythia layout

**Locations:** `_synthetic_models.py:10–13, 42–45, 113–148`; `tests/_synthetic_models.py:10–15, 44–47, 116–148`; bare `_synthetic_models` imports in `tests/test_discovery.py:6–8` and other torch tests.

The root helper lacks attention-head metadata and implements contiguous Q/K/V chunking for a model labeled `gpt_neox`. The tests-directory helper supplies four heads and the actual head-interleaved forward layout. Using the root helper's Pythia model now fails discovery for missing head metadata; adding only the metadata would still leave its forward interpretation wrong. Bare imports and manual `sys.path` insertion also allow whichever helper was imported first to remain cached under the same module name.

**Repair/test:** use one canonical, explicitly namespaced fixture implementation with verified layout/head metadata. Test import behavior independently of prior module imports and exercise both fused discovery and actual forward/intervention semantics. Do not weaken production layout validation merely to accommodate the obsolete fixture.

## Suggested repair order

1. Production memory, strict completion, and SVD-cache invariants: ESD-001–004.
2. Overlap/discovery/replay/scope correctness: ESD-005–009.
3. Scientific provenance and numerical/configuration qualification: ESD-010–016.
4. Fixture deduplication and import isolation: ESD-017.

Add focused regression tests for these triggers before relying on the existing smoke tests. No undefined-name/syntax problem was found in maintained code, and lint-only style warnings were not counted as functional defects.
