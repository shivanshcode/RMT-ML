# Static bug report v2: offline RMT analysis

## Scope and evidence

This review found 10 defects at commit `d1e0698797b031b8fdf1a383cd769a21d710f0bb`. The current working tree fixes all 10 defects. Paths and line numbers in the findings refer to the reviewed commit.

Static analysis reads code without running the application. This review traced maintained model loading, discovery, activation capture, numerical analysis, decile changes, caching, output, checkpoint analysis, and the launcher. It also examined test coverage and the earlier `bug_report.md`.

The finding descriptions preserve the original static evidence. The repair added focused CPU regression tests. Hardware behavior still requires a test on the selected backend.

All 48 Python files passed syntax parsing, including the disabled archive. The semantic review excluded `rmt_pipeline_glm.py` because it is not a supported runner. Both project launchers passed `bash -n`.

Targeted Ruff analysis found no syntax, undefined-name, or selected control-flow errors. A broader pass flagged test callbacks that execute immediately inside their loops. Inspection did not establish defects from those warnings.

The repair did not load real Hugging Face snapshots or run GPU and cluster jobs. Those items remain deployment tests.

This is a best-effort repair, not proof that no other bugs exist.

## Terms and priority

SVD factors a matrix into singular values and vectors. FP32 and FP64 denote 32-bit and 64-bit floating-point numbers. MP describes expected spectra of specified random matrices.

An activation is a signal produced by a model layer. Covariance measures how activation values vary together. A decile is one of ten ordered groups, unless the configuration changes the group count.

A checkpoint stores model weights at a training stage. Stable rank measures matrix energy relative to its largest component. A Hill estimate measures the decay rate of a distribution tail.

P1 means a broad risk of unintended model or data damage. P2 means incorrect scientific output or a broken supported execution path. P3 means a narrower library or numerical boundary defect.

All findings have status FIXED. P2 findings are ESD-V2-001 through ESD-V2-006 and ESD-V2-009. P3 findings are ESD-V2-007, ESD-V2-008, and ESD-V2-010.

## ESD-V2-001: Strict model analysis accepts a partly missing layer selection

The relevant code is `rmt/pipeline.py:88-97,99-107,216-220` and `rmt/discovery.py:343-394`.

Discovery applies the requested layer filter, but the pipeline tests only whether the entire result is empty. It does not compare discovered layers with all requested layers. Missing selections therefore create no failure record when at least one requested layer exists.

A source-derived case requests `layers=[0, 99]` from a one-layer model. Layer zero produces records, so the empty-discovery guard does not fail. If the remaining stages succeed, even strict analysis writes a complete status without identifying missing layer 99.

The manifest preserves the requested list but does not establish that the pipeline fulfilled it. The checkpoint path already rejects a completely missing probe layer. Normal per-model analysis needs the same completeness rule.

Repair and acceptance steps:

1. Compare requested layer identities with the discovered records before matrix analysis.
2. Reject missing requested layers in strict mode.
3. Record missing selections and partial status in permissive mode.
4. Test fully missing, partly missing, and valid layer selections.

## ESD-V2-002: Overlap qualification ignores the actual SVD precision

The relevant code is `rmt/overlap.py:12-29,69-75,124-129`, `rmt/linalg.py:39-49`, and `rmt/per_matrix.py:269-285,322-358`.

The shared `singular_basis_status` uses `factorization_dtype` to qualify individual singular vectors. The overlap module uses a separate helper that always assumes FP64 precision. It also uses only the input dimension instead of the larger matrix dimension.

If a permitted fallback computes FP32 factors, their FP64 storage does not restore lost accuracy. Two singular values separated by more than the FP64 threshold but less than FP32 resolution can pass the overlap guard. Their individual directions remain unresolved at the actual calculation precision.

A source-derived test supplies a 16-by-16 SVD result marked FP32 with leading values `5.000001` and `5.0`. Use distinct positive activation eigenvalues. The shared basis guard rejects the pair, but the overlap helper accepts it.

The permissive production path can therefore publish ordinary overlap and coincidence values for a basis that its own vector diagnostics reject. The run-level degraded status does not repair the matrix-level availability claim. A real GPU fallback remains a separate deployment test.

Repair and acceptance steps:

1. Reuse one precision-aware weight-basis qualification policy across vector diagnostics.
2. Use the actual factorization dtype and both matrix dimensions.
3. Test near-repeated and near-null FP32 factors against FP64 controls.
4. Make sure that unresolved factors do not produce available overlap claims.

## ESD-V2-003: Duplicate probe layers attach the wrong value to later checkpoints

The relevant code is `rmt/pipeline.py:231-235,249-251,286-304,310-322`.

`probe_layers` remains a list that can contain duplicates. The result dictionaries collapse duplicate keys, but the accumulation loop still appends once for every list entry. Output later indexes those lists as though each checkpoint appended exactly one value.

For `probe_layers=[0, 0]` and two checkpoints with means A and B, the stored list becomes `[A, A, B, B]`. The second CSV row uses index one and reports A again. Its training fraction belongs to the second checkpoint, so the trajectory is incorrect.

The same duplication affects backend and dtype columns. The existing checkpoint smoke test uses distinct probes and checks only the header. No current input guard rejects the duplicate case.

Repair and acceptance steps:

1. Reject duplicate probe layers or remove duplicates before allocating result containers.
2. Test two checkpoints with deliberately different stable ranks and repeated probe input.
3. Make sure that every output row uses measurements from its own checkpoint.

## ESD-V2-004: Checkpoint means can silently compare different matrix sets

The relevant code is `rmt/pipeline.py:259-305,329-334`.

Checkpoint analysis tests only whether each probe layer contains at least one matrix. It does not preserve or compare exact matrix identities, roles, shapes, or counts across checkpoints. It averages whichever records discovery returns at each stage.

A source-derived case starts with Q, K, V, O, and MLP matrices in a probe layer. At a later checkpoint, remove or rename one recognized projection while keeping the others. Both checkpoints still satisfy the current layer-coverage test.

The later mean now describes a different matrix set, but the status remains complete. The output does not record enough membership data to detect the changed comparison. This is a remaining coverage gap beyond the earlier ESD-004 repair for wholly absent probe layers.

Repair and acceptance steps:

1. Define an explicit matrix-selection contract for a checkpoint trajectory.
2. Compare exact identities, roles, and geometry with that contract at each checkpoint.
3. Reject incompatible coverage in strict mode or report partial coverage with explicit membership data.
4. Test a missing projection, a renamed projection, and a changed projection shape within an otherwise present layer.

## ESD-V2-005: Library model tags can bypass output-directory ownership

The relevant code is `rmt/pipeline.py:31-47,174-176,223-247,453-475` and `rmt/cli.py:166-173`.

The CLI converts model identities to safe filename tags. The public pipeline functions accept `model_tag` directly and join it into output paths without the same protection. An absolute tag or parent-directory component can place artifacts outside the claimed directory.

For example, pass `model_tag='../shared/existing'` with a fresh output directory. The manifest writer resolves that tag outside the claimed directory and can replace an existing manifest there. Claiming the fresh directory does not protect the outside file.

The same raw tag enters checkpoint artifact and status paths. This is a library-call boundary defect, not a claim that the current CLI tag sanitizer fails. Existing model labels must not act as unrestricted filesystem paths.

Repair and acceptance steps:

1. Convert library tags to safe filename components or reject path-like tags before claiming output.
2. Make sure that every resolved artifact path remains inside its intended output directory.
3. Test parent traversal, absolute paths, platform separators, and ordinary model identities.
4. Preserve sentinel files outside the temporary output directory in every rejection test.

## ESD-V2-006: Direct analysis and decile paths still erase complex components

The relevant code is `rmt/per_matrix.py:43-44`, `rmt/overlap.py:12-17`, and `rmt/decile.py:109-172,197-198`.

The SVD helper and eager discovery reject complex weights. Direct aggregation first casts `record.weight` to FP64, so the SVD helper never sees the original complex dtype. The overlap helper has the same bypass.

Decile mutation also converts live parameters to FP64 without rejecting complex input. It then copies the real reconstruction into the original parameter. A purely imaginary parameter can therefore become zero through the public mutation API.

A source-derived case uses `1j * eye(4)` and two decile groups. Metadata discovery accepts the shape, then `set_layer_svd_decile` discards the imaginary values before factorization. For aggregation, use four groups so that the group-count guard does not mask the cast defect.

This is a remaining entry-point gap in the earlier ESD-012 repair. The current production discovery path rejects complex values when it materializes them. Direct public callers remain exposed, including a path that changes model weights.

Repair and acceptance steps:

1. Reject unsupported complex input before any real cast.
2. Reject complex parameters during complete decile preflight, before any parameter changes.
3. Test purely imaginary and mixed complex values through each affected entry point.
4. Make sure that rejected decile calls preserve every parameter byte.

## ESD-V2-007: MP density and lower-edge energy fractions still fail under rescaling

The relevant code is `rmt/mp.py:45-87,258-278` and `rmt/mp_fit.py:51-59`.

The quantile solver now uses unit-scale coordinates. Density evaluation still multiplies dimensional squared support terms before dividing by dimensional factors. Those intermediate products can overflow or underflow while the true result remains representable.

For `n=m=4`, evaluate `mp_pdf` at `x=sigma=1e100`. The radicand overflows, although the correct density is finite. The corresponding scale `1e-100` makes that radicand underflow and returns zero density at an interior point.

`mp_cdf` integrates the same density, so the defect also reaches lower-edge cumulative comparisons. `small_sv_deviation` separately squares unnormalized singular values for its energy fraction. The modified-MP density helper also retains the dimensional fourth-power product.

These are remaining calculation gaps, not a repetition of the repaired root-finding tolerance or normalized fit search. Ordinary model scales do not establish correctness over the documented unit-change contract. Extreme finite scales need dedicated tests.

Repair and acceptance steps:

1. Evaluate density and cumulative probability in normalized coordinates.
2. Rescale the density only after evaluating the normalized expression.
3. Normalize magnitudes before computing dimensionless energy fractions.
4. Test normalized density, cumulative probability, and lower-edge fractions over large and small finite scales.

## ESD-V2-008: Greedy Hill-band search misses longer overlapping bands

The relevant code is `rmt/tail.py:202-236`.

The function searches for the longest contiguous flat band. After a band ends, it advances to `i = j + 1`. It never tests a band that starts inside the preceding accepted band.

A source-derived unit test injects local estimates `[1, 1, 1.19, 1.4, 1.4, 1.4, 1.4]`. Use `window=5` and `flat_tol=0.20`. The current search finds bands of lengths three and four.

The band `[1.19, 1.4, 1.4, 1.4, 1.4]` has length five and satisfies the same flatness rule. Missing it changes the width threshold from satisfied to unsatisfied. Its value also satisfies the function's extreme-tail consistency rule.

This example isolates the search logic by supplying a known local Hill curve. An additional test must exercise the full path from a valid positive sample. Changing the selection algorithm must preserve contiguous-rank and unavailable-value behavior.

Repair and acceptance steps:

1. Consider overlapping candidate bands instead of skipping their possible starts.
2. Compare short curves with an exhaustive reference search.
3. Test threshold-crossing examples and gaps of unavailable local estimates.
4. Repeat full-sample Pareto and random-matrix calibration tests after the change.

## ESD-V2-009: MP fitting treats numerical null modes as noise observations

The relevant code is `rmt/per_matrix.py:84-128`, `rmt/mp.py:161-184`, and `rmt/linalg.py:39-49`.

The pipeline qualifies numerical null modes for vector diagnostics but not for MP noise estimation. `estimate_sigma_gd_median` takes the median of all returned singular values. It has no factorization-precision threshold or unresolved-bulk status.

Consider a 64-by-64 matrix with only three nonzero singular values. Exact zeros in the remaining values give a zero median. Roundoff-sized positive values in those positions instead give a positive fitted noise scale and MP edges.

The two factor sets can agree on matrix content within factorization precision. Their apparent noise bulk depends on the solver's representation of the null space. Promoting returned factors to FP64 does not make that bulk resolvable.

A source-derived regression can supply both factor sets through the SVD seam used by `per_matrix_analysis`. The existing singular-basis guard identifies the null directions, but aggregation still publishes MP measurements without an availability field. The repair test covers exact and roundoff null modes on CPU. A GPU comparison remains a deployment test.

Repair and acceptance steps:

1. Qualify the resolvable spectrum using factorization precision and matrix dimensions before estimating MP noise.
2. Report unavailable MP measurements when the remaining sample cannot support the fit.
3. Preserve raw singular values separately from the qualified fit sample.
4. Test exact-null, roundoff-null, and resolvable ill-conditioned matrices without imposing an absolute cutoff.

## ESD-V2-010: Negative interval lengths produce valid-looking zero statistics

The relevant code is `rmt/spacing.py:121-145,148-179`.

Number variance and spectral rigidity require a positive interval length. Their public functions do not test that `L` is finite and positive. Negative lengths can produce empty counts and zero-valued statistics instead of an input error.

For example, use a nondegenerate ordered spectrum and `L=-1`. In `sigma2`, no value can satisfy both bounds of the reversed interval, so all counts are zero. In `delta3`, the same empty interval produces a zero staircase and zero residual.

A zero length also reaches a division by zero. The production aggregator uses fixed positive lengths, so the demonstrated exposure is direct library use. These results still violate the mathematical domain of the public functions.

Repair and acceptance steps:

1. Reject nonfinite and nonpositive lengths before unfolding or allocating windows.
2. Test negative, zero, NaN, and infinite lengths.
3. Preserve the existing unavailable result for positive intervals longer than the observed spectrum.

## Risks that need separate evidence

The following items are not included in the 10 confirmed source-level defects. Their effect depends on hardware, supported-model policy, or the intended status contract. Do not treat them as reproduced failures:

- Test GPU memory recovery during activation capture. `_safe_forward` allocates covariance snapshots before its retry guard, and shorter token windows cannot reduce those fixed buffers.
- Define behavior when a decile boundary splits equal singular values. Different valid singular bases can produce different interventions.
- Test a corrupted cache whose finite, sorted arrays have valid shapes but do not reconstruct the recorded weight. The loader examines metadata and shapes, not reconstruction.
- Decide whether unavailable optional analyses must make strict runs fail. The pipeline aggregates a failed power-law adapter but not its insufficient-sample unavailable result.
- Test partial numerical failures in direct decile mutation. Complete metadata preflight does not provide rollback after an earlier physical parameter changes.

## Repair validation

Keep this project isolated from `compute_optimal_analysis`. Both projects expose an incompatible package named `rmt`. Do not use the disabled archive.

The focused command `python -m pytest -q -p no:cacheprovider tests/test_bug_report_v2.py` passed 14 tests. The command with `-m "not torch"` passed 92 tests. The command with `-m torch` passed 36 tests and skipped one test.

The repair tests cover layer and checkpoint selection, safe tags, actual overlap precision, complex input, and MP availability. They also cover density scaling, overlapping Hill bands, valid interval lengths, and mutation safety.

`bash -n run_rmt.slurm` passed. Real snapshots, CUDA precision, memory recovery, and cluster execution remain deployment tests.
