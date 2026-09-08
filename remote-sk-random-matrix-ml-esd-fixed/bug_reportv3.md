# Static analysis — bug report v3

## Scope and verification

Fresh review of the **current working-tree implementation**, including its uncommitted changes. `bug_report.md`, `bug_reportv2.md`, and `to_change_env.md` were not used as review inputs. Findings are based on current code; they do not assume any old report is still applicable. No implementation files were changed.

Reviewed the supported `python -m rmt` entry point, configuration/model loading, discovery, numerical core, per-matrix and model pipelines, activation capture, decile interventions, persistent SVD cache, plots, launch script, and tests. `rmt_pipeline_glm.py` is explicitly a non-executable archive and is not treated as the production entry point. Paths/line numbers below are relative to this directory and the pre-fix review snapshot.

Validation performed:

- `python -m pytest -q -p no:cacheprovider`: **114 passed, 1 skipped**. The skip was the CUDA comparison test.
- `bash -n run_rmt.slurm`: passed.
- Small CPU-only numerical/model probes and fault injection reproduced the findings identified as reproduced below.
- Local interpreter: Python 3.13.2. No CUDA/HPC or real Hugging Face checkpoint validation was performed; `transformers` and `datasets` were unavailable locally. Architecture conclusions come from the implemented name/layout contracts and synthetic modules, not a claimed production checkpoint run.

**P1** = incorrect scientific results, silently incomplete analysis, or major production resource/compatibility failure. **P2** = important conditional correctness or reporting problem.

## Findings at a glance

| ID | Priority | Finding |
|---|---|---|
| ESD-01 | P1 | Full dense overlap matrices accumulate across the whole model |
| ESD-02 | P1 | Degraded float32 SVD results are accepted and lose precision provenance |
| ESD-03 | P1 | Text windows ignore the model's enforced context limit |
| ESD-04 | P1 | Mixtral is registered as Llama and silently loses every expert MLP |
| ESD-05 | P1 | Per-record activation capture defeats global OOM replay consistency |
| ESD-06 | P2 | Degenerate spectra abort the entire matrix analysis through spacing |
| ESD-07 | P2 | Hill plots use singular values while headline metrics use eigenvalues |
| ESD-08 | P2 | MP-bulk scalar summaries include lower-edge outliers |
| ESD-09 | P2 | Reversible decile benchmarking permanently changes training/eval mode |

---

## ESD-01 — Dense overlap artifacts defeat projection-at-a-time memory bounds

**Locations:** `rmt/pipeline.py:91–102, 125–127, 405–416`; `rmt/per_matrix.py:219–228`; `rmt/overlap.py:39–41, 88–91`.

**Trigger:** `do_overlap=True`, especially the CLI default of analyzing all layers of a large model.

**Cause:** Metadata discovery and covariance capture are projection-at-a-time, but `ovmats` retains every full `ov["overlap_matrix"]` until the final plotting stage. Each artifact has shape `min(n,m) × m` and float64 elements; it is not a small scalar/spectrum summary. Releasing `live.weight` does not release these arrays. The coincidence routine also temporarily computes a second dense overlap matrix with the same entries.

**Static size check:** For Llama-3.1-8B-style geometry (`d=4096`, KV width 1024, MLP width 14336):

- Q/O/gate/up overlaps: four arrays of 128 MiB each.
- K/V overlaps: two arrays of 32 MiB each.
- Down-projection overlap: 448 MiB.
- Total retained overlap artifacts: **1 GiB per layer, approximately 32 GiB over 32 layers**, excluding covariances, eigenvectors, SVD factors, plotting temporaries, and any CPU-resident model.

This is an allocation/lifetime calculation, not a claimed measurement from loading an 8B model. It demonstrates that peak host memory still scales with all selected projections and can exhaust an otherwise adequate analysis allocation.

**Fix direction:** Emit each heatmap while its matrix is live, spool it to disk, or retain only an explicitly downsampled visualization. Keep only scalar overlap summaries in memory across records. Reuse the matrix product between overlap/coincidence calculations where possible.

**Regression:** A many-record synthetic pipeline should release full overlap arrays between records. Use weak references or allocation instrumentation to verify that retained dense-artifact storage does not grow linearly with the number of layers. Preserve exactly-once SVD behavior.

## ESD-02 — Degraded SVDs are still used as unqualified high-precision scientific results

**Locations:** `rmt/linalg.py:96–135`; `rmt/per_matrix.py:45–69, 250–279`; `rmt/svd_cache.py:55–78`.

**Trigger:** The GPU float64 factorization fails, CPU float64 fallback also fails, and the last-resort float32 factorization succeeds.

**Cause:** `cached_svd()` correctly marks this result `factorization_dtype="float32", degraded=True`, and the cache refuses to persist it as a float64 result. However, `per_matrix_analysis()` only consults `degraded` when deciding whether to save the cache. It still computes all small-singular-value, MP, spacing, and overlap metrics and returns a normal row, even with `strict=True`.

The CSV schema contains no actual backend, factorization dtype, degraded flag, or precision-availability status. Cache hits also reconstruct an `SVDResult` without forwarding the saved backend metadata. Avoiding cache poisoning does not prevent mislabeling the current run.

**Reproduced:** Inject an `SVDResult` marked degraded/float32 into a Gaussian `(64,64)` matrix analysis with `RunConfig(strict=True, use_svd_cache=False)`. Analysis succeeds, and the returned row contains **no dtype/backend/degraded fields**.

**Impact:** The exact fallback that compromises the small-SV regime can silently enter the scientific CSV as an ordinary completed matrix. A warning in a transient log is not sufficient provenance for downstream comparisons.

**Fix direction:** Define the precision contract at the aggregation boundary. In strict mode, reject results that cannot satisfy it; in a permitted degraded mode, serialize actual precision/backend and mark affected diagnostics appropriately. Preserve provenance through cache reads as well as cache writes.

**Regression:** Fault-inject the fallback or return a degraded factorization. Verify strict failure or explicitly degraded output/status, no false float64 contract, and metadata round trips for cache hits.

## ESD-03 — Default text windows exceed supported models' positional limits

**Locations:** `rmt/config.py:57–61`; `rmt/pipeline.py:204–226`; `rmt/activations.py:217–238, 306–336, 338–369`; `rmt/perplexity.py:14–31, 70–91`.

**Trigger:** A model with an enforced context shorter than the configured/default text window. GPT-2 is explicitly supported by discovery but commonly has a 1024-position table, while activation capture defaults to 2048. Library-loaded encoder/small-context models can have even shorter limits; perplexity independently hardcodes a 1024 maximum.

**Cause:** Window construction does not consult the model's effective positional/context limit. `_safe_forward()` retries only recognized OOM errors; an out-of-range positional embedding/index error is not an OOM. With sufficient real input text, the default overlap stage therefore fails for an otherwise supported GPT-2 snapshot and strict mode aborts the run.

**Reproduced:** A synthetic GPT-2-named module with a 16-position embedding table and `config.n_positions=max_position_embeddings=16` fails activation capture with the 2048 default (`IndexError: index out of range in self`). The same capture succeeds when explicitly limited to 16. This exercises the same positional-index failure mechanism without HF dependencies.

**Fix direction:** Resolve and validate an architecture-appropriate effective context once, before expensive matrix work. Cap/validate both activation and perplexity windows and their strides against enforced limits, and serialize requested versus effective lengths. Do not rely on an OOM retry to discover a positional limit; account for architectures whose configured limits are genuinely extensible rather than blindly truncating all models.

**Regression:** Use a tiny enforced-position model and cover both capture and perplexity. Defaults should run within its limit or fail clearly at preflight, not after SVD/capture work. Add an offline GPT-2 integration test when assets/dependencies are available.

## ESD-04 — Mixtral registration silently omits expert MLP projections

**Locations:** `rmt/discovery.py:72–82, 134–141, 316–345`; the same classification is reused by activation capture and decile discovery.

**Trigger:** A model/config with `model_type="mixtral"`.

**Cause:** The registry maps Mixtral to `_LLAMA`, whose MLP patterns are `gate_proj`, `up_proj`, and `down_proj`. Mixtral expert projections use names such as:

```text
model.layers.0.block_sparse_moe.experts.0.w1
model.layers.0.block_sparse_moe.experts.0.w2
model.layers.0.block_sparse_moe.experts.0.w3
```

None matches the selected spec. Discovery simply skips them; because attention records still exist, the run can finish successfully without disclosing that every expert MLP was omitted. This also narrows `decile_scope="all"` through the already-incomplete discovered role set.

**Reproduced:** A synthetic Mixtral-configured layer containing Q/K/V/O plus one expert's w1/w2/w3 has **7 linear projections**, but `discover_weight_metadata()` returns only **4 records**, with roles Q/K/V/O. `get_model_spec()` reports the Llama spec.

**Fix direction:** Add a Mixtral-specific spec for expert projections and explicit router policy, or reject the architecture as unsupported rather than reusing a spec that silently loses most weights. Report discovered/skipped roles and reasons, especially for MoE architectures.

**Regression:** A two-expert Mixtral-shaped synthetic layer must discover Q/K/V/O and all six expert matrices with correct roles/shapes. Verify the corresponding activation and all-scope decile selection; router handling must be explicit rather than accidental.

## ESD-05 — OOM fallback can give different projections different activation corpora

**Locations:** `rmt/pipeline.py:94–102, 204–224`; `rmt/activations.py:260–301`.

**Trigger:** One later projection needs shorter forward windows under its covariance-capture memory overhead, while earlier projections succeeded at the original lengths.

**Cause:** `compute_activation_covariance()` contains restart logic intended to discard earlier results and replay every target on the same shortened windows. However, the production pipeline now calls it separately for **each `[rec]`**. The restart can only synchronize targets inside that one call, generally one projection. Earlier per-matrix results have already been computed and appended, and each subsequent call reloads the original text windows.

**Reproduced:** A two-projection synthetic model, token-weighted capture, two text batches, maximum length 8:

- Q captures normally: mean/FM observation counts **16 / 16**.
- K simulates OOM for windows longer than 4, then retries successfully: counts **8 / 8**.
- Calling `_maybe_activation_cov()` exactly as the production loop does returns both results without a cross-record replay or failure.

The per-matrix rows also do not preserve these FM counts/window identities, so output consumers cannot see that the input/context distribution changed. Equal mean/FM counts within one projection do not establish comparability between projections.

**Fix direction:** Establish a shared successful-window plan for the run, with bounded-memory preflight, or invalidate/recompute earlier rows when a later capture changes the plan. Alternatively fail the run instead of silently changing corpora. Serialize exact scored/captured window identities and counts. Preserve projection-at-a-time storage rather than solving this by retaining every dense covariance.

**Regression:** Simulate OOM only for a later projection through the actual model pipeline. All accepted projection covariances must use the same replay plan, or the run must explicitly report incompatible/failed captures. Check window contents as well as counts.

## ESD-06 — A legitimate degenerate spectrum destroys all otherwise valid matrix metrics

**Locations:** `rmt/per_matrix.py:191–207`; `rmt/spacing.py:35–49`; `rmt/pipeline.py:103–125, 374–386`.

**Trigger:** A zero matrix, scalar identity, or another matrix with a sufficiently large but constant MP-selected spectrum, with default spacing enabled.

**Cause:** The aggregator checks only `len(bulk_lam) >= 50`. `unfold()` correctly rejects fewer than two distinct levels, but the exception propagates out of the entire matrix analysis. Stable rank, entropy, singular-value summaries, and other meaningful results are discarded. Strict model analysis then fails, instead of marking only unsupported spacing diagnostics unavailable.

**Reproduced:** Both `per_matrix_analysis()` on `zeros((64,64))` and on `eye(64)`, with caching disabled and otherwise default configuration, raise:

```text
ValueError: unfolding requires at least two distinct levels
```

Smaller examples evade this only because the spacing count threshold is not reached. If fixed only in aggregation, the final spacing plot path can independently retry the same unsupported unfolding and still fail strict mode.

**Fix direction:** Treat a degenerate/unfoldable bulk as unavailable for the specific spacing statistics, with a serialized reason. Retain meaningful scalars and distinguish a mathematical non-applicability from a computational failure. Gate corresponding plots on the same availability decision. Do not manufacture artificial distinct levels.

**Regression:** Zero/identity matrices of rank at least 50 must yield valid scalar rows, explicit spacing availability, and consistent plot/run status. Also cover mixed degeneracies that legitimately retain zero spacing mass.

## ESD-07 — Hill plots are in the wrong domain for the headline Hill metrics

**Locations:** `rmt/pipeline.py:365–369`; `rmt/plots/hill.py:7–20`; `rmt/per_matrix.py:102–122`.

**Cause:** CSV headline windowed Hill metrics are correctly computed on `lam=s**2/N_cov`. The plotting pipeline still calls `plot_hill(s, ...)`, and that function applies both Hill estimators directly to the singular values. Its axis labels say only `alpha_hill`/`alpha_local`, without declaring the different domain.

**Reproduced:** For the same positive spectrum, the windowed Hill values on singular values are **twice** those on covariance eigenvalues, to floating-point accuracy (`median(alpha_nu/alpha_lambda) ≈ 2`). This follows directly from `log(lambda_i/lambda_j)=2*log(s_i/s_j)`; the constant denominator cancels.

**Impact:** A plot intended to explain a row's plateau exponent instead shows an exponent off by a factor of two, potentially implying disagreement between metrics and figures.

**Fix direction:** Plot the covariance-eigenvalue input used by the headline metric, or deliberately produce separately named and explicitly labeled nu/lambda plots. Thread domain metadata rather than inferring it from a generic array argument.

**Regression:** Capture arrays passed to plot estimators and verify equality with the per-matrix lambda-domain inputs. Check labels and exponent agreement for a deterministic Pareto-like spectrum.

## ESD-08 — Lower MP outliers contaminate quantities labeled as bulk summaries

**Locations:** `rmt/scalars.py:72–86, 124–130`; callers in `rmt/per_matrix.py:163–178`.

**Trigger:** Rectangular matrices with a positive lower MP edge and low singular-value outliers, exactly the lower-tail regime this project studies.

**Cause:** `ipr_summary()` discards the lower edge and defines bulk as `s <= nu_plus`. `bulk_mass_frac()` uses the same upper-only criterion. Thus lower outliers contribute to `ipr_bulk_mean` and to a quantity described as energy inside the bulk, even though spacing and outlier counts use the full `[nu_minus,nu_plus]` support.

**Reproduced:** With `n=8`, `m=32`, `sigma=1`, the MP support is approximately `[2.8284,8.4853]`. Use singular values `[9,8,7,6,5,4,3,0.1]`, seven coordinate right-singular vectors, and an eighth vector uniformly spread over 25 other coordinates. The actual in-support mean IPR is **1.0**, but `ipr_bulk_mean` is approximately **0.8628571** because it includes the `0.1` lower-outlier direction.

**Fix direction:** Use both fitted edges for metrics called MP-bulk summaries. If an upper-trimmed whole-body statistic is intentional, give it a distinct name/definition and retain a genuine MP-bulk metric rather than conflating the two. Consider compatibility/versioning for existing CSV consumers.

**Regression:** Plant both lower and upper outliers with different IPRs; verify neither changes the in-support mean. Test corresponding full-support versus upper-only energy fractions explicitly.

## ESD-09 — Reversible decile benchmarking does not restore model mode

**Location:** `rmt/decile.py:172–185, 210–239`.

**Trigger:** Library use where `model_factory` returns an existing training-mode model, a supported case according to the function's pristine-reuse contract. This is usually invisible in the CLI because its loader already calls `eval()`.

**Cause:** `perplexity_vs_decile()` calls `model.eval()` before baseline evaluation and never saves/restores its original training state. The nested perplexity evaluator restores only the state it sees on entry—already eval mode. Weight snapshots are restored, but training/dropout behavior remains changed after success or after exceptions, including baseline failures before the snapshot cleanup block.

**Reproduced:** Start a tiny model with `model.train()`, run a one-decile analyzed-scope sweep with a finite mocked perplexity evaluator, and inspect the same instance afterward: **`model.training` is False**.

**Fix direction:** Save the incoming mode and wrap the complete baseline/intervention flow in an outer restoration guard, including failures before snapshot creation. If mixed per-submodule modes are supported, preserve those rather than flattening the whole tree to one mode.

**Regression:** Test initially training and initially eval models on success, baseline failure, and intervention failure. Assert exact weight restoration and the intended mode-restoration contract.

## Suggested repair order

1. Repair bounded artifact lifetime and precision acceptance/provenance (ESD-01/02).
2. Fix architecture/context preflight and run-wide activation replay semantics (ESD-03/04/05).
3. Isolate mathematically unavailable spacing from the rest of a matrix result and from plotting (ESD-06).
4. Align plotting/bulk definitions and restore library-call state (ESD-07/08/09).

Keep the passing existing suite, but add production-pipeline regressions and fault injection. Parser tests and small random full-rank matrices do not exercise these conditions.
