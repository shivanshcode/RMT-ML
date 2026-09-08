# Static analysis — bug report v4

## Scope and verification

Review of the **current working tree**, including the interrupted edits, not just the committed version. All Python implementation files and tests, the asset-staging utility, launcher, requirements, and primary usage/design contracts were reviewed. `bug_reportv3.md` was used as a reconciliation checklist; findings below were checked against current code rather than assumed to persist. Paths and line numbers are relative to this directory.

**No implementation, test, or launcher files were changed during this review.** Do not revert the entire interrupted change set: some numerical-safety fixes are present and verified.

### Validation actually performed

- AST parsing of all **37 Python files**: **one syntax error**, in `pipelines/spectral_lesioning.py:477`.
- `python -m pytest -q -p no:cacheprovider`: **collection fails**, with that syntax error. This is not a passing release.
- The diagnostic subset `python -m pytest -q -p no:cacheprovider --ignore=tests/test_spectral_lesioning.py`: **75 passed**. This deliberately excludes the broken module's tests; it does not validate the runner or lesions.
- `bash -n run_hpc.slurm`: passed.
- Small CPU numerical and fault-injection probes verified the examples below. To inspect runner functions despite the syntax blocker, the runner AST was loaded in a temporary namespace with **only its broken lesion import omitted**. The numerical/configuration function bodies were unchanged. These are isolated function probes, **not successful production-runner executions**; no lesion implementation was repaired in memory or on disk.
- Local interpreter: Python **3.13.2**. No CUDA, compilation, SLURM execution, production training, or connected asset acquisition was tested. `transformers` and `datasets` were unavailable. The local environment is not certification of the pinned standalone environment.

**P1:** startup blocker or materially incorrect scientific output. **P2:** important conditional correctness/configuration problem. **P3:** lower-impact metadata defect.

## Findings at a glance

| ID | Priority | Finding | Relationship to v3 |
|---|---|---|---|
| COA4-01 | P1 | Corrupted decile guard makes the lesion module and runner unimportable | New interrupted-edit blocker |
| COA4-02 | P1 | FARMS-pooled levels still enter single-matrix spacing diagnostics | COA-01 remains |
| COA4-03 | P1 | Golden configuration still bypasses its requested FARMS stage | COA-02 remains |
| COA4-04 | P1 | Accepted fit/detector combinations still mix spectral domains and denominators | COA-03 remains |
| COA4-05 | P1 | Tracy–Widom dimensions still come from pooled observation count | COA-04 remains |
| COA4-06 | P2 | Presets still overwrite explicitly supplied aliases | COA-07 remains |
| COA4-07 | P2 | Default absolute Lanczos margin overwhelms transformer-scale spectra | COA-09 remains |
| COA4-08 | P2 | Realized-ratio float grouping fragments scaling trajectories | COA-10 remains |
| COA4-09 | P2 | Unfolding failures still suppress the unfolding-free gap ratio | COA-11 remains |
| COA4-10 | P2 | Float32 raw-moment subtraction can produce a grossly invalid covariance | Newly identified |
| COA4-11 | P2 | Capture/evaluation flatten a borrowed model's mixed train/eval modes | Newly identified |
| COA4-12 | P3 | Windowed-Hill observation support omits its final boundary observation | Newly identified |

## What changed since v3

| v3 finding | Current assessment |
|---|---|
| COA-01–04 | Still present; independently rechecked below. |
| COA-05, non-finite gradients | **Original corruption path fixed.** `trainer.py:337–343` rejects a non-finite gradient norm before the optimizer step. CPU Inf and NaN gradient injections left the parameter at `1.0`, with zero successful updates/tokens. GPU behavior was not tested. |
| COA-06, invalid/saturated perplexity | **Evaluator repaired, lesion repair incomplete.** `trainer.py:153–178` rejects non-finite scored losses and zero scored targets, removes the cap at 80, and labels exponential overflow. Losses 81 and 100 now give distinct perplexities; Inf/NaN raise. The tranche benchmark has a finite-result guard, but the decile guard is malformed and prevents importing the entire module: COA4-01. Do not call the end-to-end lesion issue fixed. |
| COA-07 | Still present: COA4-06. |
| COA-08, endless no-progress training | **Original loop fixed.** `trainer.py:391–395` rejects a complete pass without updates and bounds epoch-only mode. An all-ignored one-batch loader now raises promptly. |
| COA-09–11 | Still present: COA4-07–09. |

---

## COA4-01 — Malformed decile guard blocks every runner invocation

**Locations:** `pipelines/spectral_lesioning.py:476–478`; unconditional import in `run_experiments.py:38`.

The current source contains:

```python
if not math.isfinite(valuelk(value := value):"): # no
```

Python reports `SyntaxError: unterminated string literal (detected at line 477)`. The damage is not confined to users selecting decile lesions: importing the module fails before any function can run, and the production runner imports it unconditionally. Manifest-only runs and runs with lesioning disabled are therefore also blocked. Test collection reproduces the failure without any dataset or GPU dependency.

**Repair direction:** Replace the malformed condition with the intended finite-value check, preserving the context manager's restoration on failure. Review both benchmark guards; the nearby `"must belak finite"` message is also corrupted text, although that typo is not itself a runtime blocker.

**Regression:** Parse/import all maintained modules, exercise the actual runner's help/manifest path, and test invalid tranche/decile evaluation callbacks. Verify exact weight restoration and failure rather than a successful non-finite result. Run the **full** suite without ignoring this file.

## COA4-02 — Pooling ESD observations still fabricates level statistics

**Locations:** `run_experiments.py:346–356, 400–455`; `rmt/farms_aspect_ratio.py:315–362`.

For prepared FARMS modes, the runner assigns the pooled ESD to `eigenvalues` and then uses `spacing_eigenvalues = eigenvalues`. NNSD, Brody, gap ratio, number variance, and rigidity therefore describe sorted levels from multiple operators, not the full matrix's spectrum. Equal-sized window ESDs may be pooled for density estimation; their union does not preserve single-operator level correlations.

**Reproduced on the current functions:** Construct `A = default_rng(22).normal(size=(32,32))`, then `W = vstack((A,A))`. Select 32-by-32 FARMS windows, two row starts, one column start, analytic MP, and BBP detection. The result has **64 pooled levels, 63 spacings, 32 zero spacings, and gap ratio 0.0**. The full reduced covariance has only 32 levels and gap ratio approximately **0.501149719**.

`design.md` explicitly forbids interpreting pooled FARMS levels as one matrix's NNSD; its claim that the runner already separates these spectra is not true of this snapshot.

**Repair direction:** Maintain separate pooled ESD/tail observations and single-operator spacing observations, with compatible bulk selection in each domain. Alternatively compute each window's statistics independently and label their aggregation. Deduplicating pooled eigenvalues is not a repair for changed correlations.

**Regression:** Increasing/repeating FARMS windows must not change full-matrix spacing statistics or create more than `min(W.shape)-1` full-matrix spacings.

## COA4-03 — Golden still records FARMS settings without performing FARMS

**Locations:** `run_experiments.py:346–356, 383–392`; `rmt/factory.py:322–345`; `run_hpc.slurm:196–219`.

The raw-method branch includes `lanczos_stieltjes` and `thamm_modified_singular`. It bypasses `prepare_spectrum()` and sends raw full-matrix eigenvalues to the tail solver and ESD artifacts. Thus the default Golden combination, `aspect_ratio_mode=farms_normalized` plus Lanczos MP, ignores all FARMS controls while its saved requested configuration and documentation advertise FARMS.

**Reproduced:** For `default_rng(2).normal(size=(64,128))`, Lanczos steps 24/probes 1, FARMS window size 16, and tail minimum 4, switching aspect mode between `raw` and `farms_normalized` makes **no call to `prepare_spectrum()`**. Both results say `spectrum_mode="raw"`, have 64 ESD observations, and give the same tail exponent, approximately **11.944496039**.

**Repair direction:** Preserve original-operator Lanczos analysis while adding the separately labeled FARMS ESD/tail branch. Do not run Lanczos on an unrelated pooled array. If that combination is not implemented, reject it explicitly instead of silently ignoring the request.

**Regression:** Through the production runner, changing a meaningful FARMS window setting must change the Golden FARMS input while leaving the original operator passed to Lanczos unchanged.

## COA4-04 — Fitted support, detector output, and serialized units can still disagree

**Locations:** `rmt/factory.py:346–369, 439–500`; `run_experiments.py:353–381, 400–417, 515–521, 539–555, 632–634`.

Three related incompatibilities remain:

1. **FARMS fit with raw/shape-normalized preparation:** `dispatch_mp_fit()` builds a separate FARMS spectrum when `mp_fit_method="farms_unbiased"` but `prepared.farms` is absent. The runner still masks/counts/plots the original observations against that other spectrum's support. It labels `mp_spectrum_domain` with the observation mode, so the plot's equality check cannot expose the mismatch.
2. **Lanczos detector with prepared non-raw observations:** `dispatch_spike_detector()` analyzes the original weight operator, ignoring the supplied prepared eigenvalues/aspect ratio for this detector. Its raw-domain threshold/count is combined with the other ESD without independent detector-domain provenance.
3. **Wrong denominator metadata:** `spectrum_denominator` is always the source SVD's normalization. Canonical FARMS instead divides by the window's larger dimension; raw FARMS divides by nothing; trace normalization has a separate trace denominator per window. Shape normalization adds another rescaling not represented by that field.

**Reproduced:** Generate `W = default_rng(2).normal(size=(64,128))`, add 100 to `W[32,64]`, and select raw preparation with a FARMS MP fit using 16-by-16 windows and two starts along each axis. The prepared aspect ratio is **0.5**, the fit's is **1.0**. The sampled windows omit the planted center entry: the fit reports **zero upper outliers and bulk fraction 1.0**, but the runner's raw observations have **one value above that support and bulk fraction 0.984375**. These fields describe different observation sets, not one coherent fit.

**Repair direction:** Give observations, fits, and detectors explicit domain/geometry/normalization identities. Compare or combine them only when compatible; otherwise produce separate named outputs or reject the configuration. Coordinate this with COA4-02/03 rather than restoring one shared array for every diagnostic.

**Regression:** Test actual fit × aspect-mode × detector results, including denominators and window geometry—not just whether configuration objects construct.

## COA4-05 — Tracy–Widom mistakes the pool size for an operator dimension

**Location:** `rmt/factory.py:464–474`.

The dispatcher sets `effective_small = values.size` and derives the larger dimension from `q`. For pooled FARMS levels, this counts repeated/windowed observations as additional rows of a single Wishart operator. Adding measurements then spuriously reduces the finite-size edge correction.

**Reproduced:** With square 32-by-32 windows and variance 1, the single-window threshold is approximately **4.181043170**. Passing a 64-value pool gives **4.122577394**; repeating exactly those observations ten times gives **4.030094209**. The per-window operator geometry never changed.

**Repair direction:** Pass actual source/window dimensions explicitly. Apply TW to a specified operator or per window; define any multi-window decision/count or multiple-testing policy separately.

**Regression:** Duplicating observations must not change the per-operator threshold. Check the exact dimension pair for rectangular windows too.

## COA4-06 — Presets still overwrite explicit CLI aliases

**Locations:** `run_experiments.py:957–966, 995–1034`; aliases in `pipelines/cli_config.py:166–176, 192–199`.

Explicit-option tracking stores raw flag spellings, whereas `preset()` checks only canonical flag names. Argparse aliases are accepted and then treated as omitted during preset resolution. Accepted abbreviations have the same structural risk.

**Reproduced:**

| Arguments | Parsed value | After runtime presets |
|---|---|---|
| `--experiment-mode reproduce_paper2 --unfolding-degree 3` | degree 3 | degree 15 |
| `--overlap-mode=frobenius_projection` | Frobenius projection | Staats dual-end |

The second command uses the default Golden mode. Existing parser tests do not exercise this production resolution step.

**Repair direction:** Track explicit argparse destinations, resolving aliases to their destination; handle or disable abbreviations. Preserve canonical and paired boolean overrides.

**Regression:** Test `parse_args()` plus `_resolve_runtime_configuration()` for each alias under applicable presets, with both space-separated and `--flag=value` forms.

## COA4-07 — Default Lanczos spike margin is still in the wrong production scale

**Locations:** `rmt/lanczos_stieltjes.py:916–920, 1020–1045`; defaults in `rmt/factory.py:108–109`; initialization in `models/transformer.py:33, 118–129`.

This is a **production calibration issue**, not a claim that the explicitly documented absolute reference formula is algebraically wrong. The factor adapter normalizes by dimension but not variance; its margin remains `c*N^-delta`, with default `c=1`. Transformer entry standard deviations are around 0.02 or smaller, so this margin can dominate their covariance spectrum.

**Reproduced:** Gaussian `(128,256)` factor, seed 0, first row multiplied by 6, Lanczos steps 50/probes 1:

| Factor scale | Upper edge | Threshold | Largest eigenvalue | Spikes |
|---|---:|---:|---:|---:|
| 1 | 2.817008 | 3.114310 | 37.331197 | 1 |
| 0.02 | 0.001126803 | 0.298428582 | 0.014932479 | 0 |

The same strongly separated relative spike disappears solely because of units.

**Repair direction:** Calibrate the production adapter in a normalized covariance domain and map results back, or expose a clearly named scale-relative margin. Preserve/document the pure absolute reference convention and serialize effective units.

**Regression:** Include actual initialization/residual scales and rescaled versions of the same spiked factor, not only unit-variance Wishart tests.

## COA4-08 — Scaling plot groups by unstable realized-ratio floats

**Location:** `run_experiments.py:719–749`.

Series membership uses the exact `realized_allocation_ratio`. Architecture discretization means one requested intervention acquires different realized ratios at different budgets. The subsequent `isclose()` comparison does not recover the intervention identity; nearly equal keys can also generate duplicate curves.

**Reproduced:** The default architecture search at vocabulary size 50257 and budgets `1e15, 1e16, 1e17` gives realized ratios **0.0610740783, 0.0644938051, 0.0633483483** for requested `kappa=0.25`. Those become three one-point series instead of one three-budget trajectory.

**Repair direction:** Group by a stable requested-intervention identifier. Annotate realized ratios and collapsed/distorted allocations per point; do not claim that realized ratios are exactly matched.

**Regression:** Inspect plotted series data for a multi-budget manifest. Each requested noncollapsed intervention should have its intended budget sequence exactly once.

## COA4-09 — Gap ratio remains coupled to unrelated unfolding availability

**Location:** `run_experiments.py:420–455`.

`r_statistic(bulk_levels)` is still inside successful unfolding and its smoother-specific minimum-size gate. The adjacent-gap ratio requires neither that fit nor that many observations.

**Reproduced:** For a seed-18 Gaussian `(64,128)` factor with analytic raw-domain MP selection, inject a `ValueError` from the unfolding dispatcher. The selected 64-level bulk has raw gap ratio approximately **0.498401514**, but the runner returns `r_statistic=NaN`.

**Repair direction:** Compute the gap ratio on the selected single-operator bulk independently, with its own minimum-count/degeneracy checks. Scope unfolding failure to the statistics that actually require unfolding.

**Regression:** Force unfolding failure on a valid bulk and assert that gap ratio survives unchanged while unfolding-dependent diagnostics are unavailable.

## COA4-10 — Float32 centering can destroy the activation covariance

**Locations:** `pipelines/activation_extractor.py:45–73, 82–98`; production float32 default in `pipelines/cli_config.py`; eigendecomposition in `rmt/overlap.py:15–27`.

The accumulator forms raw float32 first/second moments, then subtracts `E[xxᵀ] - E[x]E[x]ᵀ`. For activations concentrated near a nonzero mean, this subtracts nearly equal large numbers. Symmetrization does not recover the covariance, and the downstream eigensystem checks finiteness but not covariance validity or cancellation error.

**CPU reproduction using the production accumulation dtype:**

```python
x = (1.0 + 1e-4 * np.random.default_rng(7).normal(size=(4096, 4))).astype(np.float32)
a = CovarianceAccumulator(4, dtype=torch.float32, device="cpu")
a.update(torch.from_numpy(x))
observed = a.covariance().numpy()
reference = np.cov(x.astype(np.float64), rowvar=False, bias=True)
```

The reference's eigenvalues are approximately **9.56e-9 to 1.03e-8**, all positive. The computed covariance has eigenvalues approximately **[-9.53e-7, -4.05e-7, -8.73e-8, 7.30e-7]**, with relative Frobenius error approximately **64.3**. Promoting the already damaged covariance to float64 before `eigh` cannot fix it. This is a numerical probe, not a claim that a particular trained checkpoint exhibited these activations.

**Repair direction:** Use stable centered streaming/batch-merge moments or a controlled two-pass centered calculation. Add scale-aware covariance validity/error checks and honest unavailable status. Merely clipping negative eigenvalues would conceal the inaccurate eigenvectors and does not solve the cancellation.

**Regression:** Compare against a float64 centered reference on nonzero-mean, low-variance inputs, including multiple batches. Verify both covariance error and PSD tolerance under each supported accumulation dtype.

## COA4-11 — Capture and evaluation do not restore mixed submodule modes

**Locations:** `pipelines/activation_extractor.py:298–327`; `pipelines/trainer.py:120–160`.

Both helpers save only `model.training`, call `model.eval()`, and finally call `model.train(was_training)`. The last call recursively gives every submodule the root's mode. A borrowed training model with intentionally frozen/eval submodules is permanently changed even though these are measurement helpers.

**Reproduced:** A runnable embedding/linear/dropout model starts with root training enabled and dropout explicitly in eval mode. Each helper independently changes mode flags from **[True, True, True, False]** to **[True, True, True, True]**. This affects library reuse; the runner's uniformly configured models ordinarily conceal it.

**Repair direction:** Snapshot and restore each original module's training flag, including exception paths. The sibling ESD perplexity evaluator now implements this pattern; its activation helper still has the analogous defect.

**Regression:** Mixed-mode models must retain exact flags after successful capture/evaluation, a forward failure, and an invalid-loss failure.

## COA4-12 — Hill support metadata is one observation short

**Locations:** `rmt/tail.py:173–180, 218–226`.

A window starting at one-based rank `k` averages `window` Rényi log-spacings. Its final spacing uses both `x[k+window-1]` and **`x[k+window]`**. Metadata instead ends the observation range at `last_window_start + window - 1`, omitting that final boundary observation.

**Reproduced:** For values `np.arange(1.0, 201.0)**(-0.5)` and window 20, the plateau reports ranks **5–83**, support **79**. The selected windows actually require ranks **5–84**, or **80 observations**. The exponent itself is not changed by this reporting defect. These detailed support fields belong to the library result; the runner currently exports only a subset of plateau metadata.

**Repair direction:** Distinguish observation support from the count/ranks of Rényi spacings. Include the final boundary observation when reporting observation support; do not alter the estimator to fit the erroneous metadata.

**Regression:** Reconstruct the observation indices consumed by selected windows and compare their support with the serialized range/count. The sibling implementation has the same off-by-one defect.

## Recommended repair order

1. Restore importability with the smallest targeted repair for COA4-01, then run the full suite and real runner smoke paths.
2. Repair COA4-02–05 together using separate spectral domains and actual operator/window geometry. Preserve the existing gradient/no-progress/evaluation safeguards.
3. Fix CLI precedence, production Lanczos calibration, and stable covariance accumulation.
4. Fix reporting/independent diagnostics and borrowed-model state restoration; correct the smaller Hill metadata defect.

The 75 passing isolated tests do not cover the broken production import, training fault handling, production preset precedence, or domain consistency across the runner. Add regressions at those boundaries rather than weakening existing numerical checks or declaring success from parser/default tests alone.
