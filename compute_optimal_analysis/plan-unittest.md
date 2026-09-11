# Unit-test contract

All stochastic tests use fixed seeds. Pure tests import only NumPy, SciPy, pytest, and `rmt`. Torch tests use tiny in-process models and no network or package installation.

Phase III assigns each test group to one file. `test_bug_report_regressions.py` contains static-audit regressions. They cover autocast moments, collapse, finite configuration, unavailable results, and steep bounded tails.

`test_lanczos_stieltjes.py` covers VEST, support, and reference-Ritz calibration. `test_farms_aspect_ratio.py` covers reference geometry, sampling, and fixed-ratio invariance. `test_activation_hooks.py` covers device reduction and the bridge to pure SVD. `test_cli_dispatch.py` covers source isolation, four-track parsing, SLURM, and offline contracts.

The `test_pure_rmt_*` files cover other MP, tail, spacing, scalar, and overlap methods. Model, allocator, lesion, and pure numerical tests stay separate.

## MP calibration

- The analytical density integrates to one within `1e-3` for aspect ratios `0.25`, `0.5`, and `1.0`.
- Density is zero outside support and both bounds are nonnegative and ordered.
- A Gaussian factor has an empirical nonzero covariance spectrum near analytical support. Edge tests permit finite-size Tracy-Widom changes, not a deterministic one-percent limit.
- The median estimator recovers entry standard deviation within `2 percent` on seeded matrices large enough for asymptotics.
- Eigenvalue and singular-value compatibility APIs agree exactly under the declared normalization.
- The historical KDE path uses integrated ECDF mass and returns positive ordered support. It recovers seeded Gaussian scale without selection of an optimization boundary.
- TW95 lies more than the asymptotic null edge for the calibration size. BBP tests distinguish the population threshold `1+sqrt(q)` from the sample edge `(1+sqrt(q))^2`.

## Lanczos–Stieltjes calibration

- Full-dimension, full-reorthogonalized Lanczos VEST agrees with a direct dense resolvent quadratic form within `1e-9` on a seeded positive matrix.
- Full reorthogonalization uses the released two-pass projection cleanup. A fixed seed gives deterministic Gaussian probes and recurrence metadata.
- Constant Cholesky entries `alpha=1`, `beta=0.5` produce support `[0.25,2.25]` exactly to floating precision.
- Finite VEST poles are the eigenvalues of the realized Jacobi matrix more than the threshold, and reported residues are squared first Ritz-vector components.
- The reference modified tail begins at `used - check_interval - sequence_length - 1` after convergence, replaces the suffix in Jacobi coordinates before Cholesky factorization, and records its selection mode.
- A seeded `240 x 960` factor with three population spikes more than `1+sqrt(q)` is analyzed matrix-free with 96 Lanczos steps and three probes.
- The detector returns exactly three separated right poles, positive residues, and an upper support edge within `2.5 percent` of `(1+sqrt(q))^2`.
- The result labels the `reference_ritz` rule, stores one convergence flag per probe, and returns a finite nonnegative probe-averaged density on the estimated support.
- Production factor analysis uses a bulk-edge-relative margin and gives the same spike decision after uniform factor rescaling. Direct operator analysis retains the named absolute reference mode.
- After modal spike counting, reference spike locations and residues come from the longest realized recurrence, matching the released source.
- Tests do not substitute the incorrect `(1+q)^2` expression.

## FARMS calibration

- Reference floor-stride starts, reference fixed-step starts, grid starts, pooled spectra, and metadata are deterministic for a fixed source and configuration.
- The reference ratio is sampled columns divided by sampled rows, `target_aspect_ratio=2` and `window_size=3` thus produce a `3 x 6` window.
- Every sampled window has the declared fixed shape. Pooled level count equals windows times reduced window dimension.
- Gaussian matrices with source row/column ratios `{0.1,0.25,0.5,1.0,2.0,4.0}` are sampled through square `256 x 256` windows. Analytically fitted pooled upper edges remain within `2 percent` of the common null value four.
- `shape_normalize_eigenvalues` maps the analytic raw edge to the requested target exactly and remains separately labeled from FARMS.
- Raw and canonical pooled spectra from identical windows differ exactly by `max(window_shape)`. Raw, canonical, and trace modes record one exact denominator per sampled window.
- Runner regressions send pooled FARMS levels to ESD and tail output. Spacing count and gap ratio stay with the original operator. Golden Lanczos keeps raw operator input.

## Tail calibration

- A continuous Pareto density exponent `3.0` is recovered by CSN within `2 percent` for a sufficiently large seeded sample.
- The Hill survival exponent is near `2.0`, and squaring observations halves that exponent.
- Too-short or degenerate tails return structured non-finite estimates without raising an unrelated exception.
- Fixed-cutoff MLE recovers a Pareto density exponent `3.0` within `2 percent`. Rank regression is allowed `5 percent` and must report `R^2>0.98` on the seeded large sample.
- CSN semiparametric bootstrap returns a probability in `[0,1]`, preserves the observed fit fields, and is reproducible for a fixed seed.

## Spacing calibration

- Unfolded mean spacing is one to numerical tolerance.
- Poisson levels fit Brody beta near zero, GOE bulk levels fit closer to one than Poisson levels.
- Every fitted beta remains in `[0,1]`.
- Mean adjacent-gap ratio is larger for GOE than for Poisson.
- Number variance is finite and nonnegative for supported windows.
- Chebyshev and monotone-spline unfolding each map the same central GOE spectrum to mean spacing `1.00 +/- 0.01`. The large seeded calibration requires Brody beta in `[0.90,1.00]`.
- Adaptive Gaussian unfolding has strictly increasing finite output when its local window fits the spectrum.
- Bounded CDF least squares and bounded MLE always return beta in `[0,1]`. Seeded bootstrap uncertainty is nonnegative when requested.

## Scalar and overlap calibration

- Identity stable rank is its dimension. Rank-one stable rank is one.
- Entropy of equal singular energy is `log(rank)` and rank-one entropy is zero.
- Condition number is infinite for an exactly singular matrix.
- Projection entries lie in `[0,1]`. Aligned bases produce an identity overlap matrix. Zero-rank covariance is unavailable, null modes are excluded, and tied cutoff clusters are not split.
- Top, bulk, and bottom tranche indices are disjoint and cover the spectrum.
- Identical subspaces have zero principal angles and unit principal-angle/projector scores after arbitrary basis rotation.
- A separate test covers the archived signed maximum-cosine convention and sign-invariant alternatives.
- Archive localization and participation ratios reproduce their declared norm ratios.
- Finite-dimensional Porter–Thomas Monte Carlo returns per-vector distances and probabilities with deterministic seed plumbing.

## Model and pipeline calibration

- Causal-LM logits have shape `(batch, sequence, vocabulary)` and labeled forward returns finite cross-entropy.
- Changing a future token cannot change earlier logits while dropout is disabled.
- Allocation conserves `C=c_fND`. Target-ratio mode satisfies `D/N=20`. Fixed-compute regimes satisfy `N=N*/kappa`, `D=kappa D*`.
- Captured covariance is symmetric positive semidefinite and has feature dimension equal to module input width.
- Explicit float32 accumulation allocates both moment buffers on the requested device. A calibration compares stable centered covariance with float64 for many batches, a large mean, and small variance. CUDA is an operator hardware gate.
- `compute_tensor_svd(matrix, backend="cpu")` returns a reconstructing pure `SVDResult`, CUDA driver behavior is an operator hardware gate.
- Lesion contexts restore every parameter exactly, including when evaluation raises. Non-finite tranche/decile evaluations fail rather than becoming completed results.
- FP32 non-finite gradients fail before optimizer/accounting mutation, epoch-only all-ignored loaders terminate, finite high losses remain distinct, and capture/evaluation restore every mixed submodule mode.
- Reference descending-decile lesions map decile zero to the largest group. They map the last decile to the smallest group and restore exact parameters.

## CLI and isolation calibration

- Parsing no method flags produces exactly `RMTMethodConfig()`.
- Every declared value in each of the six primary choice tuples constructs successfully, both alone and in the full Cartesian configuration set.
- Each MP fit, unfolding, tail, overlap, and spike method is dispatched on a small seeded calibration input and returns its declared result label. Standalone Lanczos assumption failures return unavailable status without discarding other metrics.
- Bounded-Pareto quantiles recover their source exponent. Rank-tail fits do not change with scale. Three levels give one gap ratio.
- Scaler overflow can decrease and try again. Token-budget schedules cross loader boundaries. Only one concurrent process can acquire output.
- The Thamm adapter must preserve raw singular edges through `s**2/max(shape)`. It must return finite compatibility scale and KS diagnostics.
- `--boundary-detector lanczos_stieltjes` resolves to MP method `lanczos_stieltjes` and spike detector `lanczos_poles`.
- An argparse-destination test covers production preset precedence. Thus, aliases and `--flag=value` override presets. Long-option abbreviation is disabled.
- Pipeline defaults resolve to the Golden method set and BF16/CUDA accelerator intent. Argument sets for all four SLURM tracks parse through the centralized schema.
- Static SLURM tests require all four track functions and the three strict offline environment exports.
- Every maintained `rmt/*.py` source is scanned case-insensitively. The framework name prohibited by the isolation contract must have zero occurrences, including comments and docstrings.
- Placeholder scanning applies to maintained project source/specifications/SLURM files and excludes all read-only reference archives.

## Offline deployment calibration

- Staging selects one canonical dataset configuration and tokenizer. It resolves fixed commit SHAs and requires explicit network acknowledgment. It writes revision and SHA-256 metadata.
- Runtime code has no remote fallback: a missing local token array raises before model training.
- `requirements.txt` uses exact direct pins only. The wheelhouse and platform-lock procedure in `other_requirements.md` is an operator prerequisite.
- The SLURM launcher examines the virtual environment, token array, and asset manifest before a track starts. It sets local cache roots.
- The SLURM launcher starts the local manifest tool. The tool examines path containment, byte sizes, and SHA-256 before a track starts.
- No unit test downloads an asset, imports a remote dataset, starts a job, or assumes an accelerator exists.

## Operator-only empirical tests

Heavy-tail ranges in trained deep layers, dual-end information localization, lesion perplexity ordering, and spectral scaling collapse require completed training executions. They are report assertions, not unit tests against untrained random weights.
