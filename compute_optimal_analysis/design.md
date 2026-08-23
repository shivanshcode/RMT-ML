# Spectral-Chinchilla design and scientific decisions

## Scope

The repository has two dependency layers. `rmt/` is a mathematically pure spectral library with no torch import. `models/` and `pipelines/` own transformer construction, training, activation hooks, and in-place model surgery. The experiment runner composes the layers but does not hide training caps or substitute synthetic observations for trained-model results.

## Mathematical conventions

Let a real matrix have shape `n x m`, reduced rank `k=min(n,m)`, and singular values in descending order. `SVDResult.eigenvalues` exposes the raw Gram values `s_i^2`; `SVDResult.covariance_eigenvalues` exposes the normalized values used by MP diagnostics. For the canonical MP view, orient the reduced covariance along the smaller dimension and normalize by `max(n,m)`. Thus

`lambda_i = s_i^2 / max(n,m)` and `q = min(n,m) / max(n,m)`.

For entry variance `sigma^2`, the continuous MP support is

`lambda_pm = sigma^2 (1 +/- sqrt(q))^2`.

The explicit compatibility API `eigenvalues_of_cov(weight=None, s=None, N=denominator)` also supports the supplied contract's `s^2/N` convention; its default is the matrix column count. Callers must record `N` whenever they use that view.

The CSN estimate `alpha` is a density exponent in `p(x) proportional to x^(-alpha)`. The Hill estimate is the positive survival exponent. For an ideal continuous Pareto law, `alpha_CSN = alpha_Hill + 1`. Squaring singular values halves the Hill survival exponent. Reports label the base variable and estimator to prevent accidental comparison of unlike exponents.

Spectral unfolding fits the smooth cumulative staircase and then renormalizes mean nearest-neighbor spacing to one. Brody `beta=0` is Poisson and `beta=1` is the GOE Wigner surmise. Number variance and Dyson-Mehta rigidity are evaluated only where the unfolded interval supports the requested window.

Activation covariance is centered by default, matching the referenced small-singular-value analysis. The extractor can return the uncentered second moment `X^T X / n` when explicitly requested. The default overlap uses squared, sign-invariant projections. Selectable alternatives use principal angles or the normalized projector trace; the sign-sensitive maximum-cosine archive convention remains an explicit diagnostic API rather than the default. Weight tranches are always indexed with singular values descending: top means largest, bottom means smallest, and bulk excludes both ends.

## Phase II and Phase III bulk/signal boundary design

### FARMS fixed-ratio pooled ESD

For a source matrix, preserve its orientation by default; transpose only when the explicit compatibility option `orient_tall=True` is selected. The released FARMS convention is `Q_F=m'/n'`, sampled columns divided by sampled rows, and `window_size=n'`, the sampled row count. The reference sampler uses either a floor-stride schedule chosen from requested row/column operation counts or a fixed-step sliding schedule. Deterministic uniform-grid and seeded-random schedules are additional, separately named alternatives. For each equal-shape window `W_ab`, compute

`lambda_i^(ab) = s_i(W_ab)^2` for byte-for-formula reference reproduction, or `s_i(W_ab)^2/max(m',n')` for the canonical covariance normalization used by cross-shape MP fitting.

The FARMS spectrum is the concatenation of these equal-length series. Because each window contributes the same number of levels, its empirical measure is the arithmetic mean of the window ESDs. Results record source/oriented/window shape, starts, the reference ratio `m'/n'`, canonical `q_F=min(m',n')/max(m',n')`, normalization, transpose state, number of windows, and estimated source coverage. Overlapping windows are correlated; pooled level spacings must therefore not be interpreted as one matrix's NNSD.

Accordingly, the experiment runner performs NNSD, Brody, gap-ratio, and number-variance calculations on the full matrix's canonical spectrum. The Paper 2 track selects that spacing bulk with codebase2's modified singular-domain curve fit after converting its empirical edges to canonical covariance units; other tracks retain the corrected analytic selector. FARMS remains available for ESD, MP, and tail comparisons without contaminating level statistics with duplicated overlapping-window levels.

`shape_normalized` instead applies the explicit analytic baseline

`lambda_tilde = lambda * 4 / (sigma^2 (1+sqrt(q))^2)`.

This maps the null upper MP edge to four but does not recover the correlations that FARMS samples. The two methods have distinct names and result metadata.

### Lanczos–Cholesky Stieltjes support

For a positive covariance operator `A` and unit probe `b`, full- or partial-reorthogonalized Lanczos produces a Jacobi matrix `J_k` with diagonal `a_j` and nonnegative off-diagonal `b_j`. Its finite vector empirical Stieltjes transform is

`s_b(z) = e_1^T (J_k-zI)^(-1) e_1`.

The paper-faithful support estimator Cholesky-factorizes `J_k=L_k L_k^T`. If sufficiently deep diagonal and subdiagonal entries of the lower bidiagonal factor converge to constants `alpha` and `beta`, the one-cut support estimate is

`gamma_minus=(alpha-beta)^2`, `gamma_plus=(alpha+beta)^2`.

Phase III follows the released Julia recurrence more closely. Gaussian probes are normalized to the sphere, full reorthogonalization uses two projection passes, and optional adaptive stopping compares successive local means and sample deviations of Jacobi coefficients. The reference tail path averages the stable suffix in Jacobi coordinates, appends two diagonal entries and one off-diagonal entry, Cholesky-factorizes that modified finite matrix, and forms a cross-probe consensus from the terminal Cholesky coefficients. The implementation reports per-probe stopping and edge diagnostics and averages the continued-fraction VEST across probe recurrences.

The default `reference_ritz` pole rule counts finite-Jacobi Ritz values outside the consensus support, matching the released detector. It also reports their first-component VEST residues and permits an optional residue floor without silently making that extension part of the reference result. The separately named `constant_tail` rule uses a finite section of the semi-infinite tail. A right pole is retained only above

`gamma_plus + C n^(-delta)`, with `C>=0` and `0<delta<1/2`, and above an optional residue threshold. Defaults reproduce the paper's numerical `C=1`, `delta=0.25`; they are tuning parameters, not universal constants.

The method assumes a positive one-cut limiting spectrum with square-root edges and finitely many separated right outliers. A failed convergence diagnostic is reported rather than converted into a claim of an exact, fluctuation-free boundary. The released repository does not contain the generic fixed-point solver described in the Phase II shorthand, so this implementation does not mislabel the Cholesky-tail estimator as that absent algorithm.

### Tracy–Widom and BBP separation

For real Wishart covariance normalized by the larger dimension, the finite Johnstone center and scale define the selectable TW1 threshold. The implementation tabulates standard TW1 right quantiles and interpolates only on the supported confidence range.

For identity-noise population covariance and canonical `q`, the BBP population transition is

`theta_c=sigma^2(1+sqrt(q))`.

The null sample edge is `sigma^2(1+sqrt(q))^2`. A supercritical population eigenvalue `theta` maps to

`lambda(theta)=sigma^2 t (1+q/(t-1))`, where `t=theta/sigma^2`.

Population and sample thresholds are never conflated in result fields.

## Multi-method semantics

- `analytic_mp` is the corrected one-parameter quantile fit.
- `kde_bulk_fit` fits MP density to a pure NumPy triangular KDE, reproducing the WeightWatcher archive's scientific structure.
- `fit_modified_mp_singular` reproduces codebase2's unconstrained singular-domain curve with fixed empirical lower edge and free amplitude/upper edge. `thamm_modified_singular` dispatches that curve for the Paper 2 track, converts its fitted support to canonical covariance units, and labels the reported variance as an upper-edge compatibility projection rather than an analytic MP fit.
- `polynomial_chebyshev`, `spline_monotone`, and `gaussian_kernel` estimate a smooth staircase. `raw_rank_order` is a diagnostic that intentionally removes spacing fluctuations.
- Brody MLE and CDF least squares are both bounded to `[0,1]`; optional seeded bootstrap estimates CDF-fit uncertainty.
- Deterministic sliding and seeded Monte Carlo number variance coexist. `Delta_3` uses exact interval integration of the empirical staircase.
- CSN, fixed-cutoff, and rank regression report density exponents. Hill reports a survival exponent. `csn_goodness_of_fit` supplies the semiparametric bootstrap probability separately from the observed KS distance.
- The production default is the Golden synthesis: FARMS canonical fixed-ratio pooling, Lanczos-Stieltjes MP support and `reference_ritz` spike counting, monotone-spline unfolding, CSN MLE, and Staats dual-end overlap. Historical paper reproductions override these choices explicitly through the CLI.

## Phase III offline and accelerator boundary

Compute jobs have no network fallback. `scripts/download_assets.py` is an operator-run staging utility that requires the explicit `--allow-network` acknowledgement on a connected host and writes a checksum manifest. Runtime data loading accepts only local NPY/NPZ token arrays. Hugging Face and framework cache roots are redirected into the repository, and all three offline environment switches are set by `run_hpc.slurm`.

The numerical boundary remains array-based: `rmt/` consumes NumPy arrays and SciPy linear operators only. The model/pipeline layer owns mixed precision, TF32, compilation, pinned-memory transfer, CUDA covariance reduction, and accelerated SVD. Activation hooks update float32 `D x D` Gram and `D`-vector buffers on the selected accelerator and transfer only the final reduced covariance to host memory. `compute_tensor_svd` similarly computes a reduced device SVD and constructs the pure `SVDResult` only after transferring the reduced factors. Lesioning can use the same CUDA SVD driver while reconstructing weights on their original device.

The supplied SLURM profile requests one accelerator per task, as specified by the deployment contract. Independent grid cells can be submitted as separate tasks or cluster arrays by the operator; the code does not claim unimplemented distributed-data-parallel semantics.

## Compute allocation

For `L=E+A/N^alpha+B/D^beta` under `C=c_f N D`, direct constrained minimization gives

`N* = (alpha A/(beta B))^(1/(alpha+beta)) (C/c_f)^(beta/(alpha+beta))`

and

`D* = (beta B/(alpha A))^(1/(alpha+beta)) (C/c_f)^(alpha/(alpha+beta))`.

This corrects the frequently transposed `alpha`/`beta` prefactor. A separate `target_tokens_per_parameter` mode enforces the empirical `D/N=20` rule exactly while conserving compute. At fixed compute, applying regime multiplier `kappa` means `D=kappa D*` and `N=N*/kappa`; changing only `D` would leave the IsoFLOP surface.

## Literature audit

- Hoffmann et al. support equal proportional scaling of model size and training tokens under compute-optimal training; `C approximately 6ND` remains an accounting approximation.
- Qiu et al. report normalized loss-curve collapse for compute-optimally scaled models and breakdown under suboptimal hyperparameter scaling. Spectral collapse in this project is a new hypothesis, not a result attributed to that paper.
- Thamm, Staats, and Rosenow find universal RMT behavior through much of trained-network spectra and explicitly caution that the large-value tail is not generally characterized by one tail index.
- Staats, Thamm, and Rosenow connect RMT deviations and activation-covariance overlap, and show that removing small singular values can increase perplexity, especially after alignment.
- Martin and Mahoney motivate MP bulk diagnostics, soft rank, heavy-tailed self-regularization, and the 5+1 phase language.
- Hu et al. define FARMS as fixed-aspect-ratio matrix subsampling and pooled ESD fitting, with window geometry as a hyperparameter.
- Abi Younes, Ding, and Trogdon derive the Lanczos–Cholesky constant-tail support and pole estimator under one-cut sample-covariance assumptions.

Primary sources: arXiv:2203.15556, arXiv:2507.02119, arXiv:2203.14661, arXiv:2410.17770, JMLR 22(165), 2021, arXiv:2506.06280, and arXiv:2504.03066.

## Reliability decisions

- Pure functions validate dimensions, finiteness, positivity, and non-degenerate spectra.
- A single `SVDResult` can be threaded through downstream analyses.
- Tail fit failures return structured non-finite estimates rather than fabricated values.
- Lesions restore original parameters in a `finally` path and each benchmark starts from a pristine state.
- Unit tests calibrate mathematics only. Trained-model hypotheses belong to experiment reports.
- Every CLI selection is validated once in `RMTMethodConfig` and serialized before optional training begins.
- The requested `10^15`, `10^16`, and `10^17` FLOP tiers are represented exactly in manifests. Any operator-selected token cap is recorded as a realized-budget deviation.
- Source construction never certifies runtime success: numerical tests, CUDA compatibility, asset checksums, and SLURM execution remain explicit operator gates on the target cluster.
