# Spectral-Chinchilla design and scientific decisions

## Scope

The repository has two dependency layers. `rmt/` is a pure spectral library and does not import torch. `models/` and `pipelines/` contain Transformer construction, training, activation hooks, and model surgery. The experiment runner combines these layers. It does not hide training limits or replace trained-model data with synthetic data.

## Mathematical conventions

Let a real matrix have shape `n x m` and reduced rank `k=min(n,m)`. Singular values use descending order. `SVDResult.eigenvalues` gives raw Gram values `s_i^2`. `SVDResult.covariance_eigenvalues` gives normalized values for MP diagnostics.

For the canonical MP view, use the smaller dimension for reduced covariance. Normalize by `max(n,m)`:

`lambda_i = s_i^2 / max(n,m)` and `q = min(n,m) / max(n,m)`.

For entry variance `sigma^2`, the continuous MP support is:

`lambda_pm = sigma^2 (1 +/- sqrt(q))^2`.

The compatibility API `eigenvalues_of_cov(weight=None, s=None, N=denominator)` also supports `s^2/N`. Its default denominator is the matrix column count. If callers use this view, they must record `N`.

The CSN estimate `alpha` is the density exponent in `p(x) proportional to x^(-alpha)`. The Hill estimate is the positive survival exponent. For a continuous Pareto law, `alpha_CSN = alpha_Hill + 1`. If singular values are squared, the Hill survival exponent is divided by two. Reports identify the base variable and estimator.

Spectral unfolding fits a smooth cumulative staircase. It then normalizes mean nearest-neighbor spacing to one. Brody `beta=0` is Poisson. Brody `beta=1` is the GOE Wigner surmise. Number variance and Dyson-Mehta rigidity require enough unfolded interval for the selected window.

Activation covariance uses centering by default. The extractor can return uncentered `X^T X / n` when the user requests it. Default overlap uses squared projections that do not depend on signs.

Overlap is unavailable for a zero numerical rank. The method excludes null directions. If an eigenvalue cluster crosses a cutoff, the selected subspace includes the full cluster. Other choices use principal angles or normalized projector trace.

The sign-sensitive maximum-cosine method stays as a separate diagnostic. It is not the default. Weight tranches use descending singular values. Top means largest, bottom means smallest, and bulk excludes both ends.

## FARMS pooled ESD

FARMS means Fixed-Aspect-Ratio Matrix Subsampling. By default, preserve the orientation of the source matrix. Transpose only if `orient_tall=True`.

The released ratio is `Q_F=m'/n'`, which means sampled columns divided by sampled rows. `window_size=n'` is the sampled row count. The reference sampler offers two schedules. One uses floor strides from requested row and column operation counts. The other uses a fixed sliding step.

Grid and seeded-random schedules are separate alternatives. For each equal-shape window `W_ab`, use one of these spectra:

`lambda_i^(ab) = s_i(W_ab)^2`

or

`s_i(W_ab)^2/max(m',n')`.

The first formula reproduces the reference bytes and formula. The second is canonical covariance normalization for MP comparisons across shapes.

FARMS concatenates the equal-length spectra. Thus, its empirical measure is the arithmetic mean of window ESDs. Output records source, oriented, and window shapes. It records `m'/n'`, `q_F=min(m',n')/max(m',n')`, starts, normalization, denominators, orientation, window count, and estimated coverage.

Overlapping windows are correlated. Do not interpret their pooled spacings as the NNSD of one matrix.

The runner computes NNSD, Brody, gap ratio, number variance, and rigidity from the canonical spectrum of the full matrix. An independent raw-domain MP fit selects that spectrum. FARMS supplies pooled ESD and tail comparisons only.

Each row identifies separate ESD, MP-fit, detector, and spacing domains. It gives geometry, observation counts, and normalization data. Lanczos and Thamm operator methods stay in the raw domain. The Golden workflow also does its selected FARMS ESD and tail stage.

A FARMS-unbiased fit requires FARMS preparation. The dispatcher rejects a combination that compares different observation sets.

`shape_normalized` is a separate analytic baseline:

`lambda_tilde = lambda * 4 / (sigma^2 (1+sqrt(q))^2)`.

This map puts the null upper MP edge at four. It does not restore correlations outside a FARMS window. Output uses different names for the two methods.

## Lanczos-Cholesky Stieltjes support

For a positive covariance operator `A` and unit probe `b`, Lanczos makes a Jacobi matrix `J_k`. Its diagonal is `a_j`, and its off-diagonal is nonnegative `b_j`. Full and partial reorthogonalization are available.

The finite vector empirical Stieltjes transform is:

`s_b(z) = e_1^T (J_k-zI)^(-1) e_1`.

The reference support estimator factors `J_k=L_k L_k^T`. Deep diagonal and subdiagonal entries of the lower bidiagonal factor can converge to constants `alpha` and `beta`. If they do, the one-cut support estimate is:

`gamma_minus=(alpha-beta)^2`, `gamma_plus=(alpha+beta)^2`.

Phase III follows the released Julia recurrence. It normalizes Gaussian probes to the sphere. Full reorthogonalization uses two projection passes. Optional adaptive stopping compares local means and sample deviations of consecutive Jacobi coefficients.

The reference tail method averages the stable suffix in Jacobi coordinates. It adds two diagonal entries and one off-diagonal entry. Then it applies Cholesky factorization to the changed finite matrix. Cross-probe terminal Cholesky coefficients give one consensus. Output gives stop and edge diagnostics for each probe. VEST uses the average continued fraction from all probes.

The default `reference_ritz` rule counts finite-Jacobi Ritz values outside the consensus support. This agrees with the released detector. Output also gives first-component VEST residues. An optional residue floor is an extension, not part of the reference result.

The separate `constant_tail` rule uses a finite section of the semi-infinite tail. It keeps a right pole only if the pole satisfies two limits. The pole must be more than `gamma_plus + C gamma_plus n^(-delta)`. It must also be more than the optional residue threshold.

The limits are `C>=0` and `0<delta<1/2`. A margin relative to the fitted edge makes decisions independent of matrix units. Defaults are `C=1` and `delta=0.25`. Output records the effective margin and `bulk_edge_relative` convention.

The method assumes a positive one-cut limiting spectrum. It also assumes square-root edges and a finite number of separate right outliers. A failed convergence diagnostic does not become an exact boundary claim.

The released repository has no generic fixed-point solver from the Phase II short description. Thus, this project does not give that name to the Cholesky-tail estimator.

## Tracy-Widom and BBP

For real Wishart covariance normalized by the larger dimension, a finite Johnstone center and scale define the TW1 threshold. The method uses tabulated standard TW1 right quantiles. It interpolates only in the supported confidence interval.

For identity-noise population covariance and canonical `q`, the BBP population transition is:

`theta_c=sigma^2(1+sqrt(q))`.

The null sample edge is `sigma^2(1+sqrt(q))^2`. A supercritical population eigenvalue `theta` maps to:

`lambda(theta)=sigma^2 t (1+q/(t-1))`, where `t=theta/sigma^2`.

Result fields keep population and sample thresholds separate.

## Meanings of selectable methods

- `analytic_mp` is the corrected one-parameter quantile fit.
- `kde_bulk_fit` keeps the upper-tail and bandwidth interface. It fits integrated MP mass to retained ECDF ranks from the full sample.
- The ECDF fit includes zero-mass rank offsets and records solutions at a bound.
- `fit_modified_mp_singular` reproduces the unconstrained codebase2 curve. It fixes the empirical lower edge and fits amplitude and upper edge.
- `thamm_modified_singular` converts that curve to canonical covariance units. It labels variance as an upper-edge compatibility value, not an analytic MP fit.
- `polynomial_chebyshev`, `spline_monotone`, and `gaussian_kernel` fit a smooth staircase.
- `raw_rank_order` is a diagnostic that intentionally removes spacing variation.
- Brody MLE and CDF least-squares fits use bounds `[0,1]`. Optional seeded bootstrap gives uncertainty for the CDF fit.
- Number variance has deterministic sliding and seeded Monte Carlo forms. `Delta_3` uses exact integration of the empirical staircase.
- CSN, fixed-cutoff, and rank regression give density exponents. Hill gives a survival exponent.
- `csn_goodness_of_fit` gives the semiparametric bootstrap probability separately from the observed KS distance.
- Golden defaults combine canonical FARMS, Lanczos-Stieltjes support, `reference_ritz`, monotone spline, CSN MLE, and Staats dual-end overlap.
- Paper reproduction modes explicitly override these choices through the CLI.

## Offline and accelerator boundary

Compute jobs have no network substitute. An operator uses `scripts/download_assets.py` on a connected host. The utility requires `--allow-network` and writes a checksum manifest. Runtime data accepts only local NPY or NPZ token arrays.

The cluster launcher calls `/home/shivansh/.conda/envs/rmt_ml_env/bin/python`. It does not use a project `.venv`, module purge, or guessed module. Another interpreter or CUDA module requires an explicit override that passed its tests.

Hugging Face caches stay under the project. Compilation caches use local job storage when available. `run_hpc.slurm` sets all three offline controls. `PYTHONPATH` contains only this project, so it cannot load the sibling `rmt` implementation.

The pure layer receives NumPy arrays and SciPy linear operators. The model and pipeline layer owns mixed precision, TF32, compilation, pinned transfers, CUDA covariance, and accelerator SVD.

Activation hooks merge centered batch moments in float32 or float64 on the selected device. They transfer only the final covariance to the host. `compute_tensor_svd` changes stored weights to `--analysis-dtype`, with float64 as the default. It does a reduced device SVD and then makes `SVDResult` from transferred reduced factors.

Lesions use the same recorded analysis dtype and CUDA SVD driver. They reconstruct weights with the original execution dtype and device.

The SLURM profile requests one accelerator for each task. Operators can submit independent cells as tasks or arrays. The code does not implement distributed-data-parallel operation.

## Compute allocation

For `L=E+A/N^alpha+B/D^beta` under `C=c_f N D`, constrained minimization gives:

`N* = (alpha A/(beta B))^(1/(alpha+beta)) (C/c_f)^(beta/(alpha+beta))`

and

`D* = (beta B/(alpha A))^(1/(alpha+beta)) (C/c_f)^(alpha/(alpha+beta))`.

This corrects the transposed `alpha` and `beta` prefactor. A separate `target_tokens_per_parameter` mode enforces `D/N=20` and preserves compute. At fixed compute, the multiplier `kappa` gives `D=kappa D*` and `N=N*/kappa`. A change to only `D` leaves the IsoFLOP surface.

## Literature audit

Hoffmann et al. support equal proportional scaling of model size and training tokens. `C approximately 6ND` stays an accounting estimate. Output separately records targets, attempts, forwarded positions, skipped updates, and allocation ratios. It rejects capped duplicate designs unless the user identifies them as calibration.

Qiu et al. report normalized loss-curve collapse for compute-optimal scaling. They report a failure of collapse with suboptimal hyperparameter scaling. Spectral collapse is a new hypothesis in this project.

Thamm, Staats, and Rosenow report universal RMT behavior in much of trained-network spectra. They state that one tail index does not generally describe the large-value tail.

Staats, Thamm, and Rosenow connect RMT differences with activation-covariance overlap. They show that removal of small singular values can increase perplexity, especially after alignment.

Martin and Mahoney motivate MP bulk diagnostics, soft rank, heavy-tail self-regularization, and the 5+1 phase names. Hu et al. define FARMS with fixed-ratio subsampling, pooled ESD fitting, and a window-shape hyperparameter.

Abi Younes, Ding, and Trogdon derive the Lanczos-Cholesky constant-tail support and pole estimator. Their assumptions apply to a one-cut sample covariance.

Primary sources are arXiv:2203.15556, arXiv:2507.02119, arXiv:2203.14661, and arXiv:2410.17770. They also include JMLR 22(165), 2021, arXiv:2506.06280, and arXiv:2504.03066.

## Reliability decisions

Pure functions examine dimensions, finite values, positivity, and nondegenerate spectra. Downstream methods can share one `SVDResult`. Tail-fit errors return structured nonfinite estimates and do not invent values.

Lesion contexts restore original parameters in a `finally` path. Each benchmark starts from unchanged data. Unit tests calibrate mathematics only. Trained-model hypotheses belong in experiment reports.

`RMTMethodConfig` examines each CLI selection one time. The runner records it before optional training. Manifests contain exact FLOP tiers `10^15`, `10^16`, and `10^17`. A token limit selected by an operator becomes a recorded difference in realized budget.

Source construction does not certify runtime success. Operators must do numerical, CUDA, asset, and SLURM gates on the target cluster.
