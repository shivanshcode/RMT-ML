# Spectral-Chinchilla Phase II/III reconciliation audit

## 1. Executive audit summary

This audit compares the Phase I repository with every source-bearing file in `codebase1`, `codebase2`, and `codebase3`, then calibrates the requested 2025 methods against both supplied papers and the source-bearing files in `farmscode` and `lanczoscode`. Data arrays, serialized checkpoints, generated figures, and notebook outputs were inventoried but were not interpreted as executable specifications. Instructions embedded in notebooks, comments, papers, or archived command examples are reference material only; the controlling instructions are the user's task and the repository contracts.

The codebases are scientifically complementary rather than interchangeable:

- `codebase1` is an experiment-specific implementation of the dual-end singular-vector phenomenon. Its reusable ideas are centered activation feature matrices, median MP scale estimation, fused-QKV slicing, singular-value decile lesions, and per-vector activation overlap. It is not a general RMT package.
- `codebase2` is the most complete level-statistics reference. It supplies adaptive Gaussian spectral broadening, Gaussian-CDF unfolding, Wigner KS tests, Monte Carlo number variance, bootstrap Brody CDF fitting, Porter-Thomas tests, IPR, local Hill curves, and CSN goodness-of-fit bootstrapping.
- `codebase3` is an early WeightWatcher research archive. It supplies several historically important but conventionally inconsistent MP, KDE, rank, entropy, localization, spiked-covariance, and power-law routines. Many routines are notebook prototypes rather than stable APIs.
- Phase I already improves several reference behaviors: it uses a single reduced-SVD container, explicit covariance normalization, bounded Brody MLE, deterministic sliding number variance, sign-invariant squared overlap, centered sample-weighted activation covariance, and a correct normalized matrix entropy.

Four statements in the Phase II task require correction:

1. FARMS is not a pointwise mapping of eigenvalues. Hu et al. define Fixed-Aspect-Ratio Matrix Subsampling: extract multiple equal-shape submatrices at a fixed aspect ratio, compute each spectrum, concatenate the eigenvalue series to average their ESDs, and fit the heavy-tail metric to that pooled spectrum.
2. The Lanczos paper does not estimate the asymptotic support from the ordinary finite tridiagonal resolvent alone. It applies Lanczos, Cholesky-factorizes the Jacobi matrix, estimates constant limiting Cholesky entries, extends that factor to a semi-infinite constant tail, and evaluates a continued fraction for the resulting Stieltjes transform.
3. In the identity-noise spiked covariance model with `q=N/M`, the population BBP threshold is `1 + sqrt(q)`, not `1 + q`. The null sample upper edge is `(1 + sqrt(q))^2`, not `(1 + q)^2`.
4. `codebase2` implements adaptive Gaussian broadening, not Chebyshev unfolding, and it does not implement Dyson-Mehta `Delta_3`. Chebyshev and monotone-spline strategies and `Delta_3` are retained as independent Spectral-Chinchilla methods, without false provenance.

The reconciliation policy is additive. Historically faithful alternatives remain selectable, while corrected methods remain the defaults. Every method reports its normalization, exponent convention, and boundary semantics.

## 2. Canonical conventions used after reconciliation

For a real matrix `W` of shape `n x m`, let `k=min(n,m)`, `d=max(n,m)`, and `q=k/d` in `(0,1]`. The reduced singular values are descending. Spectral-Chinchilla distinguishes three spectra:

- Raw Gram spectrum: `lambda_raw = s^2`.
- Canonical covariance spectrum: `lambda_cov = s^2/d`.
- Explicit legacy covariance spectrum: `lambda_N = s^2/N`, where the caller records `N`.

For entries with variance `sigma^2`, the canonical covariance MP support is

`lambda_pm = sigma^2 (1 +/- sqrt(q))^2`.

The codebase3 symbol `Q` is generally `d/k >= 1`, so its formulas use `1/Q=q`. The released FARMS code defines `Q_ratio` operationally as sampled columns divided by sampled rows and does not bound it by one. APIs therefore use `target_aspect_ratio` for that FARMS target and `aspect_ratio` for canonical `q=min(shape)/max(shape)`.

CSN and fixed-cutoff MLE values are density exponents. The mathematical Hill statistic is a survival exponent; the Hu-style `PL_Alpha_Hill` is that value plus one. Rank-regression slopes are converted to a density exponent using `alpha=1-slope`.

## 3. Discrepancy matrix by scientific module

| Area | Phase I | `codebase1` | `codebase2` | `codebase3` | Reconciliation |
|---|---|---|---|---|---|
| SVD container | Frozen `SVDResult`, descending reduced spectrum, explicit normalization | Saves `U`, `S`, and `Vh` independently; no common metadata | Recomputes SVD through model wrappers | Uses SciPy or truncated sklearn SVD, sometimes omitting one component | Keep the Phase I container and expose pooled FARMS spectra separately because they do not have one global singular basis |
| ESD scaling | Canonical `s^2/max(n,m)` plus explicit legacy view | Primarily raw singular values; MP curve is in singular-value units | Primarily raw singular values and `s^2` for power-law fits | Primarily raw `s^2`; some cells divide `WW^T` by a dimension and others trace-normalize | Record spectrum domain and denominator in every result; never compare unlabeled values |
| Centering | Weight matrices are not centered; activation covariance is centered by default | Two-pass activation mean subtraction before the feature Gram matrix | No activation covariance path | No activation covariance path | Preserve optional centered and uncentered activation moments; do not center weights unless explicitly requested |
| MP scale | Robust MP quantile matching and iterative upper-spike removal | Gavish-Donoho style median of squared singular values matched to the numerical MP median | Modified singular MP curve fits free amplitude and upper edge while fixing lower edge to an observed value | Edge-from-maximum after manually discarding spikes, plus a linear-kernel KDE residual fit for `sigma` | Expose analytic quantile, reference KDE, FARMS-pooled, and Lanczos-support methods |
| MP support | Analytic support and Johnstone edge scale | Analytic singular support with median scale | Empirical lower edge plus fitted upper edge is not constrained to one MP aspect ratio | Analytic support using reciprocal aspect notation | Keep each result's fitted assumptions explicit; never label the unconstrained codebase2 curve as an analytic MP fit |
| Tracy-Widom | Edge fluctuation scale only | Not implemented | Discussed through finite-size calibration but no reusable threshold API | Not implemented in core notebooks | Add an explicit 95 percent upper threshold as a detector, separate from MP fitting |
| Heavy tail | CSN continuous MLE with KS cutoff selection; Hill survival curve and plateau | No reusable tail solver in the supplied analysis repository | CSN fit through `powerlaw`, semiparametric KS bootstrap p-value, local inverse-slope Hill, and truncated-law comparison | `powerlaw.Fit`, distribution comparison, and calibration notebooks | Add standardized CSN, Hill, fixed-cutoff, and rank-regression dispatch; retain optional truncated-law comparison |
| Unfolding | Global polynomial or smoothing spline | No level-statistics unfolding | Adaptive local Gaussian broadening and summed Gaussian CDF | No level-statistics unfolding | Add Chebyshev, monotone smoothed spline, adaptive Gaussian, and raw-rank strategies; identify Gaussian as the codebase2 method |
| Brody fit | Bounded `[0,1]` spacing-likelihood MLE | Not implemented | Nonlinear least-squares fit to empirical spacing CDF, bootstrapped; its historical bound permits beta above one | Not implemented | Keep bounded MLE default and add bounded CDF-NLS alternative with optional bootstrap |
| Number variance | Deterministic sliding windows | Not implemented | Random-window iterative convergence adapted from `empyricalRMT` | Not implemented | Expose deterministic and Monte Carlo estimators with seeded randomness |
| `Delta_3` | Deterministic exact staircase integration already present | Absent | Absent | Absent | Retain as an independent Phase I feature and add multi-strategy unfolding support |
| Eigenvector localization | IPR and analytic-normal-reference KS test | Uses empirical maximum cosine overlap and hand-recorded three-sigma bands | IPR and Monte Carlo Porter-Thomas tests, including pooled windows | IPR-like participation and localization ratios plus experimental entropy functions | Add reference-compatible Monte Carlo PT, pooled PT, participation, and localization metrics while retaining corrected defaults |
| Activation overlap | Squared projection matrix and top/bulk/bottom averages | `Vh @ V_feature`, then maximum signed entry per singular vector; plots raw cosine against hand-calibrated bands | No activation overlap | No activation overlap | Keep squared sign-invariant overlap as the scientific default; expose a reference-max-cosine diagnostic and principal-angle/Frobenius metrics |
| Tranches | Descending top, middle bulk, and bottom fractions | Ten equal index deciles; plots reverse the ordering for presentation | No lesion pipeline | No lesion pipeline | Preserve both fraction tranches and exact balanced deciles; label ordering in output |
| Lesion protocol | Reversible count- or energy-matched top, MP-bulk, and bottom lesions | Removes one index decile at a time from every matrix of a chosen role; no equal-energy control | No spectral lesion implementation | No spectral lesion implementation | Keep reversible independent lesions and add a reference-decile mode without making it the default |
| Null ensembles | GOE, GUE, Ginibre, Wishart, Poisson, Pareto | Gaussian controls and a Julia lower-outlier teacher-student model | Network calibrations and random-vector PT samples | Gaussian, Pareto, and spiked covariance notebooks | Add population-spiked Wishart generation and retain explicit normalization |

## 4. Phase III Audit: FARMS & Lanczos Ground-Truth Calibration

### 4.1 FARMS Implementation Analysis (`farmscode` vs `rmt/farms_aspect_ratio.py`)

#### Source-bearing reference files

The audit traced the reusable FARMS implementation through `farmscode/README.md`, `LLM_Pruning/lib/sampling.py`, `LLM_Pruning/lib/esd_utils_farms.py`, `Image_Classification/sampling.py`, `Image_Classification/tempbalance_farms.py`, and `DPOT/utils/esd_utils_tbv3.py`. The remaining network definitions, optimizer code, training scripts, cached arrays, images, and task datasets provide experiment context but do not define the matrix-subsampling operator.

The complete repository inventory is classified below; path families cover every file under `farmscode/`.

| Path or complete path family | Audit classification |
|---|---|
| `README.md` | Paper/workflow overview and authoritative acronym/experiment description |
| `LLM_Pruning/main.py`, `lib/prune.py`, `lib/layerwrapper.py`, `lib/sparsegpt.py`, `lib/eval.py`, `lib/data.py`, `lib/utils.py` | LLM pruning, calibration-data, and evaluation integration; no alternative FARMS operator |
| `LLM_Pruning/lib/sampling.py`, `lib/esd_utils_farms.py` | Source-bearing linear/convolutional sampler and pooled-spectrum/tail implementation |
| `LLM_Pruning/lib/esd_utils.py` | Non-FARMS baseline ESD utilities used for ablation |
| `LLM_Pruning/scripts/*` | Shell/text experiment invocations and hyperparameter evidence, including 2000-row square windows; not library APIs |
| `LLM_Pruning/metric_cache/**/*.npy` | Generated alpha caches; outputs only |
| `Image_Classification/sampling.py`, `tempbalance_farms.py` | Source-bearing fixed/fixed-step samplers and FARMS temperature-balancing integration |
| `Image_Classification/main_tb.py`, `config.py`, `utils.py`, `sgdsnr.py` | Training/configuration integration; no additional spectral transform |
| `Image_Classification/lars_optim/*.py`, `networks/*.py` | Optimizers and model architectures; inspected for call direction, not ported into the Transformer suite |
| `Image_Classification/bash_scripts/**/*`, `requirements.txt`, `LICENSE` | Experiment commands, environment record, and license |
| `DPOT/utils/sampling.py`, `utils/esd_utils_tbv3.py` | Source-bearing operator-model sampler and ESD/tail implementation |
| Other `DPOT/utils/*.py` | Dataset, normalization, optimizer, criterion, and bookkeeping support |
| `DPOT/finetune*.py`, `evaluate*.py`, `otherpys/*.py`, `preprocess.py` | DPOT training/evaluation wrappers; no new FARMS equation |
| `DPOT/models/*.py` | Operator-network architectures; outside the causal-Transformer model contract |
| `DPOT/configs/*`, `txt_files/*`, `bashs/*`, `README.md`, `requirements.txt` | Experiment configuration and deployment context |
| `DPOT/data_generation/**/*` | External scientific-dataset preprocessing/visualization; no spectral method |
| `DPOT/torch_utils/**/*.py`, `torch_utils/ops/*.{cpp,h,cu}`, `resources/*` | Vendored accelerator operators and images; outside the pure numerical boundary |

#### Exact reference behavior

The released code confirms that FARMS is Fixed-Aspect-Ratio Matrix Subsampling and not a pointwise map of individual eigenvalues. For a matrix with `rows` and `cols`, the linear-layer samplers define

```text
num_col_samples = int(num_row_samples * Q_ratio)
```

so the repository convention is `Q_ratio = sampled columns / sampled rows`. Each submatrix contributes the raw squared singular values `s²`; all equal-shape window spectra are concatenated and sorted ascending before the power-law fit. There is no empirical variance estimator and no analytic aspect-ratio rescaling constant in the released FARMS source. Aspect-ratio bias is controlled by making every sampled matrix have the same shape.

The `fixed_sampling` path computes one floor-divided stride per dimension from a requested number of operations, caps the number of starts, and does not force the final endpoint when the floor stride misses it. The `matrix_size_dependent_sampling` path uses one fixed stride. The reference preserves source orientation. Its convolution path slices channel matrices separately and applies the historical convolution scaling before SVD.

#### Discrepancies found

Phase II documented `target_aspect_ratio` as rows divided by columns, interpreted `window_size` as the shorter dimension, used evenly spaced rounded grid starts, and transposed wide inputs by default. Those choices preserve a fixed shape but do not reproduce the released sampler for non-square targets.

The released repository also has copy-to-copy edge-case differences. One sampler clamps `num_row_samples` but can still request too many columns; another omits that clamp; one LLM copy leaves its convolution dispatch unavailable. Silent short slices can therefore produce unequal window shapes. These behaviors are implementation defects rather than scientific requirements and are not copied into the pure engine.

#### Mathematical and numerical fixes applied

- `target_aspect_ratio` now means sampled columns divided by sampled rows.
- `window_size` now means sampled rows, matching `num_row_samples`.
- `reference_fixed` exactly reproduces the floor-stride fixed-operation schedule.
- `reference_sliding` exactly reproduces the fixed-step schedule.
- `grid` and seeded `random` remain labeled alternative schedules.
- Source orientation is preserved by default; optional tall orientation remains explicit.
- Every requested window must fit and all pooled windows have identical dimensions, eliminating the reference's silent short-slice failure.
- `raw` reproduces pooled `s²`. `canonical` divides by the larger window dimension for covariance/MP comparisons, and `trace` is an explicit entropy-oriented normalization. These are named choices rather than hidden scaling.
- The standardized project container stores the pooled values descending, unlike the reference's ascending work array. This permutation leaves the ESD and all order-aware fits unchanged and preserves the repository-wide singular-spectrum convention.
- Results record requested `Q`, realized columns/rows, canonical `q=min/max`, source/oriented/window shapes, starts, coverage, and normalization.

For a fixed window shape, multiplying every eigenvalue by one constant does not change a CSN or Hill tail exponent. Canonical normalization is therefore reference-equivalent for tail shape while making MP support values comparable across window sizes.

### 4.2 Lanczos-Stieltjes Implementation Analysis (`lanczoscode` vs `rmt/lanczos_stieltjes.py`)

#### Source-bearing reference files

The numerical ground truth is concentrated in `lanczoscode/src/AuxiliaryFunctions.jl` and `lanczoscode/src/SpikeEstimation.jl`. `BEMA.jl`, `PA.jl`, and `PassYao.jl` implement comparison baselines. The simulation notebooks define sample-covariance constructions and parameter sweeps; generated tables and figures are outputs, not API contracts.

Every file under `lanczoscode/` is covered by this inventory:

| Path or complete path family | Audit classification |
|---|---|
| `README.md` | Repository entry point and reproduction map |
| `src/AuxiliaryFunctions.jl` | Source-bearing Lanczos, adaptive recurrence, Cholesky, and continued-fraction primitives |
| `src/SpikeEstimation.jl` | Source-bearing support, density, threshold, modal spike-count, and location algorithms |
| `src/BEMA.jl`, `src/PA.jl`, `src/PassYao.jl` | Bulk-eigenvalue matching, parallel analysis, and Pass–Yao comparison estimators; retained as audit baselines rather than mislabeled Lanczos components |
| `Simulations/IntroExample/IntroExample.ipynb` | Introductory construction, probe settings, and plotting workflow |
| `Simulations/Simulation 1/Simulation1.ipynb` | Support/density/spike accuracy and runtime sweeps |
| `Simulations/Simulation 2/Simulation2.ipynb` | Comparative detection and timing study |
| `Simulations/Simulation 3/Simulation3.ipynb` | Additional covariance/spike comparison study |
| `Simulations/StdSpikedCov/StdSpikedCov.ipynb` | Standard spiked-covariance density examples |
| `Simulations/**/Tables/*.csv` | Generated numerical outputs, including support errors, spike counts/locations, percentages, averages, and timings |
| `Simulations/**/Figures/*.pdf` | Generated density, probability, support, and timing figures |

#### Exact reference behavior

`LanczosTri` and `AsympLanczos` draw Gaussian probes and normalize them, which is uniform on the real unit sphere. The production path performs two full reorthogonalization passes. The adaptive routine checks local Jacobi diagonal and off-diagonal means and sample deviations in consecutive windows. The released reference uses an absolute trace-scaled stopping tolerance. The maintained implementation preserves the sequence length based on `floor(log(N)/2)`, check interval of two, and maximum iteration count containing `6 log(N)+24`, `N/4`, and `sqrt(N)`, but uses operator-scale-relative recurrence, convergence, and Cholesky-pivot defaults so rescaling a factor cannot change availability.

`CholeskyList` first retains the true finite Jacobi Cholesky factor for spike estimation. It then averages the stable suffix in Jacobi coordinates, appends two constant diagonal entries and one constant off-diagonal entry, Cholesky-factorizes that modified Jacobi matrix, and averages terminal Cholesky coefficients across probes. If those limiting bidiagonal coefficients are `a` and `b`, the one-cut support is

```text
gamma_minus = (a - b)^2
gamma_plus  = (a + b)^2.
```

`BiRef` supplies the constant-tail Stieltjes solution and `BiRel` propagates it through each finite prefix. `EstimDensity` averages the resulting imaginary parts across probes. The reference `EstimSpike` uses the absolute rule `gamma_plus + c N^-delta`; the production adapter deliberately uses the scale-relative rule `gamma_plus + c gamma_plus N^-delta` so equivalent spectra in transformer-scale units receive equivalent decisions. It takes the modal count across probes and reports locations from a qualifying recurrence. The released detector does not compute a separate fixed-point support equation and does not threshold residues, although Ritz residues are the squared first components of the tridiagonal eigenvectors and are exactly the VEST pole weights.

#### Discrepancies found

Phase II used a fixed recurrence length, averaged a terminal Cholesky window directly, evaluated density from one representative prefix, and extracted poles from an extended constant-tail finite section. These are defensible extensions, but they differ from the released implementation's adaptive Jacobi stabilization, modified-tail ordering, multi-probe density average, and finite-Ritz spike rule.

The released `EstimSpike` derives its threshold edge from the first true finite Cholesky tail rather than the consensus modified support, and its threshold-return branch references a differently capitalized variable. The production engine uses the consensus support and a validated threshold, which removes probe-order dependence and fixes the unreachable return defect.

#### Mathematical and numerical fixes applied

- The default maximum recurrence now includes the reference `N/4` term.
- Adaptive stopping exposes trace-scaled tolerance, sequence length, and check interval and retains fixed-step operation as an option.
- Full reorthogonalization remains a double projection pass; partial and disabled modes remain explicit alternatives.
- After adaptive convergence, the initial stable suffix starts at `used - jmp - seq_len - 1`; the backward tolerance search, Jacobi-coordinate averaging, two-diagonal/one-off-diagonal replacement, and subsequent Cholesky factorization preserve the reference's finite-order operation. The iteration-ceiling branch uses the released second-last-coefficient reference search.
- Terminal Cholesky coefficients are averaged across probes to define one consensus support.
- Density evaluation averages every probe's finite-prefix continued fraction.
- `reference_ritz` takes the modal finite-VEST Ritz count across probes and reports locations from the longest recurrence; `constant_tail` preserves the Phase II extended-section alternative.
- If several counts have equal modal frequency, the released dictionary traversal does not define a scientifically meaningful tie policy. The port deterministically selects the tied count nearest the probe-count mean and records all per-probe counts.
- Ritz residues are always reported and may be filtered by an optional nonnegative threshold. A zero threshold reproduces reference counting.
- The production spike threshold uses the consensus scale-relative `lambda_plus + c lambda_plus N^-delta`, not the first probe's noisy terminal coefficients; output records the effective margin and convention.
- Rectangular factors use matrix-free covariance products and exact trace scaling without materializing a dense covariance.
- Diagnostics now report per-probe recurrence lengths, convergence flags, edges, pole counts, residues, consensus coefficients, and selected pole method.

The reference notebooks frequently use 50 or 100 probes for simulation studies, while the production default is three to control Transformer-layer cost. `--lanczos-probes` exposes the reference counts when calibration accuracy is preferred over throughput.

### 4.3 HPC harness reconciliation (`run_rmt.slurm` vs `run_hpc.slurm`)

The supplied example establishes one GPU, 64 GiB, the `gpulong` partition, strict Hugging Face offline switches, tokenizer parallelism disabled, expandable CUDA allocator segments, and the successful interpreter path `/home/shivansh/.conda/envs/rmt_ml_env/bin/python`. It does not establish exact package/Python versions, a successful CUDA module load, or the availability of a versioned CUDA/Python module. Its local Pythia snapshot, raw WikiText path, legacy `python -m rmt` flags, and hard-coded ESD project path belong to the sibling application and are not inputs to this training runner.

`run_hpc.slurm` therefore calls the recorded interpreter directly (with `RMT_PYTHON` override), declares `gpulong`, loads no module by default, and never purges inherited modules or activates a guessed `.venv`. An operator may set `RMT_CUDA_MODULE` only after validating it against the live Torch stack. The launcher requires this directory through `PROJECT_ROOT`/`SLURM_SUBMIT_DIR`, sets `PYTHONPATH` to this source alone, verifies the exact local `rmt.__file__`, checks CUDA/BF16 and the asset manifest, uses writable runtime compiler caches, and writes each job to a new non-overwriting output root. It invokes only `run_experiments.py` flags defined by `pipelines/cli_config.py` and exposes `TRACK=paper1|paper2|paper3|golden|all`. The one-process runner still requests one GPU and does not imply unimplemented distributed scaling. Package changes remain inventory-driven because the unpinned ESD requirements cannot supply an honest cluster lock and its NumPy API needs conflict with this project's standalone NumPy 1.26.4 pin.

The Phase III prose calls the Golden synthesis “Chebyshev Unfolding,” while its normative Track 4 command and exhaustive CLI table select `spline_monotone`. The executable track follows the explicit command and table. Chebyshev remains the Track 2 reproduction strategy and is one flag change away for a Golden sensitivity run.

## 5. Detailed reconciliation findings

### 5.1 `codebase1`: dual-end and lesion reference

The activation hook replacement computes a running mean in a first pass, then a running average of batch Gram matrices in a second pass. It subtracts the first-pass mean, which establishes that the relevant feature matrix is centered. It does not divide each Gram matrix by its number of flattened tokens before averaging, so batches receive equal weight and scale depends on batch sample count. Phase I's accumulator instead adds sample sums and Gram matrices and divides once by the total sample count.

The paper-figure notebook defines a singular-value MP density. It estimates entry scale by matching the empirical median of squared singular values to a numerically integrated MP median, after optional removal of a leading fraction. This is aligned with Phase I's median scale estimator after converting domains.

The reference overlap routine sorts activation eigenvectors descending, computes a reduced SVD, multiplies `Vh` by the activation eigenvector matrix, and takes the maximum along each row. It does not take an absolute value before the maximum and does not square the result. Consequently, sign choices can change the plotted statistic. The repository's squared overlap is invariant to singular/eigenvector signs and matches the paper's projection interpretation more closely.

The Pythia path splits fused query, key, and value row blocks before SVD. The Llama and Pythia lesion notebooks zero one tenth of each descending singular spectrum, reconstruct the matrix, and evaluate perplexity. BERT scripts apply equivalent decile surgery either before fine-tuning or after fine-tuning. These experiments lesion all matrices matching a role simultaneously and do not match removed Frobenius energy.

The Julia theory model studies a rank-one mean plus anisotropic row covariance and predicts a lower singular outlier and its left/right overlaps. It is a useful lower-edge calibration model but not an activation-overlap API. Its package-install statements were treated as archival instructions and were not executed.

### 5.2 `codebase2`: level statistics and goodness-of-fit reference

`GaussBroadening` assigns each level a local width from neighbors `winSize` ranks away, broadens the spectrum by a sum of Gaussians, and unfolds by the associated summed Gaussian CDF. Boundary handling either repeats edge levels or drops boundary levels. This is the actual codebase2 unfolding method.

Its modified singular-value MP curve has free amplitude, upper edge, and lower edge. The fitting routine fixes the lower edge to an observed order statistic, restricts the fit near the density peak, and fits amplitude and upper edge by nonlinear least squares. This is an empirical curve fit, not a one-parameter MP variance estimate.

Brody fitting operates on the empirical spacing CDF, uses nonlinear least squares, and bootstraps spacing samples to estimate error. The historical implementation allows beta above one. Spectral-Chinchilla bounds both MLE and CDF-NLS variants to `[0,1]`, because the CLI contract defines the Poisson-to-GOE interpolation.

Number variance samples random interval centers until a rolling set of estimates converges or a maximum iteration count is reached. This differs from Phase I's deterministic evenly spaced windows. Both are retained.

Porter-Thomas analysis constructs a Monte Carlo reference CDF for normalized real Gaussian vectors, then calibrates the KS-distance distribution with additional Monte Carlo samples. A pooled variant concatenates nearby vectors. The simpler Phase I normal-CDF test is asymptotic; the new reference mode reproduces finite-dimension calibration without adding Numba.

The power-law path fits `s^2`, chooses `xmin`, and performs the CSN semiparametric bootstrap: empirical resampling below `xmin`, synthetic Pareto sampling above it, and refitting every replicate. It reports `p>=0.1` as fit compatibility. This goodness-of-fit probability is distinct from the observed KS distance and is exposed separately.

No source-bearing file in codebase2 defines Chebyshev unfolding or `Delta_3` rigidity.

### 5.3 `codebase3`: WeightWatcher research archive

Aspect ratio is stored as `Q=max(shape)/min(shape)`, while the MP law internally uses `1/Q`. Raw eigenvalues are usually `s^2` with no covariance denominator. The archive also contains cells that normalize a Gram matrix by its trace or by a dimension, so notebook plots are not all on one scale.

The simplest MP scale estimate discards a manually chosen number of largest eigenvalues, takes the largest retained eigenvalue as the MP upper edge, and solves backward for `sigma`. The automated archive fitter builds a linear-kernel KDE over eigenvalues and minimizes the residual between that KDE and the MP density while floating only `sigma`. The maintained `kde_bulk_fit` keeps its trimming/bandwidth compatibility surface but now minimizes an integrated complete-sample ECDF objective—including the rank-deficient sample's zero-mass rank offset—with a bounded SciPy optimizer, avoiding the invalid unsmoothed-density comparison at the square MP hard edge.

The archive defines stable rank as `sum(lambda)/max(lambda)`, MP soft rank as `lambda_plus/lambda_max`, a hard-rank tolerance, convolutional tensor reshapers, IPR-related participation/localization ratios, and spike location formulas. Its matrix entropy first divides a generally rectangular matrix by `trace(W)`, which is undefined for many shapes and unnecessary because the following singular-energy probabilities are scale invariant. Phase I's squared-singular-value entropy is retained, and `normalized_matrix_entropy` reproduces the archive's final division by `log(numerical_rank)` without the invalid trace step.

Power-law analysis delegates `xmin` selection and distribution comparison to the external `powerlaw` package. There is no fixed-window MLE implementation in the supplied archive. The requested `fixed_cutoff_mle` is therefore implemented as a new compatibility method, not described as codebase3 source behavior.

No source-bearing file in codebase3 implements spectral unfolding, Brody fitting, number variance, `Delta_3`, activation covariance, or activation-subspace overlap.

## 6. Verification of the 2025 methods

### 6.1 FARMS

Hu et al. define FARMS as follows for a matrix in the sampling orientation selected by the experiment:

1. Choose a submatrix shape with a fixed sampled-columns/sampled-rows ratio across layers.
2. Select multiple, potentially overlapping windows by a sliding or sampled scheme.
3. Compute eigenvalues of each `W_ij^T W_ij`.
4. Concatenate the eigenvalue series, which averages the empirical densities.
5. Compute the heavy-tail metric on that pooled ESD.

The paper treats window dimensions, target ratio, and number of windows as hyperparameters. Its reported LLM settings use square windows and grids such as `10 x 10` or `15 x 15`; it finds a `2000 x 2000` window effective for one Llama pruning setting. It explicitly acknowledges that one submatrix loses correlations outside the window and motivates multiple windows as partial coverage. It does not prove that a transformed empirical maximum is invariant to two percent at finite size.

Implementation decision: `farms_aspect_ratio.py` defaults to the released floor-stride fixed-operation schedule and preserves source orientation. It also supports the released fixed-step schedule, evenly spaced grids, and seeded random windows. Raw reference pooling and canonical/trace normalizations are explicit. `shape_normalized` is a separate analytic affine-support mapping and is never labeled FARMS.

### 6.2 Lanczos-Cholesky Stieltjes detector

Abi Younes, Ding, and Trogdon consider a positive sample covariance matrix with a one-interval asymptotic density having square-root edges and finitely many separated right outliers. Lanczos at a random unit probe returns the finite Jacobi matrix of the vector empirical spectral distribution. The Jacobi matrix is Cholesky-factorized. Under the paper's assumptions, the diagonal and subdiagonal Cholesky entries converge exponentially to constants `alpha` and `beta`, giving support estimates

`gamma_minus = (alpha-beta)^2` and `gamma_plus = (alpha+beta)^2`.

The constant tail has an analytic Stieltjes transform. Earlier finite entries are folded back by the paper's continued-fraction recurrence. Multiple probes are combined by averaging sufficiently deep Cholesky entries before constructing each transform. Spikes are poles to the right of `gamma_plus + C N^(-delta)`, with `0<delta<1/2`; the paper's numerical section uses `C=1` and `delta=0.25` and notes that this tuning remains consequential.

Implementation decision: `lanczos_stieltjes.py` implements adaptive double-reorthogonalized Lanczos, reference-ordered Jacobi-tail modification and Cholesky factorization, consensus support, multi-probe continued fractions, and finite-VEST Ritz poles. The Phase II extended constant-tail section remains an explicit alternative. It accepts a dense symmetric matrix or a matrix-free covariance operator constructed from a rectangular factor. Results state convergence and assumption diagnostics. The method is asymptotically consistent under the paper's conditions; it is not described as exact or fluctuation-free for arbitrary neural weight spectra.

## 7. CLI reconciliation and dispatch decisions

| CLI option | Runtime method | Provenance and caveat |
|---|---|---|
| `--mp-fit-method analytic_mp` | Robust MP quantile scale fit | Phase I corrected baseline |
| `--mp-fit-method thamm_modified_singular` | Adaptive-Gaussian singular ESD with empirical lower edge and fitted amplitude/upper edge | Faithful codebase2 curve fit; fitted edges are converted to canonical covariance units for bulk selection |
| `--mp-fit-method kde_bulk_fit` | Upper-trimmed, complete-sample ECDF fit of MP scale | Hardened descendant of codebase3's automated KDE notebook; finite at the square hard edge |
| `--mp-fit-method lanczos_stieltjes` | Lanczos-Cholesky support estimate | 2025 paper; requires matrix/operator input, not only eigenvalues |
| `--mp-fit-method farms_unbiased` | Analytic MP fit on FARMS pooled fixed-ratio spectra | Hu et al. subsampling plus Phase I MP fitting |
| `--unfolding-strategy polynomial_chebyshev` | Chebyshev smooth staircase fit | Independent compatibility method; not codebase2 |
| `--unfolding-strategy spline_monotone` | Smoothed spline; folded fits are replaced by the integral of a positive smoothed density, never by flattened ranks | Independent corrected method |
| `--unfolding-strategy gaussian_kernel` | Adaptive local Gaussian-CDF unfolding | codebase2 |
| `--unfolding-strategy raw_rank_order` | Empirical rank map | Diagnostic only; it suppresses spacing fluctuations |
| `--tail-solver clauset_mle` | CSN density MLE with KS cutoff selection | Phase I and codebase2 methodology |
| `--tail-solver hill_estimator` | Hill estimate with explicit survival/density labels | codebase2 and Hu et al. use related conventions |
| `--tail-solver fixed_cutoff_mle` | User/fraction cutoff continuous MLE | New compatibility method; not found in codebase3 |
| `--tail-solver rank_ordered_mle` | Tail rank-frequency regression | New compatibility method; not found in codebase1 |
| `--overlap-metric staats_dual_end` | Squared projection tranches against leading activation eigenvectors | Paper-aligned default with codebase1 comparison output |
| `--overlap-metric subspace_principal_angles` | Mean squared cosine of principal angles | Basis-invariant subspace metric |
| `--overlap-metric frobenius_projection` | Dimension-normalized projector trace overlap | Basis-invariant subspace metric |
| `--aspect-ratio-mode raw` | Canonical full-matrix covariance spectrum | Phase I |
| `--aspect-ratio-mode farms_normalized` | FARMS pooled fixed-ratio spectrum | Hu et al. |
| `--aspect-ratio-mode farms_unbiased` | Alias of `farms_normalized` | Compatibility with the earlier task wording |
| `--aspect-ratio-mode shape_normalized` | Analytic affine mapping of the MP support | Separate baseline, not FARMS |
| `--spike-detector tracy_widom_95` | Analytic upper edge plus real-Wishart 95 percent correction | Finite-size null detector |
| `--spike-detector bbp_transition` | Right outliers beyond the MP/BBP separation edge | Uses the correct square-root aspect law |
| `--spike-detector lanczos_poles` | Adaptive finite-VEST Ritz poles with residues; constant-tail sections optional | 2025 paper and calibrated reference code |

The earlier `--boundary-detector` spelling is accepted as a compatibility alias. When supplied, it deliberately takes precedence over the paired MP-fit/spike values, and the resolved configuration is serialized before execution so the override is visible.

## 8. File-by-file source audit

### 8.1 `codebase1`

- `BertFinetuning.py`: fine-tunes GLUE models, stores SVD factors, then evaluates post-training decile removal.
- `BertRemoveFinetune.py`: removes each singular decile before fine-tuning and records validation accuracy.
- `DataGenerationBert.ipynb`: two-pass centered BERT activation feature-matrix extraction.
- `DataGenerationLlama.ipynb`: selected-layer Llama activation extraction and projection-matrix SVD export.
- `DataGenerationPythia.ipynb`: Pythia activation extraction and fused-QKV slicing before SVD.
- `PerplexityLlamaSVDLayerD.ipynb`: role-wise Llama singular-decile perplexity lesions.
- `PerplexityPythiaSVDLayerD.ipynb`: Pythia Q/K/V and MLP singular-decile perplexity lesions.
- `PaperFigures/generate_figures.ipynb`: singular MP density, MP-median scale, overlap, decile-impact, and publication plots.
- `PaperFigures/include/plot_utils.py`: plotting style only.
- `TheoryModel/README`: Julia environment notes; no independent numerical method.
- `TheoryModel/theory_model.jl`: anisotropic lower-outlier teacher-student simulation and overlap measurements.
- `TheoryModel/theory_model.pluto.jl`: Pluto serialization of the same theory experiment.
- `TheoryModel/include/utils/MTUtils.jl`: utility aggregator.
- `TheoryModel/include/utils/compute/run_or_load.jl`: cached-computation helper, not a spectral estimator.
- `TheoryModel/include/utils/plot/makie_theme.jl`: plotting theme.
- `TheoryModel/include/utils/plot/plot_utils.jl`: plotting helpers.
- `finetuning.bat`: Windows batch orchestration for repeated BERT lesion/fine-tuning jobs; it adds no estimator.
- `requirements.txt`: archived environment manifest; dependencies were audited but not installed or copied wholesale.
- `PaperFigures/include/plt_style.pstyle`: plotting style only.
- `PaperFigures/include/AverageSpectraPlot.pickle`: serialized plot artifact, not source.
- `util/featureM_utility.py`: linear-module replacement and two-pass feature matrix accumulation.
- `util/finetuneUtils.py`: GLUE tokenization, fine-tuning, evaluation, and SVD persistence.
- `util/magicattr.py`: nested module attribute replacement helper.
- `util/perplexityUtil.py`: stride-window causal perplexity evaluation.
- `Data/**/*.npy`, `Figures/*`, and `TheoryModel` result/PDF artifacts: inventoried empirical outputs; no executable or mathematical contract was inferred from their serialized values.

### 8.2 `codebase2`

- `LICENSE`: licensing metadata.
- `README.md`, `data/README.md`, `figures/README.md`, `src/README.md`: repository and artifact descriptions.
- `src/rmt_utils.py`: all reusable broadening, MP, spacing, number-variance, Brody, PT, IPR, Hill, and CSN bootstrap methods described above.
- `src/nn_utils.py`: Keras model wrappers, convolutional reshapers, spectrum extraction, and power-law batch orchestration.
- `src/plot_utils.py`: plotting wrappers for spectra, spacing, number variance, PT p-values, IPR, and Hill diagnostics.
- `src/generate_figures.ipynb`: experiment assembly; introduces no separate Chebyshev or `Delta_3` method.
- `src/compute_powerlaw_pValues_example.py`: batch driver for semiparametric CSN KS calibration.
- `src/jax_and_laziness_utils.py`: JAX neural-tangent linearization experiments, outside the spectral CLI scope.
- `src/train_miniAlexNet.py`, `src/train_MLP1024.py`, `src/train_MLP512.py`, `src/train_MLP512_laziness.py`: model-training reproductions; no distinct RMT solver.
- `include/plt_style.pstyle`: plotting style only.
- `data/powerlaw_table.txt`: archived result table; it is data, not a solver definition.
- Saved Keras models, variables, KS arrays, and `figures/*`: inventoried binary/generated artifacts and excluded from source-method attribution.

### 8.3 `codebase3`

- `RMT_Util.ipynb`: shared prototype functions for MP, ranks, entropy, localization, spiked models, power laws, KDE fitting, and tensor reshaping.
- `Automated-MPFit.ipynb`, `AlexNet-pytorch-MPFit.ipynb`: KDE and MP-scale fitting experiments on AlexNet.
- `Marchenko_Pastur_Plots.ipynb`, `GaussianRMT.ipynb`: analytic and Gaussian MP calibration plots.
- `Spiked-Covariance-Model.ipynb`: finite-rank spike thresholds, locations, and sampling experiments.
- `HeavyTailedUniversalityClasses.ipynb`, `CalibrateLargeNPL.ipynb`, `CalibratePowerLawFits.ipynb`, `Keras-PowerLaws.ipynb`, `PowerLawAccuracyTest.ipynb`: heavy-tail generation, fitting calibration, and model-quality comparisons.
- `WeightWatchers.ipynb`, `WeightWatcherTest.ipynb`: exploratory aggregate model diagnostics and checks.
- `AlexNet-Baseline-Ensemble.ipynb`, `AlexNet-Baseline-wDropout.ipynb`, `AlexNet-Baseline.ipynb`, `AlexNet-Generalization-Gap.ipynb`, `AlexNet-pytorch.ipynb`, `AlexNet-RankCollapse.ipynb`, `AlexNet-Regularized.ipynb`: AlexNet training/phase experiments using the shared utility methods.
- `All-pytorch-models-wCNNs-Slices.ipynb`, `All-pytorch-models-wCNNs.ipynb`, `All-pytorch-models.ipynb`, `All-pytorch-ranks.ipynb`, `EvenMore-pytorch-models-Accuracies-Copy1.ipynb`, `EvenMore-pytorch-models-Accuracies.ipynb`, `EvenMore-pytorch-models-OSMR.ipynb`, `EvenMore-pytorch-models.ipynb`, `Inception-pytorch.ipynb`: pretrained-model surveys, tensor slicing, rank tables, and accuracy comparisons; no additional solver family.
- `LeNet5.ipynb`, `MLP3-Basline.ipynb`: small-network training experiments.
- `Keras-Save-Load-Test.ipynb`: serialization check.
- `WordEmbeddingMatrices.ipynb`: embedding-spectrum exploration.
- `accuracies`: archived scalar data, not source.
- `allennlp/analyze.py`: AllenNLP linear-layer power-law report using raw `s^2`.
- `allennlp/rank.py`: AllenNLP shape, aspect-ratio, and minimum-singular-value report.
- `allennlp/AllenNLP_PowerLaws.ipynb`, `allennlp/AllenNLP_Ranks.ipynb`: report aggregation notebooks.
- `allennlp/__init__.py`, `allennlp/README.md`: package marker and archival integration instructions.
- `allennlp/run_analyzers.sh`: shell orchestration for the two AllenNLP reports; it adds no numerical method and was not executed.
- `allennlp/*.dat`, `img/*`, and `accuracies`: archived tabular or generated outputs, not source contracts.
- `README.md`: paper links and repository identity.

## 9. Acceptance interpretation

Synthetic tests can validate formulas, dispatcher coverage, deterministic sampling, support estimation, and pure-module isolation. They cannot guarantee that trained transformer layers have a particular power-law exponent, that FARMS always improves a downstream metric, or that bottom lesions always dominate equal-energy bulk lesions. Those remain empirical report hypotheses. The source tree therefore separates numerical calibration assertions from claims that require operator-run training and benchmarking.
