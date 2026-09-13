# Spectral-Chinchilla executive summary

Spectral-Chinchilla examines relations between compute allocation and spectra from Transformer weights and activations. It contains a configurable decoder-only language model and a Chinchilla/IsoFLOP allocator. It also contains activation-covariance capture and a NumPy/SciPy random-matrix engine. Singular-value lesion experiments are reversible.

The main comparison uses undertrained, compute-optimal, and overtrained allocations at fixed compute. The system examines each applicable attention or MLP matrix. It measures Marchenko-Pastur bulk agreement, both spectral edges, tail estimates, stable rank, entropy, and unfolded level statistics. It also measures Brody interpolation, number variance, and alignment with activation-covariance eigenvectors. Lesion experiments separately remove top, bulk, or bottom tranches. They measure changes in validation loss and perplexity. A separate descending-decile mode reproduces the supplied dual-end archive.

Phase II makes each scientific method explicit. MP scale can use analytic quantiles, an integrated ECDF fit, FARMS submatrices, or a Lanczos-Cholesky support estimate. The integrated ECDF fit is safe at a hard edge and comes from the historical KDE fit. Unfolding can use Chebyshev, a monotone spline, an adaptive Gaussian CDF, or raw ranks. Tail estimates can use CSN, Hill, fixed-cutoff MLE, or rank regression. Activation alignment can use squared projections, principal angles, or a normalized projector trace.

FARMS means Fixed-Aspect-Ratio Matrix Subsampling. Multiple windows with equal shapes produce one pooled ESD. The released ratio is sampled columns divided by sampled rows. FARMS is not a pointwise normalization map.

The Lanczos method uses normalized Gaussian probes and double full reorthogonalization. It includes adaptive recurrence diagnostics and a reference modified-tail Cholesky construction. Its support is `(alpha-beta)^2` to `(alpha+beta)^2`. It averages VEST results across probes and counts finite-Jacobi Ritz spikes. Constant-tail finite-section poles are a separate alternative. Consistency requires a positive one-cut bulk with square-root edges and separate right spikes.

Phase III provides an offline HPC package that stops on invalid prerequisites. Target-token budgets control recycled partial batches and skipped updates. Records contain attempted targets, forwarded positions, and labeled `6ND` proxies. The runner rejects collapsed N/D interventions unless the user permits calibration duplicates. Each output root must be new.

A connected host stages the exact dataset and tokenizer assets with checksums. Compute jobs use offline environment variables and local token arrays. Training supports BF16 or FP16 autocast, TF32, optional compilation, pinned transfers, device covariance, and reduced device SVD. Only final covariance matrices and reduced SVD factors enter the NumPy engine. `--analysis-dtype` independently selects weight and lesion SVD precision. Its default is float64. Fit, detector, and plot output identifies its spectrum domain and availability.

The SLURM launcher contains four tracks. Three tracks reproduce historical analysis protocols on Spectral-Chinchilla models. The Golden track combines FARMS, Lanczos, monotone-spline unfolding, CSN tails, dual-end overlap, and spectral lesions.

Synthetic Wishart, GOE, Poisson, and Pareto data provide calibrations with known answers. Trained-model results are empirical hypotheses, not unit-test facts. These hypotheses include heavy-tailed deep layers, spectral collapse, and information in the smallest singular values.

The current repairs compute reduced-precision attention scores in FP32 and qualify the raw spacing spectrum. MP density, energy matching, and Lanczos diagnostics keep their results under finite unit changes. Hook setup cleans up after registration failures.

Asset staging builds and examines a separate release. It publishes the release under one staging lock and restores the old `data/` tree after a publication failure.

The `rmt/` package imports only the standard library, NumPy, and SciPy. The `models/` and `pipelines/` directories contain the training framework. Each random operation accepts a seed. Output records requested and realized allocations and the full configuration of spectral methods.
