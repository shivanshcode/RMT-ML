# Public API contract

The signatures below are fixed for this repository. All symbols under `rmt` accept and return NumPy arrays and must not import torch.

## `rmt.svd_result`

- `SVDResult(U, s, Vh, n, m, normalization=None, lambda_minus=None, lambda_plus=None, factorization_dtype="float64")` is a frozen dataclass; storage promotion does not erase FP32 factorization provenance.
- Properties: `gamma`, `aspect_ratio`, `Q`, `V`, `singular_values`, raw `eigenvalues`, normalized `covariance_eigenvalues`, and `spectral_bounds`.
- `SVDResult.reconstruct() -> np.ndarray`
- `compute_svd(matrix, *, full_matrices=False, normalization=None) -> SVDResult`
- `rmt.linalg.cached_svd(weight, full_matrices=False, *, backend="auto") -> SVDResult`; strict core isolation permits `auto` and `numpy` only.

## `rmt.ensembles`

- `goe(n, rng=None) -> np.ndarray`
- `gue(n, rng=None) -> np.ndarray`
- `ginue(n, rng=None) -> np.ndarray`
- `wishart_factor(n, m, sigma=1.0, rng=None) -> np.ndarray`
- `wishart(n, m, sigma=1.0, rng=None, *, covariance=False) -> np.ndarray`
- `spiked_covariance(n, m, spikes, *, noise_variance=1.0, rng=None, return_factor=False) -> np.ndarray`
- `spiked_wishart` is an identity alias of `spiked_covariance`.
- `poisson_levels(n, rng=None) -> np.ndarray`
- `poisson_points_2d(n, rng=None) -> np.ndarray`
- `pareto(n, alpha, xmin=1.0, rng=None) -> np.ndarray`
- `wigner_semicircle_samples(n, rng=None) -> np.ndarray`

## `rmt.mp`

- `MPFitResult` is a frozen dataclass with MP scale, edges, departures, `method`, and diagnostics.
- `SpikeDetectionResult` is a frozen dataclass with threshold, bulk edge, descending spikes, source indices, diagnostics, and `n_spikes`.
- `ModifiedMPFitResult` is a frozen dataclass for the unconstrained singular-domain archive curve.
- `marchenko_pastur_bounds(aspect_ratio, variance=1.0) -> tuple[float, float]`
- `marchenko_pastur_density(x, aspect_ratio, variance=1.0) -> np.ndarray`
- `marchenko_pastur_cdf(x, aspect_ratio, variance=1.0) -> np.ndarray | float`
- `mp_eigenvalues(weight, normalization=None) -> np.ndarray`
- `fit_marchenko_pastur(eigenvalues, aspect_ratio, *, variance=None, trim_upper=0.1) -> MPFitResult`
- `triangular_kde(values, grid, bandwidth) -> np.ndarray`
- `fit_marchenko_pastur_kde(eigenvalues, aspect_ratio, *, bandwidth=None, trim_upper=0.1, grid_size=512) -> MPFitResult`
- `fit_marchenko_pastur_farms(weight, *, target_aspect_ratio=1.0, window_size=None, row_windows=5, column_windows=5, sampling="reference_fixed", step_size=10, orient_tall=False, seed=0, trim_upper=0.1) -> MPFitResult`
- `fit_marchenko_pastur_lanczos(weight, *, steps=None, n_probes=3, tail_window=None, threshold_c=1.0, threshold_delta=0.25, residue_threshold=0.0, ridge=0.0, adaptive=True, convergence_tolerance=None, sequence_length=None, check_interval=2, pole_method="reference_ritz", rng=0) -> MPFitResult`
- `modified_mp_singular_density(x, amplitude, nu_min, nu_max) -> np.ndarray`
- `adaptive_gaussian_spectral_density(singular_values, grid, *, window=15) -> np.ndarray`
- `fit_modified_mp_singular(singular_values, *, lower_index=0, x_min=0.0, fit_peak_fraction=0.7, kernel_window=15, grid_size=512) -> ModifiedMPFitResult`
- `fit_marchenko_pastur_thamm(weight, *, lower_index=0, x_min=0.0, fit_peak_fraction=0.7, kernel_window=15, grid_size=512) -> MPFitResult`; preserves the empirical singular-domain fit, converts its edges by `s**2/max(shape)`, and marks `variance` as an upper-edge compatibility scale.
- `tracy_widom_edge_scale(n, m, variance=1.0, *, edge="upper") -> float`
- `tracy_widom_quantile(confidence=0.95) -> float`
- `tracy_widom_upper_threshold(n, m, variance=1.0, *, confidence=0.95) -> float`
- `detect_spikes_tracy_widom(eigenvalues, n, m, variance=1.0, *, confidence=0.95) -> SpikeDetectionResult`
- `bbp_population_threshold(aspect_ratio, variance=1.0) -> float`
- `bbp_sample_location(population_eigenvalue, aspect_ratio, variance=1.0) -> float`
- `detect_spikes_bbp(eigenvalues, aspect_ratio, variance=1.0, *, margin=0.0) -> SpikeDetectionResult`
- `mp_soft_rank(eigenvalues, lambda_plus) -> float`
- `mp_pdf(x, n, m, sigma) -> np.ndarray`
- `mp_bounds(n, m, sigma) -> tuple[float, float]`
- `mp_cdf(x, n, m, sigma) -> np.ndarray | float`
- `mp_median(n, m, sigma=1.0) -> float`
- `eigenvalues_of_cov(weight=None, *, s=None, n=None, m=None, N=None) -> np.ndarray`
- `mp_pdf_eig(x, n, m, sigma, N) -> np.ndarray`
- `mp_bounds_eig(n, m, sigma, N) -> tuple[float, float]`
- `mp_cdf_eig(x, n, m, sigma, N) -> np.ndarray | float`
- `estimate_sigma_gd_median(weight=None, *, s=None, n=None, m=None, discard_largest=0.0) -> float`
- `estimate_sigma_med_refined(weight=None, *, s=None, n=None, m=None, max_iter=3) -> tuple[float, float, int]`
- `usvt_hard_threshold(n, m, sigma, *, square_optimal=True) -> float`
- `small_sv_deviation(s, n, m, sigma) -> dict`

`estimate_sigma_med` is an identity alias of `estimate_sigma_gd_median`.

## `rmt.tail`

- `fit_powerlaw_csn(values, *, min_tail=50, tail_frac=0.02, max_xmin_candidates=200, xmax=None) -> dict`; finite `xmax` uses the normalized bounded-Pareto likelihood and conditional CDF.
- `csn_goodness_of_fit(values, *, n_bootstrap=250, min_tail=50, tail_frac=0.02, max_xmin_candidates=200, rng=0) -> dict`
- `fixed_cutoff_mle(values, *, xmin=None, tail_fraction=0.1, xmax=None) -> dict`; finite `xmax` uses the same bounded likelihood/CDF contract.
- `rank_ordered_mle(values, *, xmin=None, tail_fraction=0.1) -> dict`; degeneracy detection is exact/relative rather than absolute-scale dependent.
- `hill_estimator(values, k_min=5) -> tuple[np.ndarray, np.ndarray]`
- `hill_alpha_at(values, k) -> float`
- `hill_estimator_windowed(values, *, window=20, k_min=5) -> tuple[np.ndarray, np.ndarray]`
- `hill_plateau(values, *, window=20, flat_tol=0.15) -> dict`
- `select_tail_estimator(values, estimator="csn", **kwargs) -> dict`; accepted canonical names are `clauset_mle`, `hill_estimator`, `hill_windowed`, `fixed_cutoff_mle`, `rank_ordered_mle`, and `all`, with legacy `csn`/`hill` aliases. Windowed-Hill reports start/end rank, window width, and union support; its inapplicable single-cutoff `xmin`/`n_tail` fields remain unavailable.
- `powerlaw_pkg_fit(values, xmax=None) -> dict | None`; both nested models use the same finite support, `LR_trunc = log L_pure - log L_truncated`, and `LR_p` uses the one-sided boundary chi-square law for a nonnegative truncation rate.

## `rmt.scalars`

- `spectral_norm(weight=None, *, s=None) -> float`
- `frobenius_norm(weight=None, *, s=None) -> float`
- `stable_rank(weight=None, *, s=None) -> float`
- `condition_number(weight=None, *, s=None, rcond=None) -> float`
- `hard_rank(weight=None, *, s=None, tolerance=None) -> int`
- `von_neumann_entropy(weight=None, *, s=None, base=np.e) -> float`
- `matrix_entropy` is an identity alias of `von_neumann_entropy`.
- `normalized_matrix_entropy(weight=None, *, s=None) -> float`
- `spectral_entropy(svals, base=np.e) -> float`
- `row_wise_entropy(weight, base=np.e) -> float`
- `ipr(vectors, axis=0) -> np.ndarray`
- `localization_ratio(vector) -> float`
- `participation_ratio(vector) -> float`
- `ipr_summary(Vh, s, n, m, sigma) -> dict`
- `porter_thomas_ks(Vh, *, n_vectors=None) -> dict`
- `porter_thomas_monte_carlo(vectors, *, n_reference=512, pooling_window=1, rng=0) -> dict`
- `porter_thomas_monte_carlo_pooled(vectors, *, n_reference=512, pooling_window=5, rng=0) -> dict`
- `mp_softrank(s, nu_plus) -> float`
- `bulk_mass_frac(s, nu_plus) -> float`
- `decile_index_ranges(k, n_deciles=10, ascending=True) -> list[tuple[int, int]]`
- `per_decile(s, n_deciles=10) -> dict`

## `rmt.spacing`

- `unfold(levels, deg=7) -> np.ndarray`
- `unfold_spectrum(levels, *, method="polynomial", degree=7, smoothing=None, kernel_window=15, edge_mode="replicate") -> np.ndarray`
- `nn_spacing(levels, deg=7) -> np.ndarray`
- `nearest_neighbor_spacings(levels, *, unfolded=False, method="polynomial", degree=7, smoothing=None, kernel_window=15, edge_mode="replicate") -> np.ndarray`
- `brody_pdf(s, beta) -> np.ndarray`
- `brody_cdf(s, beta) -> np.ndarray`
- `fit_brody(spacings, *, method="mle", n_bootstrap=0, rng=0) -> BrodyFit`
- `fit_brody_cdf_nls(spacings, *, n_bootstrap=0, rng=0) -> BrodyFit`
- `r_statistic(levels) -> float`; three finite levels are sufficient for one adjacent-gap ratio.
- `number_variance(levels, L, *, unfolded=False, degree=7, n_windows=None, method="sliding", tolerance=1e-4, rng=0) -> float`
- `sigma2(levels, L, deg=7) -> float`
- `dyson_mehta_delta3(levels, L, *, unfolded=False, degree=7, n_windows=None) -> float`
- `delta3(levels, L, deg=7) -> float`
- `nn_spacing_ks(levels, deg=7) -> dict`
- `complex_spacing_ratio(matrix) -> dict`
- `wigner_goe_cdf(s) -> np.ndarray`
- `poisson_cdf(s) -> np.ndarray`

## `rmt.overlap`

- `activation_eigensystem(covariance) -> tuple[np.ndarray, np.ndarray]`
- `projection_overlap(weight_vectors, activation_vectors) -> np.ndarray`
- `subspace_alignment(weight_vectors, activation_vectors) -> float`
- `principal_angle_spectrum(weight_vectors, activation_vectors) -> np.ndarray`
- `principal_angle_alignment(weight_vectors, activation_vectors) -> float`
- `frobenius_projection_overlap(weight_vectors, activation_vectors) -> float`
- `reference_max_cosine_overlap(weight_vectors, activation_vectors, *, absolute=False, squared=False) -> np.ndarray`
- `evaluate_overlap_metric(weight_vectors, activation_vectors, *, metric="staats_dual_end") -> dict`
- `tranche_indices(n_values, *, top_fraction=0.1, bottom_fraction=0.1) -> dict[str, np.ndarray]`
- `dual_end_alignment(svd, covariance, *, activation_fraction=0.1, top_fraction=0.1, bottom_fraction=0.1, metric="staats_dual_end") -> dict`; returns availability/rank status, uses source-array precision for covariance qualification, rejects significant indefiniteness, excludes null modes, does not split a tied positive cluster at the target cutoff, and marks nonidentifiable weight singular subspaces unavailable.
- `overlap_analysis(weight, feature_matrix, *, svd=None) -> dict`; legacy convention with the same covariance/weight qualification and explicit availability status.
- `eigenvector_eigenvalue_coincidence(weight, feature_matrix, *, svd=None) -> dict`; legacy convention with the same qualification/status contract.
- `three_sigma_band(N, sigma_level=3.0) -> tuple[float, float]`
- `resolve_fm_key(record_name, fm_keys) -> str | None`

## `rmt.farms_aspect_ratio`

- `FARMSConfig(target_aspect_ratio=1.0, window_size=None, row_windows=5, column_windows=5, sampling="reference_fixed", n_submatrices=None, step_size=10, normalization="canonical", orient_tall=False, seed=0)` is frozen. The target ratio is sampled columns divided by sampled rows, and `window_size` is the sampled row count.
- `FARMSResult` is a frozen dataclass containing the descending pooled spectrum, source/oriented/window shapes, starts, reference and canonical aspect ratios, normalization and one exact normalization denominator per window, transpose state, coverage, and window counts.
- `fixed_ratio_window_shape(source_shape, target_aspect_ratio=1.0, window_size=None) -> tuple[int, int]`
- `farms_window_starts(source_shape, window_shape, *, row_windows=5, column_windows=5, sampling="reference_fixed", n_submatrices=None, step_size=10, rng=0) -> np.ndarray`
- `iter_farms_submatrices(weight, window_shape, starts) -> Iterator[np.ndarray]`
- `farms_spectrum(weight, config=FARMSConfig()) -> FARMSResult`
- `farms_unbiased_spectrum` is an identity alias of `farms_spectrum`.
- `shape_normalize_eigenvalues(eigenvalues, aspect_ratio, variance=1.0, *, target_upper_edge=4.0) -> np.ndarray`

## `rmt.lanczos_stieltjes`

- `LanczosResult`, `JacobiCholesky`, and `LanczosSpikeResult` are frozen dataclasses. Results include adaptive stopping state, per-probe recurrence lengths, pole method, per-probe VEST recurrences, and the effective scale-relative threshold margin.
- `covariance_linear_operator(factor, *, normalization=None) -> scipy.sparse.linalg.LinearOperator`
- `default_lanczos_steps(dimension) -> int`
- `lanczos_tridiagonalize(matrix, *, dimension=None, steps=None, probe=None, reorthogonalization="full", tolerance=None, adaptive=False, convergence_tolerance=None, sequence_length=None, check_interval=2, rng=0, return_basis=False) -> LanczosResult`
- `jacobi_cholesky(lanczos, *, ridge=0.0, pivot_tolerance=None) -> JacobiCholesky`; the default pivot threshold is operator-scale-relative.
- `estimate_constant_tail(cholesky, *, tail_window=None) -> tuple[float, float, float, float]`
- `reference_modified_cholesky(lanczos, *, tail_window=None, ridge=0.0) -> tuple[JacobiCholesky, dict[str, float | int | str]]`
- `support_from_cholesky_tail(alpha, beta) -> tuple[float, float]`
- `vector_empirical_stieltjes(lanczos, z) -> np.ndarray | complex`
- `extended_stieltjes_transform(z, diagonal, sub_diagonal, *, tail_alpha, tail_beta) -> np.ndarray | complex`
- `LanczosSpikeResult.stieltjes(z)` evaluates the representative measure matching reported poles/residues; `ensemble_stieltjes(z)` is the separately named probe average used by `density(...)`.
- `asymptotic_spectral_density(result, energies, *, eta=1e-3) -> np.ndarray`
- `finite_section_poles(cholesky, *, tail_alpha, tail_beta, threshold, residue_threshold=0.0, tail_window=None, extension_size=None) -> tuple[np.ndarray, np.ndarray]`
- `finite_vest_poles(lanczos, *, threshold, residue_threshold=0.0) -> tuple[np.ndarray, np.ndarray]`
- `detect_spikes_lanczos(matrix, *, dimension=None, steps=None, n_probes=1, reorthogonalization="full", tail_window=None, threshold_c=1.0, threshold_delta=0.25, threshold_mode="absolute", residue_threshold=0.0, ridge=0.0, extension_size=None, adaptive=True, convergence_tolerance=None, sequence_length=None, check_interval=2, pole_method="reference_ritz", rng=0) -> LanczosSpikeResult`
- `detect_spikes_from_factor(factor, **kwargs) -> LanczosSpikeResult`; the production factor adapter defaults `threshold_mode` to `bulk_edge_relative`, while direct operator calls retain the reference absolute convention.
- `lanczos_tridiagonalization`, `vector_empirical_stieltjes_transform`, and `lanczos_stieltjes_detector` are compatibility aliases.

## `rmt.factory`

- Choice tuples: `MP_FIT_METHODS`, `UNFOLDING_STRATEGIES`, `TAIL_SOLVERS`, `OVERLAP_METRICS`, `ASPECT_RATIO_MODES`, and `SPIKE_DETECTORS`.
- `RMTMethodConfig(mp_fit_method="lanczos_stieltjes", unfolding_strategy="spline_monotone", tail_solver="clauset_mle", overlap_metric="staats_dual_end", aspect_ratio_mode="farms_normalized", spike_detector="lanczos_poles", polynomial_degree=15, spline_smoothing=None, gaussian_kernel_window=15, mp_trim_upper=0.1, kde_bandwidth=None, tail_minimum=50, tail_fraction=0.1, farms_target_aspect_ratio=1.0, farms_window_size=None, farms_row_windows=5, farms_column_windows=5, farms_sampling="reference_fixed", farms_step_size=10, farms_normalization="canonical", farms_orient_tall=False, lanczos_steps=50, lanczos_probes=3, lanczos_tail_window=None, lanczos_threshold_c=1.0, lanczos_threshold_delta=0.25, lanczos_residue_threshold=0.0, lanczos_ridge=0.0, lanczos_adaptive=True, lanczos_convergence_tolerance=None, lanczos_sequence_length=None, lanczos_check_interval=2, lanczos_pole_method="reference_ritz", seed=0)` is the frozen validated method/hyperparameter contract.
- `PreparedSpectrum` is a frozen spectrum, canonical aspect ratio, mode, optional FARMS result, and diagnostics container.
- `prepare_spectrum(weight, config=RMTMethodConfig(), *, variance=1.0) -> PreparedSpectrum`
- `dispatch_mp_fit(weight, config=RMTMethodConfig(), *, variance=None) -> MPFitResult`
- `dispatch_tail_solver(eigenvalues, config=RMTMethodConfig(), **overrides) -> dict`
- `dispatch_unfolding(levels, config=RMTMethodConfig()) -> np.ndarray`
- `dispatch_overlap(weight_vectors, activation_vectors, config=RMTMethodConfig()) -> dict`
- `dispatch_spike_detector(weight, config=RMTMethodConfig(), *, variance=1.0, eigenvalues=None, aspect_ratio=None, operator_shape=None) -> SpikeDetectionResult`; Tracy--Widom finite-size corrections use `operator_shape`, never pooled observation count, and Lanczos assumption failures return a structured unavailable result.

## Model and pipeline contracts

- `TransformerConfig` is a frozen dataclass containing model dimensions and architecture choices.
- `CausalTransformer.forward(input_ids, attention_mask=None, labels=None, *, return_dict=True)` returns `CausalLMOutput` or a tuple.
- `ScalingLaw` and `Allocation` are frozen dataclasses.
- `compute_optimal_allocation(compute_budget, law=ScalingLaw(), *, target_tokens_per_parameter=None) -> Allocation`
- `allocation_for_regime(optimal, kappa, *, name=None, law=ScalingLaw()) -> Allocation`
- `isoflop_grid(compute_budgets, kappas=(0.25, 1.0, 4.0), law=ScalingLaw(), *, target_tokens_per_parameter=20.0) -> list[Allocation]`
- `estimate_transformer_parameters(config) -> int`
- `suggest_architecture(target_parameters, *, vocab_size=512, max_layers=24, width_multiple=64, max_parameters=None) -> dict`; embedding cost and all feasible lower widths are included, and one-layer searches are supported.
- `CovarianceEstimate(array, *, observation_count, accumulation_dtype, centered)` is an `np.ndarray` subclass carrying rank/precision provenance into overlap qualification.
- `CovarianceAccumulator(dimension, count=0, sum_vector=None, gram_matrix=None, device=None, dtype=float64)` maintains a stable running mean and centered M2 matrix on one device (historical buffer attribute names are retained).
- `CovarianceAccumulator.update(activations, *, valid_mask=None, max_samples=None) -> None`; moment arithmetic explicitly disables an enclosing autocast context.
- `CovarianceAccumulator.second_moment() -> Tensor`
- `CovarianceAccumulator.covariance(*, centered=True, unbiased=False) -> Tensor`
- `ActivationExtractor(model, module_filter=None, *, capture=("pre", "post"), max_samples_per_hook=None, accumulation_device="auto", accumulation_dtype="auto")`
- `ActivationExtractor.__enter__() -> ActivationExtractor`
- `ActivationExtractor.__exit__(exc_type, exc_value, traceback) -> bool`
- `ActivationExtractor.covariances(*, centered=True, unbiased=False) -> dict[str, np.ndarray]`
- `compute_tensor_svd(matrix, *, backend="auto", driver="gesvdj", normalization=None) -> SVDResult`
- `compute_activation_covariances(model, dataloader, *, device, module_filter=None, max_batches=None, centered=True, accumulation_device="auto", accumulation_dtype="auto", amp_dtype="float32") -> dict[str, np.ndarray]`
- `compute_tensor_svd(matrix, *, backend="auto", driver="gesvdj", normalization=None, analysis_dtype="float64") -> SVDResult`; analysis precision is independent of model/autocast precision.
- `lesion_matrix(weight, tranche, *, fraction=0.05, mode="count", reference_energy=None, generator=None, svd_backend="auto", svd_driver="gesvdj") -> tuple[Tensor, LesionInfo]`
- `lesion_matrix_decile(weight, decile, *, n_deciles=10, svd_backend="auto", svd_driver="gesvdj") -> tuple[Tensor, LesionInfo]`
- `spectral_lesion(model, parameter_names, tranche, *, fraction=0.05, mode="count", reference_energy=None, seed=0, svd_backend="auto", svd_driver="gesvdj", factor_cache=None, analysis_dtype="float64")` returns a restoring context manager. Reusable cache entries are verified against pristine weight bytes and the analysis contract.
- `spectral_decile_lesion(model, parameter_names, decile, *, n_deciles=10, svd_backend="auto", svd_driver="gesvdj")` returns a restoring context manager.
- `independent_lesion_benchmark(model, parameter_names, evaluate, *, tranches=("top", "bulk", "bottom"), fraction=0.05, mode="count", reference_energy=None, seed=0, svd_backend="auto", svd_driver="gesvdj") -> list[dict]`
- `independent_decile_benchmark(model, parameter_names, evaluate, *, n_deciles=10, svd_backend="auto", svd_driver="gesvdj") -> list[dict]`
- `spectral_parameter_names(model) -> list[str]`

## CLI contracts

- `SpectralCLIConfig` is a frozen serializable CLI counterpart to `RMTMethodConfig` with the same method/tuning defaults plus the optional compatibility `boundary_detector` field.
- `add_rmt_cli_arguments(parser) -> argparse.ArgumentParser`
- `rmt_config_from_namespace(namespace) -> RMTMethodConfig`
- `parse_rmt_args(arguments=None) -> RMTMethodConfig`
- `add_pipeline_cli_arguments(parser) -> argparse.ArgumentParser`
- `parse_pipeline_args(arguments=None) -> argparse.Namespace`
- `add_spectral_method_arguments` and `config_from_args` are compatibility aliases.
- The primary flags are `--mp-fit-method`, `--unfolding-strategy`, `--tail-solver`, `--overlap-metric`, `--aspect-ratio-mode`, and `--spike-detector`.
- `--overlap-mode` is an alias of `--overlap-metric`.
- `--boundary-detector` is a compatibility flag whose resolved MP/spike pair takes precedence when supplied.
- The complete runner schema additionally owns experiment/data/scaling/training/device/covariance/SVD/diagnostic/lesion flags. `README.md` is the exhaustive user-facing dictionary; `pipelines.cli_config` is the executable source of truth.

## Data and training contracts

- `CharTokenizer(alphabet)` with `from_text`, `encode`, and `decode` constructors and methods.
- `TokenSequenceDataset(tokens, sequence_length, *, stride=None)` returns dictionaries containing `input_ids`, `labels`, and `attention_mask`.
- `load_token_array(path) -> np.ndarray`
- `split_tokens(tokens, *, train_fraction=0.9) -> tuple[np.ndarray, np.ndarray]`
- `build_synthetic_corpus(length, vocab_size, *, seed=0, noise_probability=0.05) -> np.ndarray`
- `TrainConfig` is a frozen optimizer/schedule/accelerator dataclass containing AMP dtype, compile mode, TF32 policy, and nonblocking-transfer policy.
- `evaluate_language_model(model, dataloader, device, *, max_batches=None, amp_dtype="float32", non_blocking_transfers=True) -> dict[str, float]`
- `LanguageModelTrainer.fit(train_dataloader, validation_dataloader=None, *, callback=None) -> list[dict]`; token-budget schedules follow successful target-token progress across recycled loader passes, and recoverable GradScaler overflows back off without advancing tokens, scheduler, or optimizer-update counters.
- `LanguageModelTrainer.save_checkpoint(path, *, metadata=None) -> None`
