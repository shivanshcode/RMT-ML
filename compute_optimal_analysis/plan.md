# Spectral-Chinchilla implementation plan

## Architecture

1. Preserve the numerical `rmt` foundation: `SVDResult`, canonical spectrum conventions, ensembles, MP theory, tail estimators, scalar metrics, spacing/rigidity, and overlap.
2. Reconcile every source-bearing file in `codebase1`, `codebase2`, and `codebase3`; record method provenance, formula/domain differences, absent features, and rejected attributions in `differences.md`.
3. Add paper-faithful FARMS fixed-ratio submatrix pooling, including reference floor-stride/fixed-step schedules, and keep analytic shape normalization separate.
4. Add matrix-free Lanczos, double full reorthogonalization, adaptive Jacobi-coefficient stopping, reference modified-tail Cholesky analysis, one-cut support, averaged VEST evaluation, finite-Ritz pole counting, and the alternative constant-tail finite-section detector.
5. Add historical and corrected alternatives: triangular-KDE MP fitting, modified singular MP curves, TW and BBP boundaries, fixed/rank tail fits, CSN bootstrap, adaptive Gaussian unfolding, Brody CDF fitting, Monte Carlo number variance, Porter–Thomas calibration, localization ratios, and multiple overlap metrics.
6. Centralize method selection in `RMTMethodConfig`, then expose the same validated choices through `pipelines/cli_config.py`.
7. Preserve the model layer: normalization, RoPE or learned positions, multi-head or grouped-query causal attention, gated or GELU MLPs, decoder blocks, and causal-LM loss.
8. Preserve the Chinchilla allocator: analytic optimum, exact `D/N` target mode, fixed-compute regime transforms, IsoFLOP grids, and architecture parameter estimates.
9. Preserve local-only data loading, training, centered activation covariance, reversible tranche lesions, and reference descending-decile lesions; add BF16/FP16 autocast, TF32, optional compilation, in-device covariance aggregation, and accelerator SVD bridges only in `models/`/`pipelines/`.
10. Wire method metadata into the runner and lock each method with deterministic synthetic tests.
11. Stage a hermetic asset manifest with `scripts/download_assets.py`, preserve standalone direct pins, document inventory-driven cluster dependency reconciliation and caches, and keep runtime fail-closed/offline.
12. Provide the four-track `run_hpc.slurm` harness using the verified `rmt_ml_env` interpreter and `gpulong` queue; keep unverified module, account/QoS, and resource changes operator-controlled.

## Data flow

`CLI method config -> compute allocation -> transformer config -> training checkpoint -> weight/operator + activation covariance -> selected spectral methods -> independent lesions -> tabular and graphical report`

Every metric row records compute tier, regime, `kappa`, requested and realized parameter/token counts, layer, matrix role, shape, spectrum mode, MP method, spike detector, tail solver and exponent kind, unfolding strategy, overlap metric, and random seed. `spectral_method_config.json` records every method-specific hyperparameter before execution.

## Module interactions

- `rmt.farms_aspect_ratio` consumes a rectangular matrix and returns a pooled spectrum plus sampling metadata; it does not synthesize a global singular-vector basis.
- `rmt.lanczos_stieltjes` consumes a symmetric matrix, a linear operator, or a rectangular factor through `covariance_linear_operator`; it does not materialize the covariance unless the caller supplied one.
- `rmt.factory` is the only method-name dispatcher. Numerical modules contain implementations, while `pipelines.cli_config` contains parsing and compatibility aliases.
- `--boundary-detector` is a compatibility selector. If present, it takes precedence over the paired MP/spike defaults and the resolved values are serialized.
- `pipelines.activation_extractor` owns device covariance buffers and the tensor-to-`SVDResult` bridge. No device object or tensor crosses into `rmt/`.
- `pipelines.cli_config.add_pipeline_cli_arguments` is the complete runner schema; `add_rmt_cli_arguments` remains the reusable scientific subset.
- The Golden default path is Lanczos-Stieltjes MP support, monotone-spline unfolding, CSN MLE, squared dual-end projection, canonical FARMS pooling, and reference-Ritz Lanczos spike detection.

## Experiment matrix

The full research grid is the Cartesian product of compute budgets `{1e15,1e16,1e17}` and regime multipliers `{0.25,1.0,4.0}`. The production CLI defaults to the requested Golden budget `{1e16}` with all three multipliers; arbitrary positive lists are accepted. The fixed-compute transform is `(N,D)=(N*/kappa,kappa D*)`.

For every `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`, collect selected MP fit, lower/upper edge counts, selected spikes, selected tail estimate plus exponent convention, scalar metrics, Brody beta, `r`, number variance, and selected dual-end overlap. Lesions remove a configurable fraction from top, MP-bulk candidate, or bottom tranches. Count-matched and Frobenius-energy-matched modes are available; reference decile surgery is separate.

## Outputs

- `allocation_manifest.json`
- `spectral_method_config.json`
- `spectral_metrics.csv`
- `lesion_metrics.csv`
- `training_metrics.jsonl`
- `esd_powerlaw_fits.png`
- `brody_spacing_distribution.png`
- `dual_end_activation_overlap.png`
- `lesioning_perplexity_impact.png`
- `scaling_collapse_spectral_trajectory.png`
- `run_config.json`
- `runtime_environment.json`

Offline staging additionally produces `data/asset_manifest.json`, local raw JSONL shards, tokenizer files, and an NPY token stream. Those files are deployment inputs, not experiment observations.

## Completion gates

The source gate requires no framework-name occurrence anywhere under `rmt/`, no placeholder bodies in the maintained source/specification tree, synchronized signatures and tests, deterministic seed plumbing, local-only data access during jobs, and recoverable lesions. Supplied reference archives are read-only audit inputs and are excluded from the maintained-source placeholder gate. The numerical gate requires the operator to run pytest after building the pinned offline environment. The accelerator gate requires operator validation on the cluster's CUDA/driver stack. The empirical gate requires actual training runs and cannot be certified by source generation.
