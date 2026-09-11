# Spectral-Chinchilla implementation plan

## Architecture

1. Keep the numerical `rmt` foundation. It includes `SVDResult`, spectrum conventions, ensembles, MP theory, tail estimates, scalar metrics, spacing, rigidity, and overlap.
2. Examine each source file in `codebase1`, `codebase2`, and `codebase3`. Record provenance, formula differences, domain differences, absent features, and rejected attributions in `differences.md`.
3. Add FARMS pooling of fixed-ratio submatrices. Include the reference floor-stride and fixed-step schedules. Keep analytic shape normalization separate.
4. Add matrix-free Lanczos and double full reorthogonalization. Include adaptive Jacobi stopping, modified-tail Cholesky analysis, one-cut support, averaged VEST, and finite-Ritz pole counts. Keep the constant-tail finite-section detector as an alternative.
5. Add historical and corrected alternatives. They include triangular-KDE and modified-singular MP fits, TW and BBP limits, and fixed or rank tail fits. They also include CSN bootstrap, Gaussian unfolding, Brody CDF fitting, Monte Carlo number variance, Porter-Thomas calibration, localization, and overlap methods.
6. Put method selection in `RMTMethodConfig`. Give the same permitted choices through `pipelines/cli_config.py`.
7. Keep normalization and RoPE or learned positions in the model layer. Keep multi-head or grouped-query attention, both MLP types, decoder blocks, and causal-LM loss.
8. Keep the analytic Chinchilla optimum and exact `D/N` mode. Keep fixed-compute transforms, IsoFLOP grids, and estimates of architecture parameters.
9. Keep local data loading, training, centered activation covariance, tranche lesions, and reference decile lesions. Put accelerator operations only in `models/` and `pipelines/`.
10. Record method metadata in the runner. Give each method a deterministic synthetic test.
11. Use `scripts/download_assets.py` to make an isolated asset manifest. Keep standalone pins and document cluster reconciliation, caches, and offline operation.
12. Use `run_hpc.slurm` for the four tracks. It uses the recorded `rmt_ml_env` interpreter and `gpulong` queue. Operators control untested module, account, QoS, and resource changes.

## Data flow

`CLI method config -> compute allocation -> transformer config -> training checkpoint -> weight/operator + activation covariance -> selected spectral methods -> independent lesions -> tabular and graphical report`

Each metric row identifies the compute tier, regime, `kappa`, parameter counts, token counts, layer, role, shape, and spectrum mode. It also identifies each selected method, exponent kind, overlap status, rank status, and random seed. `spectral_method_config.json` records each method parameter before execution.

## Module interactions

- `rmt.farms_aspect_ratio` receives a rectangular matrix. It returns a pooled spectrum and sampling metadata. It does not make one global singular-vector basis.
- `rmt.lanczos_stieltjes` receives a symmetric matrix, linear operator, or rectangular factor through `covariance_linear_operator`. It materializes covariance only if the caller supplies it.
- `rmt.factory` is the only method-name dispatcher. Numerical modules contain methods. `pipelines.cli_config` contains parsing and compatibility aliases.
- `--boundary-detector` is a compatibility selector. If present, it overrides the paired MP and spike defaults. The runner records the resolved values.
- `pipelines.activation_extractor` owns device covariance buffers and the bridge to `SVDResult`. No device object or tensor enters `rmt/`.
- `pipelines.cli_config.add_pipeline_cli_arguments` gives the full runner schema. `add_rmt_cli_arguments` gives the reusable scientific subset.
- The Golden defaults use Lanczos-Stieltjes support and monotone-spline unfolding. They also use CSN MLE, dual-end projection, canonical FARMS pooling, and reference-Ritz spikes.

## Experiment matrix

The full grid combines compute budgets `{1e15,1e16,1e17}` with regime multipliers `{0.25,1.0,4.0}`. The CLI defaults to Golden budget `{1e16}` and all three multipliers. Users can give other positive lists. The fixed-compute transform is `(N,D)=(N*/kappa,kappa D*)`.

Examine each `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Collect the selected MP fit, edge counts, spikes, tail estimate, exponent convention, scalar metrics, `r`, and other level statistics. Also collect the selected dual-end overlap. Unavailable standalone detectors and zero-rank activation covariance do not remove unrelated metrics.

Lesions remove a selected fraction from top, MP-bulk, or bottom tranches. Count matching and Frobenius-energy matching are available. Reference decile surgery stays separate.

## Outputs

The runner writes these files:

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

Offline staging also writes `data/asset_manifest.json`, raw JSONL shards, tokenizer files, and an NPY token stream. These files are deployment inputs, not experiment observations.

## Completion gates

The source gate permits no prohibited framework name under `rmt/`. It permits no placeholder body in maintained source or specifications. Signatures, tests, seeds, local data access, and lesion recovery must agree with their contracts. Read-only reference archives are not part of the placeholder gate.

After the operator builds the pinned offline environment, the operator must do the pytest gate. The operator must do accelerator tests with the cluster CUDA and driver stack. Only completed training can satisfy the empirical gate.
