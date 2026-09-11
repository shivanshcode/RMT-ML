# Verification checklist

## Source gates

- [x] Requested directory layers and public contracts are documented.
- [x] Pure RMT modules use NumPy, SciPy, and standard Python only.
- [x] MP normalization and CSN/Hill exponent conventions are explicit.
- [x] Transformer supports RoPE or learned positions, RMSNorm or LayerNorm, SwiGLU or GELU, and MHA or GQA.
- [x] Fixed-compute undertrained, optimal, and overtrained allocation is implemented.
- [x] Activation hooks support centered covariance and uncentered second moment.
- [x] Top, bulk, and bottom lesions are independent and reversible.
- [x] Unit tests cover all requested scientific and integration groups.
- [x] Experiment orchestration defines all five required plots.
- [x] `differences.md` inventories and reconciles every source-bearing file in all three supplied archives.
- [x] FARMS is implemented as fixed-ratio submatrix pooling with deterministic/seeded sampling metadata.
- [x] Analytic shape normalization is separate from FARMS and labeled accordingly.
- [x] Lanczos–Cholesky support, VEST, continued fraction, finite-section poles, multi-probe aggregation, and convergence diagnostics are implemented.
- [x] Correct BBP population and sample thresholds are documented and implemented.
- [x] Historical KDE, modified-MP, Gaussian unfolding, Brody CDF, Monte Carlo number variance, Porter–Thomas, localization, tail, overlap, and decile-lesion alternatives are present.
- [x] All six primary method selectors are centralized and wired into `run_experiments.py`.
- [x] The runner serializes `spectral_method_config.json` and method labels in metric rows.
- [x] Governing documents and API signatures include every Phase II public symbol.
- [x] FARMS reference columns/rows geometry, floor-stride sampling, fixed-step sampling, raw pooling, and canonical pooling are distinguished.
- [x] Lanczos reference modified-tail Cholesky, adaptive recurrence diagnostics, finite-Ritz pole rule, residues, and probe-averaged VEST are implemented and documented.
- [x] Accelerator covariance reduction and reduced SVD bridges remain confined to `pipelines/`.
- [x] Central CLI parsing covers experiment, scaling, offline data, training, hardware, diagnostics, lesions, and every scientific method selector.
- [x] Standalone direct pins, inventory-driven cluster reconciliation, optional wheelhouse procedure, offline cache layout, and asset checksums are documented.
- [x] `run_hpc.slurm` uses the recorded `rmt_ml_env` interpreter and `gpulong` partition, isolates local imports, creates per-job outputs/caches, and contains all four fail-closed tracks.
- [x] Every runner invocation records package, platform, precision, accelerator, and scheduler provenance in `runtime_environment.json`.
- [x] Phase III governing documents describe the implemented defaults and signatures.
- [x] Pooled FARMS ESD/tail observations are separated from single-operator spacing, with per-domain geometry and normalization provenance.
- [x] Tracy--Widom uses operator/window dimensions. Production Lanczos margins are scale-relative and serialized.
- [x] Training, evaluation, lesions, activation covariance, CLI aliases, scaling grouping, and Hill support have v3/v4 regressions.
- [x] All COA-001..COA-016 findings in `bug_report.md` are fixed. Repairs cover MP, scaling, schedules, Lanczos, tails, overlap, collapse, and atomic ownership.

## Human execution gates

- [ ] Record `/home/shivansh/.conda/envs/rmt_ml_env` and do its tests. Do not install standalone `requirements.txt` pins into it in place.
- [ ] If the live stack is incompatible, create a separate environment and archive its reviewed `requirements-cluster.txt` and wheel inventory.
- [ ] Search `rmt/` without case sensitivity. Make sure that the prohibited framework name does not occur in source, comments, or docstrings.
- [ ] Do the full pytest suite again in the selected cluster environment. It must have zero failures. `bug_report.md` records the local CPU test.
- [ ] Record the exact NumPy/SciPy/platform versions used for numerical calibration.
- [ ] Apply the seeded three-spike detector. Make sure that it returns three poles and satisfies edge tolerance on the operator BLAS/LAPACK stack.
- [ ] Apply the six-ratio FARMS edge calibration. Make sure that it satisfies the two-percent tolerance.
- [ ] Operate the three-by-three IsoFLOP experiment matrix.
- [ ] Make sure that each output row records requested and realized compute.
- [ ] Inspect generated plots for labels, normalization, and uncertainty metadata.
- [ ] Do tests of lesion results from independently restored checkpoints.
- [ ] Record hardware, dtype, seed, dataset checksum, and wall-clock metadata.
- [ ] If installation is necessary, build a transitive wheelhouse. Do a test with the exact cluster Python ABI, Torch/CUDA build, manylinux ABI, and architecture.
- [ ] On a connected host, enter `scripts/download_assets.py --assets all --allow-network`. Examine `data/asset_manifest.json` before transfer.
- [ ] After cluster transfer, enter `scripts/download_assets.py --verify-only`. Keep the successful file count in deployment records.
- [ ] Before submission, make sure of `gpulong`, account, QoS, CPU, time, modules, and driver compatibility.
- [ ] Create `logs/` before `sbatch`. Make sure that line endings are LF. Submit here with an absolute `PROJECT_ROOT`.
- [ ] Do tests of BF16, TF32, compilation, and each requested CUDA SVD driver on the target accelerator.

## Scope notes

- [x] The `codebase*`, `farmscode`, and `lanczoscode` directories are read-only audit inputs. Historical placeholder tokens inside those archives are not maintained-project stubs and are not modified.
- [x] Trained-layer exponent ranges, FARMS downstream benefit, spectral collapse, and lesion perplexity ordering remain empirical hypotheses for operator experiments rather than source-completion claims.
- [x] Source generation does not install dependencies, do tests, train models, or operate benchmarks.
