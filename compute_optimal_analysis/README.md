# Spectral-Chinchilla

Spectral-Chinchilla trains causal Transformers with controlled compute allocation. It measures random-matrix signatures, level statistics, activation alignment, and spectral-lesion effects. Phase III compares FARMS and Lanczos with the supplied reference repositories. It also gives an offline High-Performance Computing (HPC) workflow.

## Repository map

- `rmt/` contains the pure NumPy and SciPy engine. Do not import an accelerator framework in this directory.
- `models/` contains the decoder-only Transformer and Chinchilla allocator.
- `pipelines/` contains offline data, training, activation covariance, SVD bridges, lesions, and the CLI.
- `scripts/download_assets.py` stages assets on a connected computer.
- `run_experiments.py` controls experiments and plots.
- `run_hpc.slurm` contains three paper tracks and the Golden Compute-Optimal RMT track.
- `differences.md` records source and mathematical reconciliation decisions.
- `other_requirements.md` specifies the air-gap, wheelhouse, asset, and directory contracts.

## Numerical and safety contracts

Gradient clipping gets its coefficient from an FP64 norm. Token-progress warmup cannot exceed the selected peak learning rate.

SVD results keep the actual factorization precision. Numerical rank, overlap, and Porter-Thomas output is unavailable for an unresolved null or repeated basis. Real-only APIs reject complex matrices.

Finite changes of spectral units do not change MP quantiles, modified-MP fits, or dimensionless scalar summaries. FARMS availability uses the prepared pooled spectrum. Constant-tail poles and the Stieltjes transform use the same recurrence.

Reusable lesion factors bind to matrix content and the analysis contract. Presets apply before dependent input tests.

## Cluster procedure

### 1. Record the environment

The existing ESD jobs use `/home/shivansh/.conda/envs/rmt_ml_env/bin/python`. By default, `run_hpc.slurm` calls this interpreter directly. Use `RMT_PYTHON` only for a prefix or clone that passed separate tests.

The launcher does not activate a project `.venv`. It does not purge modules or load a Python module. It does not assume a CUDA module version. Set `RMT_CUDA_MODULE` only if the site requires a module that passed its tests.

Before package changes, record the environment as specified in `other_requirements.md`. Record `pip freeze --all`, `pip check`, the explicit Conda package list, loaded modules, and operating-system data. Also record the compiler, driver, Torch CUDA build, GPU capability, BF16 support, and package versions.

Do not replace packages in the working ESD environment without an inventory and tests. The sibling Delta3 code supports `np.trapezoid` and NumPy 1.x `np.trapz`. Thus, NumPy 1.26 alone is not a known incompatibility. The full stack still requires tests.

`requirements.txt` is the original standalone environment contract. If the cluster stack is incompatible, make a separate environment. Make a reviewed `requirements-cluster.txt` from the inventory and test results. Do not infer that file from unpinned ESD requirements. Do not install standalone pins into `rmt_ml_env` in place.

Build a wheelhouse on a connected Linux host. It must match the cluster Python ABI, architecture, C library, and selected Torch and CUDA build. Read `other_requirements.md` for the standalone and wheelhouse procedures.

### 2. Stage assets on a connected computer

```bash
python scripts/download_assets.py --assets all --allow-network
```

Before partial staging, the utility examines unchanged asset families against the manifest. It examines both membership and content. It rejects added, removed, or changed files instead of certifying them again.

The utility downloads a tokenizer snapshot into a new tree before replacement. Thus, a file that upstream removed cannot stay after a revision change. Manifest paths use relative POSIX syntax and work for Windows staging and Linux use.

The utility resolves the dataset and tokenizer repositories to fixed commit SHAs. It stages `Salesforce/wikitext` configuration `wikitext-103-raw-v1` and the `openai-community/gpt2` tokenizer. It also stages raw JSONL splits, one integer token array, cache directories, and a deterministic synthetic corpus. `data/asset_manifest.json` records source revisions and SHA-256 checksums.

To examine a copied asset tree without network access, enter:

```bash
python scripts/download_assets.py --verify-only
```

For a small staging calibration, enter:

```bash
python scripts/download_assets.py --assets all --allow-network --max-wikitext-tokens 10000000
```

To make only synthetic data without network access, enter:

```bash
python scripts/download_assets.py --assets synthetic --synthetic-tokens 1000000
```

### 3. Do preflight and submit the job

Enter these commands from the deployed `compute_optimal_analysis` directory. Use the selected interpreter. Make `logs/` before `sbatch`, because SLURM opens log files before the script starts.

```bash
cd /absolute/deployed/path/to/compute_optimal_analysis
PY=/home/shivansh/.conda/envs/rmt_ml_env/bin/python
mkdir -p logs results cache/torch cache/huggingface
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

"$PY" -m pip check
"$PY" scripts/download_assets.py --root "$PWD" --verify-only
"$PY" -m pytest -q
bash -n run_hpc.slurm
if LC_ALL=C grep -q $'\r' run_hpc.slurm; then echo "run_hpc.slurm is not LF" >&2; exit 2; fi
```

Make sure that `rmt.__file__` points to this project. In a short GPU allocation, do forward, backward, evaluation, covariance, SVD-driver, and lesion tests. First use `--no-compile-model`. Do a separate compilation test before production. No GPU on a login node does not show a GPU incompatibility.

`TRACK` selects one batch track. Its default, `all`, operates all four tracks in sequence. The known queue is `gpulong`. Submit from this directory, or give an absolute `PROJECT_ROOT` that passed its tests:

```bash
sbatch --chdir="$PWD" --export=ALL,PROJECT_ROOT="$PWD",TRACK=paper1 run_hpc.slurm
sbatch --chdir="$PWD" --export=ALL,PROJECT_ROOT="$PWD",TRACK=paper2 run_hpc.slurm
sbatch --chdir="$PWD" --export=ALL,PROJECT_ROOT="$PWD",TRACK=paper3 run_hpc.slurm
sbatch --chdir="$PWD" --export=ALL,PROJECT_ROOT="$PWD",TRACK=golden run_hpc.slurm
```

The launcher requests one process and one GPU. It does not implement DDP or FSDP. Before work, it examines the interpreter, Python version, local `rmt` import, CUDA, BF16, and staged assets.

Output goes to a new `results/jobs/$SLURM_JOB_ID` directory. `OUTPUT_ROOT` can select another location. The selected location must be absent or empty and unowned.

The launcher claims an intentionally pre-created empty directory with `.job-owner`. Each child runner claims its track directory with `.run-owner.json`. `COMPILE_MODEL=0` turns compilation off for calibration.

For `RMT_CUDA_MODULE`, use only a module that passed its tests. Before production, ask the site about account, QoS, 16 CPUs, 24 hours, GPU model, and longer time limits.

### 4. Write manifests without training

Do not give `--execute` if you only want allocation and method manifests. Manifest and execution modes both require a new or empty output directory.

```bash
python run_experiments.py --output-dir results/manifest_check
```

### 5. Operate one paper track

These tracks apply each paper analysis protocol to the Spectral-Chinchilla Transformer family. They do not reproduce original checkpoints, datasets, or training histories byte for byte.

For Staats et al. dual-end overlap and singular-value lesions, enter:

```bash
python run_experiments.py --execute --dataset-path data/tokenized/wikitext-103-raw-v1_gpt2.npy --experiment-mode reproduce_paper1 --model-type causal_transformer --aspect-ratio-mode raw --mp-fit-method analytic_mp --spike-detector tracy_widom_95 --overlap-metric staats_dual_end --unfolding-strategy gaussian_kernel --tail-solver rank_ordered_mle --run-spectral-lesioning --lesion-tranches top,bulk,bottom --device cuda --output-dir results/paper1
```

For Thamm et al. unfolding, NNSD, Brody, number variance, and rigidity, enter:

```bash
python run_experiments.py --execute --dataset-path data/tokenized/wikitext-103-raw-v1_gpt2.npy --experiment-mode reproduce_paper2 --model-type causal_transformer --aspect-ratio-mode raw --mp-fit-method thamm_modified_singular --spike-detector tracy_widom_95 --unfolding-strategy polynomial_chebyshev --unfolding-degree 15 --compute-spacing-distribution --compute-number-variance --compute-delta3 --device cuda --output-dir results/paper2
```

For Martin and Mahoney heavy-tail and empirical MP diagnostics, enter:

```bash
python run_experiments.py --execute --dataset-path data/tokenized/wikitext-103-raw-v1_gpt2.npy --experiment-mode reproduce_paper3 --model-type causal_transformer --aspect-ratio-mode raw --mp-fit-method kde_bulk_fit --spike-detector tracy_widom_95 --tail-solver clauset_mle --compute-stable-rank --device cuda --output-dir results/paper3
```

### 6. Operate the Golden pipeline

```bash
python run_experiments.py --execute --dataset-path data/tokenized/wikitext-103-raw-v1_gpt2.npy --experiment-mode compute_optimal_rmt --scaling-budget-flops 1e16 --allocation-ratios 0.25 1.00 4.00 --aspect-ratio-mode farms_normalized --farms-sampling reference_fixed --mp-fit-method lanczos_stieltjes --spike-detector lanczos_poles --lanczos-steps 50 --lanczos-adaptive --lanczos-pole-method reference_ritz --unfolding-strategy spline_monotone --tail-solver clauset_mle --overlap-metric staats_dual_end --run-spectral-lesioning --device cuda --amp-dtype bfloat16 --compile-model --svd-backend cuda --covariance-device cuda --output-dir results/golden_compute_optimal
```

## CLI reference

Boolean controls have matching `--flag` and `--no-flag` forms. Each paper mode applies its documented method preset. An explicit option, including `--no-*`, overrides that preset. `custom` uses parser defaults.

### Pipeline and scaling controls

| Flag | Type and permitted values | Default | Function |
|---|---|---|---|
| `--experiment-mode` | `reproduce_paper1`, `reproduce_paper2`, `reproduce_paper3`, `compute_optimal_rmt`, `custom` | `compute_optimal_rmt` | It selects the workflow and metadata. |
| `--model-type` | `causal_transformer` | `causal_transformer` | It selects the model family. |
| `--output-dir` | path | `results` | It selects the output directory. |
| `--execute` | boolean | false | True trains and analyzes. False writes manifests. |
| `--dataset-path` | local NPY or NPZ path | `data/tokenized/wikitext-103-raw-v1_gpt2.npy` | It selects the offline token stream. |
| `--cells` | `all` or comma-separated indices | `all` | It selects manifest cells. |
| `--scaling-budget-flops` | one or more positive floats | `1e16` | It selects budgets in `C≈6ND`. |
| `--allocation-ratios` | one or more positive floats | `0.25 1.0 4.0` | It selects token multipliers. |
| `--vocab-size` | integer | `50257` | It sets the embedding and output vocabulary. |
| `--parameter-cap` | positive float or omitted | omitted | It sets a local model-size limit. |
| `--max-train-tokens` | positive float or omitted | omitted | It sets the successful-update token target. Finite loaders recycle. |
| `--allow-collapsed-allocations` | boolean | false | It permits equal realized N/D designs in capped calibration cells. |

### Offline data and training controls

| Flag | Type | Default | Function |
|---|---|---|---|
| `--sequence-length` | integer | `256` | It sets the causal context length. |
| `--batch-size` | integer | `8` | It sets sequences for each optimizer step. |
| `--dataloader-workers` | nonnegative integer | `4` | It sets local loader workers. |
| `--prefetch-factor` | positive integer | `2` | It sets prefetched batches for each worker. |
| `--pin-memory` | boolean | true | It page-locks host batches for asynchronous transfer. |
| `--learning-rate` | float | `3e-4` | It sets the AdamW peak rate. |
| `--warmup-steps` | nonnegative integer | `100` | It sets linear warmup length. |
| `--gradient-clip` | positive float | `1.0` | It sets the global gradient-norm limit. |
| `--log-every` | positive integer | `50` | It sets the evaluation and log interval. |
| `--seed` | integer | `0` | It sets seeds for training, sampling, and diagnostics. |

### Accelerator controls

| Flag | Type and permitted values | Default | Function |
|---|---|---|---|
| `--device` | `cuda`, `cpu`, `auto` | `cuda` | It selects the training and inference device. |
| `--amp-dtype` | `float32`, `float16`, `bfloat16` | `bfloat16` | It selects the autocast dtype. |
| `--compile-model` | boolean | true | It compiles the model for CUDA training. |
| `--compile-mode` | `default`, `reduce-overhead`, `max-autotune` | `default` | It selects compilation behavior. |
| `--allow-tf32` | boolean | true | It permits TensorFloat-32 matrix multiplication. |
| `--svd-backend` | `auto`, `cuda`, `cpu` | `auto` | It selects SVD processing outside the pure engine. |
| `--svd-driver` | `gesvdj`, `gesvd`, `gesvda`, `default` | `gesvdj` | It selects the CUDA linear-algebra driver. |
| `--analysis-dtype` | `float32`, `float64` | `float64` | It selects weight and lesion SVD precision. |
| `--covariance-device` | `auto`, `cuda`, `cpu` | `auto` | It selects activation-buffer placement. |
| `--covariance-dtype` | `float32`, `float64` | `float32` | It selects activation-moment precision. |

### Diagnostics and lesion controls

| Flag | Type and permitted values | Default | Function |
|---|---|---|---|
| `--activation-batches` | integer | `5` | It sets batches for covariance estimates. |
| `--validation-batches` | integer | `20` | It sets batches for perplexity. |
| `--activation-centered` | boolean | true | It subtracts activation means. |
| `--compute-activation-overlap` | boolean | true | It computes weight and activation alignment. |
| `--compute-spacing-distribution` | boolean | true | It computes unfolded NNSD and Brody beta. |
| `--compute-number-variance` | boolean | true | It computes `Σ²(10)`. |
| `--compute-stable-rank` | boolean | true | It computes stable rank. |
| `--compute-delta3` | boolean | false | It computes Dyson-Mehta `Δ₃(10)`. |
| `--compute-porter-thomas` | boolean | false | It gives pooled calibration only for an identifiable singular-vector basis. |
| `--brody-fit-method` | `mle`, `cdf_nls` | `mle` | It selects the Brody optimizer. |
| `--number-variance-method` | `sliding`, `monte_carlo` | `sliding` | It selects the interval estimator. |
| `--run-spectral-lesioning` | boolean | false | It enables reversible tranche lesions. |
| `--lesion-tranches` | subset of `top,bulk,bottom` | `top,bulk,bottom` | It selects tranches to remove. |
| `--lesion-fraction` | float in `(0,1]` | `0.05` | It sets the value fraction or energy target. |
| `--lesion-mode` | `count`, `energy` | `count` | It selects count or Frobenius-energy matching. |
| `--save-checkpoints` | boolean | false | It saves model and optimizer states. |

### Scientific method controls

| Flag | Permitted values | Default | Function |
|---|---|---|---|
| `--mp-fit-method` | `analytic_mp`, `thamm_modified_singular`, `kde_bulk_fit`, `lanczos_stieltjes`, `farms_unbiased` | `lanczos_stieltjes` | It selects bulk support and scale. |
| `--unfolding-strategy` | `polynomial_chebyshev`, `spline_monotone`, `gaussian_kernel`, `raw_rank_order` | `spline_monotone` | It selects the smooth staircase method. |
| `--tail-solver` | `clauset_mle`, `hill_estimator`, `fixed_cutoff_mle`, `rank_ordered_mle` | `clauset_mle` | It selects the tail estimator. |
| `--overlap-metric` | `staats_dual_end`, `subspace_principal_angles`, `frobenius_projection` | `staats_dual_end` | It selects subspace alignment. |
| `--overlap-mode` | values for overlap metric | same | It is a compatibility alias. |
| `--aspect-ratio-mode` | `raw`, `farms_normalized`, `farms_unbiased`, `shape_normalized` | `farms_normalized` | It selects spectrum preparation. |
| `--spike-detector` | `tracy_widom_95`, `bbp_transition`, `lanczos_poles` | `lanczos_poles` | It selects the right-spike rule. |
| `--boundary-detector` | `analytic_mp`, `tracy_widom`, `weightwatcher_kde`, `lanczos_stieltjes` | omitted | It jointly selects a compatible fit and detector. |

`farms_unbiased` requires a FARMS aspect mode. Operator-only methods keep a separately labeled raw domain.

### Unfolding, MP, and tail parameters

| Flag | Type | Default | Function |
|---|---|---|---|
| `--polynomial-degree`, `--unfolding-degree` | integer | `15` | They set polynomial degree. |
| `--spline-smoothing` | nonnegative float or omitted | omitted | It sets the spline penalty. |
| `--gaussian-kernel-window` | integer | `15` | It sets the Gaussian neighbor window. |
| `--mp-trim-upper` | float in `[0,0.5)` | `0.1` | It omits an upper fraction during scale fits. |
| `--kde-bandwidth` | positive float or omitted | omitted | It keeps the historical parameter. The fit uses an integrated ECDF objective. |
| `--tail-minimum` | integer at least 2 | `50` | It sets the minimum CSN observations. |
| `--tail-fraction` | float in `(0,1]` | `0.1` | It sets the fixed or rank tail fraction. |

The ECDF objective uses the full sample and includes zero-mass rank offsets.

### FARMS parameters

The released ratio is `Q = sampled_columns / sampled_rows`.

| Flag | Type and permitted values | Default | Function |
|---|---|---|---|
| `--farms-target-aspect-ratio` | positive float | `1.0` | It sets reference `Q`. |
| `--farms-window-size` | row count or omitted | omitted | It sets sampled rows. |
| `--farms-row-windows` | positive integer | `5` | It sets fixed-operation row starts. |
| `--farms-column-windows` | positive integer | `5` | It sets fixed-operation column starts. |
| `--farms-sampling` | `reference_fixed`, `reference_sliding`, `grid`, `random` | `reference_fixed` | It selects the start schedule. |
| `--farms-step-size` | positive integer | `10` | It sets the reference sliding stride. |
| `--farms-normalization` | `canonical`, `raw`, `trace` | `canonical` | It selects normalization for each window. |
| `--farms-orient-tall` | boolean | false | It transposes wide source matrices before sampling. |

`raw` reproduces pooled squared singular values. `canonical` divides by the larger window dimension for comparable MP scales. Constant scale does not change the heavy-tail exponent.

### Lanczos-Stieltjes parameters

| Flag | Type and permitted values | Default | Function |
|---|---|---|---|
| `--lanczos-steps` | integer at least 2 | `50` | It sets the maximum recurrence length. |
| `--lanczos-probes` | positive integer | `3` | It sets independent spherical probes. |
| `--lanczos-tail-window` | integer at least 2 or omitted | omitted | It selects a stable recurrence suffix. |
| `--lanczos-threshold-c` | nonnegative float | `1.0` | It sets `c` in `λ+ + c λ+ N^-δ`. |
| `--lanczos-threshold-delta` | float in `(0,0.5)` | `0.25` | It sets the finite-size exponent. |
| `--lanczos-residue-threshold` | nonnegative float | `0.0` | It sets the minimum VEST residue. |
| `--lanczos-ridge` | nonnegative float | `0.0` | It sets the positive-definiteness ridge. |
| `--lanczos-adaptive` | boolean | true | It enables recurrence-stability stopping. |
| `--lanczos-convergence-tolerance` | positive float or omitted | omitted | It sets the absolute stability tolerance. |
| `--lanczos-sequence-length` | positive integer or omitted | omitted | It sets the stability-window length. |
| `--lanczos-check-interval` | positive integer | `2` | It sets the stability test interval. |
| `--lanczos-pole-method` | `reference_ritz`, `constant_tail` | `reference_ritz` | It selects finite VEST or extended-tail poles. |

## Map from analyses to source

| Analysis | Source | CLI control |
|---|---|---|
| Analytic, Thamm, KDE, FARMS, and Lanczos MP fits | `rmt/mp.py` | `--mp-fit-method` and its parameters |
| Fixed-ratio pooled spectra | `rmt/farms_aspect_ratio.py` | `--aspect-ratio-mode`, `--farms-*` |
| Lanczos support, VEST density, and spikes | `rmt/lanczos_stieltjes.py` | `--spike-detector`, `--lanczos-*` |
| CSN, Hill, fixed-cutoff, and rank tails | `rmt/tail.py` | `--tail-solver`, `--tail-*` |
| Chebyshev, spline, Gaussian, and rank unfolding | `rmt/spacing.py` | `--unfolding-strategy` and its parameters |
| Brody, number variance, and `Δ₃` | `rmt/spacing.py` | Diagnostic and fit controls |
| Dual-end, angle, and projector overlap | `rmt/overlap.py` | `--overlap-metric` and overlap control |
| Stable rank and Porter-Thomas | `rmt/scalars.py` | Scalar diagnostic controls |
| Count or energy lesions | `pipelines/spectral_lesioning.py` | Lesion and SVD controls |
| Device activation covariance | `pipelines/activation_extractor.py` | Covariance and activation controls |

## Output contract

Each execution first creates `.run-owner.json` with exclusive creation. It writes four JSON records through unique atomic temporary files. They are `allocation_manifest.json`, `spectral_method_config.json`, `run_config.json`, and `runtime_environment.json`.

Manifests separate requested and realized token-to-parameter ratios. They convert executable token budgets to integer targets and reject a budget that cannot fund one target. They identify collapsed interventions by realized architecture and target. Collapse rejection applies only to duplicate designs selected by `--cells`. Full-manifest collapse metadata stays available.

Scaling plots group data by requested intervention identity. Each point gives its realized ratio. Training records successful-update targets, attempted targets after masking, forwarded positions, skipped attempts, and the applied learning rate. It records a labeled `6ND` estimate and a forwarded-position compute proxy.

Spectral rows identify separate domains and geometries for ESD, tail, MP fit, detector, and single-operator spacing. The runner does not treat pooled FARMS levels as one operator spacing sequence.

The environment record contains package, interpreter, platform, and requested precision data. During active execution, it also contains CUDA build and device data. It includes available SLURM identifiers, UTC times, elapsed time, and the staged dataset SHA-256 when present.

Execution also writes spectral and lesion CSV files and training JSON Lines. It can write checkpoints. Plots include ESD, tails, spacing, overlap, lesion effects, and scaling trajectories.

Synthetic tests calibrate formulas and software behavior. They do not prove a power law, a selected Brody value, or a lesion order in trained layers. Those values are measured research results.
