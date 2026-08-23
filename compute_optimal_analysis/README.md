# Spectral-Chinchilla

Spectral-Chinchilla trains compute-allocation-controlled causal Transformers and measures random-matrix signatures, level statistics, activation alignment, and spectral-lesion effects. Phase III calibrates FARMS and Lanczos against the supplied reference repositories and provides a strictly offline HPC workflow.

## Repository map

- `rmt/` is the pure NumPy/SciPy engine. No accelerator-framework import is permitted in this directory.
- `models/` contains the decoder-only Transformer and Chinchilla allocator.
- `pipelines/` contains offline datasets, mixed-precision training, in-device activation covariance, accelerated SVD bridges, lesions, and the centralized CLI.
- `scripts/download_assets.py` is the explicit connected-node prefetch utility.
- `run_experiments.py` is the production experiment and plotting runner.
- `run_hpc.slurm` contains the three paper tracks and the Golden Compute-Optimal RMT track.
- `differences.md` records all source-level and mathematical reconciliation decisions.
- `other_requirements.md` defines the air-gap, wheelhouse, asset, and directory contract.

## Guide to run

### 1. Stage the exact environment

Use CPython 3.10 and the exact package versions in `requirements.txt`. On an internet-connected Linux staging machine matching the cluster ABI:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip download --dest wheelhouse --requirement requirements.txt
python -m pip install --no-index --find-links wheelhouse --requirement requirements.txt
python -m pip check
python -m pip freeze --all > wheelhouse/resolved-environment.txt
sha256sum wheelhouse/*.whl > wheelhouse/SHA256SUMS
```

For an A100 or H100, place the CUDA 12.1 build of `torch==2.4.1` and all dependency wheels in `wheelhouse/`.

### 2. Prefetch assets on a connected node

```bash
python scripts/download_assets.py --assets all --allow-network
```

This resolves the current dataset and tokenizer repositories to immutable commit SHAs, stages `Salesforce/wikitext` configuration `wikitext-103-raw-v1`, the `openai-community/gpt2` tokenizer, raw JSONL splits, a contiguous integer token array, cache directories, a deterministic synthetic corpus, and `data/asset_manifest.json` with source revisions and SHA-256 checksums.

Verify a copied asset tree without network access:

```bash
python scripts/download_assets.py --verify-only
```

For a compact staging calibration:

```bash
python scripts/download_assets.py --assets all --allow-network --max-wikitext-tokens 10000000
```

Synthetic-only generation performs no network access:

```bash
python scripts/download_assets.py --assets synthetic --synthetic-tokens 1000000
```

### 3. Install and submit on the air-gapped cluster

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --no-index --find-links wheelhouse --requirement requirements.txt
mkdir -p logs results
python -m pytest -q
bash -n run_hpc.slurm
sbatch run_hpc.slurm
```

The validation commands are operator-run preflight steps. Source generation does not execute them.

`TRACK` selects a single batch track; default `all` executes all four sequentially:

```bash
sbatch --export=ALL,TRACK=paper1 run_hpc.slurm
sbatch --export=ALL,TRACK=paper2 run_hpc.slurm
sbatch --export=ALL,TRACK=paper3 run_hpc.slurm
sbatch --export=ALL,TRACK=golden run_hpc.slurm
```

The supplied batch header requests one accelerator, matching the portable cluster contract. Sites may override the resource directive, but the current runner assigns one process to one accelerator and does not claim data-parallel scaling. Before launching a track, the batch script verifies every staged asset against `data/asset_manifest.json`.

### 4. Manifest-only configuration

Omit `--execute` to write allocation and resolved-method manifests without loading data or training:

```bash
python run_experiments.py --output-dir results/manifest_check
```

### 5. Standalone paper tracks

These tracks reproduce each paper's analytical protocol on the Spectral-Chinchilla causal-Transformer family. They do not claim byte-identical reproduction of the papers' original checkpoints, datasets, or training histories.

Staats et al. dual-end overlap and singular-value lesions:

```bash
python run_experiments.py --execute --dataset-path data/tokenized/wikitext-103-raw-v1_gpt2.npy --experiment-mode reproduce_paper1 --model-type causal_transformer --aspect-ratio-mode raw --mp-fit-method analytic_mp --spike-detector tracy_widom_95 --overlap-metric staats_dual_end --unfolding-strategy gaussian_kernel --tail-solver rank_ordered_mle --run-spectral-lesioning --lesion-tranches top,bulk,bottom --device cuda --output-dir results/paper1
```

Thamm et al. unfolding, NNSD, Brody, number variance, and rigidity:

```bash
python run_experiments.py --execute --dataset-path data/tokenized/wikitext-103-raw-v1_gpt2.npy --experiment-mode reproduce_paper2 --model-type causal_transformer --aspect-ratio-mode raw --mp-fit-method thamm_modified_singular --spike-detector tracy_widom_95 --unfolding-strategy polynomial_chebyshev --unfolding-degree 15 --compute-spacing-distribution --compute-number-variance --compute-delta3 --device cuda --output-dir results/paper2
```

Martin and Mahoney heavy-tail and empirical MP diagnostics:

```bash
python run_experiments.py --execute --dataset-path data/tokenized/wikitext-103-raw-v1_gpt2.npy --experiment-mode reproduce_paper3 --model-type causal_transformer --aspect-ratio-mode raw --mp-fit-method kde_bulk_fit --spike-detector tracy_widom_95 --tail-solver clauset_mle --compute-stable-rank --device cuda --output-dir results/paper3
```

### 6. Golden Compute-Optimal RMT pipeline

```bash
python run_experiments.py --execute --dataset-path data/tokenized/wikitext-103-raw-v1_gpt2.npy --experiment-mode compute_optimal_rmt --scaling-budget-flops 1e16 --allocation-ratios 0.25 1.00 4.00 --aspect-ratio-mode farms_normalized --farms-sampling reference_fixed --mp-fit-method lanczos_stieltjes --spike-detector lanczos_poles --lanczos-steps 50 --lanczos-adaptive --lanczos-pole-method reference_ritz --unfolding-strategy spline_monotone --tail-solver clauset_mle --overlap-metric staats_dual_end --run-spectral-lesioning --device cuda --amp-dtype bfloat16 --compile-model --svd-backend cuda --covariance-device cuda --output-dir results/golden_compute_optimal
```

## Exhaustive CLI reference

Boolean flags use paired `--flag` and `--no-flag` forms. Defaults shown are parser defaults; `reproduce_paper1` and `compute_optimal_rmt` enable lesions and activation overlap, while the other modes enable their required diagnostics.

### Pipeline control and scaling

| Flag | Type and allowed values | Default | Purpose |
|---|---|---|---|
| `--experiment-mode` | `reproduce_paper1`, `reproduce_paper2`, `reproduce_paper3`, `compute_optimal_rmt`, `custom` | `compute_optimal_rmt` | Workflow dispatcher and run metadata |
| `--model-type` | `causal_transformer` | `causal_transformer` | Model family |
| `--output-dir` | path | `results` | Metrics, manifests, checkpoints, and figures |
| `--execute` | boolean | false | Train and analyze; false writes manifests only |
| `--dataset-path` | local NPY/NPZ path | `data/tokenized/wikitext-103-raw-v1_gpt2.npy` | Offline token stream |
| `--cells` | `all` or comma-separated indices | `all` | Manifest cells to execute |
| `--scaling-budget-flops` | one or more positive floats | `1e16` | Requested budgets in `C≈6ND` |
| `--allocation-ratios` | one or more positive floats | `0.25 1.0 4.0` | Undertrained, optimal, and overtrained token multipliers |
| `--vocab-size` | integer | `50257` | Embedding and output vocabulary |
| `--parameter-cap` | positive float or omitted | omitted | Explicit local model-size cap |
| `--max-train-tokens` | positive float or omitted | omitted | Explicit local token cap |

### Offline loading and training

| Flag | Type and allowed values | Default | Purpose |
|---|---|---|---|
| `--sequence-length` | integer | `256` | Causal training context length |
| `--batch-size` | integer | `8` | Sequences per optimizer step |
| `--dataloader-workers` | nonnegative integer | `4` | Local loader workers |
| `--prefetch-factor` | positive integer | `2` | Batches prefetched per worker |
| `--pin-memory` | boolean | true | Page-lock host batches for asynchronous transfers |
| `--learning-rate` | float | `3e-4` | AdamW peak learning rate |
| `--warmup-steps` | nonnegative integer | `100` | Linear warmup length |
| `--gradient-clip` | positive float | `1.0` | Global gradient-norm ceiling |
| `--log-every` | positive integer | `50` | Validation and log interval |
| `--seed` | integer | `0` | Training, sampling, and diagnostic seed |

### Hardware acceleration

| Flag | Type and allowed values | Default | Purpose |
|---|---|---|---|
| `--device` | `cuda`, `cpu`, `auto` | `cuda` | Training and inference device |
| `--amp-dtype` | `float32`, `float16`, `bfloat16` | `bfloat16` | Autocast dtype |
| `--compile-model` | boolean | true | Enable model compilation for CUDA training |
| `--compile-mode` | `default`, `reduce-overhead`, `max-autotune` | `default` | Compilation strategy |
| `--allow-tf32` | boolean | true | Permit TensorFloat-32 matrix multiplication |
| `--svd-backend` | `auto`, `cuda`, `cpu` | `auto` | SVD execution backend outside the pure engine |
| `--svd-driver` | `gesvdj`, `gesvd`, `gesvda`, `default` | `gesvdj` | CUDA linear-algebra driver |
| `--covariance-device` | `auto`, `cuda`, `cpu` | `auto` | Activation moment-buffer placement |
| `--covariance-dtype` | `float32`, `float64` | `float32` | Activation moment accumulation dtype |

### Diagnostics and lesions

| Flag | Type and allowed values | Default | Purpose |
|---|---|---|---|
| `--activation-batches` | integer | `5` | Batches used for covariance estimates |
| `--validation-batches` | integer | `20` | Batches used for perplexity evaluations |
| `--activation-centered` | boolean | true | Subtract activation means |
| `--compute-activation-overlap` | boolean | true | Compute weight/activation alignment |
| `--compute-spacing-distribution` | boolean | true | Compute unfolded NNSD and Brody beta |
| `--compute-number-variance` | boolean | true | Compute `Σ²(10)` |
| `--compute-stable-rank` | boolean | true | Compute stable rank |
| `--compute-delta3` | boolean | false | Compute Dyson-Mehta `Δ₃(10)` |
| `--compute-porter-thomas` | boolean | false | Run pooled eigenvector calibration |
| `--brody-fit-method` | `mle`, `cdf_nls` | `mle` | Brody optimizer |
| `--number-variance-method` | `sliding`, `monte_carlo` | `sliding` | Number-variance interval estimator |
| `--run-spectral-lesioning` | boolean | false | Run reversible tranche lesions |
| `--lesion-tranches` | comma-separated subset of `top,bulk,bottom` | `top,bulk,bottom` | Tranches to remove |
| `--lesion-fraction` | float in `(0,1]` | `0.05` | Singular-value fraction or energy target |
| `--lesion-mode` | `count`, `energy` | `count` | Match removed count or Frobenius energy |
| `--save-checkpoints` | boolean | false | Save model and optimizer states |

### Scientific method selection

| Flag | Allowed values | Default | Purpose |
|---|---|---|---|
| `--mp-fit-method` | `analytic_mp`, `thamm_modified_singular`, `kde_bulk_fit`, `lanczos_stieltjes`, `farms_unbiased` | `lanczos_stieltjes` | Bulk support and scale method; the Thamm option fits free amplitude/upper singular edge with an empirical lower edge |
| `--unfolding-strategy` | `polynomial_chebyshev`, `spline_monotone`, `gaussian_kernel`, `raw_rank_order` | `spline_monotone` | Smooth staircase method |
| `--tail-solver` | `clauset_mle`, `hill_estimator`, `fixed_cutoff_mle`, `rank_ordered_mle` | `clauset_mle` | Heavy-tail estimator |
| `--overlap-metric` | `staats_dual_end`, `subspace_principal_angles`, `frobenius_projection` | `staats_dual_end` | Subspace alignment |
| `--overlap-mode` | same values as overlap metric | same | Compatibility alias |
| `--aspect-ratio-mode` | `raw`, `farms_normalized`, `farms_unbiased`, `shape_normalized` | `farms_normalized` | Spectrum preprocessing |
| `--spike-detector` | `tracy_widom_95`, `bbp_transition`, `lanczos_poles` | `lanczos_poles` | Right-spike rule |
| `--boundary-detector` | `analytic_mp`, `tracy_widom`, `weightwatcher_kde`, `lanczos_stieltjes` | omitted | Joint compatibility selector |

### Unfolding, MP, and tail tuning

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--polynomial-degree`, `--unfolding-degree` | integer | `15` | Polynomial/Chebyshev degree |
| `--spline-smoothing` | nonnegative float or omitted | omitted | Spline smoothing penalty |
| `--gaussian-kernel-window` | integer | `15` | Adaptive Gaussian neighbor window |
| `--mp-trim-upper` | float in `[0,0.5)` | `0.1` | Upper fraction omitted in scale fitting |
| `--kde-bandwidth` | positive float or omitted | omitted | Triangular KDE bandwidth |
| `--tail-minimum` | integer at least 2 | `50` | Minimum CSN tail observations |
| `--tail-fraction` | float in `(0,1]` | `0.1` | Fixed/rank tail fraction |

### FARMS tuning

The released reference convention is `Q = sampled_columns / sampled_rows`.

| Flag | Type and allowed values | Default | Purpose |
|---|---|---|---|
| `--farms-target-aspect-ratio` | positive float | `1.0` | Fixed reference `Q` |
| `--farms-window-size` | row count or omitted | omitted | Sampled rows |
| `--farms-row-windows` | positive integer | `5` | Fixed-operation row starts |
| `--farms-column-windows` | positive integer | `5` | Fixed-operation column starts |
| `--farms-sampling` | `reference_fixed`, `reference_sliding`, `grid`, `random` | `reference_fixed` | Window-start schedule |
| `--farms-step-size` | positive integer | `10` | Reference sliding stride |
| `--farms-normalization` | `canonical`, `raw`, `trace` | `canonical` | Per-window spectrum normalization |
| `--farms-orient-tall` | boolean | false | Transpose wide source matrices before sampling |

`raw` reproduces the released pooling of squared singular values. `canonical` divides by the larger window dimension so MP scales are comparable. A constant scaling leaves the heavy-tail exponent unchanged.

### Lanczos-Stieltjes tuning

| Flag | Type and allowed values | Default | Purpose |
|---|---|---|---|
| `--lanczos-steps` | integer at least 2 | `50` | Maximum recurrence length |
| `--lanczos-probes` | positive integer | `3` | Independent spherical probes |
| `--lanczos-tail-window` | integer at least 2 or omitted | omitted | Stable recurrence suffix |
| `--lanczos-threshold-c` | nonnegative float | `1.0` | Constant in `λ+ + cN^-δ` |
| `--lanczos-threshold-delta` | float in `(0,0.5)` | `0.25` | Finite-size exponent |
| `--lanczos-residue-threshold` | nonnegative float | `0.0` | Minimum VEST residue |
| `--lanczos-ridge` | nonnegative float | `0.0` | Positive-definiteness ridge |
| `--lanczos-adaptive` | boolean | true | Reference recurrence-stability stopping |
| `--lanczos-convergence-tolerance` | positive float or omitted | omitted | Absolute stability tolerance |
| `--lanczos-sequence-length` | positive integer or omitted | omitted | Stability-window length |
| `--lanczos-check-interval` | positive integer | `2` | Stability-check interval |
| `--lanczos-pole-method` | `reference_ritz`, `constant_tail` | `reference_ritz` | Finite VEST or extended-tail poles |

## Algorithm-to-CLI map

| Analysis family | Implementation | CLI control |
|---|---|---|
| Analytic, Thamm empirical-singular, KDE, FARMS, and Lanczos MP fits | `rmt/mp.py` | `--mp-fit-method` plus tuning flags |
| Fixed-ratio pooled spectra | `rmt/farms_aspect_ratio.py` | `--aspect-ratio-mode`, `--farms-*` |
| Lanczos support, VEST density, and spikes | `rmt/lanczos_stieltjes.py` | `--spike-detector`, `--lanczos-*` |
| CSN, Hill, fixed-cutoff, and rank tails | `rmt/tail.py` | `--tail-solver`, `--tail-*` |
| Chebyshev, spline, Gaussian, and rank unfolding | `rmt/spacing.py` | `--unfolding-strategy` and tuning flags |
| Brody, number variance, and `Δ₃` | `rmt/spacing.py` | diagnostic and fit-method flags |
| Dual-end, angle, and projector overlap | `rmt/overlap.py` | `--overlap-metric`, overlap boolean |
| Stable rank and Porter-Thomas | `rmt/scalars.py` | scalar diagnostic booleans |
| Count/energy lesions | `pipelines/spectral_lesioning.py` | lesion and SVD flags |
| In-device activation covariance | `pipelines/activation_extractor.py` | covariance and activation flags |

## Output contract

Every run writes `allocation_manifest.json`, `spectral_method_config.json`, `run_config.json`, and `runtime_environment.json`. The environment record captures package versions, interpreter/platform details, requested precision, CUDA build/device metadata when execution is active, available SLURM identifiers, the staged dataset SHA-256 when present in the asset manifest, and UTC/elapsed timing. Execution additionally writes spectral and lesion CSV files, training JSON Lines, optional checkpoints, ESD/tail plots, spacing plots, overlap heatmaps, lesion-impact plots, and scaling trajectories.

Synthetic tests calibrate formulas and software behavior; they do not guarantee a trained layer has a power law, a particular Brody parameter, or stronger bottom-than-bulk lesion damage. Those remain measured research outcomes.
