# rmt: Random Matrix Theory analysis of LLM weight matrices

`rmt` is an offline tool for spectra from Large Language Model (LLM) weight matrices. A spectrum is the set of singular values or eigenvalues. The tool can examine matrices during or after training. It supports methods from three sources:

1. Thamm, Staats, and Rosenow, PRE 2022: windowed Hill estimates and level statistics.
2. Martin and Mahoney, JMLR 2021: the power-law tail index `α` of the ESD.
3. Staats, Thamm, and Rosenow, arXiv:2410.17770v3: MP fits, small-value differences, activation overlap, and decile ablation.

An empirical spectral distribution (ESD) is the distribution of measured spectrum values. Marchenko-Pastur (MP) theory gives the expected bulk spectrum for specified random matrices.

## Design

The scientific core uses only NumPy and SciPy. It contains `config`, `linalg`, `ensembles`, `mp`, `tail`, `scalars`, `spacing`, and `overlap`. It does not import torch.

The model layer imports torch and Hugging Face only when it accesses models. This layer contains `discovery`, `activations`, `decile`, `perplexity`, `per_matrix`, `pipeline`, `model_io`, `plots`, and `baselines`.

A singular value decomposition (SVD) factors one matrix into singular values and vectors. `linalg.cached_svd` sends large SVD operations to torch and CUDA on an A100. It uses NumPy for other operations. It always returns NumPy arrays. `per_matrix_analysis` does one SVD for each matrix and gives the result to each analysis.

## Installation

```bash
pip install -e .                 # core (numpy, scipy)
pip install -e ".[torch,plots]"  # add torch / transformers / datasets / matplotlib
```

## Operation

Before analysis of a real model, do the analytic self-test:

```bash
python -m rmt --selftest         # exit 0 if the RMT core matches known ground truth
```

`run_rmt.slurm` gives the A100 batch procedure. For a full offline analysis, enter this command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m rmt \
  --models ./models/Llama-3.1-8B \
  --output_dir ./RMT_Local_Outputs \
  --layers 0 4 9 14 19 24 29 \
  --backend auto --do_overlap --do_spacing --do_powerlaw
```

Each model claims a new or empty identity-hashed directory with `.run-owner.json`. Concurrent or later reuse stops before output files can mix. Library callers must supply one safe filename component as `model_tag`.

Real-text analysis stops if it cannot load the matching tokenizer. For synthetic tests, enable text and tokenizer substitutes separately:

```bash
python -m rmt --models ./models/tiny-test --allow_fallback_text --allow_fallback_tokenizer
```

## Frontier model sweep

`frontier_models.json` contains the reviewed model list. The default list contains public causal language models with compatible projection layouts. Qwen3.5 and Qwen3.8 are not in the executable list because their hybrid DeltaNet matrices need a new scientific mapping.

The analysis does not generate text or apply a chat template. Thus, Qwen thinking mode is not active. Perplexity measures the next-token logits for raw WikiText-2.

Keep the pinned compute environment unchanged. Make a small overlay environment for the newer model loaders:

```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate rmt-ml
if [[ ! -x "$HOME/venvs/rmt-frontier/bin/python" ]]; then
    python -m venv --system-site-packages "$HOME/venvs/rmt-frontier"
fi
"$HOME/venvs/rmt-frontier/bin/python" -m pip install \
    -r requirements-frontier.txt
"$HOME/venvs/rmt-frontier/bin/python" -m pip check
```

The sweep downloads one pinned model snapshot at a time. It deletes that model after all selected tasks finish or after a task fails. A restart skips each task that has a matching `complete.json` record.

By default, weight analysis covers all layers in two-layer tasks. Decile perplexity covers the first, middle, and final layers in separate one-layer tasks. Use `--decile-layer-strategy all` only when the extra run time is acceptable.

First, display the plan:

```bash
$HOME/venvs/rmt-frontier/bin/python -m rmt.frontier_sweep --dry-run
```

Set the results and snapshot roots before the full run. Put the snapshot root on a separate filesystem when one is available:

```bash
export RMT_RESULTS_ROOT=/path/to/scratch/rmt-results
export RMT_MODELS_ROOT=/path/to/scratch/rmt-models
export RMT_TMPDIR=/path/to/scratch/rmt-tmp
export RMT_SNAPSHOT_ROOT=/path/to/backup/rmt-snapshots
nohup "$HOME/venvs/rmt-frontier/bin/python" -m rmt.frontier_sweep \
    --gpu-index 1 \
    --layers-per-task 2 \
    --decile-layer-strategy representative \
    --snapshot-interval 3600 \
    > frontier-sweep.log 2>&1 &
echo $!
```

The sweep makes an incremental snapshot each hour and after each task. Unchanged files share storage between snapshots. A snapshot on `/home` protects against a stopped process, but it does not protect against loss of the `/home` filesystem.

The optional 8B entries require BF16 to fit one 23 GiB L4. BF16 reduces the fidelity of the smallest singular values. Enter `--include-optional` only after a calibration run and a storage increase.

For models up to 4B, request at least 100 GiB of scratch storage. For a future Qwen3.8-27B implementation, request 250 GiB because its checkpoint alone is approximately 55.6 GB. That model also needs a multi-GPU loader and a DeltaNet analysis before this tool can evaluate it.

If `/scratch/$USER` is writable, `run_frontier_scratch.slurm` puts models, temporary files, results, caches, and snapshots there. It runs Qwen3-4B-Instruct by default. From the repository root, enter `GPU_INDEX=1 bash launch_batch.sh frontier-scratch`.

The optional Qwen3-8B run uses the same file. Set `RMT_FRONTIER_MODELS=qwen3-8b-exploratory` before the launch command. This run uses BF16 and can still exceed one L4 during a large FP64 factorization.

## Output for each model

For each model `<tag>`, the tool writes these artifacts:

- `<tag>_matrix_metrics.csv` contains one row for each matrix. `EXECUTIVE_SUMMARY.md` gives the full schema.
- `<tag>_summary.json` contains model-level aggregates.
- `<tag>_run_status.json` starts before analysis and finishes after the requested stages.
- `<tag>_run_manifest.json` records the resolved configuration, model source, requested source dtype, and live parameter dtypes before analysis.
- `<tag>_perplexity.json` contains decile-ablation perplexity when `--do_perplexity` is active.
- `<tag>_stable_rank_per_epoch.csv` contains epoch tracking and actual backend and dtype data.
- `<tag>_checkpoint_status.json` records partial checkpoint coverage in permissive mode.
- `<tag>_weightwatcher_status.json` records `complete`, `unavailable`, or `failed` when the user requests WeightWatcher.
- The `<tag>/` directory contains ESD, Hill, nearest-neighbor spacing, heatmap, and summary plots.

A precision decrease makes a permissive execution partial. In strict mode, a partial requested stage causes a nonzero exit. Strict model analysis rejects any missing requested layer.

Checkpoint analysis rejects duplicate probe layers. The first checkpoint defines exact matrix names, roles, shapes, and counts. Later coverage changes stop strict analysis or produce explicit partial status.

## Tests

```bash
pytest -m "not torch"   # pure-science tests (no torch needed)
pytest                  # full suite (adds tiny in-process torch models)
```

## Fixed conventions

The eigenvalue domain is `λ = ν²/N`, with `C = WWᵀ/N`. By default, `N = #cols`. The noise estimate `σ̂` uses the Gavish-Donoho median estimator.

Deciles use ascending order. Decile 1 contains the smallest values. The final decile contains the largest values. `n_deciles` must not exceed the singular-value count of a selected matrix. Lesions process tied parameter aliases one time. Distinct fused Q, K, and V blocks stay separate.

Fused QKV uses a layout identified from the architecture. GPT-2 uses contiguous blocks. GPT-NeoX and Pythia use head-interleaved blocks. The tool rejects an unknown fused layout.

CSN `α` is a density exponent. Hill `α = 1/H` is a survival exponent. For a pure law, `α_csn = α_hill + 1`. For Hill estimates, `α(λ) = α(ν)/2`.

A finite `xmax` uses a normalized bounded-Pareto likelihood and CDF. Random controls use the selected main estimator. They record estimator kind, cutoff, support, and other applicable metadata. The main windowed Hill estimate uses `λ`. It records rank and window support instead of one false cutoff.

Repeated levels keep zero spacing mass. The full spacing sample is normalized to a mean of one. Continuous Brody fits use only positive spacings and state this condition.

Persistent SVD caching requires `--use_svd_cache` and is off by default. Thus, normal library and test calls do not write to the source tree. Cache records include backend, factorization dtype, and degraded status. Array dtypes must agree with metadata. Singular values must use descending order.

A malformed, inconsistent, unsorted, or corrupt cache record is a miss. The tool computes and writes a replacement. A degraded float32 result does not satisfy a float64 contract. Independent decile factors bind to weight content and precision.

ESD histograms have at most 400 bins. This limit also applies to spectra with a very small empirical IQR. Learned-position context limits include positive reserved-position offsets.

Discovery reads metadata before matrix data. It records source dtype, fused-QKV layout, and head overrides. Analysis and covariance capture process one projection at a time. Decile changes make sure of all metadata before they materialize one physical parameter.

Exact projection names such as `multihead_attention`, `pre_norm_attention`, and `embed_projection` stay valid. Exclusion applies only to actual embedding and output-head path components. Unresolved activation eigenspaces and weight singular subspaces are unavailable. The tool restores the original training mode of each borrowed submodule.

Number variance and rigidity use and record `RunConfig.seed`. Only the required analytic gate uses fixed `rmt.config.SEED` calibration.

MP density, cumulative probability, modified-MP fits, and dimensionless scalar summaries do not change with spectral units. MP fitting uses only singular values that the actual SVD precision resolves. A row states when too few values remain.

Random controls use the selected SVD backend and record separate precision data. All vector diagnostics use the actual factorization dtype and both matrix dimensions. Real-only APIs reject complex input before mutation.

The Hill plateau search examines overlapping bands. Number variance and rigidity reject nonpositive or nonfinite interval lengths.

The histogram code applies its bin limit before integer conversion. Each requested optional failure adds a status and reason to the execution status. `rmt_pipeline_glm.py` is a non-executable archive. Use `python -m rmt` as the supported entry point.
