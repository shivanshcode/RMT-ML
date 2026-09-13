# Folder guide

This directory is one working copy of the RMT pipeline. It was made on 2026-08-02 from duplicate folders in `~/Downloads`. Some historical output paths in this guide are not in this repository checkout.

## Current code

| Path | Description |
|---|---|
| `rmt/` | This maintained package includes the float64 contract, SVD cache, decile operations, activation capture, tail fits, self-tests, ESD limits, CLI, plots, and backends. |
| `rmt/plots/`, `rmt/baselines/` | These directories contain plots and the WeightWatcher baseline. |
| `tests/` | This directory contains pytest tests, CLI tests, and v4 numerical and state-safety tests. |
| `docs/` | The historical working copy contained `design.md`, `plan.md`, `plan-unittest.md`, and `signatures_api.md`. This repository checkout does not contain `docs/`. |
| `run_rmt.slurm` | This is the current LF SLURM script. It requests 64 GiB and contains OOM notes. |
| `rmt_pipeline_glm.py` | This is a historical monolith that does not operate. It directs users to `python -m rmt`. |
| `test_rmt_critical.py`, `tests/synthetic_models.py` | The first file is a root test. The second file contains shared synthetic models. |

## Historical model output in `RMT_Local_Outputs/`

The inventory that follows describes the consolidated working copy. This repository checkout does not contain `RMT_Local_Outputs/`.

| Directory | Model | Analyzed layers | Notes |
|---|---|---|---|
| `llama-3.1-8b/` | Llama-3.1-8B | 0, 10, 25 | It contains 90 files from the current execution. |
| `pythia-160m-fixed-layers-0-4-9/` | pythia-160m | 0, 4, 9 | It contains 80 files and `alpha_mean` 3.818. |
| `pythia-160m-run1-layers-0-5-10/` | pythia-160m | 0, 5, 10 | It contains 80 files and `alpha_mean` 3.985. |
| `legacy/` | bert, llama, pythia-160m | Not applicable | It contains 15 older files with another output format. Use `ls -a` for directories that start with `.`. |

The two Pythia directories contain different executions with different layers. Neither directory contains all data from the other directory. The consolidation kept both directories.

Each current execution directory contains `esd/`, `hill/`, `overlap/`, `spacing/`, and `qkv/`. The Pythia directory also contains `perplexity/`. Each directory contains `_matrix_metrics.csv` and `_summary.json`. Files keep the generated `._models_<tag>_` prefix. Only the names of the parent directories changed.

## Historical archive in `legacy/`

The consolidated working copy had `legacy/`, but this repository checkout does not. `legacy/` contained an earlier generation of the project. It contains the 33K-line `ai_studio_code.py` monolith, its SLURM script, and old job logs. It also contains 113 files in `legacy/RMT_Local_Outputs/`. Those files cover alexnet, vgg16, pythia-410m, pythia-160m, bert, and Llama-3_1-8B.

This result set differs from `RMT_Local_Outputs/legacy/`. Both sets have the name `legacy`, but their contents differ. The consolidation kept both sets.

## Output ownership and memory

A production output directory includes a digest of the resolved model snapshot. It must not exist, or it must be empty. An exclusive `.run-owner.json` claim prevents concurrent use.

An execution manifest records the resolved configuration and precision before analysis. The status file starts with `running`. The pipeline writes its final value after all requested stages.

Checkpoint CSV files use a separate exclusive claim and atomic replacement. Duplicate probes stop before loading. The first checkpoint defines exact matrix membership and geometry. In permissive mode, missing or changed coverage gives partial status.

A library `model_tag` must contain one safe filename component. It cannot select a path outside the owned output directory.

Discovery reads only metadata until it processes one matrix. Activation capture processes one projection and preserves all train and evaluation modes. Decile sweeps change and restore one model. They require non-degraded float64 SVD results.

Before mutation, a decile sweep makes sure of all metadata. It materializes one physical parameter at a time and removes tied aliases. It rejects an empty partition tranche. Histogram grids have a fixed bound. Text windows obey learned-position offsets.

Synthetic text and synthetic hash tokenization require separate options. Persistent SVD caching is off by default. Small-model tests use the single `tests.synthetic_models` fixture. The fixture includes head-interleaved GPT-NeoX and Pythia QKV behavior.

The sibling `compute_optimal_analysis` project has another, incompatible package named `rmt`. Operate each project from its own root and in a separate process. Do not combine their roots on `PYTHONPATH`.

## Historical notes

`~/Downloads` is a Git repository, so its `git status` includes changes in this directory. The archive `~/Downloads/to_port_sk-random-matrix-ml-fixed (2).zip` contains the state before consolidation. Removed items went to the Trash and not through `rm`, so recovery was possible.
