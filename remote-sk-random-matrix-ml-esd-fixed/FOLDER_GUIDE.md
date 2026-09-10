# Folder guide

Single working copy of the RMT pipeline. Created 2026-08-02 by consolidating
several scattered duplicate folders in `~/Downloads`.

## Code (the current, latest version)

| Path | What it is |
|---|---|
| `rmt/` | The maintained package. Includes the float64 precision contract, corruption-tolerant SVD caching, alias-safe decile SVDs, qualified/mixed-mode-safe activation capture, bounded-tail fitting, fixed-seed selftesting, bounded ESD bins, context-limit handling, config-driven CLI, plot wiring, and backend dispatch. |
| `rmt/plots/`, `rmt/baselines/` | Plotting and WeightWatcher baseline submodules. |
| `tests/` | Pytest suite, including end-to-end CLI and v4 numerical/state-safety regressions. |
| `docs/` | `design.md`, `plan.md`, `plan-unittest.md`, `signatures_api.md`. |
| `run_rmt.slurm` | Current LF-normalized SLURM job script (64G mem, OOM notes). |
| `rmt_pipeline_glm.py` | Non-executable historical monolith. It exits with guidance to use `python -m rmt`. |
| `test_rmt_critical.py`, `_synthetic_models.py` | Root-level test helpers. |

## Model outputs — `RMT_Local_Outputs/`

| Folder | Model | Layers analysed | Notes |
|---|---|---|---|
| `llama-3.1-8b/` | Llama-3.1-8B | 0, 10, 25 | 90 files. Current run. |
| `pythia-160m-fixed-layers-0-4-9/` | pythia-160m | 0, 4, 9 | 80 files. `alpha_mean` 3.818. |
| `pythia-160m-run1-layers-0-5-10/` | pythia-160m | 0, 5, 10 | 80 files. `alpha_mean` 3.985. |
| `legacy/` | bert, llama, pythia-160m | — | 15 files. Older run, different output format. Subfolders start with `.` so use `ls -a`. |

The two pythia folders are **different runs over different layers**, not duplicates —
neither is a superset of the other. Both were kept deliberately.

Each current run folder contains `esd/`, `hill/`, `overlap/`, `spacing/`, `qkv/`
(and `perplexity/` for pythia), plus a `_matrix_metrics.csv` and `_summary.json`.
The files inside still carry the pipeline's auto-generated `._models_<tag>_` prefix;
only the containing folders were renamed for readability.

## Archive — `legacy/`

Earlier generation of the project: `ai_studio_code.py` (33K monolith), its SLURM
script, old job logs, and `legacy/RMT_Local_Outputs/` (113 files covering
alexnet, vgg16, pythia-410m, pythia-160m, bert, Llama-3_1-8B).

Note this is a **different** result set from `RMT_Local_Outputs/legacy/` — same
name, different contents. Both retained.

## Runtime ownership and memory

Production output directories include a digest of the resolved model snapshot and must be absent or empty. An exclusive `.run-owner.json` claim prevents concurrent reuse; the status JSON begins as `running` and is finalized after all requested stages. Requested best-effort baselines receive their own status artifact and make the overall status partial when unavailable. Matrix discovery is metadata-only until each matrix is processed; activation covariance is captured one projection at a time without flattening mixed train/eval state; and decile sweeps mutate/restore one pristine model while requiring non-degraded float64 SVDs, prevalidating metadata, materializing one physical parameter at a time, deduplicating tied aliases, and rejecting empty partition tranches before mutation. Histogram grids are globally bounded and text windows honor learned-position offsets. Synthetic text and synthetic hash-tokenization are separate explicit opt-ins. Tiny-model tests use the single canonical `tests.synthetic_models` fixture, including verified head-interleaved GPT-NeoX/Pythia QKV semantics.

The sibling `compute_optimal_analysis` project has an incompatible package with the same `rmt` import name. Run each project from its own root in a separate Python process and never combine their roots on `PYTHONPATH`.

## Notes

- `~/Downloads` is itself a git repo, so changes here show up in its `git status`.
- Full backup of the pre-cleanup state: `~/Downloads/to_port_sk-random-matrix-ml-fixed (2).zip`.
- Everything removed during cleanup went to the Trash, not `rm` — still recoverable.
