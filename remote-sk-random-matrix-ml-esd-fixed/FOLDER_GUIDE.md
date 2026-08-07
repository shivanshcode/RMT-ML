# Folder guide

Single working copy of the RMT pipeline. Created 2026-08-02 by consolidating
several scattered duplicate folders in `~/Downloads`.

## Code (the current, latest version)

| Path | What it is |
|---|---|
| `rmt/` | The package — 20 modules. This is the **only** copy. Includes the REPORT § correctness fixes (fp32/float64 precision contract, reversible activation capture, config-driven CLI, eigh dedup, plot wiring, device placement). |
| `rmt/plots/`, `rmt/baselines/` | Plotting and WeightWatcher baseline submodules. |
| `tests/` | 14 pytest files, including the end-to-end CLI regression test. |
| `docs/` | `design.md`, `plan.md`, `plan-unittest.md`, `signatures_api.md`. |
| `run_rmt.slurm` | Current SLURM job script (64G mem, OOM notes). |
| `rmt_pipeline_glm.py` | Standalone single-file pipeline (older monolith, kept for reference). |
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

## Notes

- `~/Downloads` is itself a git repo, so changes here show up in its `git status`.
- Full backup of the pre-cleanup state: `~/Downloads/to_port_sk-random-matrix-ml-fixed (2).zip`.
- Everything removed during cleanup went to the Trash, not `rm` — still recoverable.
