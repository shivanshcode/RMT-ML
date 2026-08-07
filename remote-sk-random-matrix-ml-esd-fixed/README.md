# rmt — Random Matrix Theory analysis of LLM weight matrices

A model-agnostic, fully-offline toolkit for analyzing the singular-value /
eigenvalue spectra of LLM weight matrices during and after training. It
implements the methods of three source papers:

1. **Thamm, Staats & Rosenow, PRE 2022** (windowed Hill estimator, level statistics).
2. **Martin & Mahoney, JMLR 2021** (power-law tail index α of the ESD).
3. **Staats, Thamm & Rosenow, arXiv:2410.17770v3** (Marchenko–Pastur fit, small
   singular-value deviations, activation-covariance overlap, decile ablation).

## Design in one paragraph

The scientific core (`config, linalg, ensembles, mp, tail, scalars, spacing,
overlap`) is **pure numpy/scipy** — no torch — so it is fast to unit-test and
trustworthy. The model-I/O layer (`discovery, activations, decile, perplexity,
per_matrix, pipeline, model_io, plots, baselines`) adds torch/HF only where
real models are touched. Every heavy SVD goes through a single dispatcher
(`linalg.cached_svd`) that uses the A100 (torch+cuda) for large matrices and
numpy otherwise, but **always returns numpy**, so downstream math never sees a
GPU tensor. `per_matrix_analysis` computes **exactly one SVD per matrix** and
threads it to every sub-analysis.

## Install

```bash
pip install -e .                 # core (numpy, scipy)
pip install -e ".[torch,plots]"  # add torch / transformers / datasets / matplotlib
```

## Run

Always gate a real run on the analytic self-check:

```bash
python -m rmt --selftest         # exit 0 if the RMT core matches known ground truth
```

Full offline analysis (see `run_rmt.slurm` for the A100 batch version):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m rmt \
  --models ./models/Llama-3.1-8B \
  --output_dir ./RMT_Local_Outputs \
  --layers 0 4 9 14 19 24 29 \
  --backend auto --do_overlap --do_spacing --do_powerlaw
```

## Outputs (per model `<tag>`)

- `<tag>_matrix_metrics.csv` — one row per matrix, full schema (see `EXECUTIVE_SUMMARY.md`).
- `<tag>_summary.json` — model-level aggregates.
- `<tag>_perplexity.json` — decile-ablation perplexity (if `--do_perplexity`).
- `<tag>_stable_rank_per_epoch.csv` — epoch tracking (via `pipeline.analyze_checkpoints`).
- plots under `<tag>/` (ESD, Hill, NN-spacing, heatmaps, summary).

## Tests

```bash
pytest -m "not torch"   # 66 pure-science tests (no torch needed)
pytest                  # full suite (adds tiny in-process torch models)
```

## Conventions (locked)

- Eigenvalue domain `λ = ν²/N`, `C = WWᵀ/N`, default `N = #cols`.
- σ̂ = Gavish–Donoho median estimator.
- Deciles ascending: decile 1 = smallest 10%, decile 10 = largest.
- Fused QKV split into contiguous thirds `Q=[:d], K=[d:2d], V=[2d:]`.
- Exponents: CSN α is the **density** exponent; Hill α = 1/H is the **survival**
  exponent; for a pure law `α_csn = α_hill + 1` and `α(λ) = α(ν)/2` for Hill.
