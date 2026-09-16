# rmt

> Random Matrix Theory analysis of LLM weight matrices — Marchenko–Pastur bulk, heavy-tailed
> exponents, level-spacing universality and Porter–Thomas eigenvector tests, scored against
> matched-shape Monte-Carlo null bands.

![version](https://img.shields.io/badge/version-1.0.0-blue)
![python](https://img.shields.io/badge/python-%E2%89%A53.9-blue)
![tests](https://img.shields.io/badge/tests-294%20collected%20%7C%20263%20offline%20%7C%20293%20pass%201%20skip-brightgreen)
![offline](https://img.shields.io/badge/network-offline%20by%20construction-informational)
![license](https://img.shields.io/badge/license-not%20specified-lightgrey)

`rmt` takes **one SVD per weight matrix** (`rmt/linalg.py:44`) and threads that single
factorization through every spectral analysis, producing one flat CSV row per matrix
(`rmt/per_matrix.py:29`). Deviations are reported as z-scores against null bands Monte-Carlo'd at
the *same matrix shape* (`rmt/nulls.py:291`), never against an asymptotic constant.

Validated on synthetic ensembles with known laws (GOE, GUE, GSE, Poisson, COE, CUE, CSE, Wishart,
spiked Wishart) and run end-to-end on **meta-llama/Llama-3.1-8B**. See
[Analysis](#analysis-benchmark-and-test-results) for what the numbers actually support — including
[§A.6](#a6-where-this-may-be-wrong), which lists the places a referee should push.

> **Note** Every accuracy, power and control figure in this
> repository was measured at Chebyshev unfolding **degree 7** (the `RunConfig` default,
> `rmt/config.py:86`). The committed Llama-3.1-8B results and the null bands they are scored
> against were produced at **degree 15** (`run_rmt_calibrated_v4.slurm:110`). The two
> configurations are not interchangeable: raising the degree from 7 to 15 moves the Σ² null mean by
> up to **12 %** (2.3 band sd) on the production shapes, while leaving Δ₃ and ⟨r⟩ essentially
> unchanged. Details and the measurement are in [§A.1](#a1-what-was-tested-with-what-and-at-which-settings).

---

## Features

- **Single-SVD pipeline** — every statistic derives from one `cached_svd()` per matrix
  (`rmt/linalg.py:44`; GPU via torch, or numpy, always returning numpy float64), so nothing is
  recomputed per statistic.
- **Marchenko–Pastur layer** — density, CDF, edges and median in both the singular-value and
  eigenvalue (λ = ν²/N) domains (`rmt/mp.py:42,63,68,89,102–143`), with the Gavish–Donoho median σ
  estimator (`rmt/mp.py:143`) and a USVT hard threshold (`rmt/mp.py:196`).
- **Heavy-tail exponents** — CSN (density, `rmt/tail.py:20`) and Hill (survival, `rmt/tail.py:72`)
  exponents plus a windowed-Hill plateau diagnostic (`rmt/tail.py:134`) that refuses to call an MP
  bulk a power law.
- **Level-spacing universality** — Atas ⟨r⟩ (`rmt/spacing.py:102`), spectral unfolding (global
  Chebyshev fit or Thamm's adaptive Gaussian kernel, auto-selected with a cross-check gate,
  `rmt/spacing.py:639,693,723`), NN-spacing KS vs Wigner-GOE/Poisson (`rmt/spacing.py:809`), Brody β
  (`rmt/spacing.py:866`), number variance Σ²(L) (`rmt/spacing.py:1004`) and Dyson–Mehta rigidity
  Δ₃(L) (`rmt/spacing.py:1080`).
- **Exact finite-L references** — `rmt/reference.py:107,127` supply the exact Σ²/Δ₃ laws for
  β = 1, 2, 4 and Poisson from the two-level cluster function, so small-L comparisons are not made
  against an L→∞ asymptote (which is kept separately at `rmt/reference.py:151,165`).
- **Calibrated Porter–Thomas test** — Monte-Carlo null for the KS statistic at each dimension
  (`rmt/porter_thomas.py:83,121,155`), so the p-value is Uniform(0,1) under the null instead of a
  fixed distance threshold.
- **Matched-shape null bands** — `rmt/nulls.py:291` Monte-Carlos the pipeline's own statistic
  vector (`rmt/nulls.py:217`) on iid Wishart matrices of the same shape; z-scores
  (`rmt/nulls.py:343`) are thresholded with a Student-t Bonferroni critical value
  (`rmt/nulls.py:359`), not `norm.ppf`.
- **Destructive controls** — entry/row/column shuffles and a Gaussian match
  (`rmt/controls.py:65,78,92,130`), with a decision ladder (`rmt/controls.py:308`) separating
  "learned structure" from heavy tails or heteroscedasticity.
- **Offline by construction** — `OfflineGuard` (`rmt/config.py:149`) asserts the HF offline
  environment; no network use at run time.
- **Model-agnostic discovery** — walks `named_modules()`, classifies Linear/Conv1D weights into
  Q/K/V/O/G/U/D roles, skips embeddings and the LM head, and splits fused QKV
  (`rmt/discovery.py:263,298`); the registry covers llama/mistral/mixtral, gpt_neox/pythia,
  qwen2/qwen3, bert/roberta, gpt2, plus a never-None generic fallback (`rmt/discovery.py:133–138`).

---

## Installation

`requirements.txt` is a **conda explicit spec of the exact environment the committed results were
produced in** — `name=version=build` triples, not pip requirement lines, so `pip install -r` will
not read it. Recreate it with conda:

```bash
git clone <this-repo> && cd to-port-remote-shree-sk-random-matrix-ml-rmt-stats-esd-fixed

conda create --name rmt_ml_env --file requirements.txt
conda activate rmt_ml_env

pip install -e .            # installs the `rmt` package and the `rmt` console script
```

The spec is **platform-locked to `linux-64`** (its header says so) and pins Python 3.10.18,
**numpy 2.1.2 and scipy 1.15.3** — which is exactly what the job logs record
(`rmt_bands_436776.log:1–3`), so the analysis environment is now reproducible from the repository.
It also pins torch 2.7.1+cu118 with the `nvidia-*-cu11` runtime, transformers 5.12.1, tokenizers
0.22.2, huggingface-hub 1.19.0, accelerate 1.14.0, datasets 5.0.0, safetensors 0.5.3,
sentencepiece 0.2.1, matplotlib 3.10.3, pandas 2.3.3, seaborn 0.13.2, and both optional baselines
(weightwatcher 0.7.7, powerlaw 2.0.0). **`pytest` 9.1.1 is included**, so the SLURM gate needs no
extra install. Both SLURM scripts activate this env by name (`run_stage2_bands.slurm:42`).

On another platform, or to run only the scientific core, the packaged dependency set is numpy +
scipy alone (`pyproject.toml` `dependencies`) — torch and transformers are needed solely to load a
model. `Generator` RNG streams are stable across numpy versions, so a different numpy will
reproduce the Monte-Carlo draws; it will not necessarily reproduce the last digits of an SVD, which
come from whichever LAPACK the build links. (For reference, the figures in this README were
re-verified locally under numpy 1.24.3 / scipy 1.10.1 / Python 3.11.4 and agreed with the committed
CSVs to the digits quoted.)

Optional extras declared in `pyproject.toml`, for a minimal install that does not use the pinned
environment:

```bash
pip install -e ".[torch]"       # torch>=2.0, transformers, datasets
pip install -e ".[plots]"       # matplotlib, pandas
pip install -e ".[baselines]"   # weightwatcher, powerlaw
pip install -e ".[dev]"         # pytest
```

Fetch the model once on a machine with network access (the Llama 3.1 licence must be accepted):

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir models/Llama-3.1-8B
```

Verify the install:

```bash
python -m rmt --selftest                   # 11 analytic checks, expect "selftest overall: PASS"
python -m pytest tests/ -q -m "not torch"  # 263 passed, 31 deselected
python -m pytest tests/ -q                 # 294 collected: 293 passed, 1 skipped (needs torch)
```

The 11 self-test checks are, in order (`rmt/selftest.py:22–77`): MP pdf normalisation, σ recovery
on Wishart, CSN α on Pareto(3), windowed-Hill Pareto-vs-MP discrimination, ⟨r⟩ on GOE, ⟨r⟩ on
Poisson, NN-spacing GOE-vs-Poisson ordering, unfolding health, the Δ₃ GOE log-law, Porter–Thomas
p-value uniformity, and fused-QKV feature-matrix key resolution.

---

## Usage

### Command line

```bash
# fast analytic self-check, no model required
python -m rmt --selftest

# full per-model analysis — the exact recorded Llama-3.1-8B configuration
# (run_rmt_calibrated_v4.slurm:217–234)
python -m rmt \
  --models meta-llama/Llama-3.1-8B \
  --model_path ./models/Llama-3.1-8B \
  --output_dir rmt_results \
  --dtype fp32 --backend auto \
  --unfold_method auto --unfold_deg 15 --unfold_win 15 \
  --spacing_domain sval --spacing_bulk_mode auto \
  --sigma2_n_windows 200000 --delta3_L 5 10 50 --sigma2_L 5 10 20 \
  --do_spacing --do_porter_thomas --do_brody --do_powerlaw \
  --no-do_overlap --no-do_perplexity \
  --layers 0 8 16 24 31 --seed 0
```

An installed copy also exposes the console script `rmt` (equivalent to `python -m rmt`).
Output lands in `<output_dir>/<model_tag>/`: `<tag>_matrix_metrics.csv` (133 columns),
`<tag>_summary.json`, the `<tag>_summary.png` figure, and PNG directories
(`esd/ hill/ spacing/ rigidity/ porter_thomas/ porter_thomas_decile/ qkv/`).

### Calibration and scoring

These are the commands that actually produced the committed `cal/` artefacts, in the order they
were run, transcribed from the shell history of the HPC session
(`~/remote-shree-sk-random-matrix-ml-rmt-stats-esd-fixed`, conda env `rmt_ml_env`).

```bash
# --- 2026-08-29, interactive on the login node ------------------------------
python check_repo.py                       # md5 gate: the tree matches the intended overlay
python -m rmt --selftest                   # 11 analytic checks
python -m pytest tests/ -q -m "not torch"  # 263 passed, 31 deselected

# GATE 0b: the 3-way estimator benchmark. NOTE the OLD_REPO path actually used.
OLD_REPO=../sk-random-matrix-ml-rmt-stats-esd-fixed NEW_REPO=. FIX_REPO=. \
    python benchmark3.py 6 --out cal/benchmark_results_3way.csv

python rmt_null_calibration.py ensembles  --reps 8 --out cal
python rmt_null_calibration.py invariance --shape 1024x1024 --out cal
python rmt_null_calibration.py bulkfrac   --shape 4096x4096 \
    --fracs 0.6 0.7 0.8 1.0 --reps 20 --out cal

# --- 2026-09-01, interactive ------------------------------------------------
python rmt_null_calibration.py power --shape 4096x4096 --reps 30 --out cal
```

The expensive stage was submitted as a batch job rather than run inline
(`run_stage2_bands.slurm:79–86`). The committed bands cover all **four** Llama shapes, and
`--unfold-deg` must match the pipeline's `--unfold_deg` or the band and the observation are
different estimators:

```bash
python rmt_null_calibration.py nulls \
    --shapes 4096x4096 1024x4096 14336x4096 4096x14336 \
    --reps 40 \
    --bulk-frac 1.0 \
    --unfold-method auto \
    --unfold-deg 15 \
    --porter-thomas --pt-vectors 200 \
    --out cal
```

`--pt-samples` is not passed, so the PT band uses the default of 3000 Monte-Carlo draws against the
pipeline's 5000 (`rmt_null_calibration.py:631`) — see [§A.4.10](#a4-weaknesses-where-the-suite-is-fragile-or-blind).

The scoring and control stages are driven by `run_rmt_calibrated_v4.slurm` (stages 7 and 8); their
standalone forms are:

```bash
python rmt_null_calibration.py score --csv rmt_results/<tag>/<tag>_matrix_metrics.csv \
    --bands cal/null_bands.csv --out cal                                # the table you report
python rmt_null_calibration.py controls --npy weights_npy/<matrix>.npy --reps 30 \
    --bulk-frac 1.0 --out cal
```

> The `power`, `ensembles` and `controls` subcommands have **no `--unfold-deg` flag**
> (`rmt_null_calibration.py:604–668`), so they are hard-wired to the `spacing_statistics` default of
> degree 7 (`rmt/nulls.py:218`). Only `nulls` can be run at degree 15. This is the provenance split
> described in [§A.1](#a1-what-was-tested-with-what-and-at-which-settings).

On a SLURM cluster the two heavy stages are scripted. The scripts support a dependency chain,
but the recorded runs were **two independent submissions** on different days:

```bash
sbatch run_stage2_bands.slurm          # null bands   (2026-08-29; jobs 426284/426285, re-run 436776)
sbatch run_rmt_calibrated_v4.slurm     # everything   (2026-09-04; jobs 436779/436780)

# the chained form the scripts are written for, if you start from nothing:
jid=$(sbatch --parsable run_stage2_bands.slurm)
sbatch --dependency=afterok:$jid run_rmt_calibrated_v4.slurm
```

`run_stage2_bands.slurm` requests `--partition=gpulong --gres=gpu:1 --mem=32G --time=8:00:00` and
**does not use the GPU** — the whole stage is numpy/scipy, and the `--gres` line is there only
because the partition requires it (`run_stage2_bands.slurm:19–23`). A CPU queue will schedule
sooner. (Its own comment block still says 64G while the directive says 32G; the directive is what
ran.)

Measured wall time for the committed artefacts (`rmt_bands_436776.log`,
`RMT_calibrated_v4_436780.log`): the four null bands took 1019 s + 39 s + 1457 s + 1736 s ≈ **1.2 h**
of band-building (the script requests 8 h; `run_rmt_calibrated_v4.slurm:120` budgets 3–5 h for a
cold build on a slower node), and the v4 job finished **≈6.3 h** after the bands job.

### Library

```python
import numpy as np
from rmt import cached_svd, RunConfig
from rmt import mp as MP, spacing as SP, tail as TAIL
from rmt import nulls as NU

W = np.random.default_rng(0).standard_normal((512, 256)) / np.sqrt(512)

# one SVD, reused everywhere
svd = cached_svd(W)
s = svd.s

# Marchenko-Pastur bulk
sigma = MP.estimate_sigma_gd_median(s=s, n=512, m=256)
lo, hi = MP.mp_bounds(512, 256, sigma)
n_outliers = int((s > hi).sum())

# heavy tail (CSN density exponent)
alpha = TAIL.fit_powerlaw_csn(s ** 2)["alpha"]

# level statistics on the unfolded spectrum
xi, branch = SP.unfold_auto(np.sort(s), deg=7)
r_stat = SP.r_statistic(np.sort(s))
d3 = SP.delta3(None, 10, unfolded=xi)
s2 = SP.sigma2(None, 10, unfolded=xi, method=branch)

# score against a matched-shape null band
band = NU.null_band((512, 256), n_reps=8, seed=0)
z = NU.zscore(d3, band["delta3_L10"])
thr = NU.critical_value(n_reps=8, n_tests=11)   # Student-t Bonferroni, not norm.ppf
```

Whole-model analysis from Python:

```python
from rmt.model_io import load_model, load_tokenizer
from rmt.pipeline import analyze_one_model

model = load_model("meta-llama/Llama-3.1-8B", model_path="./models/Llama-3.1-8B", dtype="fp32")
csv_path, rows = analyze_one_model(
    model, "meta-llama_Llama-3.1-8B", "rmt_results/meta-llama_Llama-3.1-8B",
    tokenizer=load_tokenizer("meta-llama/Llama-3.1-8B", model_path="./models/Llama-3.1-8B"),
    layers=[0, 8, 16, 24, 31], dtype="fp32", do_spacing=True, do_porter_thomas=True,
)
```

---

## Configuration

Every field of the `RunConfig` dataclass (`rmt/config.py:41`) becomes a CLI flag automatically
(`rmt/cli.py:51–95`), so the parser can never drift from the config. Booleans get `--flag` /
`--no-flag` pairs via `argparse.BooleanOptionalAction` (`rmt/cli.py:81`).

| Flag | Default | Meaning | Defined at |
|---|---|---|---|
| `--models` | `["meta-llama/Llama-3.1-8B"]` | Model tags to analyse | `rmt/config.py:45` |
| `--model_path` | `None` | Local snapshot directory | `rmt/config.py:46` |
| `--output_dir` | `./rmt_results` | Output root | `rmt/config.py:47` |
| `--layers` | `[]` (all) | Layer indices to analyse | `rmt/config.py:48` |
| `--dtype` | `fp32` | `fp16` \| `bf16` \| `fp32`; mixing dtypes changes the spectrum | `rmt/config.py:51` |
| `--backend` | `auto` | `auto` \| `numpy` \| `torch` SVD dispatch | `rmt/config.py:53` |
| `--gpu_svd_min_dim` | `1024` | `auto` goes to GPU above this `max(n,m)` | `rmt/config.py:54` |
| `--sigma_estimator` | `gd_median` | Only Gavish–Donoho median is dispatched | `rmt/config.py:56`, `rmt/cli.py:29–32` |
| `--alpha_estimator` | `all` | `csn` \| `hill` \| `hill_windowed` \| `all` | `rmt/config.py:69` |
| `--unfold_method` | `auto` | `auto` \| `cheb` \| `gauss` \| `poly` | `rmt/config.py:84` |
| `--unfold_deg` / `--unfold_win` | `7` / `15` | Global-fit degree; local-kernel window | `rmt/config.py:86,85` |
| `--spacing_domain` | `sval` | Unfold singular values or eigenvalues | `rmt/config.py:87` |
| `--spacing_bulk_mode` | `center` | `auto` \| `center` \| `mp` \| `none` | `rmt/config.py:100`, `rmt/cli.py:38` |
| `--spacing_bulk_center_frac` | `0.7` | Centred fraction kept by `center` | `rmt/config.py:101` |
| `--delta3_L` / `--sigma2_L` | `5 10 50` / `5 10 20` | L grids written to the CSV | `rmt/config.py:115,116` |
| `--sigma2_n_windows` | `200000` | Fixed window count (0 restores the adaptive rule) | `rmt/config.py:110` |
| `--do_brody` / `--brody_bootstrap` | `True` / `200` | Brody β and its bootstrap error | `rmt/config.py:111,112` |
| `--do_porter_thomas` / `--pt_n_samples` / `--pt_alpha` | `True` / `5000` / `0.05` | PT test | `rmt/config.py:75,77,78` |
| `--pt_max_vectors` | `None` (all) | Cap PT cost on wide models | `rmt/config.py:79` |
| `--do_overlap` / `--do_perplexity` / `--do_ipr` | `True` / `False` / `True` | Optional blocks | `rmt/config.py:59,118,74` |
| `--text_path` | `./wikitext-2-raw/wiki.test.raw` | Local text for perplexity/activations | `rmt/config.py:126` |
| `--offline` | `True` | Engage `OfflineGuard` | `rmt/config.py:141` |
| `--seed` | `0` | RNG discipline for Σ², Δ₃, Brody, PT | `rmt/config.py:143` |

Note the default `--sigma2_L 5 10 20` never emits Σ²(50); the Σ²(50) numbers discussed in
[§A.4](#a4-weaknesses-where-the-suite-is-fragile-or-blind) exist only in the benchmark CSVs.
`--delta3_L 5 10 50` *does* emit Δ₃(50).

Environment variables honoured by the SLURM scripts and `OfflineGuard` (`rmt/config.py:152–156`):

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
MODEL_PATH=./models/Llama-3.1-8B   # run_rmt_calibrated_v4.slurm:104
RUN_STAGE2=auto|always|never       # skip the 3-5 h band build (slurm:120-122)
OLD_REPO=... NEW_REPO=... FIX_REPO=...   # benchmark3.py:30-32 comparison trees
```

---

## API

Importing `rmt` pulls in only the numpy/scipy core; torch-dependent modules load lazily. Verified
torch-free: `rmt/config.py`, `mp`, `tail`, `scalars`, `spacing`, `nulls`, `reference`,
`porter_thomas`, `controls`, `ensembles`, `overlap`.

```python
__all__ = ["config", "ensembles", "mp", "tail", "scalars", "spacing", "overlap", "linalg",
           "SVDResult", "cached_svd", "RunConfig", "TOL", "OfflineGuard", "get_logger",
           "__version__"]
```

| Function | Module (file:line) | Purpose |
|---|---|---|
| `cached_svd(weight, full_matrices=False, *, backend="auto", gpu_min_dim=1024)` | `rmt/linalg.py:44` | The single SVD entry point; always returns numpy float64 |
| `discover_weight_matrices(model, layer_indices=None, *, spec=None, dtype="float64", num_heads=None)` | `rmt/discovery.py:298` | One `MatrixRecord` per analysable 2-D weight |
| `per_matrix_analysis(record, fm_dict=None, *, cfg=None, ...)` | `rmt/per_matrix.py:29` | One matrix → one flat CSV row |
| `analyze_one_model(model, model_tag, output_dir, *, tokenizer=None, **cfg_flags)` | `rmt/pipeline.py:28` | Whole model → `(csv_path, rows)` |
| `analyze_checkpoints(checkpoint_loader, checkpoint_fracs, probe_layers, output_dir, model_tag, **cfg_flags)` | `rmt/pipeline.py:98` | Stable rank across training checkpoints |
| `mp_bounds(n, m, sigma)` / `mp_pdf` / `mp_median` / `estimate_sigma_gd_median` / `usvt_hard_threshold` | `rmt/mp.py:63,42,89,143,196` | Marchenko–Pastur layer |
| `fit_powerlaw_csn(values, ...)` / `hill_estimator` / `hill_plateau` | `rmt/tail.py:20,72,134` | Tail exponents |
| `unfold(levels, deg=7, *, method="auto", win_size=15)` / `unfold_auto` | `rmt/spacing.py:693,723` | Spectral unfolding |
| `r_statistic` / `nn_spacing_ks` / `brody_beta` / `sigma2` / `delta3` | `rmt/spacing.py:102,809,866,1004,1080` | Level statistics |
| `sigma2_exact(L, beta=1)` / `delta3_exact(L, beta=1)` | `rmt/reference.py:107,127` | Exact finite-L laws |
| `sigma2_asymptote(L, beta=1)` / `delta3_asymptote(L, beta=1)` | `rmt/reference.py:151,165` | The L→∞ Dyson–Mehta forms, kept only for comparison |
| `spacing_statistics(levels, *, unfold_method="auto", unfold_deg=7, ...)` | `rmt/nulls.py:217` | The one statistic vector shared by data and null |
| `null_band(shape, *, n_reps=30, kind="wishart", seed=0, bulk_frac=1.0, **stat_kw)` | `rmt/nulls.py:291` | Matched-shape Monte-Carlo band |
| `zscore(value, band_entry)` / `critical_value(n_reps, n_tests=1, alpha=0.05)` | `rmt/nulls.py:343,359` | Scoring and the correct threshold |
| `ks_pvalue_mc(levels, shape, ...)` | `rmt/nulls.py:416` | Parametric-bootstrap KS p-value |
| `circular_levels(n, beta=1, rng=None, unfolded=True)` | `rmt/nulls.py:148` | COE/CUE/CSE eigenphases, exactly unfolded |
| `porter_thomas_summary(Vh, *, n_vectors=None, n_samples=5000, alpha=0.05, ...)` | `rmt/porter_thomas.py:155` | PT p-values and `pt_frac_random` |
| `control_suite(W, *, band=None, bulk_mode="center", ...)` / `interpret` | `rmt/controls.py:266,308` | Destructive controls and verdicts |
| `run(seed=1234, verbose=True)` | `rmt/selftest.py:20` | The 11-check analytic gate |

---

## Provenance of the committed artefacts

| File | Produced by | n / reps | Unfolding deg | Bulk selection |
|---|---|---|---|---|
| `benchmark_results.csv` | `benchmark.py` (legacy OLD-vs-NEW) | 6 seeds nominal | 7 | none, except MP trim on the spiked case (`benchmark.py:152`) |
| `benchmark_results_3way.csv` | `benchmark3.py`, external `NEW_REPO`/`FIX_REPO` | 6 seeds | 7 | `strip_edge_outliers` for FIXED (`benchmark3.py:131`) |
| `cal/benchmark_results_3way.csv` | `benchmark3.py` with `NEW_REPO=FIX_REPO=.`; run interactively 2026-08-29 and re-generated by v4 GATE 0b 2026-09-04 (`run_rmt_calibrated_v4.slurm:135`). Only the committed copy exists | 6 seeds | 7 | as above |
| `cal/ensemble_benchmark.csv` | `rmt_null_calibration.py ensembles --reps 8 --out cal` (2026-08-29; re-run by v4 stage 1) | 8 (GSE 4, circular 4) | 7 | none |
| `cal/null_bands.csv` | `sbatch run_stage2_bands.slurm` → `nulls … --unfold-deg 15` (2026-08-29, re-run 2026-09-04) | 40 (PT band 8) | **15** | `bulk_frac 1.0` |
| `cal/null_bands_deg7.csv` | same, `--unfold-deg 7` | 40 | 7 | `bulk_frac 1.0` |
| `cal/power_table.csv` | `rmt_null_calibration.py power --shape 4096x4096 --reps 30 --out cal` (2026-09-01; re-run by v4 stage 3) | 30-rep band, 200 KS nulls | 7 | `bulk_frac 1.0` |
| `cal/invariance.csv` | `rmt_null_calibration.py invariance --shape 1024x1024 --out cal` (2026-08-29) | 1 | 7 | — |
| `cal/bulk_frac_sweep.csv` | `rmt_null_calibration.py bulkfrac --shape 4096x4096 --fracs 0.6 0.7 0.8 1.0 --reps 20 --out cal` (2026-08-29) | 20 | 7 | 0.6/0.7/0.8/1.0 |
| `cal/controls.csv` | v4 stage 8 `controls` — **last matrix only** | 30-rep band | 7 | `bulk_frac 1.0` (centre cut = no-op) |
| `rmt_results/.../_matrix_metrics.csv` | v4 stage 6, `python -m rmt --unfold_deg 15 --spacing_bulk_mode auto` | 1 | **15** | `auto` = `strip_edge_outliers`, 2–63 levels dropped |
| `cal/zscores.csv` | v4 stage 7, `rmt_null_calibration.py score` | — | — | measurement deg 15 vs deg-15 band |
| `calc_results/*` | 400×400 demo, 20 reps, `bulk_frac 0.7` | 20 | 7 | centre 0.7 |
| *environment for all of the above* | conda env `rmt_ml_env`, pinned in `requirements.txt`: Python 3.10.18, numpy 2.1.2, scipy 1.15.3, torch 2.7.1+cu118, linux-64 | — | — | — |

`cal/controls.csv` retains only the last of the **15** matrices the controls stage actually ran
(`RMT_calibrated_v4_436780.log:349–768`); the other 14 survive only in the log.

**Three gaps between this checkout and the tree that produced the artefacts.** These do not
invalidate the numbers, but a reader trying to reproduce them should know about them up front.

1. `python check_repo.py` — the repository's own md5 gate — **passes on this checkout (22/22), but
   its pins were refreshed after the fact.** Between the gate last passing on 2026-08-29 and the
   re-pin, `rmt_null_calibration.py` grew by 542 bytes and `tests/test_nulls.py` and
   `tests/test_spacing.py` by 248 and 242, and the pinned `run_rmt_calibrated.slurm` and
   `run_rmt_calibrated_v2.slurm` were superseded by `run_rmt_calibrated_v4.slurm`. The pins now
   describe **this** tree, not the tree that produced the CSVs, and nothing records what the
   intervening edits to the calibration driver changed. A green gate here means "consistent with
   itself", not "identical to the analysis tree".
2. The shell history records `sbatch run_rmt_calibrated_v2.slurm` (2026-09-01) but **no submission
   of `run_rmt_calibrated_v4.slurm`**, and no `controls` invocation — yet the v4 logs and
   `cal/controls.csv` exist, dated 2026-09-04. The final pipeline run happened in a session that is
   not in the captured history, so its exact submission cannot be confirmed from the record; the
   flags it used are read here from `run_rmt_calibrated_v4.slurm` and its own logs.
3. The interactive `benchmark3.py` run used `OLD_REPO=../sk-random-matrix-ml-rmt-stats-esd-fixed`,
   while `run_rmt_calibrated_v4.slurm:135` defaults to `../remote-sk-random-matrix-ml-esd-fixed`.
   Those are different trees — see [§A.4.11](#a4-weaknesses-where-the-suite-is-fragile-or-blind).

---

## Analysis: benchmark and test results

All numbers below were read programmatically from the committed CSVs and logs, and every claim was
re-verified against the code that produced it. File and line references are given so a reader can
check each one.

### A.1 What was tested, with what, and at which settings

| Ensemble / case | File | Size, replicas | Deg | Known answer |
|---|---|---|---|---|
| GOE eigenvalues | `benchmark_results_3way.csv` | N = 1500, 6 seeds | 7 | `reference.sigma2_exact/delta3_exact`, β = 1 |
| Poisson levels | same | n = 6000, 6 seeds | 7 | Σ² = L, Δ₃ = L/15 |
| Wishart singular values (ν) | same | 3584×1024, 6 seeds | 7 | β = 1 bulk |
| Wishart eigenvalues (λ = ν²/N) | same | 3584×1024, 6 seeds | 7 | β = 1 bulk |
| Wishart + 20 graded spikes (ν) | same | 3584×1024, 6 seeds, amp 4·(i+1)^−0.7 | 7 | β = 1 bulk + planted outliers |
| Square Wishart (ν), Llama q/k/v/o shape | same | 1024×1024, 6 seeds | 7 | β = 1 bulk |
| Square Wishart (λ) | same | 1024×1024, 6 seeds | 7 | β = 1 bulk |
| GOE / GUE / GSE / Poisson | `cal/ensemble_benchmark.csv` | N = 1200 / 1200 / 600 / 6000; 8 reps (GSE 4) | 7 | Mehta asymptotes |
| COE / CUE / CSE | same | n = 800 / 800 / 400, 4 reps | 7 (unfolding **exact**) | same |
| Planted deviations | `cal/power_table.csv` | 4096×4096, 30-rep band, 200 KS nulls | 7 | detection limits |
| Haar / Student-t / localized vectors | `benchmark_results.csv` (`porter_thomas` rows) | N = 256, 1024, 4096; 40 vectors each | — | PT null is Uniform(0,1) |

There is **no "GOE + spike" case** in the suite. Spikes are planted on Wishart matrices
(`benchmark3.py:70–76`) and, in the power table, on an iid Gaussian 4096×4096 matrix
(`rmt_null_calibration.py:287–290`).

**The degree-7 / degree-15 split.** `cal/null_bands.csv` and `cal/null_bands_deg7.csv` are the same
40-replica experiment at the two degrees, so the cost of the choice is directly measurable:

| Shape | Σ²(20) null mean, deg 7 | deg 15 | exact (1.0491) | Δ₃(10) deg 7 → deg 15 | ⟨r⟩ shift |
|---|---|---|---|---|---|
| 4096×4096 | 1.0678 (+1.8 %) | 1.0334 (−1.5 %) | — | 0.2358 → 0.2362 (+0.15 %) | 0.000 % |
| 1024×4096 | 1.0773 (+2.7 %) | 1.0264 (−2.2 %) | — | 0.2355 → 0.2356 (+0.04 %) | 0.000 % |
| 14336×4096 | 1.1872 (+13.2 %) | 1.0441 (−0.5 %) | — | 0.2343 → 0.2348 (+0.20 %) | 0.000 % |
| 4096×14336 | 1.1704 (+11.6 %) | 1.0346 (−1.4 %) | — | 0.2340 → 0.2342 (+0.10 %) | 0.000 % |

Two things follow, and they pull in opposite directions.

1. **Degree 15 is the right call for the tall shapes.** At degree 7 the *null band itself* is
   +13 % biased on 14336×4096, so a band built at degree 7 and a pipeline run at degree 15 would be
   incomparable. The `nulls` subcommand's own help text says as much
   (`rmt_null_calibration.py:619–624`) and quotes +14.4 % / +2.0 %, consistent with the +13.2 % /
   −0.5 % measured here.
2. **Degree 15 is untested by the accuracy suite and is warned against by the design record.**
   `design.md` §D3 tabulates the degree scan and shows the trade-off explicitly: too much capacity
   *absorbs the count fluctuations Σ² is measuring* — GOE Σ²(50) goes +2.0 % at deg 7 to −2.8 % at
   deg 15, square-ν Σ²(50) +2.4 % to −11.7 %. D3's conclusion is "degree 7 is at or near the optimum
   for every RMT ensemble tested", and `RunConfig.unfold_deg` remains 7.

Net: Σ² z-scores in `cal/zscores.csv` are computed with an estimator that partially suppresses the
long-range fluctuation it is meant to measure. Because the band uses the same estimator, the bias
largely cancels in z and the scores are not *wrong*; but the **power** of Σ² at degree 15 is lower
than anything the power table (degree 7) reports, by an unmeasured amount. Δ₃ and ⟨r⟩ are degree-
robust (≤ 0.2 % and 0.000 %) and carry no such caveat.

### A.2 Correctness: where the estimators agree with theory

**The unfolding is monotone and unit-mean in every case.** For the FIXED column of
`benchmark_results_3way.csv` the mean unfolded spacing ⟨s⟩ is 0.9996–1.0044 and the fraction of
non-monotone spacings is exactly 0.0 in all seven cases. This rules out gross unfolding failure. It
does **not** rule out a residual smooth density modulation, which leaves both diagnostics intact
while inflating Σ² — precisely what `outstanding-issues.md` O5 attributes the Σ² excess to. Treat
these two numbers as a floor, not a certificate.

**Δ₃ is the reliable statistic.** Against the exact finite-L laws, the fixed estimator is within
**3.2 %** in every one of the 28 Δ₃ rows and within 1.3 % for GOE and Poisson. The largest Δ₃ error
is 3.19 % at L = 50 on the square Wishart (ν) case.

**The circular ensembles separate estimator error from asymptote error.** COE has an exactly
uniform eigenphase density (`rmt/nulls.py:148–179`), so `xi = n·θ/2π` is an exact unfolding and any
discrepancy is estimator or formula error only. COE reproduces GOE's apparent Δ₃ deficit
(`cal/ensemble_benchmark.csv`):

| Δ₃ vs L→∞ asymptote | L = 3 | L = 5 | L = 10 | L = 20 | L = 50 |
|---|---|---|---|---|---|
| GOE, N=1200, 8 reps (unfolding estimated) | +25.7 % | +9.8 % | +2.2 % | +0.8 % | −0.3 % |
| COE, n=800, 4 reps (unfolding exact) | +24.6 % | +8.7 % | +3.5 % | +1.4 % | −2.4 % |
| CUE (β = 2), 4 reps | −1.0 % | −0.3 % | +0.7 % | +3.2 % | +2.6 % |
| CSE (β = 4), 4 reps | −4.8 % | −3.0 % | −0.8 % | +0.9 % | +2.3 % |

The deviation is selective by Dyson index — large for β = 1, absent for β = 2 and 4 — which no
estimator bug can produce. It is the β = 1 asymptote that is wrong at small L. In a replica-noise
sense this is also the most significant result in the file: the GOE and COE Δ₃(3) and Δ₃(5) rows
carry |t| = 6.4 to 26.1, whereas the GSE/CSE β = 4 rows carry |t| = 4.6 to 11.0 in the *opposite*
direction, and essentially nothing else in the file is significant (see §A.4).

Scored against `reference.delta3_exact` instead of the asymptote, GOE Δ₃(5) is within **0.85 %** in
`benchmark_results_3way.csv` (N = 1500, 6 seeds) and **−1.42 %** in `cal/ensemble_benchmark.csv`
(N = 1200, 8 reps). Both are within 1.1 replica standard errors of the exact law. The two files are
independent runs of different sizes; quote whichever you cite by name.

**The asymptote is measurably outside the null band.** In `cal/null_bands.csv` the Δ₃(5) asymptote
falls **outside** the 2.5–97.5 % band on all four production shapes, and Δ₃(10) on three of four.
This is the operational justification for `rmt.reference`: at these level counts, scoring against
the asymptote would flag every matrix in the model as deviant before any weight was analysed.

**⟨r⟩ is accurate everywhere** — within 0.75 % of the Atas reference (`rmt/reference.py:51`) for all
seven ensembles in `cal/ensemble_benchmark.csv` (GOE 0.5337 vs 0.5307; GUE 0.5957 vs 0.5996; GSE
0.6704 vs 0.6744; Poisson 0.3850 vs 0.3863; worst case COE +0.74 %). ⟨r⟩ needs no unfolding, which
is why it is the one statistic that does not move at all between degree 7 and degree 15.

**Outlier handling works.** On Wishart + 20 graded spikes, the old code's Σ²(20) is 2.691 against an
exact 1.049 (+156 %); the MP bulk trim (NEW, `benchmark3.py:129`) gives 1.041 (−0.7 %) and the
adaptive strip of 5 levels (FIXED, `benchmark3.py:131`) gives 1.037 (−1.2 %). Both repair the case.

**The coordinate-transform search is the decisive fix.** In the eigenvalue domain the old estimator
collapses; the fixed one applies a `sqrt` transform (`rmt/spacing.py:639`) and recovers:

| Σ²(L), Wishart λ = ν²/N (3584×1024) | exact | OLD | files(3) NEW | FIXED |
|---|---|---|---|---|
| L = 10 | 0.9087 | 1.1521 (+26.8 %) | 1.0929 (+20.3 %) | 0.9314 (+2.5 %) |
| L = 20 | 1.0491 | 1.4299 (+36.3 %) | 1.4567 (+38.9 %) | 1.0810 (+3.0 %) |
| L = 50 | 1.2348 | 2.1677 (+75.6 %) | 2.0090 (+62.7 %) | 1.1966 (−3.1 %) |

On the **square** 1024×1024 λ case the old estimator fails catastrophically — Σ²(50) = 77.38 against
an exact 1.235, a factor of 62.7 — because the MP cut is a no-op on a square matrix
(`rmt/spacing.py:348–356`) and the hard edge at zero stays in the sample. FIXED returns 1.208
(−2.2 %). Worst-case error across all 56 Σ²/Δ₃ rows falls from 62.7 % (files(3) NEW) to 4.8 %
(FIXED).

**The Porter–Thomas test is now calibrated.** From the `porter_thomas` rows of
`benchmark_results.csv` (40 vectors per cell):

| N | Student-t(3) accepted as random, old `D < 0.1` | …under the MC test | Haar accepted, MC test |
|---|---|---|---|
| 256 | 45 % | 22.5 % | 100 % |
| 1024 | 67.5 % | 0 % | 92.5 % |
| 4096 | 87.5 % | 0 % | 95 % |

against a 95 % design target at α = 0.05. For balance: vectors localized on 5 % of coordinates are
rejected 100 % of the time by *both* rules, so the old rule failed specifically on heavy tails, not
on localization.

**Exact invariances hold to machine precision** (`cal/invariance.csv`): transpose (1.15e-14),
random sign flips (0.0), row/column permutation (9.55e-15) and a ×2.7 rescale (Δ₃ 3.58e-15, ⟨r⟩
1.11e-15), all against a 2.0e-10 tolerance. Two further rows in that file are worth quoting and are
easy to miss: an fp32 SVD differs from fp64 by 3.07e-05 of a spacing (tolerance 0.01, passes), but
**bf16 entry rounding moves the spacings by 0.085 of a mean spacing** — recorded with no tolerance
and no pass flag. That is the quantitative reason `--dtype fp32` is the default and why a bf16
checkpoint cannot be used for short-range level statistics.

### A.3 The Llama-3.1-8B result

This is the part the previous README omitted. 35 matrices (layers 0, 8, 16, 24, 31 × Q/K/V/O/G/U/D),
four shapes, degree 15, `--spacing_bulk_mode auto`, scored against 40-replica bands
(`cal/zscores.csv`). The correct threshold is **|z| > 4.30** (Student-t, 40 replicas, 385 tests);
the score stage prints 3.83 from `norm.ppf` (`rmt_null_calibration.py:472`, issue O8).

| Statistic | mean z | median z | range | matrices with \|z\| > 4.30 |
|---|---|---|---|---|
| ⟨r⟩ | −0.12 | 0.00 | −2.67 … +3.33 | 0 / 35 |
| KS vs Wigner-GOE | +0.64 | +0.47 | −1.73 … +3.84 | 0 / 35 |
| Brody β | −0.08 | −0.11 | −2.62 … +4.68 | 1 / 35 |
| Δ₃(5) | +0.02 | +0.05 | −4.58 … +3.16 | 1 / 35 |
| Δ₃(10) | +0.20 | +0.40 | −4.38 … +2.28 | 1 / 35 |
| Δ₃(50) | +0.35 | +0.37 | −2.50 … +4.21 | 0 / 35 |
| Σ²(5) | +1.50 | +0.59 | −1.76 … +22.97 | 2 / 35 |
| Σ²(10) | +2.82 | +0.88 | −1.81 … +57.51 | 2 / 32 |
| Σ²(20) | +5.83 | +1.18 | −1.46 … +146.05 | 3 / 32 |
| `pt_frac_random` | −13.42 | −3.33 | −95.40 … −1.04 | **16 / 35** |
| `pt_p_mean` | −10.48 | −5.67 | −40.32 … −1.84 | **21 / 35** |

379 of the 385 z-values are finite; the 6 missing entries are Σ²(10) and Σ²(20) on the three
matrices whose unfolding fell to the `gauss` branch, where the code correctly refuses L beyond its
calibrated reach (`SIGMA2_RELIABLE_LMAX["gauss"] = 5`, `rmt/spacing.py:56`). The score stage warned
about the mixed branch (`RMT_calibrated_v4_436780.log:277`).

Three readings, in order of how much they are supported.

**1. Short-range level statistics are indistinguishable from an iid Wishart matrix of the same
shape.** ⟨r⟩, the Wigner-GOE KS distance, Brody β, Δ₃(5) and Δ₃(10) each exceed the threshold on at
most one matrix out of 35 — and it is the *same* matrix each time (layer-16 k_proj). Against 385
Bonferroni-corrected tests, this is a null result. The local correlations of Llama-3.1-8B weight
matrices lie in the β = 1 universality class.

**2. The Porter–Thomas block fires hard and is the most robust deviation in the run.**
`pt_frac_random` — the fraction of right singular vectors whose Porter–Thomas null is *not*
rejected at α = 0.05 (`rmt/porter_thomas.py:160`) — averages 0.774 against a null band of
0.94–0.96, i.e. roughly 23 % of singular vectors are non-Porter-Thomas where 5 % is expected. By
role: V 0.721, D 0.725, O 0.707, K 0.734, Q 0.750, U 0.882, G 0.900. `pt_p_mean` runs 0.27–0.44
against a null of 0.50. These z-scores are *conservative*: almost all of the PT band width is the
binomial noise of testing only 200 vectors (√(0.95·0.05/200) = 0.0154, against band sds of
0.0100–0.0198), whereas the pipeline tests all 1024 or 4096 vectors (binomial sd 0.0068 / 0.0034).
The true deviation is larger than the table says by roughly a factor of 1.5–6. This is the one
headline claim the run actually supports.

**3. The Σ² excursions are almost certainly an edge artefact, not long-range structure.** The
largest is layer-0 q_proj at z = +146.1 — and the same matrix has **1474 of 4096 singular values
above the MP edge** (36 % of the spectrum), of which the adaptive strip removed 62, because
`strip_edge_outliers` caps total removal at `max_strip_frac = 0.03` (`rmt/spacing.py:266,284`).
Δ₃(10) on that same matrix sits at z = −0.69, perfectly null. Across the run `n_right_outliers`
ranges from 41 to 1474 (mean fraction 0.130), so every matrix carries a detached population the
global Chebyshev fit must absorb. This is the signature the power table calls out in the
Student-t(3) row: Σ²(10) z = +50 with every other statistic inside the band, and
`run_rmt_calibrated_v4.slurm:205–211` already flags it as a known artefact. Do not report the Σ²
excursions as evidence of long-range rigidity.

**Controls.** `stage_controls` ran on **15 matrices** (`RMT_calibrated_v4_436780.log:349–768`; only
the last survives in `cal/controls.csv`). Every one of the 15 was flagged heteroscedastic: row-sd
coefficient of variation 0.051–0.406 against an iid expectation of 0.006–0.011 (a factor of 9–37),
and excess kurtosis 0.35–17.43. Verdicts from the decision ladder (`rmt/controls.py:308`, threshold
|z| > 2):

| Statistic | consistent-with-random | variance-profile | entry-distribution | structure-beyond-controls |
|---|---|---|---|---|
| ⟨r⟩ | 13 | 0 | 0 | 2 |
| KS vs GOE | 12 | 2 | 0 | 1 |
| Δ₃(5) | 13 | 0 | 0 | 2 |
| Δ₃(10) | 14 | 0 | 0 | 1 |
| Δ₃(50) | 5 | 4 | 0 | 6 |
| Σ²(5) | 11 | 3 | 0 | 1 |
| Σ²(10) | 2 | 11 | 0 | 2 |
| Σ²(20) | 1 | 12 | 0 | 2 |

The entry shuffle leaves Σ²(20) inside the band on all 15 matrices (|z| ≤ 2.82, and ≤ 1.42 on 13 of
them), while the row shuffle exceeds |z| = 2 on 11 of 15 and often *overshoots* the real value
(layer-8 q_proj: real +6.84, row +16.63, col +0.65). Read together: **the Σ² excursions are produced
by the per-row variance profile, not by the entry distribution and not by learned correlations.**
That is a clean mechanistic result and the strongest argument in the repository for O7 (wire in a
variance-profile-matched null).

**Two caveats that keep this from being a finished claim**, both serious:

- The controls stage runs at **degree 7 and `bulk_frac 1.0`** (no `--unfold-deg` flag exists for it;
  `rmt_null_calibration.py:653–661`, `rmt/nulls.py:218`), whereas the scored rows are at **degree 15
  with `auto` stripping**. The two are different estimators on different level sets. Concretely:
  layer-16 q_proj scores z(Σ²(20)) = +157 in the controls stage and z = +1.3 in `cal/zscores.csv`.
  The control verdicts therefore do **not** transfer to the z-score table as it stands.
- `interpret()` compares **absolute** z-values (`rmt/controls.py:322`), so a control that produces a
  deviation of the *opposite sign* still counts as an explanation. Layer-31 q_proj: real Σ²(20)
  z = −4.31, row shuffle +3.75, col shuffle −3.05 → verdict "variance-profile" on the strength of a
  control that moved the statistic the other way. Six of the fifteen Δ₃(50) "structure-beyond-
  controls" verdicts span z = −4.52 to +19.70, i.e. both signs, which is not what a single physical
  mechanism looks like.

### A.4 Weaknesses: where the suite is fragile or blind

**1. Σ² is the weak statistic, and it is biased at both ends of L.** The fixed estimator's
worst-case Σ² error per case is 2.2–4.8 %, against ≤ 3.2 % for Δ₃. Across the seven benchmark cases
(FIXED vs exact, sign test over cases):

| L | Σ² mean rel. error | high in | sign-test p | Δ₃ mean rel. error | high in | sign-test p |
|---|---|---|---|---|---|---|
| 5 | −0.1 % | 3/7 | 1.00 | −0.4 % | 2/7 | 0.45 |
| 10 | **+1.8 %** | **7/7** | **0.016** | −0.5 % | 2/7 | 0.45 |
| 20 | +1.7 % | 5/7 | 0.45 | −0.1 % | 4/7 | 1.00 |
| 50 | **−3.1 %** | **0/7** | **0.016** | **+1.4 %** | **7/7** | **0.016** |

Three systematic patterns, of which the previous README reported only the first two:

- **Mid-L Σ² excess.** Σ²(10) is high in all seven cases, mean +1.8 % (issue O5).
- **L = 50 Σ² deficit.** Σ²(50) is low in all seven cases, mean −3.1 %, worst −4.8 % on the spiked
  case (issue O4). With ~1000 levels the effective number of independent windows at L = 50 is only
  ~20, and the estimator excludes L/2 at each end, exactly where the unfolding residual is largest.
- **L = 50 Δ₃ *excess*, previously unreported.** Δ₃(50) is high in all seven cases, +0.4 % to
  +3.2 %, mean +1.4 %. The two long-range statistics are biased in opposite directions at L = 50, so
  "Δ₃ is reliable" is a statement about L ≤ 20, not about L = 50. Note that the pipeline's default
  `--delta3_L 5 10 50` **does** emit Δ₃(50) while `--sigma2_L 5 10 20` does not emit Σ²(50), so this
  is the bias that actually reaches the results table.

`SIGMA2_RELIABLE_LMAX["cheb"] = 50` (`rmt/spacing.py:56`) permits Σ² out to L = 50 for the global
fit, which is inconsistent with the O4 evidence above; the constant is a statement about the
unfolding's reach, not about the window-sampling bias, and the two are not the same limit.

**2. The individual deviations are not resolved by the replica counts used.** In
`cal/ensemble_benchmark.csv`, dividing each `sd` by √reps gives the standard error of the reported
mean. Only 11 of the 70 Σ²/Δ₃ rows exceed |t| = 2, and nine of those are the Δ₃-vs-asymptote rows at
L = 3 and L = 5 discussed in §A.2. At L ≥ 10 the only rows past |t| = 2 are COE Σ²(20) (t = −2.11)
and Poisson Δ₃(20) (t = +2.10). Both rows that break the suite's own printed PASS criterion of
"|err| < 5 % for L ≥ 10" (`rmt_null_calibration.py:176`) — GOE Σ²(50) −5.6 % and GSE Σ²(50) −8.3 % —
sit at t = −1.25 and t = −1.57. **The failures are real as a pattern across cases and not resolved
within any single case.** Present them as sign tests, not as point estimates.

This has a sharp consequence for O5. `benchmark_results_3way.csv` gives GOE Σ²(10) = +3.14 % and
Σ²(20) = +3.29 % (N = 1500, 6 seeds), which is where "~3 % high" comes from; the independent GOE run
in `cal/ensemble_benchmark.csv` gives **−0.05 %** and **−0.98 %** (N = 1200, 8 reps). The exact and
asymptotic Σ² references agree to 0.006 % so the two are directly comparable, and they differ by
3.2 % — about 1.2 pooled standard errors. **Two runs of the same estimator disagree on the sign of
the mid-L Σ² bias.** The +3 % figure is one run's point estimate, not a measured property.

**3. One frequently quoted number is not reproducible from any committed file.** The claim that a
degree scan "reduces GOE Σ²(20) from +4.9 % to +2.9 % at degree 15" appears in
`outstanding-issues.md` O5 attributed to `design.md` §D3. §D3's degree table contains **no Σ²(20)
row at all**, and no +4.9 %/+3.6 %/+2.9 % figures; the only Σ²(20) numbers in §D3 are "4.7 % to
9.4 %", describing the *rejected* per-spectrum degree search regressing. §D3's actual L = 50 rows
show degree 15 making Σ² **worse** (GOE +2.0 % → −2.8 %, square-ν +2.4 % → −11.7 %). Use the
measurable degree-7/degree-15 band comparison in §A.1 instead.

**4. Σ² fires on things that are not structure.** From `cal/power_table.csv` (z against a 30-replica
band at 4096×4096, degree 7):

| Planted | ⟨r⟩ | KS-GOE | Brody β | Δ₃(10) | Σ²(10) | p_mc(KS) |
|---|---|---|---|---|---|---|
| iid Gaussian (null) | +1.4 | −0.1 | −0.5 | **+2.2** | +1.0 | 0.358 |
| +20 rank-1 spikes (amp 3) | +0.9 | −0.8 | +1.0 | +0.3 | **+4.4** | 0.677 |
| 2 independent blocks | −28.1 | +27.8 | −20.9 | +27.4 | +22.5 | 0.005 |
| 95 % GOE + 5 % Poisson | −1.9 | +3.9 | −4.8 | +5.9 | **+138.6** | 0.005 |
| 90 % GOE + 10 % Poisson | −4.5 | +6.5 | −6.1 | +14.1 | +25.1 | 0.005 |
| 80 % GOE + 20 % Poisson | −12.1 | +16.5 | −13.3 | +31.4 | +44.5 | 0.005 |
| 80 % sparse | −0.7 | +1.8 | −1.0 | −0.2 | +1.3 | 0.030 |
| Student-t(3) entries | +1.7 | −1.1 | +1.8 | −1.3 | **+50.1** | 0.741 |

Student-t(3) entries have finite variance, so Tao–Vu universality guarantees β = 1 local statistics
— and every spacing statistic correctly says "random". Σ²(10) nevertheless reports z = +50. The
mechanism is specific: t(3) has an **infinite fourth moment**, so the extreme singular values are
not Tracy–Widom and detach from the edge; with no stripping in this code path those detached levels
distort the global fit, and Σ² reports the distortion. **Δ₃(10) is the honest long-range detector;
treat Σ² as advisory.** The p_mc column bottoms out at 1/201 = 0.005 (`--n-null 200`), so 0.005 is a
floor, not a measured value.

**5. The suite is blind to entry-level structure.** Heavy tails and 80 % sparsity are invisible to
every spacing statistic (all |z| < 4.3 excluding Σ², p_mc 0.74 and 0.03). "The bulk is still GOE" is
therefore **not** evidence that weights are Gaussian — it is a statement about local correlations
only. On Llama the tail block does carry an entry-distribution signal: `hill_is_powerlaw` is true on
22 of 35 matrices, `alpha_hill_lambda` runs 1.49–13.84 (median 3.31), and the stable rank is a
median 4.5 % of `min(n, m)`. Those, not the spacing statistics, are what any claim about the entry
distribution has to rest on.

**6. Detection limits are coarse, and two "detections" are artefacts.** Against |z| > 4.30, a
Poisson admixture is caught at 10 % by four of the five spacing statistics excluding Σ² (⟨r⟩ −4.5,
KS +6.5, Brody −6.1, Δ₃ +14.1; Σ² +25.1 makes five if it is counted) and at 5 % only by Δ₃ (+5.9)
and Brody (−4.8). Twenty rank-1 spikes are caught by Σ² alone, at z = +4.40 against a 4.30
threshold — marginal, and by the §A.4.4 mechanism it is the un-stripped outliers being detected, not
the low-rank structure. A ~2 % Poisson admixture would pass unnoticed.

**A construction caveat on the admixture rows.** The "GOE + Poisson" cases are built by *deleting
the top f·k levels and replacing them with Uniform(min, max) draws*
(`rmt_null_calibration.py:296–300`). That is not a density-matched superposition of an independent
Poisson sequence: it simultaneously truncates the upper edge and injects levels at a density that
does not follow the MP profile. The resulting Σ²(10) = +138.6 at 5 % admixture is therefore
predominantly a density-distortion artefact, and the admixture fractions should not be read as
calibrated sensitivities to a genuine Poisson component.

**7. A "|z| > 2 = detected" rule produces false positives — and it is what the code prints.** The
pure iid null itself scores Δ₃(10) z = +2.2 in the power table, and in the 400×400 demo
(`calc_results/power_table.csv`) the iid null scores ⟨r⟩ +3.3, Brody +3.5 and Δ₃(10) −4.1. Both
`stage_power` (`rmt_null_calibration.py:312`) and `stage_controls`
(`rmt_null_calibration.py:589`) print "|z| > 2 = detected". Use the Student-t Bonferroni threshold
(`rmt/nulls.py:359`): 4.30 for 40-replica bands and 385 tests; 3.67 for the 30-replica, 44-test
control table; 3.63 for the 40-test power table. The value the score stage prints, 3.83, comes from
`norm.ppf` and is anti-conservative (issue O8, `rmt_null_calibration.py:472`); the v4 script
computes and prints the correct 4.30 immediately afterwards
(`RMT_calibrated_v4_436780.log:318,328`).

**8. Brody β has a measurable null bias that is not 1.** The null-band means are 0.9535–0.9606
across the four production shapes, and β = 1 lies **outside** the 2.5–97.5 % band at 14336×4096
(0.9158–0.9932). Any claim of the form "trained weights have β < 1" must be scored against ~0.954,
not against 1. The z-score machinery does this automatically; a reader comparing raw `brody_beta`
values against 1 will not.

**9. The band and the measurement do not see exactly the same levels, and the guard does not
catch it.** The bands were built at `bulk_frac = 1.0` (no trimming) while the pipeline ran
`--spacing_bulk_mode auto`, which strips 2–63 levels per matrix. `run_rmt_calibrated_v4.slurm:112`
asserts these are equivalent; they are equivalent only on a spectrum with nothing detached. The
score stage's guard (`rmt_null_calibration.py:422–434`) explicitly discards `"nan"` from the CSV's
`bulk_center_frac` column, and `bulk_mode = "auto"` writes NaN there (`rmt/per_matrix.py:193–195`),
so the guard silently does not fire. The size of the mismatch is small — interpolating
`cal/bulk_frac_sweep.csv`, a 1.5 % edge cut moves Σ²(20) by ~0.2 %, about 0.04 band sd, and the
measured correlation between levels stripped and z(Σ²) across the 35 matrices is only +0.23 — so
this is a latent defect rather than a live error in the present numbers. It will not stay small on a
model with heavier edges.

**10. Replica counts are thin in places.** GSE and the three circular ensembles use 4 replicas
(`rmt_null_calibration.py:157,169`), the Porter–Thomas bands 8 (`reps = max(3, a.reps // 5)`,
`rmt_null_calibration.py:250`). The PT band also tests only the top 200 singular vectors with 3000
MC samples while the pipeline tests every vector with 5000, so the PT band sd (0.0100–0.0198) is
1.5–6× the pipeline's own binomial sampling noise (0.0034 at 4096 vectors, 0.0068 at 1024). PT
z-scores are conservative lower bounds rather than calibrated deviations — which strengthens §A.3.2
rather than weakening it, since the PT deviation is large and negative.

**11. Both 3-way benchmark copies have provenance problems.** `cal/benchmark_results_3way.csv` was
run with `NEW_REPO=FIX_REPO=.` (`run_rmt_calibrated_v4.slurm:135`), so its NEW and FIXED columns
execute the same code and differ only in level selection (max |NEW − FIXED| = 0.027); it cannot show
the improvement over files(3). The root `benchmark_results_3way.csv` does — its NEW column differs
from cal's in exactly 27 of 70 rows, with max NEW-vs-exact error 62.7 % against cal's 4.5 %.
Neither reproduces the OLD column of the legacy `benchmark_results.csv`, although `benchmark3.py`'s
docstring (`benchmark3.py:3–5`) says it should. The first reason is simply that the two files are
not contemporaneous: **`benchmark3.py` did not exist when `benchmark_results.csv` was produced** —
that file is the output of `benchmark.py` in an earlier generation of the code — so the docstring's
claim was never testable against a matching run, and there is no second copy of the 3-way file to
compare against. On top of that, the shell history shows **at least two different `OLD_REPO` trees
were in play.** The interactive run on 2026-08-29 used
`OLD_REPO=../sk-random-matrix-ml-rmt-stats-esd-fixed`, whereas `run_rmt_calibrated_v4.slurm:135`
defaults to `../remote-sk-random-matrix-ml-esd-fixed`, and `benchmark.py:20` defaults to a third
path (`/home/claude/mycode/remote-sk-random-matrix-ml-esd-fixed`). Note this does *not* explain the
root-versus-`cal` split: those two files' OLD columns agree to 3.9e-13, so they shared one OLD tree.
It plausibly explains the legacy file. Independently of which tree it was, the data show that
**the legacy run and the 3-way runs did not see the same level realisations**: the deterministic Δ₃ column differs in 28 of 28 rows (up to
18.1 % on square-λ Δ₃(20)), the unfolded ⟨s⟩ differs in all 7 cases, and the replica standard
deviations differ by as much as 28× (Poisson Δ₃(5): 0.00024 vs 0.00677). Every discrepancy is under
one replica sd, so the two files are statistically consistent — but the seed count and/or the source
trees differed, and that is not recoverable from what is committed. Also note `benchmark3.py:132–134`
records `branch`, `transform` and `outliers_stripped` from the **last seed only**.

**12. The demo calibration in `calc_results/` is too small to be used.** Its bands are 400×400 with
20 replicas and a 0.7 centred cut, leaving 280 levels; the band's own Σ²(20) mean is 0.895 against
an exact 1.049 (−15 %) with sd 0.133. Its ensemble check fails GOE Σ²(20) (−8.4 %), CSE Σ²(10)
(−6.4 %) and COE Δ₃(10) (+5.1 %), and a 5 % Poisson admixture goes undetected (p = 0.55). These
files are reference artefacts only (issue O13) — never score a 4096-level matrix against them.
Ironically, the demo is the one configuration in the repository where the pipeline's
`bulk_center_frac` (0.7) and the band's `bulk_frac` (0.7) actually match.

### A.5 What the code can and cannot do

**Can.**

- Reproduce the exact finite-L Σ² and Δ₃ laws for β = 1, 2, 4 and Poisson to ≤ 3.2 % (Δ₃) and
  ≤ 4.8 % (Σ²) at L ≤ 50, on seven ensembles including both Wishart domains and a square matrix.
- Detect and repair the eigenvalue-domain collapse (Σ²(50) off by a factor of 63) via automatic
  coordinate selection, and the spiked-spectrum collapse (+156 %) via adaptive edge stripping.
- Produce a genuinely calibrated Porter–Thomas p-value at any dimension, replacing a fixed KS
  distance that accepted 87.5 % of Student-t(3) vectors as random at N = 4096.
- Score a real matrix against a Monte-Carlo null of the *same shape and level count* rather than an
  asymptotic constant, with the correct Student-t Bonferroni threshold available.
- Distinguish, via the destructive controls, a deviation produced by the entry distribution from one
  produced by the row/column variance profile.
- Do all of the above offline, deterministically for a given seed, on matrices up to 14336×4096.

**Cannot.**

- See the entry distribution through any level statistic. Heavy tails and 80 % sparsity are
  invisible; that is universality, not a defect, but it means the tail-exponent and
  variance-profile blocks must carry every claim about entries.
- Separate "long-range rigidity" from "detached outliers distorting the global unfolding". This is
  the binding limitation on the Llama result: `strip_edge_outliers` caps removal at 3 % of the
  spectrum, and the matrices with the largest Σ² excursions have 12–36 % of their singular values
  outside the MP edge.
- Score a matrix against a variance-profile-matched null. `rmt.controls.variance_profile_gaussian`
  exists (`rmt/controls.py:181`) but is not wired into `per_matrix` or `nulls` (issue O7), so the
  only null that reaches `cal/zscores.csv` is iid Wishart — which every one of the 15 controlled
  matrices was explicitly flagged as the wrong null for.
- Resolve a ~2 % Poisson admixture, a rank-20 spike structure by anything but Σ², or a 2–3 %
  estimator bias at the replica counts used (4–8 for ensembles, 40 for bands).
- Be quoted for Σ² at L = 50 (systematically low by 2–5 %) or, at degree 15, for Σ² power at any L
  (the degree-15 estimator's sensitivity has not been measured).
- Use bf16 or fp16 weights for level statistics: bf16 rounding moves spacings by 8.5 % of a mean
  spacing.

### A.6 Where this may be wrong

A referee should push hardest on these, in order:

1. **The degree-15 results are validated by a degree-7 benchmark.** Nothing in this repository
   measures the accuracy or the power of the estimator configuration that produced
   `rmt_results/…_matrix_metrics.csv`. Re-running `ensembles`, `power` and `controls` at degree 15
   is the single highest-value missing experiment, and the CLI currently cannot do it.
2. **The Σ² excursions on Llama are attributed to un-stripped outliers on circumstantial evidence.**
   The argument is that Δ₃(10) is null on the same matrices, that the power table shows the same
   signature for Student-t entries, and that the affected matrices carry 12–36 % of their spectrum
   outside the MP edge. It has not been demonstrated directly by re-running the pipeline with a
   larger `max_strip_frac`. The one direct comparison available — layer-16 q_proj at z = +157
   unstripped (degree 7) versus z = +1.3 with 10 levels stripped (degree 15) — confounds two
   changes at once.
3. **The control verdicts are computed on a different estimator from the scored rows** (§A.3), so
   the "variance-profile" conclusion, though mechanistically persuasive, has not been established
   for the numbers actually reported.
4. **The Δ₃(50) "structure-beyond-controls" verdicts on 6 of 15 matrices are the only surviving
   claim, and they are not coherent.** The z-values span −4.52 to +19.70. Δ₃(50) also carries a
   +1.4 % estimator bias (§A.4.1), and on the 1024-level k_proj matrices L = 50 is at the very edge
   of the window-count limit (`n/20 ≈ 51`). Do not build a physical claim on this row without first
   re-running it at larger L-reliability margin.
5. **The Porter–Thomas deviation is real but its interpretation is open.** `pt_frac_random = 0.774`
   says 23 % of singular vectors fail a Gaussian-amplitude test. That is consistent with learned
   localization, but it is equally consistent with the same heteroscedasticity that explains Σ²: a
   row-variance profile induces non-Gaussian singular-vector amplitudes. **No destructive control
   was run on the Porter–Thomas block** — `control_suite` computes spacing statistics only
   (`rmt/controls.py:299`). This is the most important missing control in the repository.
6. **The power table's Poisson-admixture construction perturbs the density** (§A.4.6), so the
   quoted detection limits are not limits on a density-matched admixture.
7. **`interpret()` ignores sign** (`rmt/controls.py:322`), inflating the "variance-profile" count.
8. **Replica counts of 4–8 cannot resolve the 2–5 % effects the analysis discusses** (§A.4.2).
   Several statements in `outstanding-issues.md` compare a mean over 6 seeds against a *per-replica*
   sd rather than the standard error of the mean, which understates their own evidence by √n; O5's
   "~1.2σ" for GOE Σ²(10) is t ≈ 2.5 on the correct comparison.
9. **The checkout is not provably the tree that produced the results.** `check_repo.py` passes,
   but only because its hashes were re-pinned to the current files after three of them had drifted
   (`rmt_null_calibration.py`, `tests/test_nulls.py`, `tests/test_spacing.py`) and two pinned SLURM
   scripts had been replaced by `run_rmt_calibrated_v4.slurm`. The gate therefore certifies the
   present tree, not the one the CSVs came from. Compounding it, the shell history contains no
   submission of `run_rmt_calibrated_v4.slurm` — only of a `run_rmt_calibrated_v2.slurm` that is
   not in the repository — so the v4 logs are the only record of what stages 6–8 ran. If the
   analysis is to be re-derived, re-run the whole calibration campaign against these pins rather
   than assuming the committed CSVs correspond to them. (The *environment* half of this problem is
   solved: `requirements.txt` pins the exact conda env, and its numpy/scipy versions match the job
   logs.)
10. **`rmt/reference.py`'s docstring cites `tests/test_reference.py`, which does not exist**
   (the file is `tests/test_reference_and_unfolding.py`) — harmless, but it is the kind of stale
   cross-reference that suggests others should be checked.
11. **`outstanding-issues.md` O2 ("`strip_edge_outliers` is NOT wired into `per_matrix`") is stale.**
    It is wired in through `spacing_bulk_mode="auto"` (`rmt/per_matrix.py:179–191` →
    `rmt/spacing.py:364`), and the committed run used it. The live defect is the 3 % cap, not the
    absence of the call.

### A.7 Unit tests as verification

`pytest` collects **294 test cases** from **255 test functions**; 263 cases run without torch and 31
are torch-marked. With torch installed the full run is 293 passed, 1 skipped. The suite is written
as regression pins for the defects above, not as coverage padding:

| Module | Functions / collected | Pins |
|---|---|---|
| `test_spacing.py` | 34 / 42 | ⟨r⟩, unfolding validity, KS/Brody discrimination, Δ₃/Σ² log-laws, cheb-beats-gauss at long range |
| `test_nulls.py` | 22 / 30 | Band construction; `ks_pvalue_mc` uniform under the null where `scipy.stats.kstest` is not; β = 1 asymptote low at small L |
| `test_reference_and_unfolding.py` | 17 / 23 | Exact finite-L laws; λ-domain `sqrt` transform; spike stripping; `critical_value` is Student-t |
| `test_controls.py` | 19 / 23 | Each shuffle preserves/destroys what it claims; verdict ladder; band bulk-fraction mismatch is refused |
| `test_docs_and_references.py` | 17 / 23 | Docstring claims turned into assertions |
| `test_mp.py` / `test_tail.py` / `test_scalars.py` | 15 / 11 / 13 (= collected) | MP layer, α_CSN − α_Hill = 1, scalar identities |
| `test_sigma2_determinism.py` | 13 / 15 | Fixed 200 000-window Σ² is deterministic; MC error ∝ 1/√n |
| `test_bulk_selection.py` | 14 / 17 | The MP cut is a no-op on square matrices |
| `test_sigma2_lmax_calibration.py` / `test_crosscheck_calibration.py` | 8 / 5 (= collected) | Executable calibrations for `SIGMA2_RELIABLE_LMAX` and `AUTO_CROSSCHECK_TOL` |
| `test_porter_thomas.py` | 11 / 13 | p-value uniform at every dimension; localized and heavy-tailed vectors rejected |
| torch-marked: `test_activations` / `test_decile` / `test_discovery` / `test_pipeline_smoke` | 5 / 7 / 11 / 8 = **31** | Model-I/O paths against tiny in-process models |

The earlier README listed the function counts under a heading of 294, mixing the two units; both are
given here. The SLURM gate's comment "263 passed / 4 skipped"
(`run_rmt_calibrated_v4.slurm:132`) is stale — the correct offline result is 263 passed, 31
deselected, and `RMT_calibrated_v4_436779.log` records the gate correctly aborting a run when the
tests failed.

### A.8 Bottom line

The estimators reproduce the exact finite-L laws to within ~3 % for Δ₃ and ~5 % for Σ² at
**degree 7**, and the one catastrophic failure mode of the old pipeline — the eigenvalue domain — is
fixed. On Llama-3.1-8B, the short-range level statistics (⟨r⟩, Wigner-GOE KS, Brody β, Δ₃(5),
Δ₃(10)) are a clean null against matched-shape bands: **the local correlations of these weight
matrices are β = 1 universal.** The Σ²(10)/Σ²(20) excursions are large but are reproduced by a
row-shuffle control and co-occur with 12–36 % of the spectrum sitting outside the MP edge, so they
should be read as heteroscedasticity plus an unfolding artefact, not as long-range structure. The
one deviation that survives on its own terms is **Porter–Thomas**: 23 % of singular vectors fail the
Gaussian-amplitude test where 5 % is expected, on a conservatively-scored band — and it has no
destructive control behind it yet.

What the suite cannot do is see the entry distribution: heavy tails, sparsity and heteroscedasticity
are invisible to every level statistic it computes. Any claim about *trained* structure therefore
needs the destructive controls run at the production settings and a variance-profile-matched null,
and the latter is generated but not yet wired in (O7).

---

## Contributing

1. The analytic gate must stay green: `python -m rmt --selftest` (11 checks, `rmt/selftest.py:22–77`)
   and `python -m pytest tests/ -q -m "not torch"` (263 cases, no torch needed).
2. Torch-dependent groups (`test_activations`, `test_decile`, `test_discovery`,
   `test_pipeline_smoke` — 31 cases) are marked `pytest.mark.torch` (`pytest.ini`) and use the tiny
   in-process models in `tests/_synthetic_models.py` — never a downloaded checkpoint.
3. Keep the scientific core torch-free. `rmt/config.py`, `mp`, `tail`, `scalars`, `spacing`,
   `nulls`, `reference`, `porter_thomas`, `controls`, `ensembles` and `overlap` import numpy/scipy
   only (verified).
4. A constant that is "calibrated numerically" needs an executable calibration next to it; see
   `tests/test_sigma2_lmax_calibration.py` and `tests/test_crosscheck_calibration.py`.
5. Never score against an L→∞ asymptote at small L. Use `rmt.reference` or a `null_band`.
6. **A calibration stage and the pipeline must run the same estimator.** If you add a knob to
   `spacing_statistics`, add the matching flag to *every* subcommand of `rmt_null_calibration.py`,
   not only to `nulls`. The current `power`/`ensembles`/`controls` stages cannot be run at the
   production unfolding degree, which is the largest open methodological gap (§A.6.1).
7. A calibration stage that runs over several inputs must not overwrite its own output — the
   controls stage ran on 15 matrices and `cal/controls.csv` retains 1.
8. Design decisions are recorded in `design.md` (D1–D19), open defects in `outstanding-issues.md`
   (O1–O13), and resolved ones in `resolved-issues.md`. Add to them rather than relitigating — and
   when citing one, check the cited section actually contains the number (§A.4.3).
9. Run `python check_repo.py` before and after a calibration campaign, and update its pinned
   hashes **in the same commit** that changes a pinned file. It is the only mechanism in the repo
   that ties an artefact to a tree state, and it is only worth anything if the pin moves with the
   file; re-pinning later, as was done once already, buys a green gate and no guarantee.
10. Commit the exact submission script and the job logs alongside the CSVs an artefact run
    produces. `cal/` currently contains outputs from three different days and two different
    `OLD_REPO` trees with nothing in the files themselves to say so.
11. Architecture: `docs/ARCHITECTURE.md`. Entry points and module pages: `docs/modules/`.
    Setup, reproduction, results and known issues: this file. It was drafted as `README4.md`
    and supersedes three earlier drafts (`README.md`, `README2.md`, `README3.md`) that were
    removed, so anything citing one of those is out of date.

---

## License

**No licence file is present in this repository** and `pyproject.toml` declares no `license`
field, so no licence is granted. The bundled third-party material has its own terms:

- Portions of `rmt/spacing.py` and `rmt/porter_thomas.py` are ports of the reference
  implementation from **Thamm, Staats & Rosenow (2022)**, `src/rmt_utils.py`.
- The analysed checkpoint, **meta-llama/Llama-3.1-8B**, is covered by the Llama 3.1 Community
  License. No weights are redistributed here.
