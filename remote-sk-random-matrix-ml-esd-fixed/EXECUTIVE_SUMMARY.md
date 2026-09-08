# Executive summary — `rmt`

## What this code does

`rmt` analyzes the spectra of LLM weight matrices to characterize how training
shapes them, using three complementary Random-Matrix-Theory lenses. For every
analyzable 2-D weight (attention Q/K/V/O, MLP gate/up/down, fused QKV split by
an architecture-verified contiguous or head-interleaved layout) it computes **one SVD** and derives, in a single flat CSV row:

**Marchenko–Pastur bulk + noise scale.** The Gavish–Donoho median estimator
gives σ̂; the MP edges (ν₋, ν₊) and their eigenvalue images (ν₋²/N, ν₊²/N) bound
the random-noise bulk. Singular values above ν₊ are "signal" outliers; mass
below ν₋ measures small-singular-value depletion — the central thesis of Paper 3.

**Power-law tail (Paper 2).** The CSN maximum-likelihood + KS fit returns the
density exponent α on λ=ν² (and on ν), with the standard Hill survival exponent
on both domains, and the Paper-1 *windowed* Hill estimator with a plateau
diagnostic that distinguishes a genuine heavy tail from an MP edge.

**Scalars.** Stable rank, spectral/row-wise entropy, IPR (localization), MP
soft-rank (ν₊/ν_max), bulk-mass fraction, and a 10-bin ascending per-decile
breakdown of entropy and stable-rank.

**Bulk universality (level statistics).** The Atas r-statistic, nearest-neighbour
spacing KS vs Wigner-GOE / Poisson, Dyson–Mehta Δ₃(L) and number variance Σ²(L)
at L=10, 50, plus the complex spacing ratio for the Ginibre test on square matrices.

**Activation-covariance overlap (Paper 3).** When activations are captured, the
overlap O_k = maxⱼ|v_k·f_j| between right singular vectors and activation-covariance
eigenvectors, plus the eigenvector/eigenvalue coincidence summaries.

**Decile ablation + epoch tracking.** Perplexity after zeroing each singular-value
decile (scope = all-matrices-of-type or only-analyzed), and stable-rank tracking of
a settable layer list across training checkpoints (every ~10% of iterations).

## Architecture and why it is trustworthy

The scientific core is pure numpy/scipy and is validated against closed-form RMT
ground truth (GOE/GUE/Ginibre/Wishart/Pareto) in 76 fast unit tests that need no
torch. The torch/HF layer is exercised by 29 more tests using tiny in-process
models — no network, no downloads. A single SVD dispatcher (`cached_svd`) drives
the A100 for large matrices but always hands numpy back to the analysis, and
`per_matrix_analysis` is proven to call it exactly once per matrix. Everything
runs fully offline (HF offline env vars set before any model touch).

The v2 repair suite additionally covers fail-closed tokenizer provenance, strict run status, complete-mean spacing, lambda-domain windowed Hill support, pristine reversible decile sweeps, metadata-first bounded processing, and precision-qualified SVD caching. Run the current suite locally for an environment-specific count; historical pass totals are not a release guarantee.

## How to run

1. Install: `pip install -e ".[torch,plots]"` (core works with just numpy/scipy).
2. Gate: `python -m rmt --selftest` — must exit 0 before any real run.
3. Analyze (offline):
   ```
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m rmt \
     --models ./models/Llama-3.1-8B --output_dir ./RMT_Local_Outputs \
     --layers 0 4 9 14 19 24 29 --backend auto \
     --do_overlap --do_spacing --do_powerlaw --do_perplexity
   ```
4. On the cluster: submit `run_rmt.slurm` (A100, offline, selftest-gated).

Set `--layers` to any list of layer indices (start/middle/end). Validate on
`pythia-410m` by passing it in `--models`; the discovery registry handles its
fused `query_key_value` automatically.

## Output files (per model `<tag>`)

`<tag>_matrix_metrics.csv` (one row per matrix), `<tag>_summary.json`, `<tag>_run_status.json`,
`<tag>_perplexity.json`, `<tag>_stable_rank_per_epoch.csv`, and plots under `<tag>/`.

## CSV columns (groups)

identity (`name, short, layer_idx, n, m, is_square, N_cov`) ·
MP bulk (`sigma_med, sigma_med_refined, n_iter_sigma, mp_minus/plus[/_eig],
n_*_outliers, frac_*_outliers`) · small-SV (`ks_lower, n_below_minus,
frac_mass_below_minus, excess_small_sv`) · tail (`alpha, xmin, ks_D, n_tail,
alpha_on_nu, alpha_hill_nu, alpha_hill_lambda, hill_plateau_alpha/width/start/end/window/support,
hill_is_powerlaw, LR_trunc, LR_p, alpha_rand, max_ev_rand`) · scalars
(`row_wise_entropy, spectral_entropy, stable_rank, mp_softrank, bulk_mass_frac,
max/min/mean/median_sval, ipr_top10_mean, ipr_bulk_mean, pt_ks_mean,
pt_frac_random`) · bulk stats (`r_statistic_mean, nn_KS_GOE, nn_KS_Poisson,
delta3_L10/L50, sigma2_L10/L50, complex_r_abs_mean, complex_r_cos_mean`) ·
overlap (`max/mean_overlap, overlap_at_top/bottom_sval, rho_*,
max_overlap_with_top_eigenvector, argmax_singular_for_top_eigenvector,
diagonal_coincidence`) · per-decile (`entropy_decile_1..10, srk_decile_1..10`).
