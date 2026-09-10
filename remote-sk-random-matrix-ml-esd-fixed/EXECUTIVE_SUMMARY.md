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
overlap O_k = maxⱼ|v_k·f_j| between right singular vectors and identifiable positive activation-covariance eigenvectors, plus the eigenvector/eigenvalue coincidence summaries. Rank-zero/unresolved activation eigenspaces and repeated/null weight singular subspaces are explicitly unavailable rather than basis-dependent.

**Decile ablation + epoch tracking.** Perplexity after zeroing each singular-value
decile (scope = all-matrices-of-type or only-analyzed), and backend-dispatched stable-rank tracking with actual backend/dtype provenance across training checkpoints (every ~10% of iterations). Tied parameter aliases are lesioned once and oversized partition counts fail before mutation.

## Architecture and why it is trustworthy

The scientific core is pure numpy/scipy and is validated against closed-form RMT
ground truth (GOE/GUE/Ginibre/Wishart/Pareto) in the current pure-science test suite. The torch/HF layer is exercised using tiny in-process models—no network or downloads. Run the suite for environment-specific counts rather than relying on historical totals. A single SVD dispatcher (`cached_svd`) drives
the A100 for large matrices but always hands numpy back to the analysis, and
`per_matrix_analysis` is proven to call it exactly once per matrix. Everything
runs fully offline (HF offline env vars set before any model touch).

The repair suite additionally covers fail-closed tokenizer provenance, strict run/stage status, fixed-seed selftesting, complete-mean spacing, bounded-Pareto likelihoods, matched random-control estimators, lambda-domain windowed Hill support (including its final boundary observation), alias-safe reversible decile sweeps, metadata-first bounded processing, corruption-tolerant precision-qualified SVD caching, and atomic output ownership. ESD bins are globally bounded for tiny-IQR spectra; text contexts account for learned-position offsets; activation capture preserves mixed submodule modes and accepts explicitly named projections without numeric layer indices. Run the current suite locally for an environment-specific count; historical pass totals are not a release guarantee.

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
`<tag>_perplexity.json`, `<tag>_stable_rank_per_epoch.csv`, optional WeightWatcher result/status JSON, and plots under `<tag>/`.

## CSV columns (groups)

identity (`name, short, layer_idx, n, m, is_square, N_cov`) ·
MP bulk (`sigma_med, sigma_med_refined, n_iter_sigma, mp_minus/plus[/_eig],
n_*_outliers, frac_*_outliers`) · small-SV (`ks_lower, n_below_minus,
frac_mass_below_minus, excess_small_sv`) · tail (`alpha, xmin, ks_D, n_tail,
alpha_on_nu, alpha_hill_nu, alpha_hill_lambda, hill_plateau_alpha/width/start/end/window/support,
hill_is_powerlaw, LR_trunc, LR_p, powerlaw_pkg_status/reason, alpha_rand plus estimator/kind/cutoff/plateau-union-support/KS metadata, max_ev_rand`) · scalars
(`row_wise_entropy, spectral_entropy, stable_rank, mp_softrank, bulk_mass_frac,
max/min/mean/median_sval, ipr_top10_mean, ipr_bulk_mean, pt_ks_mean,
pt_frac_random`) · bulk stats (`spacing_seed, r_statistic_mean, nn_KS_GOE, nn_KS_Poisson,
delta3_L10/L50, sigma2_L10/L50, complex_r_abs_mean, complex_r_cos_mean`) ·
overlap (`max/mean_overlap, overlap_at_top/bottom_sval, rho_*,
max_overlap_with_top_eigenvector, argmax_singular_for_top_eigenvector,
diagonal_coincidence, overlap_status, activation_covariance_rank`) · per-decile (`entropy_decile_1..10, srk_decile_1..10`).
