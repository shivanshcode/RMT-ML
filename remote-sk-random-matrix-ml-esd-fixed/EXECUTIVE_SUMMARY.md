# Executive summary for `rmt`

## Function

`rmt` examines LLM weight matrices with three Random Matrix Theory (RMT) method groups. It does one singular value decomposition (SVD) for each applicable two-dimensional weight. Applicable weights include attention, MLP, and fused QKV projections. It identifies fused contiguous or head-interleaved layouts from the architecture. The tool writes results for one matrix in one flat CSV row.

The Marchenko-Pastur group estimates `σ̂` with the Gavish-Donoho median. MP edges `(ν₋, ν₊)` and eigenvalue images `(ν₋²/N, ν₊²/N)` define the random-noise bulk. Values more than `ν₊` are signal outliers. Mass less than `ν₋` measures small-value depletion, which is the main statement of Paper 3.

The power-law group applies the CSN maximum-likelihood and KS fit to `λ=ν²` and `ν`. It also gives the standard Hill survival exponent in both domains. The Paper 1 windowed Hill estimate includes a plateau diagnostic. This diagnostic separates a stable heavy tail from an MP edge.

Scalar output includes stable rank, spectral entropy, row entropy, IPR, MP soft rank, and bulk-mass fraction. It also includes ten ascending decile groups for entropy and stable rank.

Level statistics include the Atas `r` statistic and nearest-neighbor KS values for Wigner-GOE and Poisson. They include Dyson-Mehta `Δ₃(L)` and number variance `Σ²(L)` at `L=10, 50`. Square matrices also receive the Ginibre complex spacing ratio.

If activation capture is active, the tool computes `O_k = maxⱼ|v_k·f_j|`. It also computes summaries for coincidence between eigenvectors and eigenvalues. The tool reports unavailable results for zero-rank or unresolved activation eigenspaces. It does the same for repeated or null weight singular subspaces. Thus, output does not depend on an arbitrary basis.

Decile ablation sets each singular-value decile to zero and measures perplexity. Scope can include all matrices of one type or only analyzed matrices. Epoch tracking uses the selected backend and records actual backend and dtype information. By default, probes occur at intervals of approximately ten percent of training. Tied parameter aliases receive one lesion. An excessive partition count stops before mutation.

## Architecture and tests

The scientific core uses only NumPy and SciPy. Tests compare it with closed-form RMT results for GOE, GUE, Ginibre, Wishart, and Pareto data. Tests of the torch and Hugging Face layer use small in-process models without downloads. Current local test output gives the applicable test count. Old test counts are not a release promise.

`cached_svd` sends large SVD operations to an A100 and returns NumPy arrays. `per_matrix_analysis` calls it one time for each matrix. Offline environment variables are active before model access.

The repair tests cover tokenizer provenance, strict stage status, fixed-seed self-tests, and spacing with a mean from the full sample. They cover bounded-Pareto likelihoods, matching random estimators, and full windowed Hill support in the lambda domain. They also cover alias-safe decile sweeps, bounded metadata processing, SVD cache corruption, and atomic output ownership.

ESD bins have a fixed upper limit for a very small IQR. Text windows include learned-position offsets. Activation capture keeps mixed submodule modes and accepts exact projection names without numeric layer indices.

## Procedure

1. Install the tool with `pip install -e ".[torch,plots]"`. The core requires only NumPy and SciPy.
2. Enter `python -m rmt --selftest`. Do not analyze a real model unless this command returns exit code 0.
3. For offline analysis, enter this command:

   ```
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m rmt \
     --models ./models/Llama-3.1-8B --output_dir ./RMT_Local_Outputs \
     --layers 0 4 9 14 19 24 29 --backend auto \
     --do_overlap --do_spacing --do_powerlaw --do_perplexity
   ```

4. On the cluster, submit `run_rmt.slurm`. The job uses an A100, offline mode, and the required self-test.

`--layers` accepts a list of layer indices. For a Pythia test, give `pythia-410m` to `--models`. The discovery registry processes its fused `query_key_value` layout.

## Output files

For each `<tag>`, output includes `<tag>_matrix_metrics.csv`, `<tag>_summary.json`, and `<tag>_run_status.json`. It also includes `<tag>_run_manifest.json` and optional `<tag>_perplexity.json`. Epoch output uses `<tag>_stable_rank_per_epoch.csv` and checkpoint status. WeightWatcher output and status are optional. The `<tag>/` directory contains plots.

## CSV column groups

Identity and provenance columns are `name, short, layer_idx, n, m, is_square, N_cov, source_dtype, svd_factorization_dtype`.

MP columns are `sigma_med, sigma_med_refined, n_iter_sigma, mp_minus/plus[/_eig], n_*_outliers, frac_*_outliers`. Small-value columns are `ks_lower, n_below_minus, frac_mass_below_minus, excess_small_sv`.

Tail columns include `alpha, xmin, ks_D, n_tail, alpha_on_nu, alpha_hill_nu, alpha_hill_lambda`. They include all `hill_plateau_*` and support fields. They also include `hill_is_powerlaw, LR_trunc, LR_p, powerlaw_pkg_status/reason`. Random-control columns include `alpha_rand` and its estimator, kind, cutoff, support, KS, and plateau metadata. `max_ev_rand` contains the maximum random eigenvalue.

Scalar columns are `row_wise_entropy, spectral_entropy, stable_rank, mp_softrank, bulk_mass_frac`. They include summary singular values, IPR groups, and Porter-Thomas groups.

Bulk columns are `spacing_seed, r_statistic_mean, nn_KS_GOE, nn_KS_Poisson`. They include `delta3_L10/L50, sigma2_L10/L50, complex_r_abs_mean, complex_r_cos_mean`.

Overlap columns include maximum, mean, top-value, and bottom-value overlap. They include all `rho_*` values and coincidence values. They also include `overlap_status` and `activation_covariance_rank`.

Per-decile columns are `entropy_decile_1..10` and `srk_decile_1..10`.
