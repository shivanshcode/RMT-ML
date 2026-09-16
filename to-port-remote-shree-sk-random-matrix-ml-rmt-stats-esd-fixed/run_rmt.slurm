#!/bin/bash
#SBATCH --job-name=RMT_All_Models
#SBATCH --output=RMT_All_Models_%j.log
#SBATCH --error=RMT_All_Models_%j.error
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00

# ---------------------------------------------------------------------------
# RMT analysis of LLM weight matrices — fully OFFLINE A100 run.
#
# WHAT CHANGED vs the previous version of this script
#   * --do_porter_thomas is now ON. It was --no-do_porter_thomas before, so the
#     Porter-Thomas block never ran at all and pt_* columns were all NaN.
#   * New spacing/unfolding flags are passed explicitly rather than relying on
#     defaults, so the run is self-documenting:
#       --unfold_method auto    global Chebyshev staircase fit, falling back to
#                               Thamm's adaptive Gaussian broadening if that fit
#                               is not monotone / unit-mean-spacing
#       --unfold_deg 7          degree of that global fit
#       --unfold_win 15         Thamm's window for the Gaussian fallback
#       --spacing_domain sval   unfold the SINGULAR VALUES (Thamm's convention)
#       --spacing_bulk_only     restrict level statistics to the MP bulk
#       --delta3_L / --sigma2_L the L grids written to the CSV
#   * --do_brody adds the Brody beta (Poisson 0 <-> Wigner-GOE 1) per matrix.
#
# PLOTS PRODUCED (under $OUT/<model>/)
#   esd/                    ESD + MP overlay                (unchanged)
#   hill/                   Hill estimator + tail fit       (unchanged)
#   spacing/                P(s) vs Wigner-GOE/Poisson, with CDF inset   [NEW]
#   rigidity/               Sigma^2(L) and Delta_3(L) vs GOE/Poisson     [NEW]
#   porter_thomas/          PT p-value per singular vector, vs nu and rank [NEW]
#   porter_thomas_decile/   PT mean-p and non-rejected fraction / decile  [NEW]
#   qkv/  overlap/          heatmaps                        (unchanged)
#   <model>_summary.png     model-level summary             (unchanged)
#
# COST NOTE: the spacing block costs ~3.5 s per matrix at k=4096 levels (the
#   SVD itself is ~33 s), and Porter-Thomas ~4 s once the Monte-Carlo null for a
#   given width is built (it is cached across matrices). Set --pt_max_vectors to
#   cap the per-matrix PT cost on very wide models; leave unset to test all
#   singular vectors, which is what the per-decile plot wants.
#
# MEMORY NOTES (if you hit an oom-kill):
#   * raise --mem above (host RAM), e.g. 64G -> 128G for the 8B model;
#   * analyze fewer layers at once (shorten LAYERS);
#   * lower --perplexity_tokens and --fm_max_length / --n_text_batches;
#   * use --decile_scope analyzed (ablate only analyzed matrices, not all);
#   * or drop --do_perplexity for the largest models and run it separately.
# NOTE: only true causal LMs load via AutoModelForCausalLM. Encoder models
#   like bert-base-uncased are not causal LMs; analyze those separately with an
#   encoder loader rather than in this causal-LM batch.
# ---------------------------------------------------------------------------

cd /home/shivansh/sk-random-matrix-ml-rmt-stats-esd-fixed/

# --- offline guarantees (no network at run time) ---------------------------
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- environment (edit to match your cluster) ------------------------------
module load cuda 2>/dev/null || true

# Activate the conda env. In a non-interactive SLURM shell `conda activate`
# is not set up, so source conda's profile first (edit the path to your conda
# install), OR skip activation entirely and call the env's python directly.
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate rmt_ml_env
    PY=python
else
    # fallback: call the env interpreter directly (no activation needed)
    PY="$CONDA_BASE/envs/rmt/bin/python"
fi
echo "[RMT] using python: $($PY -c 'import sys; print(sys.executable)')"

# Local model snapshots (must already be on disk; never downloaded).
# These are causal LMs. Analyze encoder models (bert, etc.) separately.
MODELS=(
  "./models/Llama-3.1-8B"
)

# Track start / middle / end layers (settable list passed straight through).
LAYERS="0 9 25"

OUT=./RMT_Local_Outputs

# ---------------------------------------------------------------------------
# 1) GATE: run the analytic self-check first. If it fails, do NOT touch the
#    real matrices — abort the whole job. The gate now also covers unfolding
#    validity, the Delta_3 GOE log-law, and Porter-Thomas calibration.
# ---------------------------------------------------------------------------
echo "[RMT] running selftest gate ..."
$PY -m rmt --selftest || { echo "[RMT] selftest FAILED — aborting"; exit 1; }
echo "[RMT] selftest passed."

# ---------------------------------------------------------------------------
# 2) Full analysis with everything enabled.
#    --backend auto uses torch+cuda SVD for large matrices, numpy otherwise.
# ---------------------------------------------------------------------------
$PY -m rmt \
  --models "${MODELS[@]}" \
  --output_dir ${OUT} \
  --layers ${LAYERS} \
  --dtype fp32 \
  --backend auto \
  --alpha_estimator all \
  --sigma_estimator gd_median \
  --N_cov_mode cols \
  --hill_window 20 \
  --n_text_batches 5 \
  --text_path ./wikitext-2-raw/wiki.test.raw \
  --do_overlap \
  --do_qkv_heatmap \
  --do_ipr \
  --do_powerlaw \
  --do_perplexity --decile_scope all \
  --n_deciles 10 \
  --no-do_complex_spacing \
  \
  `# ---- level statistics: spacing, number variance, Delta_3 ----` \
  --do_spacing \
  --unfold_method auto \
  --unfold_deg 7 \
  --unfold_win 15 \
  --spacing_domain sval \
  --spacing_bulk_only \
  --delta3_L 5 10 20 50 \
  --sigma2_L 5 10 20 50 \
  --do_brody --brody_bootstrap 200 \
  \
  `# ---- Porter-Thomas (was disabled before; now on) ----` \
  --do_porter_thomas \
  --pt_n_samples 5000 \
  --pt_alpha 0.05

echo "[RMT] batch processing complete."

# ---------------------------------------------------------------------------
# 3) Report what was written, so a truncated run is obvious in the log.
# ---------------------------------------------------------------------------
for d in esd hill spacing rigidity porter_thomas porter_thomas_decile qkv overlap; do
    n=$(find ${OUT} -type d -name "$d" -exec sh -c 'ls -1 "$1" | wc -l' _ {} \; 2>/dev/null \
        | awk '{s+=$1} END {print s+0}')
    printf '[RMT] %-22s %s figures\n' "$d/" "$n"
done
find ${OUT} -name '*_matrix_metrics.csv' -print
