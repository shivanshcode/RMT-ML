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
# Based on the original ai_studio_code.slurm; adapted to the `rmt` package.
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

cd /home/shivansh/remote-sk-random-matrix-ml-esd-fixed/

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
  "./models/pythia-160m"
)

# Track start / middle / end layers (settable list passed straight through).
LAYERS="0 5 10"

# ---------------------------------------------------------------------------
# 1) GATE: run the analytic self-check first. If it fails, do NOT touch the
#    real matrices — abort the whole job.
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
  --output_dir ./RMT_Local_Outputs \
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
  --do_spacing \
  --do_ipr \
  --do_powerlaw \
  --do_perplexity --decile_scope all \
  --no-do_complex_spacing \
  --no-do_porter_thomas

echo "[RMT] batch processing complete."
