#!/bin/bash
#SBATCH --job-name=rmt_frontier_scratch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${RMT_PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
SCRATCH_BASE="${RMT_SCRATCH_BASE:-/scratch/$USER/RMT-ML}"

mkdir -p "$SCRATCH_BASE"
if [[ ! -d "$SCRATCH_BASE" || ! -w "$SCRATCH_BASE" ]]; then
    echo "The scratch directory is not writable: $SCRATCH_BASE" >&2
    echo "Ask the administrator to create a writable scratch directory for $USER." >&2
    exit 2
fi

export RMT_PROJECT_ROOT="$PROJECT_ROOT"
export RMT_RESULTS_ROOT="${RMT_RESULTS_ROOT:-$SCRATCH_BASE/results/frontier}"
export RMT_SNAPSHOT_ROOT="${RMT_SNAPSHOT_ROOT:-$SCRATCH_BASE/snapshots/frontier}"
export RMT_MODELS_ROOT="${RMT_MODELS_ROOT:-$SCRATCH_BASE/models/frontier}"
export RMT_TMPDIR="${RMT_TMPDIR:-$SCRATCH_BASE/tmp/frontier}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_BASE/cache}"
export HF_HOME="${HF_HOME:-$SCRATCH_BASE/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

# Start with the 4B model. Set RMT_FRONTIER_MODELS to select another catalog ID.
export RMT_FRONTIER_MODELS="${RMT_FRONTIER_MODELS:-qwen3-4b-instruct-2507}"
export RMT_WEIGHT_LAYER_STRATEGY="${RMT_WEIGHT_LAYER_STRATEGY:-representative}"
export RMT_DECILE_LAYER_STRATEGY="${RMT_DECILE_LAYER_STRATEGY:-representative}"
export RMT_REPRESENTATIVE_LAYERS="${RMT_REPRESENTATIVE_LAYERS:-3}"
export RMT_LAYERS_PER_TASK="${RMT_LAYERS_PER_TASK:-1}"
export RMT_PERPLEXITY_TOKENS="${RMT_PERPLEXITY_TOKENS:-512}"
export RMT_ACTIVATION_BATCHES="${RMT_ACTIVATION_BATCHES:-1}"
export RMT_ACTIVATION_LENGTH="${RMT_ACTIVATION_LENGTH:-512}"

mkdir -p \
    "$PROJECT_ROOT/logs" \
    "$RMT_RESULTS_ROOT" \
    "$RMT_SNAPSHOT_ROOT" \
    "$RMT_MODELS_ROOT" \
    "$RMT_TMPDIR" \
    "$HUGGINGFACE_HUB_CACHE"

exec bash "$PROJECT_ROOT/run_frontier_calibration.slurm"
