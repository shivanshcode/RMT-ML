#!/bin/bash
#SBATCH --job-name=compute_optimal_scratch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
SCRATCH_BASE="${RMT_SCRATCH_BASE:-/scratch/$USER/RMT-ML}"
RUN_ID="${SLURM_JOB_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

mkdir -p "$SCRATCH_BASE"
if [[ ! -d "$SCRATCH_BASE" || ! -w "$SCRATCH_BASE" ]]; then
    echo "The scratch directory is not writable: $SCRATCH_BASE" >&2
    echo "Ask the administrator to create a writable scratch directory for $USER." >&2
    exit 2
fi

export PROJECT_ROOT
export OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH_BASE/results/compute/compute_grid_$RUN_ID}"
export RMT_SNAPSHOT_ROOT="${RMT_SNAPSHOT_ROOT:-$SCRATCH_BASE/snapshots/compute/$RUN_ID}"
export RMT_TMPDIR="${RMT_TMPDIR:-$SCRATCH_BASE/tmp/compute/$RUN_ID}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_BASE/cache}"
export HF_HOME="${HF_HOME:-$SCRATCH_BASE/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_BASE/cache/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SCRATCH_BASE/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH_BASE/cache/triton}"

# This capped grid trains three model/data regimes at each of three model scales.
export RMT_SCALING_BUDGETS="${RMT_SCALING_BUDGETS:-3e16 1e17 3e17}"
export RMT_ALLOCATION_RATIOS="${RMT_ALLOCATION_RATIOS:-0.25 1.0 4.0}"
export RMT_PARAMETER_CAP="${RMT_PARAMETER_CAP:-250000000}"
export RMT_MAX_TRAIN_TOKENS="${RMT_MAX_TRAIN_TOKENS:-1000000}"
export RMT_STAGE_TOKENS="${RMT_STAGE_TOKENS:-10000000}"
export RMT_SNAPSHOT_INTERVAL="${RMT_SNAPSHOT_INTERVAL:-3600}"
export RMT_SNAPSHOT_RETENTION="${RMT_SNAPSHOT_RETENTION:-12}"

mkdir -p \
    "$PROJECT_ROOT/logs" \
    "$OUTPUT_ROOT" \
    "$RMT_SNAPSHOT_ROOT" \
    "$RMT_TMPDIR" \
    "$HUGGINGFACE_HUB_CACHE" \
    "$TORCH_HOME" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR"

exec bash "$PROJECT_ROOT/run_compute_calibration.slurm"
