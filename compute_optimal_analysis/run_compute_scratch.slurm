#!/bin/bash
#SBATCH --job-name=compute_all_tracks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
PY="${RMT_PYTHON:-$HOME/micromamba/envs/rmt-ml/bin/python}"
SCRATCH_BASE="${RMT_SCRATCH_BASE:-/scratch/$USER/RMT-ML}"
RUN_ID="${SLURM_JOB_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

if [[ ! -x "$PY" ]]; then
    echo "The compute Python interpreter is not executable: $PY" >&2
    exit 2
fi
mkdir -p "$SCRATCH_BASE"
if [[ ! -d "$SCRATCH_BASE" || ! -w "$SCRATCH_BASE" ]]; then
    echo "The scratch directory is not writable: $SCRATCH_BASE" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs
ASSET_ROOT="${RMT_ASSET_ROOT:-$SCRATCH_BASE/assets/compute}"
DATASET="$ASSET_ROOT/data/tokenized/wikitext-103-raw-v1_gpt2.npy"
if [[ ! -f "$ASSET_ROOT/data/asset_manifest.json" || ! -f "$DATASET" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "Stage the full assets on a connected login node before submission." >&2
        exit 2
    fi
    "$PY" scripts/download_assets.py \
        --root "$ASSET_ROOT" \
        --assets all \
        --allow-network \
        --max-wikitext-tokens "${RMT_STAGE_TOKENS:-0}"
fi
"$PY" scripts/download_assets.py --root "$ASSET_ROOT" --verify-only

export PROJECT_ROOT
export RMT_PYTHON="$PY"
export RMT_ASSET_ROOT="$ASSET_ROOT"
export DATASET_PATH="$DATASET"
export TRACK="${TRACK:-all}"
# Torch 2.4 Inductor has known BF16 graph-dtype failures on this model.  Eager
# CUDA still trains on the GPU and is the safe production default.
export COMPILE_MODEL="${COMPILE_MODEL:-0}"
export RMT_SCALING_BUDGETS="${RMT_SCALING_BUDGETS:-2.07e16 2.07e17 2.07e18}"
export RMT_ALLOCATION_RATIOS="${RMT_ALLOCATION_RATIOS:-0.25 1.0 4.0}"
export RMT_MAX_TRAIN_TOKENS="${RMT_MAX_TRAIN_TOKENS:-20000000}"
export RMT_SEQUENCE_LENGTH="${RMT_SEQUENCE_LENGTH:-512}"
export RMT_BATCH_SIZE="${RMT_BATCH_SIZE:-2}"
export RMT_DATALOADER_WORKERS="${RMT_DATALOADER_WORKERS:-8}"
export RMT_PREFETCH_FACTOR="${RMT_PREFETCH_FACTOR:-4}"
export RMT_ACTIVATION_BATCHES="${RMT_ACTIVATION_BATCHES:-20}"
export RMT_VALIDATION_BATCHES="${RMT_VALIDATION_BATCHES:-100}"
export RMT_LOG_EVERY="${RMT_LOG_EVERY:-1000}"
export RMT_THOROUGH_ANALYSIS="${RMT_THOROUGH_ANALYSIS:-1}"
export RMT_SAVE_CHECKPOINTS="${RMT_SAVE_CHECKPOINTS:-1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRATCH_BASE/results/compute/all_tracks_$RUN_ID}"
export RMT_RUNTIME_CACHE="${RMT_RUNTIME_CACHE:-$SCRATCH_BASE/tmp/compute/$RUN_ID}"
export RMT_SNAPSHOT_ROOT="${RMT_SNAPSHOT_ROOT:-$SCRATCH_BASE/snapshots/compute/$RUN_ID}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_BASE/cache}"
export HF_HOME="${HF_HOME:-$SCRATCH_BASE/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_BASE/cache/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SCRATCH_BASE/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH_BASE/cache/triton}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    export SLURM_JOB_ID="$RUN_ID"
    export SLURM_SUBMIT_DIR="$PROJECT_ROOT"
    export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"
    export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-1}"
fi

mkdir -p \
    "$OUTPUT_ROOT" \
    "$RMT_RUNTIME_CACHE" \
    "$RMT_SNAPSHOT_ROOT" \
    "$HUGGINGFACE_HUB_CACHE" \
    "$HF_DATASETS_CACHE" \
    "$TORCH_HOME" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$TRITON_CACHE_DIR"

SNAPSHOT_INTERVAL="${RMT_SNAPSHOT_INTERVAL:-3600}"
SNAPSHOT_RETENTION="${RMT_SNAPSHOT_RETENTION:-12}"

snapshot_results() {
    local reason="$1"
    local stamp temporary final
    stamp="$(date -u +%Y%m%dT%H%M%S.%NZ)"
    temporary="$RMT_SNAPSHOT_ROOT/.snapshot-$stamp"
    final="$RMT_SNAPSHOT_ROOT/snapshot-$stamp"
    mkdir -p "$temporary"
    cp -a --reflink=auto "$OUTPUT_ROOT/." "$temporary/"
    printf '%s\n' "$reason" > "$temporary/.snapshot-reason"
    mv "$temporary" "$final"

    mapfile -t old_snapshots < <(
        find "$RMT_SNAPSHOT_ROOT" -mindepth 1 -maxdepth 1 -type d \
            -name 'snapshot-*' -printf '%T@ %p\n' |
            sort -rn |
            tail -n "+$((SNAPSHOT_RETENTION + 1))" |
            cut -d' ' -f2-
    )
    local old
    for old in "${old_snapshots[@]}"; do
        rm -rf -- "$old"
    done
}

snapshot_loop() {
    while kill -0 "$1" 2>/dev/null; do
        sleep "$SNAPSHOT_INTERVAL"
        if kill -0 "$1" 2>/dev/null; then
            snapshot_results hourly
        fi
    done
}

ANALYSIS_PID=""
WATCH_PID=""
terminate_children() {
    if [[ -n "$ANALYSIS_PID" ]] && kill -0 "$ANALYSIS_PID" 2>/dev/null; then
        kill -TERM "$ANALYSIS_PID" 2>/dev/null || true
    fi
}
trap terminate_children TERM INT

bash "$PROJECT_ROOT/run_hpc.slurm" &
ANALYSIS_PID=$!
snapshot_loop "$ANALYSIS_PID" &
WATCH_PID=$!

set +e
wait "$ANALYSIS_PID"
RETURN_CODE=$?
set -e
kill "$WATCH_PID" 2>/dev/null || true
wait "$WATCH_PID" 2>/dev/null || true
snapshot_results final

if [[ "$RETURN_CODE" -eq 0 ]]; then
    echo "All requested tracks completed: $OUTPUT_ROOT"
else
    echo "The track sequence failed with status $RETURN_CODE: $OUTPUT_ROOT" >&2
fi
exit "$RETURN_CODE"
