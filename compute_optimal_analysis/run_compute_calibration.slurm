#!/bin/bash
#SBATCH --job-name=compute_optimal_calibration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
PY="${RMT_PYTHON:-$HOME/micromamba/envs/rmt-ml/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "The compute Python interpreter is not executable: $PY" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs results cache/torch cache/huggingface

DATASET="data/tokenized/wikitext-103-raw-v1_gpt2.npy"
if [[ ! -f data/asset_manifest.json || ! -f "$DATASET" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "Stage the assets on a connected login node before submission." >&2
        exit 2
    fi
    "$PY" scripts/download_assets.py \
        --root "$PROJECT_ROOT" \
        --assets all \
        --allow-network \
        --max-wikitext-tokens "${RMT_STAGE_TOKENS:-10000000}"
fi
"$PY" scripts/download_assets.py --root "$PROJECT_ROOT" --verify-only

RUN_ID="${SLURM_JOB_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/compute_calibration_$RUN_ID}"
SNAPSHOT_ROOT="${RMT_SNAPSHOT_ROOT:-$HOME/compute-optimal-snapshots/$RUN_ID}"
TEMP_ROOT="${RMT_TMPDIR:-/tmp/$USER/compute-optimal-$RUN_ID}"
mkdir -p "$OUTPUT_ROOT" "$SNAPSHOT_ROOT" "$TEMP_ROOT"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_INDEX:-1}"
fi
export PYTHONPATH="$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/cache/torch}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/cache/huggingface}"
export TMPDIR="$TEMP_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

SNAPSHOT_INTERVAL="${RMT_SNAPSHOT_INTERVAL:-3600}"
SNAPSHOT_RETENTION="${RMT_SNAPSHOT_RETENTION:-6}"
BUDGET_TEXT="${RMT_SCALING_BUDGETS:-${RMT_SCALING_BUDGET:-1e13}}"
RATIO_TEXT="${RMT_ALLOCATION_RATIOS:-${RMT_ALLOCATION_RATIO:-1.0}}"
read -r -a SCALING_BUDGETS <<< "$BUDGET_TEXT"
read -r -a ALLOCATION_RATIOS <<< "$RATIO_TEXT"
if [[ "${#SCALING_BUDGETS[@]}" -eq 0 || "${#ALLOCATION_RATIOS[@]}" -eq 0 ]]; then
    echo "At least one scaling budget and allocation ratio are required." >&2
    exit 2
fi

snapshot_results() {
    local reason="$1"
    local stamp temporary final
    stamp="$(date -u +%Y%m%dT%H%M%S.%NZ)"
    temporary="$SNAPSHOT_ROOT/.snapshot-$stamp"
    final="$SNAPSHOT_ROOT/snapshot-$stamp"
    mkdir -p "$temporary"
    cp -a --reflink=auto "$OUTPUT_ROOT/." "$temporary/"
    printf '%s\n' "$reason" > "$temporary/.snapshot-reason"
    mv "$temporary" "$final"

    mapfile -t old_snapshots < <(
        find "$SNAPSHOT_ROOT" -mindepth 1 -maxdepth 1 -type d \
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

"$PY" run_experiments.py \
    --execute \
    --model-type causal_transformer \
    --dataset-path "$DATASET" \
    --vocab-size 50257 \
    --experiment-mode compute_optimal_rmt \
    --scaling-budget-flops "${SCALING_BUDGETS[@]}" \
    --allocation-ratios "${ALLOCATION_RATIOS[@]}" \
    --parameter-cap "${RMT_PARAMETER_CAP:-5000000}" \
    --max-train-tokens "${RMT_MAX_TRAIN_TOKENS:-1000000}" \
    --aspect-ratio-mode farms_normalized \
    --farms-sampling reference_fixed \
    --farms-target-aspect-ratio 1.0 \
    --farms-normalization canonical \
    --mp-fit-method lanczos_stieltjes \
    --spike-detector lanczos_poles \
    --lanczos-steps 50 \
    --lanczos-adaptive \
    --lanczos-pole-method reference_ritz \
    --unfolding-strategy spline_monotone \
    --tail-solver clauset_mle \
    --overlap-metric staats_dual_end \
    --compute-spacing-distribution \
    --compute-number-variance \
    --run-spectral-lesioning \
    --lesion-tranches top,bulk,bottom \
    --save-checkpoints \
    --device cuda \
    --amp-dtype bfloat16 \
    --no-compile-model \
    --allow-tf32 \
    --svd-backend cuda \
    --svd-driver gesvdj \
    --covariance-device cuda \
    --covariance-dtype float32 \
    --pin-memory \
    --output-dir "$OUTPUT_ROOT" &
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
    echo "Compute calibration completed: $OUTPUT_ROOT"
else
    echo "Compute calibration failed with status $RETURN_CODE: $OUTPUT_ROOT" >&2
fi
exit "$RETURN_CODE"
