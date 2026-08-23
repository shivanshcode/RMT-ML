#!/bin/bash
#SBATCH --job-name=spectral_chinchilla
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load cuda/12.1 python/3.10

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "Missing .venv; stage and install the offline wheelhouse before submission." >&2
    exit 2
fi
source .venv/bin/activate
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${PROJECT_ROOT}/cache/huggingface/hub"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/tokenized/wikitext-103-raw-v1_gpt2.npy}"
TRACK="${TRACK:-all}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Offline Python interpreter is not executable: $PYTHON" >&2
    exit 2
fi
if [[ ! -f "$DATASET_PATH" ]]; then
    echo "Offline token array is missing: $DATASET_PATH" >&2
    exit 2
fi
if [[ ! -f "data/asset_manifest.json" ]]; then
    echo "Asset manifest is missing: data/asset_manifest.json" >&2
    exit 2
fi

"$PYTHON" scripts/download_assets.py --root "$PROJECT_ROOT" --verify-only

mkdir -p \
    logs \
    results/paper1 \
    results/paper2 \
    results/paper3 \
    results/golden_compute_optimal

COMMON_ARGS=(
    --execute
    --model-type causal_transformer
    --dataset-path "$DATASET_PATH"
    --vocab-size 50257
    --device cuda
    --amp-dtype bfloat16
    --compile-model
    --allow-tf32
    --svd-backend cuda
    --svd-driver gesvdj
    --covariance-device cuda
    --covariance-dtype float32
    --pin-memory
)

run_paper1() {
    echo "[Spectral-Chinchilla] Track 1: Staats dual-end overlap and lesions"
    srun --ntasks=1 "$PYTHON" run_experiments.py \
        "${COMMON_ARGS[@]}" \
        --experiment-mode reproduce_paper1 \
        --aspect-ratio-mode raw \
        --mp-fit-method analytic_mp \
        --spike-detector tracy_widom_95 \
        --overlap-metric staats_dual_end \
        --unfolding-strategy gaussian_kernel \
        --tail-solver rank_ordered_mle \
        --run-spectral-lesioning \
        --lesion-tranches top,bulk,bottom \
        --output-dir results/paper1
}

run_paper2() {
    echo "[Spectral-Chinchilla] Track 2: Thamm level statistics"
    srun --ntasks=1 "$PYTHON" run_experiments.py \
        "${COMMON_ARGS[@]}" \
        --experiment-mode reproduce_paper2 \
        --aspect-ratio-mode raw \
        --mp-fit-method analytic_mp \
        --spike-detector tracy_widom_95 \
        --unfolding-strategy polynomial_chebyshev \
        --unfolding-degree 15 \
        --compute-spacing-distribution \
        --compute-number-variance \
        --compute-delta3 \
        --output-dir results/paper2
}

run_paper3() {
    echo "[Spectral-Chinchilla] Track 3: Martin-Mahoney heavy-tail diagnostics"
    srun --ntasks=1 "$PYTHON" run_experiments.py \
        "${COMMON_ARGS[@]}" \
        --experiment-mode reproduce_paper3 \
        --aspect-ratio-mode raw \
        --mp-fit-method kde_bulk_fit \
        --spike-detector tracy_widom_95 \
        --tail-solver clauset_mle \
        --compute-stable-rank \
        --output-dir results/paper3
}

run_golden() {
    echo "[Spectral-Chinchilla] Track 4: compute-optimal RMT synthesis"
    srun --ntasks=1 "$PYTHON" run_experiments.py \
        "${COMMON_ARGS[@]}" \
        --experiment-mode compute_optimal_rmt \
        --scaling-budget-flops 1e16 \
        --allocation-ratios 0.25 1.00 4.00 \
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
        --output-dir results/golden_compute_optimal
}

case "$TRACK" in
    paper1)
        run_paper1
        ;;
    paper2)
        run_paper2
        ;;
    paper3)
        run_paper3
        ;;
    golden)
        run_golden
        ;;
    all)
        run_paper1
        run_paper2
        run_paper3
        run_golden
        ;;
    *)
        echo "TRACK must be paper1, paper2, paper3, golden, or all" >&2
        exit 2
        ;;
esac

echo "[Spectral-Chinchilla] requested track set completed"
