#!/bin/bash
#SBATCH --job-name=spectral_chinchilla
#SBATCH --partition=gpulong
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Submit from compute_optimal_analysis, or export its validated absolute path.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from compute_optimal_analysis or export PROJECT_ROOT}}"
cd "$PROJECT_ROOT"
PROJECT_ROOT="$(pwd -P)"
if [[ ! -f run_experiments.py || ! -f pipelines/cli_config.py ]]; then
    echo "PROJECT_ROOT is not compute_optimal_analysis: $PROJECT_ROOT" >&2
    exit 2
fi

# This interpreter path is recorded by the successful sibling ESD job.  Calling
# it directly avoids loading a conflicting Python module or guessing a Conda base.
PYTHON="${RMT_PYTHON:-/home/shivansh/.conda/envs/rmt_ml_env/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Cluster interpreter is not executable: $PYTHON" >&2
    exit 2
fi

# The existing Torch installation may bundle its CUDA runtime.  Load a module
# only when the operator has validated that it is required for this environment.
# No module purge is performed, and no Python module is loaded.
if [[ -n "${RMT_CUDA_MODULE:-}" ]]; then
    if ! type module >/dev/null 2>&1; then
        echo "RMT_CUDA_MODULE is set but the module command is unavailable" >&2
        exit 2
    fi
    module load "$RMT_CUDA_MODULE"
fi

# Keep this project's rmt package isolated from the sibling ESD package.
export PYTHONPATH="$PROJECT_ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${PROJECT_ROOT}/cache/huggingface/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

RUNTIME_CACHE="${RMT_RUNTIME_CACHE:-${SLURM_TMPDIR:-${PROJECT_ROOT}/cache/runtime/${SLURM_JOB_ID:?Missing SLURM_JOB_ID}}}"
export TORCHINDUCTOR_CACHE_DIR="${RUNTIME_CACHE}/inductor"
export TRITON_CACHE_DIR="${RUNTIME_CACHE}/triton"
export TMPDIR="${RUNTIME_CACHE}/tmp"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/tokenized/wikitext-103-raw-v1_gpt2.npy}"
if [[ "$DATASET_PATH" != /* ]]; then
    DATASET_PATH="${PROJECT_ROOT}/${DATASET_PATH}"
fi
TRACK="${TRACK:-all}"
COMPILE_MODEL="${COMPILE_MODEL:-1}"

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "Offline token array is missing: $DATASET_PATH" >&2
    exit 2
fi
if [[ ! -f data/asset_manifest.json ]]; then
    echo "Asset manifest is missing: data/asset_manifest.json" >&2
    exit 2
fi

# A job owns one fresh result root.  This prevents concurrent or repeated jobs
# from overwriting manifests, metrics, and checkpoints from another run.
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/jobs/${SLURM_JOB_ID:?Missing SLURM_JOB_ID}}"
if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
fi
mkdir -p "$OUTPUT_ROOT"
OUTPUT_OWNER="$OUTPUT_ROOT/.job-owner"
# Claim ownership atomically.  Directory creation alone cannot distinguish two
# jobs racing on the same explicit root, and empty pre-created roots are valid.
if ! ( set -o noclobber; printf '%s\n' "${SLURM_JOB_ID}" > "$OUTPUT_OWNER" ) 2>/dev/null; then
    echo "Refusing to reuse already-owned OUTPUT_ROOT: $OUTPUT_ROOT" >&2
    exit 2
fi
if find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 ! -name '.job-owner' -print -quit | grep -q .; then
    rm -f "$OUTPUT_OWNER"
    echo "Refusing to reuse nonempty OUTPUT_ROOT: $OUTPUT_ROOT" >&2
    exit 2
fi

"$PYTHON" - "$PROJECT_ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
print("python executable:", sys.executable)
print("python version:", sys.version)
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required")

import matplotlib
import numpy
import scipy
import torch
import rmt

actual = Path(rmt.__file__).resolve()
expected = root / "rmt" / "__init__.py"
print("rmt source:", actual)
if actual != expected:
    raise SystemExit(f"wrong rmt package: expected {expected}, got {actual}")
if not torch.cuda.is_available():
    raise SystemExit("the selected Torch installation cannot access the allocated GPU")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("the launcher requests BF16, but this GPU/Torch stack does not support it")
print("torch:", torch.__version__, "torch CUDA build:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

"$PYTHON" scripts/download_assets.py --root "$PROJECT_ROOT" --verify-only

COMMON_ARGS=(
    --execute
    --model-type causal_transformer
    --dataset-path "$DATASET_PATH"
    --vocab-size 50257
    --device cuda
    --amp-dtype bfloat16
    --allow-tf32
    --svd-backend cuda
    --svd-driver gesvdj
    --covariance-device cuda
    --covariance-dtype float32
    --pin-memory
)
case "$COMPILE_MODEL" in
    1|true|yes) COMMON_ARGS+=(--compile-model) ;;
    0|false|no) COMMON_ARGS+=(--no-compile-model) ;;
    *) echo "COMPILE_MODEL must be 1/0, true/false, or yes/no" >&2; exit 2 ;;
esac

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
        --output-dir "$OUTPUT_ROOT/paper1"
}

run_paper2() {
    echo "[Spectral-Chinchilla] Track 2: Thamm level statistics"
    srun --ntasks=1 "$PYTHON" run_experiments.py \
        "${COMMON_ARGS[@]}" \
        --experiment-mode reproduce_paper2 \
        --aspect-ratio-mode raw \
        --mp-fit-method thamm_modified_singular \
        --spike-detector tracy_widom_95 \
        --unfolding-strategy polynomial_chebyshev \
        --unfolding-degree 15 \
        --compute-spacing-distribution \
        --compute-number-variance \
        --compute-delta3 \
        --output-dir "$OUTPUT_ROOT/paper2"
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
        --output-dir "$OUTPUT_ROOT/paper3"
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
        --output-dir "$OUTPUT_ROOT/golden_compute_optimal"
}

case "$TRACK" in
    paper1) run_paper1 ;;
    paper2) run_paper2 ;;
    paper3) run_paper3 ;;
    golden) run_golden ;;
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

echo "[Spectral-Chinchilla] requested track set completed: $OUTPUT_ROOT"
