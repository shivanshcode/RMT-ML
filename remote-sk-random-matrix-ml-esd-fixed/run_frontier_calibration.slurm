#!/bin/bash
#SBATCH --job-name=rmt_frontier_calibration
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
PY="${RMT_FRONTIER_PYTHON:-$HOME/venvs/rmt-frontier/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "The frontier Python interpreter is not executable: $PY" >&2
    exit 2
fi
"$PY" - <<'PY'
import torch
import transformers

if transformers.__version__ != "4.57.6":
    raise SystemExit(
        f"transformers 4.57.6 is required, but {transformers.__version__} is installed"
    )
from transformers import Qwen2ForCausalLM, Qwen3ForCausalLM  # noqa: F401
print("frontier loader:", transformers.__version__, "torch:", torch.__version__)
PY

RESULTS_ROOT="${RMT_RESULTS_ROOT:-$HOME/rmt-results}"
SNAPSHOT_ROOT="${RMT_SNAPSHOT_ROOT:-$HOME/rmt-result-snapshots}"
MODELS_ROOT="${RMT_MODELS_ROOT:-$HOME/rmt-model-stage}"
TEMP_ROOT="${RMT_TMPDIR:-/tmp/$USER/rmt-frontier}"
mkdir -p "$PROJECT_ROOT/logs" "$RESULTS_ROOT" "$SNAPSHOT_ROOT" "$MODELS_ROOT" "$TEMP_ROOT"

MODEL_TEXT="${RMT_FRONTIER_MODELS:-qwen2.5-0.5b-instruct qwen3-0.6b qwen3-1.7b deepseek-r1-distill-qwen-1.5b}"
read -r -a MODELS <<< "$MODEL_TEXT"

GPU_ARGS=()
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    GPU_ARGS=(--gpu-index "${GPU_INDEX:-1}")
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
exec "$PY" -m rmt.frontier_sweep \
    --models "${MODELS[@]}" \
    "${GPU_ARGS[@]}" \
    --models-root "$MODELS_ROOT" \
    --results-root "$RESULTS_ROOT" \
    --snapshot-root "$SNAPSHOT_ROOT" \
    --temp-root "$TEMP_ROOT" \
    --weight-layer-strategy "${RMT_WEIGHT_LAYER_STRATEGY:-representative}" \
    --representative-layers "${RMT_REPRESENTATIVE_LAYERS:-3}" \
    --layers-per-task "${RMT_LAYERS_PER_TASK:-1}" \
    --decile-layer-strategy "${RMT_DECILE_LAYER_STRATEGY:-representative}" \
    --perplexity-tokens "${RMT_PERPLEXITY_TOKENS:-512}" \
    --activation-batches "${RMT_ACTIVATION_BATCHES:-1}" \
    --activation-length "${RMT_ACTIVATION_LENGTH:-512}" \
    --snapshot-interval "${RMT_SNAPSHOT_INTERVAL:-3600}" \
    --snapshot-retention "${RMT_SNAPSHOT_RETENTION:-6}"
