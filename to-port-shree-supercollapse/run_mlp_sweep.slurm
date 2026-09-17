#!/bin/bash
#SBATCH --job-name=sc_mlp
#SBATCH --output=slurm_logs/sc_mlp_%A_%a.log
#SBATCH --error=slurm_logs/sc_mlp_%A_%a.error
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-23

# ---------------------------------------------------------------------------
# Supercollapse, MLP task — one array task per (width, seed).
#
#   24 tasks = 8 widths x 3 seeds. Each task trains ONE model to its
#   compute-optimal horizon and writes one CSV to local_logs/.
#
# WHY AN ARRAY AND NOT ONE LONG JOB
#   The horizon scales steeply with width: D=384 is ~33k steps, D=2048 is
#   ~1.08M. Run serially that is ~30h per seed, ~4 days for three — right at
#   the wall-clock limit with no margin. As an array the tasks run in
#   parallel, each one is independently resumable, and a single failure does
#   not cost you the sweep.
#
# WHY 3 SEEDS
#   Supercollapse is the claim that the spread ACROSS WIDTHS falls below the
#   seed-to-seed noise floor. With one seed there is no noise floor to compare
#   against — you would only be showing ordinary collapse. The authors used 5.
#
# MODE
#   sbatch run_mlp_sweep.slurm            -> muP (tag: mlp)
#   sbatch --export=ALL,MODE=no_mup run_mlp_sweep.slurm   -> ablation
#
# RESUME
#   Re-running skips any (D, seed) whose CSV already exists. To redo only the
#   failed tasks:  sbatch --array=7,19,22 run_mlp_sweep.slurm
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/home/shivansh/supercollapse
cd "$REPO"
mkdir -p slurm_logs local_logs ckpts

# --- offline guarantees (no network at run time) ---------------------------
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# JAX grabs 75% of the card up front by default. Harmless on a dedicated GPU,
# but this makes the job a good neighbour and avoids surprises if the node is
# shared.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# --- weight-matrix checkpoints for the RMT suite ---------------------------
# Set SAVE_CKPT=0 to train without checkpoints.
export SAVE_CKPT="${SAVE_CKPT:-1}"
export CKPT_DIR="${CKPT_DIR:-$PWD/ckpts}"

# --- environment -----------------------------------------------------------
# Do NOT module-load cuda here: the CUDA 12 runtime comes from the pip wheels
# inside the conda env, and a module can shadow them via LD_LIBRARY_PATH.
module purge 2>/dev/null || true

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate shreesupercollapse
    PY=python
else
    PY="$CONDA_BASE/envs/shreesupercollapse/bin/python"
fi
echo "[sc] python: $($PY -c 'import sys; print(sys.executable)')"

# jax's CUDA 12 runtime ships inside the pip wheels (site-packages/nvidia/*/lib).
# `module purge` above clears LD_LIBRARY_PATH, so without this the dynamic loader
# cannot resolve libcusparse.so.12 / libcublas / libcudnn and JAX silently falls
# back to CPU. Pointing at the wheels (rather than /usr/local/cuda-12.1) keeps the
# job independent of whatever CUDA a given node happens to have installed.
NVLIBS="$($PY -c "
import os, glob, nvidia
base = os.path.dirname(nvidia.__file__)
print(':'.join(sorted(glob.glob(os.path.join(base, '*', 'lib')))))
" 2>/dev/null)"
if [ -n "$NVLIBS" ]; then
    export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[sc] CUDA libs from wheels: $(echo "$NVLIBS" | tr ':' '\n' | wc -l) dirs"
else
    echo "[sc] WARNING: could not locate nvidia wheel libs; falling back to system CUDA"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# --- what did we actually get? ---------------------------------------------
echo "[sc] node: $(hostname)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || \
    echo "[sc] WARNING: nvidia-smi unavailable"

# --- sanity gate: fail loudly and early, not 20 minutes in ------------------
$PY - <<'PYEOF' || { echo "[sc] environment check FAILED — aborting"; exit 1; }
import sys, jax, flax
print(f"[sc] jax {jax.__version__}  flax {flax.__version__}")
devs = jax.devices()
print(f"[sc] devices: {devs}")
assert any(d.platform == "gpu" for d in devs), "no GPU visible to JAX"
from flax.nnx.training.optimizer import _opt_state_variables_to_state  # noqa
print("[sc] environment OK")
PYEOF

# --- resolve this array task to a (width, seed) ----------------------------
WIDTHS=(384 512 645 812 1024 1290 1625 2048)
SEEDS=(0 1 2)

TID=${SLURM_ARRAY_TASK_ID:-0}
W_IDX=$(( TID / ${#SEEDS[@]} ))
S_IDX=$(( TID % ${#SEEDS[@]} ))
D=${WIDTHS[$W_IDX]}
S=${SEEDS[$S_IDX]}

MODE="${MODE:-mup}"
if [ "$MODE" = "no_mup" ]; then
    TAG=mlp_no_mup
    EXTRA="model.match_mup_at_D=384"
else
    TAG=mlp
    EXTRA=""
fi

echo "[sc] task ${TID}: D=${D} seed=${S} tag=${TAG} node=$(hostname)"

# --- skip if already finished ----------------------------------------------
if ls local_logs/${TAG}_D${D}_N5_seed${S}_*.csv >/dev/null 2>&1; then
    echo "[sc] already have a log for D=${D} seed=${S} — skipping."
    exit 0
fi

# --- train -----------------------------------------------------------------
# `-cn local` carries the paper settings (N=5, lr=1e-3, linear schedule,
# readout_lr_mult=1) and the fitted scale/exponent that derive the horizon.
srun $PY main.py -cn local \
    wandb_tag=${TAG} \
    model.D=${D} \
    seed=${S} \
    ${EXTRA}

echo "[sc] task ${TID} done: D=${D} seed=${S}"
ls -la local_logs/${TAG}_D${D}_N5_seed${S}_*.csv
