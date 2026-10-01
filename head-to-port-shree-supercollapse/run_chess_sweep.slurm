#!/bin/bash
#SBATCH --job-name=sc_chess
#SBATCH --output=slurm_logs/sc_chess_%A_%a.log
#SBATCH --error=slurm_logs/sc_chess_%A_%a.error
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10-00:00:00
#SBATCH --array=0-39

# ---------------------------------------------------------------------------
# Supercollapse, chess transformer — one array task per (width, seed).
#
#   40 tasks = 8 widths x 5 seeds. Each trains ONE model to its compute-optimal
#   horizon and writes one CSV to local_logs/.
#
# WHY 5 SEEDS, NOT THE MLP SWEEP'S 3
#   experiments/chess.yaml uses [0,1,2,3,4] and says why: "5 seeds is useful
#   for accurating fitting L0, necessary for the best collapse". The MLP round
#   bore this out — the entire muP-vs-no-muP effect size turned on how well L0
#   was pinned, and muP's L0 doubled when one width was dropped. L0 is fitted
#   to the seed-AVERAGED Pareto points, so seeds buy fit quality directly.
#   Also reproduces the authors' logs/chess.pkl exactly (8 widths x 5 seeds).
#
# MODE
#   sbatch run_chess_sweep.slurm                            -> muP  (tag: chess)
#   sbatch --export=ALL,MODE=no_mup run_chess_sweep.slurm   -> ablation
#
# RESUME
#   Re-running skips any (D, seed) whose CSV already exists. To redo only the
#   failures:  sbatch --array=7,19,22 run_chess_sweep.slurm
#
# COST — read this before queueing
#   The horizon scales steeply: D=768 is ~10B tokens, D=2048 ~19B, i.e. roughly
#   150k and 290k steps at 65,536 tokens/step. This is 10-30x the MLP sweep's
#   compute. --time is 10 days per task (gpulong allows up to 50); tasks run in
#   parallel so wall-clock is set by the largest, not the sum. Run ONE task
#   first (--array=0) and read the step rate off the log before committing all
#   24, then again for both parameterizations.
#
# DIFFERENCES FROM run_mlp_sweep.slurm
#   * N=3 not N=5 (affects the resume glob).
#   * match_mup_at_D=768, the smallest chess width (MLP used 384).
#   * Reads a dataset from disk. The MLP task generated data on device, so this
#     is the first time file I/O matters at all — see the staging block below.
#   * Entry point is main_chess.py, not main.py, so checkpoints include
#     attention. If you ever switch back to main.py mid-sweep the muP and
#     no-muP ladders stop being comparable.
#   * --mem 64G not 32G: L=128 x B=512 activations plus the memmap page cache.
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

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# --- weight-matrix checkpoints for the RMT suite ---------------------------
export SAVE_CKPT="${SAVE_CKPT:-1}"
export CKPT_DIR="${CKPT_DIR:-$PWD/ckpts}"

# --- environment -----------------------------------------------------------
# Do NOT module-load cuda: the CUDA 12 runtime comes from the pip wheels inside
# the conda env, and a module can shadow them via LD_LIBRARY_PATH.
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
# `module purge` clears LD_LIBRARY_PATH, so without this the loader cannot
# resolve libcusparse.so.12 / libcublas / libcudnn and JAX silently falls back
# to CPU — which on this sweep would mean weeks, not hours, before you noticed.
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

echo "[sc] node: $(hostname)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || \
    echo "[sc] WARNING: nvidia-smi unavailable"

# --- sanity gate: fail loudly and early ------------------------------------
$PY - <<'PYEOF' || { echo "[sc] environment check FAILED — aborting"; exit 1; }
import sys, jax, flax
print(f"[sc] jax {jax.__version__}  flax {flax.__version__}")
devs = jax.devices()
print(f"[sc] devices: {devs}")
assert any(d.platform == "gpu" for d in devs), "no GPU visible to JAX"
from flax.nnx.training.optimizer import _opt_state_variables_to_state  # noqa
print("[sc] environment OK")
PYEOF

# --- dataset present? ------------------------------------------------------
for f in train.bin test.bin meta.pkl; do
    [ -f "$REPO/chess/$f" ] || { echo "[sc] ERROR: chess/$f missing — run prepare_chess.py"; exit 2; }
done

# --- stage the dataset on node-local disk ----------------------------------
# data.py re-opens the memmap on EVERY batch (~290k times for D=2048). That is
# fine on the node's local RAID (/ has ~760G free) and potentially crippling
# over Lustre. Copy once PER NODE, not per task: an array can place several
# tasks on one node and they should share the copy, so this is behind a lock.
#
# Set STAGE_DATA=0 to train straight off Lustre. Worth measuring once: if the
# step rate is the same either way, train.bin is fitting in page cache and this
# block is dead weight.
STAGE_DATA="${STAGE_DATA:-1}"
DS_PATH="$REPO/chess"
if [ "$STAGE_DATA" = "1" ]; then
    STAGE=/tmp/${USER}_chess
    mkdir -p "$STAGE"
    (
      flock -x 9
      if [ ! -f "$STAGE/.done" ]; then
        echo "[sc] staging dataset to $STAGE on $(hostname)"
        cp "$REPO/chess/train.bin" "$REPO/chess/test.bin" \
           "$REPO/chess/meta.pkl" "$STAGE/"
        touch "$STAGE/.done"
      else
        echo "[sc] reusing staged dataset on $(hostname)"
      fi
    ) 9>"$STAGE.lock"
    DS_PATH="$STAGE"
fi
echo "[sc] ds_path=$DS_PATH"
# NOTE: no cleanup trap here on purpose. With a shared per-node copy, the first
# task to finish would delete the data out from under its neighbours. Sweep up
# afterwards with:
#   srun -p short -t 00:05:00 bash -c 'rm -rf /tmp/'$USER'_chess*'

# --- resolve this array task to a (width, seed) ----------------------------
WIDTHS=(768 896 1024 1152 1280 1536 1792 2048)
SEEDS=(0 1 2 3 4)

TID=${SLURM_ARRAY_TASK_ID:-0}
W_IDX=$(( TID / ${#SEEDS[@]} ))
S_IDX=$(( TID % ${#SEEDS[@]} ))
D=${WIDTHS[$W_IDX]}
S=${SEEDS[$S_IDX]}

MODE="${MODE:-mup}"
if [ "$MODE" = "no_mup" ]; then
    TAG=chess_no_mup
    EXTRA="model.match_mup_at_D=768"
else
    TAG=chess
    EXTRA=""
fi

echo "[sc] task ${TID}: D=${D} seed=${S} tag=${TAG} node=$(hostname)"

# --- skip if already finished ----------------------------------------------
# N3 here, not N5 — the chess ladder is 3 blocks.
if ls local_logs/${TAG}_D${D}_N3_seed${S}_*.csv >/dev/null 2>&1; then
    echo "[sc] already have a log for D=${D} seed=${S} — skipping."
    exit 0
fi

# --- train -----------------------------------------------------------------
# main_chess.py -> train_chess.py -> checkpoint_chess.py, so the checkpoints
# carry all 18 block matrices (Q/K/V/O + U/D per block) rather than the 6 the
# MLP saver would keep. main.py / train.py / checkpoint.py are untouched and
# still drive the MLP ladder.
#
# `-cn chess_local` carries the paper's chess settings (N=3, L=128, lr=0.01,
# B=512, embed_lr_mult=10, linear schedule) and the fitted scale/exponent that
# derive the horizon.
srun $PY main_chess.py -cn chess_local \
    wandb_tag=${TAG} \
    ds_path=${DS_PATH} \
    model.D=${D} \
    seed=${S} \
    ${EXTRA}

echo "[sc] task ${TID} done: D=${D} seed=${S}"
ls -la local_logs/${TAG}_D${D}_N3_seed${S}_*.csv
