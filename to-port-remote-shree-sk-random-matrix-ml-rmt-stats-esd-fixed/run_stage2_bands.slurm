#!/bin/bash
#SBATCH --job-name=rmt_bands
#SBATCH --output=rmt_bands_%j.log
#SBATCH --error=rmt_bands_%j.error
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00

# ===========================================================================
# Stage 2 — matched-shape null bands. Everything downstream depends on this.
#
# RESOURCE REQUEST: deliberately identical to RMT_All_Models.slurm, which is
# known to schedule on this cluster. The previous version added
# --cpus-per-task=16, which Slurm rejected with "Requested node configuration
# is not available" -- the gpulong nodes do not have 16 allocatable cores per
# task. Do not re-add it without checking `sinfo -p gpulong -o "%c %m %G"`.
#
# THE GPU IS NOT USED. rmt_null_calibration.py, rmt/nulls.py, rmt/spacing.py
# and rmt/porter_thomas.py import numpy and scipy only -- no `import torch`,
# no CUDA path in this stage. The --gres line is here purely because gpulong
# requires it. If your site has a CPU partition, switching to it will start
# sooner and free the card:  #SBATCH --partition=<cpu queue>   (drop --gres)
#
# MEMORY: 64G is far more than needed. Measured peak RSS for a float64 economy
# SVD is ~2.1x the matrix; the largest shape here is 14336x4096 = 0.47 GB, so
# ~1-2 GB per replica, run sequentially. 64G matches the known-good request.
# ===========================================================================

set -euo pipefail

# --- location --------------------------------------------------------------
# Submit from the repo root and this resolves itself. Hard-code the path here
# if you submit from elsewhere.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "[RMT] working dir: $PWD"

# --- environment -----------------------------------------------------------
module load cuda 2>/dev/null || true

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
ENV_NAME="rmt_ml_env"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    PY=python
else
    # Fallback: call the env interpreter directly. NOTE it must be the SAME
    # env as above; the snippet this was adapted from activated rmt_ml_env but
    # fell back to envs/rmt, silently running a different environment.
    PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
fi

echo "[RMT] python: $($PY -c 'import sys; print(sys.executable)')"
$PY -c 'import numpy, scipy; print(f"[RMT] numpy {numpy.__version__}  scipy {scipy.__version__}")'

# --- threading -------------------------------------------------------------
# No --cpus-per-task, so SLURM_CPUS_PER_TASK is 1. Left uncapped, OpenBLAS/MKL
# spawn one thread per physical core on the NODE, not per allocated core, and
# thrash against the cgroup. 8 is a safe compromise on a shared GPU node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
[ "$OMP_NUM_THREADS" = "1" ] && export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
echo "[RMT] BLAS threads: $OMP_NUM_THREADS"

mkdir -p cal

# --- gate ------------------------------------------------------------------
echo "[RMT] selftest gate ..."
$PY -m rmt --selftest || { echo "[RMT] selftest FAILED — aborting"; exit 1; }

# --- stage 2 ---------------------------------------------------------------
echo "[RMT] starting stage 2 at $(date)"

$PY rmt_null_calibration.py nulls \
    --shapes 4096x4096 1024x4096 14336x4096 4096x14336 \
    --reps 40 \
    --bulk-frac 1.0 \
    --unfold-method auto \
    --unfold-deg 15 \
    --porter-thomas --pt-vectors 200 \
    --out cal

echo "[RMT] done at $(date)"
ls -la cal/null_bands.csv

# ---------------------------------------------------------------------------
# CHECK BEFORE USING THE BANDS
#   1. Shapes must be the four real Llama shapes, not 400x400. If you see
#      400x400 you are reading the old demo file from calc_results/.
#   2. The unfolding branch must be a SINGLE value per shape. Mixed branches
#      mean some replicas took the local kernel and their sigma2_* are not
#      comparable with the rest.
#   3. k/v_proj (1024x4096) gives 1024 levels against 4096 for the other
#      shapes, so its band sd will be visibly wider. Physical, not a bug --
#      it is why bands are per shape and why sigma2_L20 must never be
#      compared across shapes.
#   4. delta3_L5 will be flagged "in band? NO" against the printed closed-form
#      constant. Expected: that constant is the L->infinity asymptote, which
#      is ~10% low at L=5. Score against rmt.reference.delta3_exact instead.
# ---------------------------------------------------------------------------
