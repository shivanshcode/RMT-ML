#!/bin/bash
#SBATCH --job-name=sc_dev
#SBATCH --output=slurm_logs/sc_dev_%j.log
#SBATCH --error=slurm_logs/sc_dev_%j.error
#SBATCH --partition=gpulong
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00

# ---------------------------------------------------------------------------
# Figure 1c / 3c: collapse deviation Delta(x) vs per-model noise floor
# sigma(x,p), for the muP sweep and the no-muP ablation.
#
#   sbatch run_collapse_deviation.slurm
#
# This is CPU-only work and takes seconds. Deliberately NOT modelled on
# run_mlp_sweep.slurm's GPU path:
#
#   * No --gres. Nothing here imports jax, so no GPU is requested. The
#     partition is gpulong only because that is the one 2_HANDOFF.md confirms
#     exists on this cluster; if there is a CPU partition, point at it instead
#     — that is the single line to change.
#   * No NVLIBS block. run_mlp_sweep.slurm needs it because jax's CUDA runtime
#     lives in the pip wheels and `module purge` hides it. Carrying that here
#     would only imply a GPU dependency that does not exist.
#   * No job array. Both pickles run in one task; they take seconds each and
#     splitting them would lose the comparison figure, which is the point.
#
# The honest note: this is small enough to run on a login node. The Slurm
# wrapper exists so it is reproducible and logged alongside the sweeps, not
# because it needs the scheduler.
#
# OVERRIDES (all optional)
#   SC_REPO       repo root                      default /home/shivansh/supercollapse
#   SC_LOGS       directory holding the pickles  default $SC_REPO/logs
#   SC_OUT        output directory               default $SC_REPO/figures_out
#   ANALYSIS_ENV  conda env name                 default shreesupercollapse
#   SC_MIN_D      smallest width to keep         default 384
#   SC_GRID       grid points                    default 201
#   SC_SMOOTH     display-only rolling median    default 0 (off)
#
#   sbatch --export=ALL,SC_SMOOTH=9,ANALYSIS_ENV=rmt run_collapse_deviation.slurm
# ---------------------------------------------------------------------------

set -euo pipefail

REPO="${SC_REPO:-/home/shivansh/supercollapse}"
cd "$REPO"
mkdir -p slurm_logs

LOGS="${SC_LOGS:-$REPO/logs}"
OUT="${SC_OUT:-$REPO/figures_out}"
MIN_D="${SC_MIN_D:-384}"
GRID="${SC_GRID:-201}"
SMOOTH="${SC_SMOOTH:-0}"
ENV_NAME="${ANALYSIS_ENV:-shreesupercollapse}"

mkdir -p "$OUT"

# Headless node: no display, and matplotlib must not try to find one.
export MPLBACKEND=Agg
# The power-law fit is ~100 small L-BFGS restarts. BLAS threads buy nothing
# here and oversubscribing a shared node is rude.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- environment -----------------------------------------------------------
module purge 2>/dev/null || true

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    PY=python
else
    PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
fi
echo "[sc] node:   $(hostname)"
echo "[sc] python: $($PY -c 'import sys; print(sys.executable)')"

# --- import gate: fail in 2 seconds, not 2 minutes -------------------------
# matplotlib is the one that can genuinely be missing from the training env,
# which was built for jax and nothing else.
$PY - <<'PYEOF' || { echo "[sc] dependency check FAILED — aborting"; exit 1; }
import importlib, sys
missing = []
for m in ("numpy", "pandas", "scipy", "matplotlib"):
    try:
        importlib.import_module(m)
    except ImportError:
        missing.append(m)
if missing:
    print(f"[sc] missing: {', '.join(missing)}")
    print(f"[sc] fix with: pip install {' '.join(missing)}")
    sys.exit(1)
try:
    importlib.import_module("seaborn")
except ImportError:
    print("[sc] note: seaborn absent — matplotlib defaults will be used")
print("[sc] dependencies OK")
PYEOF

# --- which pickles do we actually have? ------------------------------------
MUP="$LOGS/mlp_mup.pkl"
NOMUP="$LOGS/mlp_no_mup.pkl"

if [ ! -f "$MUP" ]; then
    echo "[sc] ERROR: $MUP not found."
    echo "[sc] Build it first:  python make_logs.py --tag mlp --out logs/mlp.pkl"
    exit 2
fi

COMMON=(--out-dir "$OUT" --min-D "$MIN_D" --grid "$GRID" --smooth "$SMOOTH" \
        --per-model --no-show)

# --- per-sweep figures -----------------------------------------------------
echo "[sc] === muP sweep ==="
$PY plot_collapse_deviation.py "$MUP" --labels muP --name mlp "${COMMON[@]}"

if [ -f "$NOMUP" ]; then
    echo "[sc] === no-muP ablation ==="
    $PY plot_collapse_deviation.py "$NOMUP" --labels no-muP --name mlp_no_mup \
        "${COMMON[@]}"

    # --- the actual deliverable: 2_HANDOFF.md section 10 item 1 ------------
    # Each ladder gets its OWN L0 fit, as the paper does. Read the verdict line
    # at the end of this log.
    echo "[sc] === muP vs no-muP ==="
    $PY plot_collapse_deviation.py "$MUP" "$NOMUP" \
        --labels muP no-muP --name mlp_mup_vs_nomup "${COMMON[@]}"
else
    echo "[sc] NOTE: $NOMUP not found — skipping the comparison."
    echo "[sc] Build it with: python make_logs.py --tag mlp_no_mup --out logs/mlp_no_mup.pkl"
fi

echo "[sc] done. Outputs in $OUT"
ls -la "$OUT"
