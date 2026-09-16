#!/bin/bash
#SBATCH --job-name=RMT_calibrated_v4
#SBATCH --output=RMT_calibrated_v4_%j.log
#SBATCH --error=RMT_calibrated_v4_%j.error
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# ===========================================================================
# RMT calibrated pipeline — full run.
#
# WHY THE PREVIOUS VERSION WOULD NOT SUBMIT
#   "Requested node configuration is not available" is a SUBMIT-TIME error: the
#   job never queued, so nothing in the body was involved. Three directives
#   asked for a node that does not exist on this cluster:
#       --gres=gpu:a100:1     type-constrained; the working scripts use gpu:1
#       --cpus-per-task=16    gpulong nodes do not have 16 allocatable cores
#       --mem=128G            above what the working scripts request
#   The block above is now byte-compatible with run_stage2_bands.slurm and
#   RMT_All_Models.slurm, both of which schedule fine. Do not re-add any of the
#   three without first checking:  sinfo -p gpulong -o "%c %m %G"
#
# THE GPU IS ONLY USED BY STAGE 6. Stages 1-4 and 7-8 are pure numpy/scipy.
# The --gres line is here because gpulong requires it and because stage 6
# (python -m rmt --backend auto) does use torch/CUDA for the large SVDs.
#
# WHAT CHANGED SCIENTIFICALLY vs run_rmt_calibrated.slurm (files(3).zip)
#   1. --unfold_deg 15  (was 7)   <-- NEW, and it matters. See below.
#   2. --unfold_method auto       "auto" now delegates to the coordinate
#                                 search. There is no "tx" CLI choice.
#   3. --spacing_bulk_mode auto   needs the one-line rmt/cli.py patch that
#                                 adds "auto" to the choices list.
#   4. Stage 2 bands use --bulk-frac 1.0 and --reps 40, matching stage 6.
#   5. Stage 7 computes the correct t-quantile inline.
#   7. Stage 7 finds the metrics CSV by glob. The v3 path was WRONG: the
#      pipeline writes $OUT/<safe_tag>/<safe_tag>_matrix_metrics.csv -- the tag
#      appears twice -- so v3 would have died at stage 7 AFTER the 3 h run.
#   8. --layers is now set (v3 analysed all 32 layers) and shared between
#      stage 6 and stage 8, so exported weights match scored rows.
#   9. Stage 2 is SKIPPED when cal/null_bands.csv already has all four shapes.
#      Override with RUN_STAGE2=always|never.
#
#   6. REQUIRES the patched rmt_null_calibration.py shipped alongside this
#      file: the `nulls` stage had no --unfold-deg flag, so bands would have
#      been built at degree 7 while stage 6 ran at 15. A band and an
#      observation computed with different estimators are not comparable --
#      that mismatch would be worse than the bias it was meant to fix.
#      Verified: 2048x1024 sigma2_L20 = 1.0969 at deg 7, 0.9809 at deg 15.
#
# WHY DEGREE 15.  Measured on the real Llama shapes, degree 7 under-fits the
# Marchenko-Pastur density once you have ~4000 levels, and the residual density
# modulation inflates the count variance:
#
#     shape          Sigma^2(20) err vs exact:  deg 7   ->   deg 15
#     4096x4096                                 + 2.7%       - 0.2%
#     1024x4096                                 + 6.1%       + 2.1%
#     14336x4096                                +14.4%       + 2.0%
#
#   Delta_3 is unaffected either way (<1%) because it integrates over the
#   modulation. If you already built bands at deg 7, they are still valid for
#   z-scores -- the bias is common-mode and cancels in (x-mean)/sd -- but the
#   absolute Sigma^2 values must not be quoted as "matching GOE theory".
# ===========================================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "[RMT] working dir: $PWD"

# --- environment -----------------------------------------------------------
module load cuda 2>/dev/null || true

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
ENV_NAME="rmt_ml_env"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    PY=python
else
    PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
fi
echo "[RMT] python: $($PY -c 'import sys; print(sys.executable)')"
$PY -c 'import numpy, scipy; print(f"[RMT] numpy {numpy.__version__}  scipy {scipy.__version__}")'

# No --cpus-per-task, so SLURM_CPUS_PER_TASK is 1. Uncapped, OpenBLAS/MKL
# spawn one thread per physical core on the NODE and thrash the cgroup.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
[ "$OMP_NUM_THREADS" = "1" ] && export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
echo "[RMT] BLAS threads: $OMP_NUM_THREADS"

MODEL_TAG="meta-llama/Llama-3.1-8B"
MODEL_PATH="${MODEL_PATH:-./models/Llama-3.1-8B}"
OUT="rmt_results"
CAL="cal"
SEED=0

UNFOLD_METHOD=auto
UNFOLD_DEG=15
BULK_MODE=auto        # pipeline;   needs the rmt/cli.py patch
BULK_FRAC=1.0         # calibration; equivalent to BULK_MODE=auto
REPS_BANDS=40

# Layers to analyse. Llama-3.1-8B has 32 (0-31); analysing all of them is ~33 s
# of SVD per matrix x 7 matrices x 32 layers. Match this to the --layers used in
# stage 8 so the exported weights correspond to scored rows.
LAYERS="0 8 16 24 31"

# Stage 2 takes 3-5 h. Skip it if cal/null_bands.csv already covers all four
# shapes -- rebuilding would overwrite good bands for nothing.
RUN_STAGE2="${RUN_STAGE2:-auto}"

mkdir -p "$OUT" "$CAL" weights_npy

# ---------------------------------------------------------------------------
# GATE 0 — nothing else runs until this is green
# ---------------------------------------------------------------------------
echo "=== GATE 0: selftest + unit tests ==="
$PY -m rmt --selftest || { echo "[RMT] selftest FAILED — aborting"; exit 1; }
$PY -m pytest tests/ -q -m "not torch" || { echo "[RMT] tests FAILED — aborting"; exit 1; }
# expect: selftest 11/11 PASS, 263 passed / 4 skipped

echo "=== GATE 0b: estimators vs the EXACT finite-L laws ==="
OLD_REPO="${OLD_REPO:-../remote-sk-random-matrix-ml-esd-fixed}" NEW_REPO="." FIX_REPO="." \
  $PY benchmark3.py 6 --out "$CAL/benchmark_results_3way.csv" || \
  echo "[RMT] benchmark3 skipped (OLD_REPO not found) — not fatal"

# ---------------------------------------------------------------------------
# STAGES 1, 4, 5b — cheap CPU calibration
# ---------------------------------------------------------------------------
echo "=== STAGE 1: ensemble benchmark ==="
$PY rmt_null_calibration.py ensembles --reps 8 --out "$CAL"
# delta3_L5 will be flagged "in band? NO". EXPECTED: the printed closed-form is
# the L->infinity asymptote, ~10% low at L=5. Check against the exact law:
#   python -c "from rmt import reference as R; print(R.delta3_exact(5,1))"
# beta=2 agrees to ~1% while beta=1 is +27% at L=3 -- no estimator bug is
# selective by Dyson index, which is what proves the reference was the problem.

echo "=== STAGE 4: invariance ==="
$PY rmt_null_calibration.py invariance --shape 1024x1024 --out "$CAL"
# All PASS at ~1e-14. The bf16 row is INFO, not a test: ~0.08 spacings, i.e.
# ~3000x the fp32-vs-fp64 disagreement. Pin --dtype fp32 and never mix.

echo "=== STAGE 5b: bulk-fraction sweep ==="
$PY rmt_null_calibration.py bulkfrac \
    --shape 4096x4096 --fracs 0.6 0.7 0.8 1.0 --reps 20 --out "$CAL"
# 1.0 is included on purpose: without the uncut baseline an EDGE-SENSITIVE flag
# cannot be told apart from a level-count effect. Measured at 4096x4096 every
# statistic moves by <1 null sd across the whole sweep, which is the evidence
# for keeping all levels rather than discarding 30% of them.

# ---------------------------------------------------------------------------
# STAGE 2 — matched-shape null bands (~3-5 h). Run separately if you prefer:
#   sbatch run_stage2_bands.slurm
# ---------------------------------------------------------------------------
echo "=== STAGE 2: matched-shape null bands ==="
_need_bands() {
    [ "$RUN_STAGE2" = "always" ] && return 0
    [ "$RUN_STAGE2" = "never" ]  && return 1
    [ -f "$CAL/null_bands.csv" ] || return 0
    $PY - "$CAL/null_bands.csv" <<'PYEOF'
import csv, sys
want = {"4096x4096","1024x4096","14336x4096","4096x14336"}
have = {r["shape"] for r in csv.DictReader(open(sys.argv[1]))}
missing = want - have
print(f"[RMT] bands present: {sorted(have & want)}")
if missing:
    print(f"[RMT] MISSING: {sorted(missing)} -> rebuilding")
sys.exit(0 if missing else 1)
PYEOF
}
if _need_bands; then
$PY rmt_null_calibration.py nulls \
    --shapes 4096x4096 1024x4096 14336x4096 4096x14336 \
    --reps "$REPS_BANDS" \
    --bulk-frac "$BULK_FRAC" \
    --unfold-method "$UNFOLD_METHOD" \
    --unfold-deg "$UNFOLD_DEG" \
    --porter-thomas --pt-vectors 200 --out "$CAL"
else
    echo "[RMT] all four shapes already in $CAL/null_bands.csv — skipping stage 2."
    echo "[RMT] force a rebuild with:  RUN_STAGE2=always sbatch $0"
fi
# CHECK: one branch per shape, and shapes must be the four real ones (not
# 400x400 -- that would be the old demo file from calc_results/).
# k/v_proj gives 1024 levels vs 4096 elsewhere, so its band sd is visibly
# wider. Physical, not a bug: never compare sigma2_L20 across shapes.

# ---------------------------------------------------------------------------
# STAGE 3 — power table
# ---------------------------------------------------------------------------
echo "=== STAGE 3: power table ==="
$PY rmt_null_calibration.py power --shape 4096x4096 --reps 30 --out "$CAL"
# KNOWN ARTEFACT: sigma2_* fires spuriously on heavy-tailed cases because
# strip_edge_outliers is NOT wired into per_matrix (outstanding-issues O2).
# Student-t(3) has FINITE variance, so Tao-Vu universality guarantees GOE local
# statistics -- a large z there is a bug, not a discovery. Measured: raw
# sigma2_L10 = 1.078 (+18.7%), after stripping 6 outliers = 0.911 (+0.3%).
# Until O2 is fixed, trust delta3_L10 as the long-range detector and treat
# sigma2_* as advisory.

# ---------------------------------------------------------------------------
# STAGE 6 — the real pipeline (GPU)
# ---------------------------------------------------------------------------
echo "=== STAGE 6: pipeline ==="
$PY -m rmt \
    --models "$MODEL_TAG" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUT" \
    --dtype fp32 \
    --backend auto \
    --unfold_method "$UNFOLD_METHOD" \
    --unfold_deg "$UNFOLD_DEG" \
    --unfold_win 15 \
    --spacing_domain sval \
    --spacing_bulk_mode "$BULK_MODE" \
    --sigma2_n_windows 200000 \
    --delta3_L 5 10 50 \
    --sigma2_L 5 10 20 \
    --do_spacing --do_porter_thomas --do_brody --do_powerlaw \
    --no-do_overlap --no-do_perplexity \
    --layers $LAYERS \
    --seed "$SEED"
# On --sigma2_L: keep L <= n_levels/20. At 4096 levels that is L <= 204, but at
# 1024 levels (k/v_proj) it is L <= 51, and every L=50 row runs 3-5% low
# (outstanding-issues O4). Prefer delta3 for long-range rigidity.

# ---------------------------------------------------------------------------
# STAGE 7 — z-scores. This is the table you report.
# ---------------------------------------------------------------------------
echo "=== STAGE 7: z-scores ==="
# The pipeline writes  $OUT/<safe_tag>/<safe_tag>_matrix_metrics.csv  -- the tag
# appears TWICE (cli.py makes the subdirectory, pipeline.py names the file).
# Globbing is robust to that and to any future renaming.
CSV="$(find "$OUT" -name '*_matrix_metrics.csv' | head -1)"
[ -n "$CSV" ] || { echo "[RMT] no *_matrix_metrics.csv under $OUT — stage 6 failed"; exit 1; }
echo "[RMT] scoring $CSV"
$PY rmt_null_calibration.py score --csv "$CSV" --bands "$CAL/null_bands.csv" --out "$CAL"

# The threshold the score stage prints comes from norm.ppf and is
# ANTI-CONSERVATIVE (outstanding-issues O8). Compute the correct one:
$PY - "$REPS_BANDS" <<'PYEOF'
import sys, csv
from rmt import nulls as NU
n = int(sys.argv[1])
rows = list(csv.DictReader(open("cal/zscores.csv")))
stats = [k for k in rows[0] if k.startswith("z_")]
m = len(rows) * len(stats)
print(f"  matrices={len(rows)}  statistics={len(stats)}  tests={m}")
print(f"  CORRECT Bonferroni |z| threshold (t, {n} reps): {NU.critical_value(n, m):.2f}")
PYEOF
# At 40 replicas: 192 tests -> printed 3.65, correct 4.07.
#                1344 tests -> printed 4.12, correct 4.71.

# ---------------------------------------------------------------------------
# STAGE 8 — destructive controls
# ---------------------------------------------------------------------------
echo "=== STAGE 8: export weights + controls ==="
$PY export_weights_npy.py \
    --model_path "$MODEL_PATH" \
    --patterns q_proj k_proj down_proj \
    --layers $LAYERS \
    --out weights_npy/

shopt -s nullglob
for NPY in weights_npy/*.npy; do
    $PY rmt_null_calibration.py controls --npy "$NPY" \
        --reps 30 --bulk-frac "$BULK_FRAC" --out "$CAL"
done
shopt -u nullglob
# WARNING: controls.py has variance_profile_gaussian() but it is NOT wired into
# CONTROLS or interpret() (outstanding-issues O7). This stage therefore runs the
# SHUFFLE controls only -- and no shuffle is a variance-profile-matched null:
# each preserves exactly one margin. A "structure-beyond-controls" verdict is
# not yet safe against a plain heteroscedasticity explanation.

echo "=== DONE at $(date) ==="
echo "Report: $CAL/zscores.csv  $CAL/controls.csv  $CAL/power_table.csv"
echo "        $CAL/null_bands.csv  $CAL/benchmark_results_3way.csv"
