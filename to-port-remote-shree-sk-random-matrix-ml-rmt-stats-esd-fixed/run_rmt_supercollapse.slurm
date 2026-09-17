#!/bin/bash
#SBATCH --job-name=rmt_supercollapse
#SBATCH --array=0-15
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/rmt_sc_%A_%a.out
#SBATCH --error=slurm_logs/rmt_sc_%A_%a.err
#
# One array task per (width, parameterization): 8 widths x 2 modes = 16 tasks.
# Inside a task, the 36 (seed, fraction) units run one per CPU.
#
#   array 0-7   -> tag mlp          (muP)
#   array 8-15  -> tag mlp_no_mup   (ablation)
#
# Results land exactly here (muP and the ablation in SEPARATE roots):
#   RMT_Out_D1024/mlp_D1024_N5_seed0_frac0p50/
#   nomup_RMT_Out_D1024/mlp_no_mup_D1024_N5_seed0_frac0p50/
#
# Submit:
#   mkdir -p slurm_logs
#   sbatch --array=0 run_rmt_supercollapse.slurm          # test one width first
#   sbatch --array=1-15 run_rmt_supercollapse.slurm

set -uo pipefail

# --- EDIT THESE -------------------------------------------------------------
RMT_DIR=$HOME/remote-shree-sk-random-matrix-ml-rmt-stats-esd-fixed             # holds rmt/ and rmt_supercollapse.py
CKPT_DIR=$HOME/supercollapse/ckpts
OUT_ROOT=$HOME/remote-shree-sk-random-matrix-ml-rmt-stats-esd-fixed                              # RMT_Out_D* / nomup_RMT_Out_D* go here
RMT_ENV=rmt_ml_env                                 # conda env with torch + scipy
# ----------------------------------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RMT_ENV"

# one BLAS thread per worker, or the 18 workers fight over the socket
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

cd "$RMT_DIR"

WIDTHS=(384 512 645 812 1024 1290 1625 2048)
TAGS=(mlp mlp_no_mup)
FRACS=(0.00 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00)

TID=${SLURM_ARRAY_TASK_ID:-0}
D=${WIDTHS[$(( TID % 8 ))]}
TAG=${TAGS[$(( TID / 8 ))]}

# muP -> RMT_Out_D<D> ; ablation -> nomup_RMT_Out_D<D>
if [[ "$TAG" == "mlp" ]]; then
    OUT="${OUT_ROOT}/RMT_Out_D${D}"
else
    OUT="${OUT_ROOT}/nomup_RMT_Out_D${D}"
fi
NCPU=${SLURM_CPUS_PER_TASK:-4}

mkdir -p "$OUT"
rm -rf "$OUT"/.tmp_t${TID}_*         # scratch left by a killed/timed-out run
echo "task $TID: tag=$TAG D=$D out=$OUT cpus=$NCPU node=$(hostname) start=$(date)"

run_unit() {
    S=$1; F=$2
    FT=${F/./p}                                   # 0.05 -> 0p05
    DEST="${OUT}/${TAG}_D${D}_N5_seed${S}_frac${FT}"
    [[ -f "$DEST/done" ]] && { echo "skip seed=$S frac=$F"; return 0; }

    # Analyse into a private scratch dir, then move the result into place. The
    # script writes a per-width roll-up CSV at the top of --out, so the 36
    # workers sharing one --out would race on that single file.
    TMP=$(mktemp -d "${OUT}/.tmp_t${TID}_${S}_${FT}_XXXX")
    python rmt_supercollapse.py \
        --ckpt-dir "$CKPT_DIR" --D "$D" --N 5 --tag "$TAG" \
        --seeds "$S" --fracs "$F" --out "$TMP" --backend numpy \
        > "$TMP/unit.log" 2>&1
    rc=$?

    if [[ $rc -eq 0 && -d "$TMP/${TAG}_D${D}_N5_seed${S}_frac${FT}" ]]; then
        rm -rf "$DEST"
        mv "$TMP/${TAG}_D${D}_N5_seed${S}_frac${FT}" "$DEST"
        mv "$TMP"/*.csv "$DEST"/ 2>/dev/null       # the roll-up CSVs
        mv "$TMP/unit.log" "$DEST"/
        touch "$DEST/done"
        rm -rf "$TMP"
        echo "ok   seed=$S frac=$F -> $DEST"
    else
        echo "FAIL seed=$S frac=$F (rc=$rc) see $TMP/unit.log" >&2
    fi
    return 0
}
export -f run_unit
export OUT TAG D TID CKPT_DIR

for S in 0 1 2; do for F in "${FRACS[@]}"; do echo "$S $F"; done; done \
    | xargs -P "$NCPU" -n 2 bash -c 'run_unit "$0" "$1"'

echo "task $TID done=$(date)  folders=$(ls -d "$OUT"/${TAG}_D${D}_*/ 2>/dev/null | wc -l)/36"
