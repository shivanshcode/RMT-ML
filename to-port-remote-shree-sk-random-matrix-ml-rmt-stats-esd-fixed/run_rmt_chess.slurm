#!/bin/bash
#SBATCH --job-name=rmt_chess
#SBATCH --array=0-15
#SBATCH --partition=defaultq
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/rmt_chess_%A_%a.out
#SBATCH --error=slurm_logs/rmt_chess_%A_%a.err
#
# RMT analysis of the CHESS TRANSFORMER checkpoints.
# Same structure as run_rmt_supercollapse.slurm; four things differ, all of
# them forced by the chess ladder rather than by taste:
#
#   1. N=3 blocks, not 5      -> layers 0..2, 6 matrices per checkpoint
#   2. 5 seeds, not 3         -> 60 (seed, frac) units per task, not 36
#      (and 18 matrices per checkpoint, not 6 --- so each unit is ~3x the work)
#   3. widths 768..2048       -> the transformer ladder
#   4. --partition=defaultq   -> gpulong is GPU-only by cluster policy and this
#                                is pure CPU work (--backend numpy)
#
# WHAT IS IN A CHESS CHECKPOINT
#   Written by checkpoint_chess.py: all 18 block matrices --- per block,
#   attn.{query,key,value,output}_proj (Q/K/V/O) and mlp.fc{1,2} (U/D) ---
#   plus embed / pos_embed / readout, which rmt.discovery skips unless
#   --include-embed is passed. Read back by rmt_supercollapse_chess.py.
#
#   The MLP ladder keeps checkpoint.py + rmt_supercollapse.py untouched, so
#   nothing already computed for mlp_* changes.
#
#   The muP and no-muP chess sweeps MUST both be trained with
#   checkpoint_chess.py. A ladder with 6 matrices cannot be compared against
#   one with 18.
#
#   array 0-7   -> tag chess          (muP)
#   array 8-15  -> tag chess_no_mup   (ablation)
#
# Results:
#   RMT_Out_chess_D1024/chess_D1024_N3_seed0_frac0p50/
#   nomup_RMT_Out_chess_D1024/chess_no_mup_D1024_N3_seed0_frac0p50/
#
# Submit:
#   mkdir -p slurm_logs
#   sbatch --array=0 run_rmt_chess.slurm        # one width first
#   sbatch --array=1-7 run_rmt_chess.slurm      # rest of muP
#   sbatch --array=8-15 run_rmt_chess.slurm     # ablation, once it has run
#
#   NOTE: this account has MaxSubmit=15, and Slurm counts array tasks
#   individually. --array=0-15 is 16 submissions and will be REJECTED.
#   Split it, as above.

set -uo pipefail

# --- EDIT THESE -------------------------------------------------------------
RMT_DIR=$HOME/remote-shree-sk-random-matrix-ml-rmt-stats-esd-fixed   # holds rmt/ and rmt_supercollapse_chess.py
CKPT_DIR=$HOME/supercollapse/ckpts
OUT_ROOT=$HOME/remote-shree-sk-random-matrix-ml-rmt-stats-esd-fixed  # RMT_Out_chess_D* go here
RMT_ENV=rmt_ml_env                                                   # conda env with torch + scipy
# ----------------------------------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RMT_ENV"

# one BLAS thread per worker, or the 18 workers fight over the socket
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg

cd "$RMT_DIR"

WIDTHS=(768 896 1024 1152 1280 1536 1792 2048)
TAGS=(chess chess_no_mup)
FRACS=(0.00 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00)
SEEDS=(0 1 2 3 4)
NBLOCKS=3

TID=${SLURM_ARRAY_TASK_ID:-0}
D=${WIDTHS[$(( TID % 8 ))]}
TAG=${TAGS[$(( TID / 8 ))]}

if [[ "$TAG" == "chess" ]]; then
    OUT="${OUT_ROOT}/RMT_Out_chess_D${D}"
else
    OUT="${OUT_ROOT}/nomup_RMT_Out_chess_D${D}"
fi
NCPU=${SLURM_CPUS_PER_TASK:-4}
# ANALYSE EVERYTHING BY DEFAULT: the 18 block matrices plus the two embeddings
# and the readout (21 in total). Set INCLUDE_EMBED=0 to analyse only the 18.
#   sbatch --export=ALL,INCLUDE_EMBED=0 --array=2 run_rmt_chess.slurm
INCLUDE_EMBED="${INCLUDE_EMBED:-1}"
[ "$INCLUDE_EMBED" = "0" ] && INCLUDE_EMBED=""

# NOTE: --layers is deliberately NOT passed. The embeddings and readout live
# outside model.layers, so extract_layer_index returns -1 for them and any
# --layers filter would drop them. Omitting it analyses every layer and every
# matrix, which is the point.

mkdir -p "$OUT"
rm -rf "$OUT"/.tmp_t${TID}_*         # scratch left by a killed/timed-out run
echo "task $TID: tag=$TAG D=$D N=$NBLOCKS out=$OUT cpus=$NCPU node=$(hostname) start=$(date)"

# Fail early and legibly rather than 60 times inside xargs.
if ! ls "$CKPT_DIR"/${TAG}_D${D}_N${NBLOCKS}_seed*_frac*.npz >/dev/null 2>&1; then
    echo "ERROR: no checkpoints matching ${TAG}_D${D}_N${NBLOCKS}_seed*_frac*.npz in $CKPT_DIR" >&2
    echo "       (is the training sweep for this width finished?)" >&2
    exit 2
fi

run_unit() {
    S=$1; F=$2
    FT=${F/./p}                                   # 0.05 -> 0p05
    DEST="${OUT}/${TAG}_D${D}_N${NBLOCKS}_seed${S}_frac${FT}"
    [[ -f "$DEST/done" ]] && { echo "skip seed=$S frac=$F"; return 0; }

    # Analyse into a private scratch dir, then move the result into place. The
    # script writes a per-width roll-up CSV at the top of --out, so the 60
    # workers sharing one --out would race on that single file.
    TMP=$(mktemp -d "${OUT}/.tmp_t${TID}_${S}_${FT}_XXXX")
    python rmt_supercollapse_chess.py \
        --ckpt-dir "$CKPT_DIR" --D "$D" --N "$NBLOCKS" --tag "$TAG" \
        --seeds "$S" --fracs "$F" --out "$TMP" --backend numpy \
        ${INCLUDE_EMBED:+--include-embed} \
        > "$TMP/unit.log" 2>&1
    rc=$?

    if [[ $rc -eq 0 && -d "$TMP/${TAG}_D${D}_N${NBLOCKS}_seed${S}_frac${FT}" ]]; then
        rm -rf "$DEST"
        mv "$TMP/${TAG}_D${D}_N${NBLOCKS}_seed${S}_frac${FT}" "$DEST"
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
export OUT TAG D TID CKPT_DIR NBLOCKS INCLUDE_EMBED

NUNITS=$(( ${#SEEDS[@]} * ${#FRACS[@]} ))
for S in "${SEEDS[@]}"; do for F in "${FRACS[@]}"; do echo "$S $F"; done; done \
    | xargs -P "$NCPU" -n 2 bash -c 'run_unit "$0" "$1"'

echo "task $TID done=$(date)  folders=$(ls -d "$OUT"/${TAG}_D${D}_*/ 2>/dev/null | wc -l)/${NUNITS}"
