#!/bin/bash
# Reproduce supercollapse on the MLP task. Fully offline.
#   ./run_mlp_sweep.sh          -> muP sweep (widths x seeds)
#   ./run_mlp_sweep.sh no_mup   -> muP ablation (collapse should degrade)
set -euo pipefail

export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

WIDTHS=(384 512 645 812 1024 1290 1625 2048)
SEEDS=(0 1 2)
MODE="${1:-mup}"

if [ "$MODE" = "no_mup" ]; then
  TAG=mlp_no_mup; EXTRA="model.match_mup_at_D=384"
else
  TAG=mlp;        EXTRA=""
fi

mkdir -p local_logs
for D in "${WIDTHS[@]}"; do
  for S in "${SEEDS[@]}"; do
    if ls local_logs/${TAG}_D${D}_N5_seed${S}_* >/dev/null 2>&1; then
      echo "skip D=$D seed=$S (already done)"; continue
    fi
    echo "=== D=$D seed=$S tag=$TAG ==="
    python main.py -cn local wandb_tag=$TAG model.D=$D seed=$S $EXTRA
  done
done

echo "Done. Now: python make_logs.py --tag $TAG --out logs/${TAG}.pkl"
