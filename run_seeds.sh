#!/usr/bin/env bash
# run_seeds.sh - run one model type over several seeds, SEQUENTIALLY on ONE GPU.
#
#   bash run_seeds.sh train <model> <gpu> <seed> [<seed> ...]
#   bash run_seeds.sh dump  <model> <gpu> <seed> [<seed> ...]
#
#   <model> : headline | nocpa | nograph | relvel
#   <gpu>   : the single free GPU index (check nvidia-smi first)
#
# Examples:
#   bash run_seeds.sh train relvel   2 42 123 456
#   bash run_seeds.sh train headline 2 789 1011 1213 1415 1617
#   bash run_seeds.sh dump  relvel   2 42 123 456
#
# Safety: `train` skips any seed whose checkpoint (val_best.pth) already exists
# and `dump` skips any seed whose .npz already exists, so nothing is overwritten.
# The chain stops at the first failure. Set DRY=1 to only print the commands.
#
# Recipe (identical for every model): obs/pred 10/10, gru_layers 2, top_k 10,
# Huber delta 0.05, lr 5e-4 - the FINAL headline recipe.

set -u

if [ "$#" -lt 4 ]; then
  sed -n '2,22p' "$0"; exit 1
fi

MODE="$1"; MODEL="$2"; GPU="$3"; shift 3
SEEDS=("$@")

case "$MODEL" in
  headline) SCRIPT="train_cpagrn_huberloss.py";          ARCH="cpagrn";  TAGP="CPAGRN_v5_huber" ;;
  nocpa)    SCRIPT="train_cpagrn_nocpa_huberloss.py";    ARCH="nocpa";   TAGP="CPAGRN_v5_nocpa_huber" ;;
  nograph)  SCRIPT="train_cpagrn_nograph_huberloss.py";  ARCH="nograph"; TAGP="CPAGRN_v5_nograph_huber" ;;
  relvel)   SCRIPT="train_cpagrn_relvel_huberloss.py";   ARCH="relvel";  TAGP="CPAGRN_v5_relvel_huber" ;;
  *) echo "unknown model '$MODEL' (headline|nocpa|nograph|relvel)"; exit 1 ;;
esac
case "$MODE" in train|dump) ;; *) echo "mode must be train or dump"; exit 1 ;; esac

run() {
  echo ">>> $*"
  if [ "${DRY:-0}" = "1" ]; then return 0; fi
  "$@"
}

for SEED in "${SEEDS[@]}"; do
  TAG="${TAGP}_d05_gru2_lr5e4_obs10_pred10_s${SEED}"
  if [ "$MODE" = "train" ]; then
    if [ -f "checkpoints/${TAG}/val_best.pth" ]; then
      echo "--- skip (checkpoint exists): ${TAG}"; continue
    fi
    run python "$SCRIPT" --obs_len 10 --pred_len 10 --gru_layers 2 --top_k 10 \
        --huber_delta 0.05 --lr 5e-4 --seed "$SEED" --tag "$TAG" --gpu_num "$GPU" \
      || { echo "FAILED: ${TAG}"; exit 1; }
  else
    if [ -f "error_dumps/${TAG}__test.npz" ]; then
      echo "--- skip (dump exists): ${TAG}"; continue
    fi
    if [ ! -f "checkpoints/${TAG}/val_best.pth" ]; then
      echo "--- missing checkpoint, cannot dump: ${TAG}"; exit 1
    fi
    run python dump_errors.py --arch "$ARCH" --tag "$TAG" --gpu_num "$GPU" \
      || { echo "FAILED: ${TAG}"; exit 1; }
  fi
done
echo "done: $MODE $MODEL seeds: ${SEEDS[*]}"
