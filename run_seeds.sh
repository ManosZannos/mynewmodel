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
# Safety: a run counts as COMPLETE only if its train.log contains the final line
# 'Done. Best val' (val_best.pth alone is written from epoch 1, so it proves nothing).
#   train: skips complete runs; if a run directory exists but is INCOMPLETE the script
#          stops before launching anything and tells you to move/delete it by hand
#          (re-running into it would append to the old log and overwrite its checkpoint).
#   dump : skips seeds whose .npz exists; refuses to dump an incomplete run.
# The chain stops at the first failure. Set DRY=1 to only print the commands.
#
# PAR=<n> (default 1) runs up to n seeds CONCURRENTLY on the same GPU. Use it only
# if the GPU is not saturated by one run (check GPU-Util in nvidia-smi) and if the
# cluster policy allows several processes on your single GPU. In parallel mode the
# stdout/stderr of each job goes to par_<TAG>.out, a failure does not stop the
# other jobs, and a final check lists every seed whose output file is missing.
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

PAR="${PAR:-1}"
CMDS=(); TAGS=(); CHECKS=()

is_complete() { [ -f "checkpoints/$1/train.log" ] && grep -q "Done\. Best val" "checkpoints/$1/train.log"; }

for SEED in "${SEEDS[@]}"; do
  TAG="${TAGP}_d05_gru2_lr5e4_obs10_pred10_s${SEED}"
  if [ "$MODE" = "train" ]; then
    if is_complete "$TAG"; then
      echo "--- skip (complete run exists): ${TAG}"; continue
    fi
    if [ -d "checkpoints/${TAG}" ]; then
      echo "!!! INCOMPLETE run directory exists: checkpoints/${TAG}"
      echo "    (no 'Done. Best val' line in its train.log). Move or delete it manually, then re-run."
      exit 1
    fi
    CMDS+=("python $SCRIPT --obs_len 10 --pred_len 10 --gru_layers 2 --top_k 10 --huber_delta 0.05 --lr 5e-4 --seed $SEED --tag $TAG --gpu_num $GPU")
    CHECKS+=("checkpoints/${TAG}/val_best.pth")
  else
    if [ -f "error_dumps/${TAG}__test.npz" ]; then
      echo "--- skip (dump exists): ${TAG}"; continue
    fi
    if ! is_complete "$TAG"; then
      echo "--- run missing or INCOMPLETE (no 'Done. Best val' in train.log), cannot dump: ${TAG}"; exit 1
    fi
    CMDS+=("python dump_errors.py --arch $ARCH --tag $TAG --gpu_num $GPU")
    CHECKS+=("error_dumps/${TAG}__test.npz")
  fi
  TAGS+=("$TAG")
done

if [ "${#CMDS[@]}" -eq 0 ]; then echo "nothing to do"; exit 0; fi

if [ "${DRY:-0}" = "1" ]; then
  for c in "${CMDS[@]}"; do echo ">>> (dry, PAR=$PAR) $c"; done
  exit 0
fi

if [ "$PAR" -le 1 ]; then
  for i in "${!CMDS[@]}"; do
    echo ">>> ${CMDS[$i]}"
    eval "${CMDS[$i]}" || { echo "FAILED: ${TAGS[$i]}"; exit 1; }
  done
else
  for i in "${!CMDS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
    echo ">>> (parallel, PAR=$PAR) ${CMDS[$i]}   [output -> par_${TAGS[$i]}.out]"
    eval "${CMDS[$i]}" > "par_${TAGS[$i]}.out" 2>&1 &
    sleep 5
  done
  wait
fi

FAILED=0
for i in "${!CHECKS[@]}"; do
  if [ "$MODE" = "train" ]; then
    is_complete "${TAGS[$i]}" || { echo "NOT COMPLETE (job failed?): ${TAGS[$i]}"; FAILED=1; }
  elif [ ! -f "${CHECKS[$i]}" ]; then
    echo "MISSING OUTPUT (job failed?): ${CHECKS[$i]}"; FAILED=1
  fi
done
if [ "$FAILED" = "1" ]; then exit 1; fi
echo "done: $MODE $MODEL seeds: ${SEEDS[*]}"