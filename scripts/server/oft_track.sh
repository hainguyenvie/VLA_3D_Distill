#!/usr/bin/env bash
# Standard OpenVLA-OFT track on LIBERO-Object: self-distillation of the official checkpoint (frozen base = teacher on
# the nominal view, LoRA student) with the matched control.   oft_track.sh <seed> <arm>
#   a1  control: same loop without visual perturbations      a2  method: training states re-rendered under random
#   camera / light / sensor perturbations for the student (vec_env.VIEW_AUG)
# SMOKE=1 runs a one-iteration mini run first (catches wiring errors before the full run takes the GPU for hours).
# Launch detached: SMOKE=1 setsid nohup bash repo/scripts/server/oft_track.sh 7 a2 > logs/oft_a2_s7.launch.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
SEED="$1"; ARM="$2"
CKPT="$W/checkpoints/moojink__openvla-7b-oft-finetuned-libero-object"
case "$ARM" in
  a1) NAME="oft_a1_clean_s$SEED"; EXTRA="" ;;
  a2) NAME="oft_a2_viewaug_s$SEED"; EXTRA="--view_aug" ;;
  *) log "unknown arm $ARM"; exit 1 ;;
esac
COMMON="--ckpt $CKPT --suite libero_object --state_source mixed --max_steps 280 --batch_size 8 --grad_accum 1 --lr ${LR:-1e-4} --lr_min ${LR_MIN:-1e-5} --grad_checkpointing --eval_trials 10 --seed $SEED --num_envs ${NUM_ENVS:-5} $EXTRA"
if [ "${SMOKE:-0}" = 1 ]; then
  bash "$HERE/run_py.sh" 0 scripts/train_oft_distill.py --out "outputs/week1/smoke_$NAME" $COMMON --iters 1 --states_per_iter 48 \
    --episodes_per_batch 3 --eval_every 0 > "logs/smoke_$NAME.log" 2>&1 || { log "SMOKE_FAILED $NAME"; exit 1; }
  log "smoke passed for $NAME"
fi
bash "$HERE/run_py.sh" 0 scripts/train_oft_distill.py --out "outputs/week1/$NAME" $COMMON --iters "${ITERS:-20}" \
  --states_per_iter 2048 --eval_every "${EVAL_EVERY:-2}" > "logs/$NAME.log" 2>&1
log "$NAME exit $?"
