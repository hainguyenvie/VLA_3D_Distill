#!/usr/bin/env bash
# First 2 x 2 comparison on LIBERO-Object: {student states, teacher states} x {no 3D, privileged depth}.
# Same student init, teacher, loss, states per iteration, updates and seed in all four arms.
#   matrix_2x2.sh <seed>        three processes at a time (B2, B4, and B2' followed by B3)
# Launch detached: setsid nohup bash repo/scripts/server/matrix_2x2.sh 7 > logs/matrix_s7.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
SEED="${1:-7}"
COMMON="--mode rkl --iters 20 --states_per_iter 2048 --batch_size 8 --grad_accum 1 --lr 1e-4 --eval_every 2 --eval_trials 10 --seed $SEED"
DEPTH="--spatial depth --spatial_layer 24 --lambda_spatial 0.5"
arm() {  # arm <name> <args...>
  local name="$1"; shift
  bash "$HERE/train_opd.sh" "${name}_s$SEED" $COMMON "$@" > "$W/logs/${name}_s$SEED.log" 2>&1
  log "$name exit $?"
}
arm b2_student_states --state_source student &
sleep 90  # stagger model loading
arm b4_student_states_depth --state_source student $DEPTH &
sleep 90
( arm b2p_teacher_states --state_source teacher; arm b3_teacher_states_depth --state_source teacher $DEPTH ) &
wait
log "MATRIX_DONE"
