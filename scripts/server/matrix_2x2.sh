#!/usr/bin/env bash
# 2 x 2 comparison on LIBERO-Object: {student states, teacher states} x {no 3D, privileged depth}.
# Same student init, teacher, loss, states per iteration, updates and seed in all four arms.
#   matrix_2x2.sh <seed> [arms...]     arms default to all four: b2 b4 b2p b3 (each one process, started 90 s apart)
#   MODE=fkl|rkl (default fkl), ITERS (default 20), EVAL_EVERY (default 2), NUM_ENVS per arm (default 5),
#   LR (default 1e-4), LR_MIN (default 0 = constant; > 0 = cosine decay to it)
#   other suites: TAG=<run name prefix> plus SUITE / STUDENT / TEACHER as in train_opd.sh
# Launch detached: MODE=fkl setsid nohup bash repo/scripts/server/matrix_2x2.sh 7 > logs/matrix_fkl_s7.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
SEED="${1:-7}"; shift || true
ARMS="${*:-b2 b4 b2p b3}"
MODE="${MODE:-fkl}"
export NUM_ENVS="${NUM_ENVS:-5}"
COMMON="--mode $MODE --iters ${ITERS:-20} --states_per_iter 2048 --batch_size 8 --grad_accum 1 --lr ${LR:-1e-4} --lr_min ${LR_MIN:-0} --grad_checkpointing --eval_every ${EVAL_EVERY:-2} --eval_trials 10 --seed $SEED"
DEPTH="--spatial depth --spatial_layer 24 --lambda_spatial 0.5"
arm() {  # arm <name> <args...>
  local name="${TAG:-}${1}_${MODE}_s$SEED"; shift
  bash "$HERE/train_opd.sh" "$name" $COMMON "$@" > "$W/logs/$name.log" 2>&1
  log "$name exit $?"
}
for a in $ARMS; do
  case "$a" in
    b2)  arm b2_student_states --state_source student & ;;
    b4)  arm b4_student_states_depth --state_source student $DEPTH & ;;
    b2p) arm b2p_teacher_states --state_source teacher & ;;
    b3)  arm b3_teacher_states_depth --state_source teacher $DEPTH & ;;
    # mixed drivers (batches of episodes alternate between student and teacher), without / with visual
    # perturbations of the training rollouts (teacher sees the nominal view), without / with the depth loss
    b2m) arm b2m_mixed_states --state_source mixed & ;;
    v1)  arm v1_mixed_viewaug --state_source mixed --view_aug & ;;
    v2)  arm v2_mixed_viewaug_depth --state_source mixed --view_aug $DEPTH & ;;
    v0)  arm v0_student_viewaug --state_source student --view_aug & ;;
    *) log "unknown arm $a" ;;
  esac
  sleep 90  # stagger model loading
done
wait
log "MATRIX_DONE"
