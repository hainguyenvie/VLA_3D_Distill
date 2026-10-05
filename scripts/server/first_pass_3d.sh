#!/usr/bin/env bash
# First 3D-vs-no-3D read-out at a matched budget of 10 iterations, scored on LIBERO-Plus.
# Waits for the two no-3D arms (B2, B2') to log iteration 10, stops them, starts B3 (teacher states + depth,
# 10 iterations), and evaluates the iteration-10 no-3D checkpoints on LIBERO-Plus. B4 keeps running; its
# iteration-10 checkpoint and B3's are evaluated by plus_adapters.sh.
# Launch detached: setsid nohup bash repo/scripts/server/first_pass_3d.sh > logs/first_pass_3d.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
O=outputs/week1
for run in b2_student_states_rkl_s7 b2p_teacher_states_rkl_s7; do
  until grep -q '"iter": 10,' "$O/$run/train_log.jsonl" 2>/dev/null; do sleep 30; done
  log "$run reached iteration 10"
done
for run in b2_student_states_rkl_s7 b2p_teacher_states_rkl_s7; do
  ps -eo pid,comm,args | awk -v pat="$O/$run" '$2 ~ /^python/ && index($0, pat) {print $1}' | while read -r p; do kill "$p"; log "stopped $run (pid $p)"; done
done
sleep 20
(MODE=rkl ITERS=10 NUM_ENVS=5 bash "$HERE/matrix_2x2.sh" 7 b3 > "$W/logs/matrix_rkl_b3_s7.log" 2>&1 &)
bash "$HERE/plus_adapters.sh" b2_student_states_rkl_s7:10 b2p_teacher_states_rkl_s7:10
log "FIRST_PASS_NO3D_EVALS_DONE"
