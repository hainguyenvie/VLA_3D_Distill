#!/usr/bin/env bash
# Per-task look at saved adapters of a LIBERO-Object distillation run: the given tasks, all 50 benchmark initial
# states each, greedy, one adapter after another.   per_task_evals.sh <run name> <tasks, comma-separated> <iteration> [...]
# Launch detached: setsid nohup bash repo/scripts/server/per_task_evals.sh b2_student_states_rkl_s7 0,3,5 10 14 18 > logs/per_task_b2.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
RUN="$1"; TASKS="$2"; shift 2
O=outputs/week1
BASE="$W/checkpoints/Haozhan72__Openvla-oft-SFT-libero-object-traj1"
for it in "$@"; do
  out="$O/${RUN}_iter$(printf %04d "$it")_tasks${TASKS//,/_}"
  [ -f "$out/summary.json" ] && { log "skip $out"; continue; }
  bash "$HERE/run_py.sh" 0 scripts/eval_libero.py --ckpt "$BASE" --lora "$O/$RUN/adapter_iter$(printf %04d "$it")" \
    --suite libero_object --tasks "$TASKS" --num_envs "${NUM_ENVS:-6}" --no_steps --out "$out" || log "FAILED $out"
done
log "PER_TASK_EVALS_DONE $RUN"
