#!/usr/bin/env bash
# Final scoring of finished LIBERO-Object distillation runs, one job at a time: the last saved adapter on the full
# standard suite (500 episodes, greedy) and on the fixed LIBERO-Plus sample.   final_evals.sh <run name> [...]
# Waits for each run to log TRAIN_DONE. Launch detached:
#   setsid nohup bash repo/scripts/server/final_evals.sh b4_student_states_depth_rkl_s7 > logs/final_evals.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
O=outputs/week1
BASE="$W/checkpoints/Haozhan72__Openvla-oft-SFT-libero-object-traj1"
for run in "$@"; do
  until grep -q "TRAIN_DONE" "logs/$run.log" 2>/dev/null; do sleep 60; done
  last="$(ls -d "$O/$run"/adapter_iter* | sort | tail -1)"; it="$(basename "$last" | sed 's/adapter_iter0*//')"
  out="$O/${run}_iter$(printf %04d "$it")_eval500"
  if [ ! -f "$out/summary.json" ]; then
    bash "$HERE/run_py.sh" 0 scripts/eval_libero.py --ckpt "$BASE" --lora "$last" --suite libero_object \
      --num_envs "${NUM_ENVS:-7}" --no_steps --out "$out" || log "FAILED $out"
  fi
  bash "$HERE/plus_adapters.sh" "$run:$it"
done
log "FINAL_EVALS_DONE $*"
