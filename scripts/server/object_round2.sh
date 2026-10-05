#!/usr/bin/env bash
# LIBERO-Object, once the two no-3D arms (B2, B2') of a seed have finished: start B3 (teacher states + depth) in
# their place, and score the final no-3D adapters on the full standard suite (500 episodes, greedy).
#   object_round2.sh <seed>
# Launch detached: setsid nohup bash repo/scripts/server/object_round2.sh 7 > logs/object_round2_s7.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
SEED="${1:-7}"; O=outputs/week1
BASE="$W/checkpoints/Haozhan72__Openvla-oft-SFT-libero-object-traj1"
RUNS="b2_student_states_rkl_s$SEED b2p_teacher_states_rkl_s$SEED"
for run in $RUNS; do
  until grep -q "TRAIN_DONE" "logs/$run.log" 2>/dev/null; do sleep 60; done
  log "$run finished"
done
(MODE=rkl NUM_ENVS=5 bash "$HERE/matrix_2x2.sh" "$SEED" b3 > "logs/matrix_rkl_b3_s$SEED.log" 2>&1 &)
sleep 120
for run in $RUNS; do
  last="$(ls -d "$O/$run"/adapter_iter* | sort | tail -1)"
  out="$O/${run}_$(basename "$last" | sed 's/adapter_//')_eval500"
  [ -f "$out/summary.json" ] && { log "skip $out"; continue; }
  bash "$HERE/run_py.sh" 0 scripts/eval_libero.py --ckpt "$BASE" --lora "$last" --suite libero_object \
    --num_envs "${NUM_ENVS:-7}" --no_steps --out "$out" || log "FAILED $out"
done
log "OBJECT_ROUND2_DONE"
