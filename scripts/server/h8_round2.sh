#!/usr/bin/env bash
# Round 2 on the 8x H200 node: consolidate the OFT result (A2 = self-distillation on re-rendered views, A1 = matched
# control). One card each: Object seed 8 (A1, A2) and seed 7 on Spatial, Goal and Long (A1, A2), each followed by
# its final evaluations (standard suite, 500 episodes; LIBERO-Plus of the same suite, 60 tasks per perturbation type).
# Launch detached: setsid nohup bash repo/scripts/server/h8_round2.sh > logs/h8_round2.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
python3 "$HERE/hf_fetch.py" checkpoints moojink/openvla-7b-oft-finetuned-libero-spatial moojink/openvla-7b-oft-finetuned-libero-goal \
  moojink/openvla-7b-oft-finetuned-libero-10 || { log "FETCH_FAILED"; exit 1; }
arm() {  # arm <gpu> <suite> <seed> <a1|a2>
  local g="$1" suite="$2" seed="$3" a="$4" short tag name
  case "$suite" in libero_spatial) short=spatial ;; libero_object) short=object ;; libero_goal) short=goal ;; libero_10) short=10 ;; esac
  tag=""; [ "$suite" = libero_object ] || tag="${short}_"
  [ "$a" = a1 ] && name="oft_${tag}a1_clean_s$seed" || name="oft_${tag}a2_viewaug_s$seed"
  SUITE="$suite" GPU="$g" NUM_ENVS=5 SMOKE=0 bash "$HERE/oft_track.sh" "$seed" "$a"
  SUITE="$suite" GPU="$g" NUM_ENVS=6 bash "$HERE/oft_final_evals.sh" "$name" > "logs/oft_final_evals_$name.log" 2>&1
  log "ARM_DONE $name"
}
arm 0 libero_object 8 a2 & arm 1 libero_object 8 a1 &
arm 2 libero_spatial 7 a2 & arm 3 libero_spatial 7 a1 &
arm 4 libero_goal 7 a2 & arm 5 libero_goal 7 a1 &
arm 6 libero_10 7 a2 & arm 7 libero_10 7 a1 &
wait
log "H8_ROUND2_DONE"
