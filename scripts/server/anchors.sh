#!/usr/bin/env bash
# Anchors of one LIBERO suite in the SimpleVLA-RL protocol (discrete-token OFT, greedy, 512 steps): for each
# checkpoint the policy-wrapper gate, then the evaluation. One job at a time; finished evaluations are skipped.
#   anchors.sh <gpu> <suite> <trials per task> <run name>=<folder under checkpoints/> [...]
# Launch detached: setsid nohup bash repo/scripts/server/anchors.sh 1 libero_10 20 long_teacher_rl=Haozhan72__openvla-oft-libero10-traj1-rl > logs/long_anchors.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
GPU="$1"; SUITE="$2"; TRIALS="$3"; shift 3
O="outputs/week1"
for spec in "$@"; do
  name="${spec%%=*}"; ckpt="$W/checkpoints/${spec#*=}"
  [ -f "$W/$O/$name/summary.json" ] && { log "skip $name"; continue; }
  until [ -f "$ckpt/.hf_fetch.json" ]; do sleep 60; done  # the download may still be running
  bash "$HERE/run_py.sh" "$GPU" scripts/check_policy.py --ckpt "$ckpt" --suite "$SUITE" || { log "GATE_FAILED $name"; continue; }
  bash "$HERE/run_py.sh" "$GPU" scripts/eval_libero.py --ckpt "$ckpt" --suite "$SUITE" --trials "$TRIALS" \
    --num_envs "${NUM_ENVS:-8}" --no_steps --out "$O/$name" || log "FAILED $name"
done
log "ANCHORS_DONE $SUITE"
