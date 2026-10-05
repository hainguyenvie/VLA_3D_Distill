#!/usr/bin/env bash
# LIBERO-Long in the VLA-OPD setting: student = 1-traj SFT, teacher = the SimpleVLA-RL checkpoint trained from that
# student (same action bins, no transfer needed). Gate: the teacher must reproduce its published level in this
# harness (>= MIN_TEACHER_SR over 200 episodes; paper 91.7%); then the no-3D on-policy arm is started.
#   long_round1.sh <seed> [log pattern to wait for first, as "<log file>:<pattern>"]
# Launch detached: setsid nohup bash repo/scripts/server/long_round1.sh 7 logs/object_round2_s7.log:OBJECT_ROUND2_DONE > logs/long_round1_s7.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
SEED="${1:-7}"; AFTER="${2:-}"
STUDENT=Haozhan72__Openvla-oft-SFT-libero10-traj1
TEACHER=Haozhan72__openvla-oft-libero10-traj1-rl
NUM_ENVS="${NUM_ENVS:-7}" bash "$HERE/anchors.sh" 0 libero_10 20 "long_teacher_rl=$TEACHER"
SR="$(python3 -c "import json; print(json.load(open('outputs/week1/long_teacher_rl/summary.json'))['success_rate'])" 2>/dev/null || echo 0)"
log "teacher success rate on LIBERO-Long: $SR"
python3 -c "import sys; sys.exit(0 if float('$SR') >= float('${MIN_TEACHER_SR:-0.85}') else 1)" || { log "LONG_GATE_FAILED"; exit 1; }
if [ -n "$AFTER" ]; then until grep -q "${AFTER#*:}" "${AFTER%%:*}" 2>/dev/null; do sleep 60; done; fi
SUITE=libero_10 STUDENT="$STUDENT" TEACHER="$TEACHER" TAG=long_ MODE=rkl ITERS="${ITERS:-40}" EVAL_EVERY=4 NUM_ENVS=6 \
  bash "$HERE/matrix_2x2.sh" "$SEED" b2
log "LONG_ROUND1_DONE"
