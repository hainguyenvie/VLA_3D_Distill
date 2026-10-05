#!/usr/bin/env bash
# Day 1: evaluation reproduction on LIBERO-Object (discrete-token OpenVLA-OFT, SimpleVLA-RL protocol).
# One job at a time: student (B0) with teacher labels, teacher with student labels, full-data SFT (B1), then a
# same-seed repeat of B0 (stability gate). NUM_ENVS defaults to 8 (RAM limit of the shared L40 machine).
# Launch detached: setsid nohup bash repo/scripts/server/day1_evals.sh 0 > logs/day1.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
GPU="$1"
C="$W/checkpoints"
STUDENT="$C/Haozhan72__Openvla-oft-SFT-libero-object-traj1"
TEACHER="$C/RLinf__RLinf-OpenVLAOFT-GRPO-LIBERO-object"
FULLSFT="$C/Haozhan72__Openvla-oft-SFT-libero-object-trajall"
O="outputs/week1"

run() {  # run <name> <args...>: skipped when its summary.json already exists; a failed step stops the queue
  local name="$1"; shift
  [ -f "$W/$O/$name/summary.json" ] && { log "skip $name"; return 0; }
  bash "$HERE/run_py.sh" "$GPU" scripts/eval_libero.py --suite libero_object --out "$O/$name" --num_envs "${NUM_ENVS:-8}" "$@" \
    || { log "QUEUE_STOPPED at $name"; exit 1; }
}

run b0_object_student --ckpt "$STUDENT" --label teacher="$TEACHER" --depth
run teacher_object --ckpt "$TEACHER" --label student="$STUDENT" --depth
run b1_object_fullsft --ckpt "$FULLSFT" --no_steps
run b0_object_student_repeat --ckpt "$STUDENT" --no_steps
log "QUEUE_DONE"
