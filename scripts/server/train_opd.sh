#!/usr/bin/env bash
# On-policy / teacher-state distillation runs on LIBERO-Object (student = 1-traj SFT, teacher = RLinf GRPO).
#   train_opd.sh <name> [train_opd.py args...]        e.g.  train_opd.sh b2_opd_s7 --state_source student --seed 7
# GPUs: student on card 0; teacher on TEACHER_GPU (default 0, same card). Output: outputs/week1/<name>.
# Launch detached: setsid nohup bash repo/scripts/server/train_opd.sh b2_opd_s7 > logs/b2_opd_s7.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
NAME="$1"; shift
C="$W/checkpoints"
TEACHER_GPU="${TEACHER_GPU:-0}"
if [ "$TEACHER_GPU" = 0 ]; then CARDS=0; TDEV=cuda:0; else CARDS="0,$TEACHER_GPU"; TDEV=cuda:1; fi
bash "$HERE/run_py.sh" "$CARDS" scripts/train_opd.py --suite libero_object --out "outputs/week1/$NAME" \
  --student "$C/Haozhan72__Openvla-oft-SFT-libero-object-traj1" --teacher "$C/RLinf__RLinf-OpenVLAOFT-GRPO-LIBERO-object" \
  --student_device cuda:0 --teacher_device "$TDEV" --num_envs "${NUM_ENVS:-8}" "$@"
