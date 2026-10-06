#!/usr/bin/env bash
# On-policy / teacher-state distillation runs. Default: LIBERO-Object, student = 1-traj SFT, teacher = full-data SFT
# (454 demos, 96% success in this harness) with its token distribution transferred to the student's bins.
#   train_opd.sh <name> [train_opd.py args...]        e.g.  train_opd.sh b2_opd_s7 --state_source student --seed 7
# Other suites: SUITE=libero_10 STUDENT=<folder under checkpoints/> TEACHER=[rebin:]<folder under checkpoints/>
# GPUs: student on card 0; teacher on TEACHER_GPU (default 0, same card). Output: outputs/week1/<name>.
# Launch detached: setsid nohup bash repo/scripts/server/train_opd.sh b2_opd_s7 > logs/b2_opd_s7.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
NAME="$1"; shift
C="$W/checkpoints"
GPU="${GPU:-0}"; TEACHER_GPU="${TEACHER_GPU:-$GPU}"  # GPU picks the card on multi-GPU machines
if [ "$TEACHER_GPU" = "$GPU" ]; then CARDS="$GPU"; TDEV=cuda:0; else CARDS="$GPU,$TEACHER_GPU"; TDEV=cuda:1; fi
STUDENT="${STUDENT:-Haozhan72__Openvla-oft-SFT-libero-object-traj1}"
TEACHER="${TEACHER:-rebin:Haozhan72__Openvla-oft-SFT-libero-object-trajall}"
case "$TEACHER" in rebin:*) TEACHER="rebin:$C/${TEACHER#rebin:}" ;; *) TEACHER="$C/$TEACHER" ;; esac
bash "$HERE/run_py.sh" "$CARDS" scripts/train_opd.py --suite "${SUITE:-libero_object}" --out "outputs/week1/$NAME" \
  --student "$C/$STUDENT" --teacher "$TEACHER" \
  --student_device cuda:0 --teacher_device "$TDEV" --num_envs "${NUM_ENVS:-8}" "$@"
