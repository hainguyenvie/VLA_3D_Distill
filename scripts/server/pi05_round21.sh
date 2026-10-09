#!/usr/bin/env bash
# pi0.5 round 21: configuration v2 after the per-task diagnosis of round 19 — full mode + no twin moved (--cf_unique) +
# teacher-expert agreement (--cf_agree 0.5) + hindsight phase segmentation (--hindsight: pre-grasp pairs only in the
# segment in which the episode itself went for the target; fixes the drawer-first Goal task 3) + no execution gate (relocate
# pairs on every task; the gate removed the most useful tasks) + objects recoloured at p 0.5; NOGRIP=1 also masks the
# gripper of the relocate pairs (Long task 0 dropped objects in transit). Pairs on at most 15% of the states.
#   [NOGRIP=1] [FRAC=0.15] [TAG=..] pi05_round21.sh <seed> <gpu object> <gpu spatial> <gpu goal> <gpu long>    (a gpu of "-" skips that suite)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$1"; T="${TAG:-}"
V2="--cf_mode full --cf_frac ${FRAC:-0.15} --cf_unique --cf_agree 0.5 --hindsight --obj_tint 0.5 --lambda_cf 0 --cf_coshift_post 0.15,0.4 --seed $S"
[ "${NOGRIP:-0}" = 1 ] && V2="$V2 --relocate_no_grip"
arm() {  # arm <gpu, "-" = skip> <suite> <max steps> <run>
  [ "$1" = - ] && return 0
  SUITE="$2" MAX_STEPS="$3" bash "$HERE/pi05_arm.sh" "$1" "$4" $V2 &
}
arm "$2" libero_object 280 "p05_v2${T}_s$S"
arm "$3" libero_spatial 220 "p05sp_v2${T}_s$S"
arm "$4" libero_goal 300 "p05gl_v2${T}_s$S"
arm "$5" libero_10 520 "p05lg_v2${T}_s$S"
wait
. "$HERE/env.sh"
log "PI05_ROUND21_DONE s$S"
