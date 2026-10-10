#!/usr/bin/env bash
# pi0.5 round 24: configuration v5, one configuration for the four suites = the main configuration of the paper (round 19
# without the gate: full mode, --cf_unique, --cf_agree 0.5, recolouring 0.5, pairs on 15% of the states) + the label fixes
# that do not over-filter:
#   --no_cotarget_swap: retarget never swaps the target with another object the task carries (Long "put both ...");
#   --joint_tol 0.01: pre-grasp pairs only once the articulated fixtures (drawers, knobs, doors) stop moving before the
#     target's first lift (Goal task 3: pi0.5 pushes the drawer open without closing the gripper); objects bumped by the hand
#     do not count, unlike --scene_tol + --hindsight (round 22), which removed most retarget pairs on Object (swap 36.5);
#   --cf_fixture 0.25: furniture nudged 1-3 cm and labelled by the teacher's own chunk (round 23; Goal swap 38.5).
#   [FRAC=0.15] [TINT=0.5] [TAG=..] pi05_round24.sh <seed> <gpu object> <gpu spatial> <gpu goal> <gpu long>   ("-" skips)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$1"; T="${TAG:-}"
V5="--cf_mode full --cf_frac ${FRAC:-0.15} --cf_unique --cf_agree 0.5 --no_cotarget_swap --joint_tol 0.01 --cf_fixture ${FIX:-0.25} --obj_tint ${TINT:-0.5} --lambda_cf 0 --cf_coshift_post 0.15,0.4 --seed $S"
arm() {  # arm <gpu, "-" = skip> <suite> <max steps> <run>
  [ "$1" = - ] && return 0
  SUITE="$2" MAX_STEPS="$3" bash "$HERE/pi05_arm.sh" "$1" "$4" $V5 &
}
arm "$2" libero_object 280 "p05_v5${T}_s$S"
arm "$3" libero_spatial 220 "p05sp_v5${T}_s$S"
arm "$4" libero_goal 300 "p05gl_v5${T}_s$S"
arm "$5" libero_10 520 "p05lg_v5${T}_s$S"
wait
. "$HERE/env.sh"
log "PI05_ROUND24_DONE s$S"
