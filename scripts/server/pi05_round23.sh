#!/usr/bin/env bash
# pi0.5 round 23: configuration v4 = v3 (round 22) + furniture pairs (--cf_fixture 0.25: at any phase the furniture is
# nudged 1-3 cm, render only, and labelled by the teacher's own chunk there; v3 lost the base's robustness to where LIBERO
# puts the cabinet at each reset: Goal task 3 at a 1-2 cm offset 55-65 vs base 85-95). v3 = v2 (round 21) + two fixes of the counterfactual labels found by the per-task
# failure analysis of Goal / Long:
#   --scene_tol 0.01: the pre-grasp approach segment starts only after the episode stops changing the rest of the scene
#     (Goal task 3: pi0.5 pushes the top drawer open without closing the gripper, so the gripper rule took the drawer
#     phase for the approach to the bowl and the retarget pairs taught "go to the bowl" there: task 3 swap 0, standard 70);
#   --no_cotarget_swap: retarget never swaps the target with another object the task carries (Long task 0 "put both the
#     soup and the sauce in the basket": the sauce under the hand labelled "go to the soup" taught hesitating at grasps).
#   [FRAC=0.15] [TINT=0.5] [SCOPE=vlm] [TAG=..] pi05_round23.sh <seed> <gpu object> <gpu spatial> <gpu goal> <gpu long>    (a gpu of "-" skips that suite)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$1"; T="${TAG:-}"
V4="--cf_mode full --cf_frac ${FRAC:-0.15} --cf_unique --cf_agree 0.5 --hindsight --scene_tol 0.01 --no_cotarget_swap --cf_fixture ${FIX:-0.25} --obj_tint ${TINT:-0.5} --lambda_cf 0 --cf_coshift_post 0.15,0.4 --seed $S"
[ -n "${SCOPE:-}" ] && V4="$V4 --lora_scope $SCOPE"
arm() {  # arm <gpu, "-" = skip> <suite> <max steps> <run>
  [ "$1" = - ] && return 0
  SUITE="$2" MAX_STEPS="$3" bash "$HERE/pi05_arm.sh" "$1" "$4" $V4 &
}
arm "$2" libero_object 280 "p05_v4${T}_s$S"
arm "$3" libero_spatial 220 "p05sp_v4${T}_s$S"
arm "$4" libero_goal 300 "p05gl_v4${T}_s$S"
arm "$5" libero_10 520 "p05lg_v4${T}_s$S"
wait
. "$HERE/env.sh"
log "PI05_ROUND23_DONE s$S"
