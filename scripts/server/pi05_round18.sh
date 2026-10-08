#!/usr/bin/env bash
# pi0.5 round 18: the retarget counterfactual (target alone moved before the grasp, hand unchanged; label = privileged
# scripted approach to above the target, masked once there) on the two suites whose swap cell is a target-choice failure.
# Upper bound (scripted approach, then pi0.5): Object swap 59.0 vs pi0.5 18.5. Pairs capped at 15% of the states.
#   pi05_round18.sh <gpu object> <gpu goal>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/pi05_arm.sh" "$1" p05_ocd_retarget_s7 --cf_mode retarget --cf_frac 0.15 --lambda_cf 0 --seed 7 &
SUITE=libero_goal MAX_STEPS=300 bash "$HERE/pi05_arm.sh" "$2" p05gl_ocd_retarget_s7 --cf_mode retarget --cf_frac 0.15 --lambda_cf 0 --seed 7 &
wait
. "$HERE/env.sh"
log "PI05_ROUND18_DONE"
