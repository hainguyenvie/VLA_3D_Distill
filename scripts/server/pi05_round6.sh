#!/usr/bin/env bash
# pi0.5 round 6: what drives the coshift gain (seed 7, same budget / evaluations as rounds 3-5), one arm per card.
#   off:    the same counterfactual on the states of the frozen pi0.5 (its own rollouts) instead of the student's,
#           i.e. ECT-style pairs built around expert-like states rather than on-policy ones
#   free:   p_swap 0: the target only moves to free spots, never onto another object's spot
#   nocons: lambda_cf 0: no consistency term between the two worlds, only flow matching in both
#   pi05_round6.sh <gpu off> <gpu free> <gpu nocons>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/pi05_arm.sh" "$1" p05_ocd_coshift_off_s7 --cf_mode coshift --lambda_cf 1.0 --state_source teacher --seed 7 &
bash "$HERE/pi05_arm.sh" "$2" p05_ocd_coshift_free_s7 --cf_mode coshift --lambda_cf 1.0 --p_swap 0 --seed 7 &
bash "$HERE/pi05_arm.sh" "$3" p05_ocd_coshift_nocons_s7 --cf_mode coshift --lambda_cf 0 --seed 7 &
wait
. "$HERE/env.sh"
log "PI05_ROUND6_DONE"
