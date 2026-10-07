#!/usr/bin/env bash
# pi0.5 round 10: coshift after the grasp (hand + held target + container moved together; label = the nominal chunk for
# carrying and placing). LIBERO-Spatial's swap cell moves the plate (the place target), and Object's position-cell
# failures are mostly lost in transit, which pre-grasp coshift does not touch. Same budget / evaluations as before.
#   pi05_round10.sh <gpu spatial post s7> <gpu object both s7>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUITE=libero_spatial MAX_STEPS=220 bash "$HERE/pi05_arm.sh" "$1" p05sp_ocd_coshift_post_s7 --cf_mode coshift --coshift_phase post \
  --lambda_cf 1.0 --seed 7 &
bash "$HERE/pi05_arm.sh" "$2" p05_ocd_coshift_both_s7 --cf_mode coshift --coshift_phase both --lambda_cf 1.0 --seed 7 &
wait
. "$HERE/env.sh"
log "PI05_ROUND10_DONE"
