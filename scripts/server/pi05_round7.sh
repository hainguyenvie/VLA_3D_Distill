#!/usr/bin/env bash
# pi0.5 round 7: rotation and coshift together (mix: each query is one or the other, half / half), two seeds; rotation
# lifted the swap cell, coshift the position cell and LIBERO-Plus Robot. Same budget / evaluations as rounds 3-6.
#   pi05_round7.sh <gpu mix s7> <gpu mix s8>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/pi05_arm.sh" "$1" p05_ocd_mix_s7 --cf_mode mix --cf_theta 0.1,0.3 --lambda_cf 1.0 --seed 7 &
bash "$HERE/pi05_arm.sh" "$2" p05_ocd_mix_s8 --cf_mode mix --cf_theta 0.1,0.3 --lambda_cf 1.0 --seed 8 &
wait
. "$HERE/env.sh"
log "PI05_ROUND7_DONE"
