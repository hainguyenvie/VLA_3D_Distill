#!/usr/bin/env bash
# pi0.5 round 16 (Object): relocate with the rim-aware placer (the first Object / Long relocate runs learned from a placer
# that drove objects into the basket wall: 7/20 even with the basket unmoved; now 20/20, 18/20 moved). With and without
# the 15% cap on counterfactual pairs, seed 7.   pi05_round16.sh <gpu uncapped> <gpu capped>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/pi05_arm.sh" "$1" p05_ocd_relocate4_s7 --cf_mode relocate --cf_coshift_post 0.15,0.4 --lambda_cf 0 --seed 7 &
bash "$HERE/pi05_arm.sh" "$2" p05_ocd_relocate4cap_s7 --cf_mode relocate --cf_frac 0.15 --cf_coshift_post 0.15,0.4 --lambda_cf 0 --seed 7 &
wait
. "$HERE/env.sh"
log "PI05_ROUND16_DONE"
