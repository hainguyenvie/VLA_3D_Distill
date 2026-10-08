#!/usr/bin/env bash
# pi0.5 round 15: relocate with its counterfactual pairs capped at 15% of the states (as on Spatial, where it worked; on
# Object 60% and on Long 27% of the states carried scripted labels and the standard tasks collapsed) and the container
# moved by at least 15 cm. Object relocate s7, Object coreloc s7, Long relocate s7.
#   pi05_round15.sh <gpu obj relocate> <gpu obj coreloc> <gpu long relocate>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CAP="--cf_frac 0.15 --cf_coshift_post 0.15,0.4 --lambda_cf 0"
bash "$HERE/pi05_arm.sh" "$1" p05_ocd_relocate3_s7 --cf_mode relocate $CAP --seed 7 &
bash "$HERE/pi05_arm.sh" "$2" p05_ocd_coreloc3_s7 --cf_mode coreloc $CAP --seed 7 &
SUITE=libero_10 MAX_STEPS=520 bash "$HERE/pi05_arm.sh" "$3" p05lg_ocd_relocate3_s7 --cf_mode relocate $CAP --seed 7 &
wait
. "$HERE/env.sh"
log "PI05_ROUND15_DONE"
