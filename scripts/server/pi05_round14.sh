#!/usr/bin/env bash
# pi0.5 round 14: (Object) relocate alone, and coshift before the grasp + relocate while carrying — the position cell's
# failures are partly carrying the target from an unusual spot by the memorised pick->basket motion; (Spatial) a third
# seed of relocate. Same budget / evaluations as before.   pi05_round14.sh <gpu obj relocate> <gpu obj coreloc> <gpu sp s9>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/pi05_arm.sh" "$1" p05_ocd_relocate_s7 --cf_mode relocate --lambda_cf 0 --seed 7 &
bash "$HERE/pi05_arm.sh" "$2" p05_ocd_coreloc_s7 --cf_mode coreloc --lambda_cf 0 --seed 7 &
SUITE=libero_spatial MAX_STEPS=220 bash "$HERE/pi05_arm.sh" "$3" p05sp_ocd_relocate_s9 --cf_mode relocate --lambda_cf 0 --seed 9 &
wait
. "$HERE/env.sh"
log "PI05_ROUND14_DONE"
