#!/usr/bin/env bash
# pi0.5 round 12 (LIBERO-Spatial): the relocate counterfactual — while carrying, the plate alone is moved to a free spot
# and the label is a privileged scripted placer's chunk (the hand-to-goal relation changes, which no exact-label
# transform can do). LIBERO-Spatial's swap cell fails by putting the bowl where the plate usually is. Two seeds, same
# budget / evaluations as rounds 8 and 10 (base s7: swap 42.0).   pi05_round12.sh <gpu s7> <gpu s8>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export SUITE=libero_spatial MAX_STEPS=220
bash "$HERE/pi05_arm.sh" "$1" p05sp_ocd_relocate_s7 --cf_mode relocate --lambda_cf 0 --seed 7 &
bash "$HERE/pi05_arm.sh" "$2" p05sp_ocd_relocate_s8 --cf_mode relocate --lambda_cf 0 --seed 8 &
wait
. "$HERE/env.sh"
log "PI05_ROUND12_DONE"
