#!/usr/bin/env bash
# pi0.5 round 17 (LIBERO-Spatial, what drives the relocate gain? swap 42 -> 64.5 / 63.0): relocated worlds labelled by
# pi0.5 itself (no privileged teacher), and the scripted placer's label in the nominal world (no relocation). Seed 7,
# same budget / evaluations.   pi05_round17.sh <gpu self-teacher> <gpu placer-nominal>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export SUITE=libero_spatial MAX_STEPS=220
bash "$HERE/pi05_arm.sh" "$1" p05sp_ocd_relocate_selfteach_s7 --cf_mode relocate --relocate_label teacher --lambda_cf 0 --seed 7 &
bash "$HERE/pi05_arm.sh" "$2" p05sp_ocd_placer_nominal_s7 --cf_mode relocate --cf_coshift_post 0,0 --lambda_cf 0 --seed 7 &
wait
. "$HERE/env.sh"
log "PI05_ROUND17_DONE"
