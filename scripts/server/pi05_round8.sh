#!/usr/bin/env bash
# pi0.5 round 8 (does it carry over?): the same three arms on LIBERO-Spatial with nothing changed but the suite and the
# openpi horizon (220): on-policy distillation without counterfactual, rotation, coshift. pi0.5 on the Spatial
# LIBERO-PRO swap cell: 42.0 (standard 98.5).   pi05_round8.sh <gpu base> <gpu rotate> <gpu coshift>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export SUITE=libero_spatial MAX_STEPS=220
bash "$HERE/pi05_arm.sh" "$1" p05sp_opd_base_s7 --no_cf --seed 7 &
bash "$HERE/pi05_arm.sh" "$2" p05sp_ocd_rotate_s7 --cf_mode rotate --cf_theta 0.1,0.3 --lambda_cf 1.0 --seed 7 &
bash "$HERE/pi05_arm.sh" "$3" p05sp_ocd_coshift_s7 --cf_mode coshift --lambda_cf 1.0 --seed 7 &
wait
. "$HERE/env.sh"
log "PI05_ROUND8_DONE"
