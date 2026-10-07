#!/usr/bin/env bash
# pi0.5 round 11 (Object): a third seed (9) for the arms the comparison rests on: on-policy distillation without
# counterfactual, the same with ordinary 2D augmentation, coshift and rotation. Same budget / evaluations as before.
#   pi05_round11.sh <gpu base> <gpu aug> <gpu coshift> <gpu rotate>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/pi05_arm.sh" "$1" p05_opd_base_s9 --no_cf --seed 9 &
bash "$HERE/pi05_arm.sh" "$2" p05_opd_aug_s9 --no_cf --img_aug --seed 9 &
bash "$HERE/pi05_arm.sh" "$3" p05_ocd_coshift_s9 --cf_mode coshift --lambda_cf 1.0 --seed 9 &
bash "$HERE/pi05_arm.sh" "$4" p05_ocd_rotate_s9 --cf_mode rotate --cf_theta 0.1,0.3 --lambda_cf 1.0 --seed 9 &
wait
. "$HERE/env.sh"
log "PI05_ROUND11_DONE"
