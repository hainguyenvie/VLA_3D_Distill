#!/usr/bin/env bash
# pi0.5 round 9 (Object, same budget / evaluations): is it just augmentation? on-policy distillation with ordinary 2D
# image augmentation (openpi LIBERO recipe) and no counterfactual, two seeds; and a second seed of the simplest coshift
# (free spots only, no swap with another object), the best arm of round 6.
#   pi05_round9.sh <gpu base+aug s7> <gpu base+aug s8> <gpu coshift free s8>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/pi05_arm.sh" "$1" p05_opd_aug_s7 --no_cf --img_aug --seed 7 &
bash "$HERE/pi05_arm.sh" "$2" p05_opd_aug_s8 --no_cf --img_aug --seed 8 &
bash "$HERE/pi05_arm.sh" "$3" p05_ocd_coshift_free_s8 --cf_mode coshift --lambda_cf 1.0 --p_swap 0 --seed 8 &
wait
. "$HERE/env.sh"
log "PI05_ROUND9_DONE"
