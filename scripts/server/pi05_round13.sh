#!/usr/bin/env bash
# pi0.5 round 13 (LIBERO-Long, does relocate carry over?): on-policy distillation without counterfactual and the relocate
# counterfactual (container moved while carrying, scripted-placer label; objects already in the container move with it),
# seed 7, Long horizon 520, same budget / evaluations. pi0.5 on the Long swap cell: 9.0 (standard 94.0).
#   pi05_round13.sh <gpu base> <gpu relocate>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export SUITE=libero_10 MAX_STEPS=520
bash "$HERE/pi05_arm.sh" "$1" p05lg_opd_base_s7 --no_cf --seed 7 &
bash "$HERE/pi05_arm.sh" "$2" p05lg_ocd_relocate_s7 --cf_mode relocate --lambda_cf 0 --seed 7 &
wait
. "$HERE/env.sh"
log "PI05_ROUND13_DONE"
