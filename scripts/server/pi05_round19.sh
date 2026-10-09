#!/usr/bin/env bash
# pi0.5 round 19: one configuration on the four LIBERO suites, no per-suite choice. Scripted-teacher counterfactual pairs
# (mode <mode>: rr = retarget before the grasp, relocate while carrying; full = retarget or coshift before, relocate while
# carrying) with the three automatic checks of the privileged teacher:
#   - an object with a twin of the same kind in the scene is never moved (--cf_unique: the instruction can then only name it
#     by where it is; this alone turns retarget off on Spatial and on two Long tasks);
#   - a pair is kept only if the same teacher, asked in the factual world, heads where pi0.5 does (--cf_agree 0.5);
#   - relocate pairs only on the tasks whose placer passes the execution gate (closed loop in the relocated world, success
#     >= 60%, n >= 5; scripts/check_relocate_gate.py on 20 base episodes per task; REL_* = its lists, -1 when none passes).
# Objects recoloured at p 0.5 (appearance variation, against the position -> appearance trade-off). Pairs on at most 15%
# of the states, no pairwise consistency term.
#   REL_OBJECT=.. REL_SPATIAL=.. REL_GOAL=.. REL_LONG=.. pi05_round19.sh <mode> <seed> <gpu object> <gpu spatial> <gpu goal> <gpu long>
# (a gpu of "-" skips that suite)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="$1"; S="$2"
T="${TAG:-}"  # suffix of the run names, for variants (e.g. TAG=nogate)
UNI="--cf_mode $MODE --cf_frac 0.15 --cf_unique --cf_agree 0.5 --obj_tint 0.5 --lambda_cf 0 --cf_coshift_post 0.15,0.4 --seed $S"
arm() {  # arm <gpu, "-" = skip> <suite> <max steps> <run> <relocate tasks>
  [ "$1" = - ] && return 0
  SUITE="$2" MAX_STEPS="$3" bash "$HERE/pi05_arm.sh" "$1" "$4" $UNI --relocate_tasks "${5:?gate list missing}" &
}
arm "$3" libero_object 280 "p05_uni${T}_${MODE}_s$S" "${REL_OBJECT:-}"
arm "$4" libero_spatial 220 "p05sp_uni${T}_${MODE}_s$S" "${REL_SPATIAL:-}"
arm "$5" libero_goal 300 "p05gl_uni${T}_${MODE}_s$S" "${REL_GOAL:-}"
arm "$6" libero_10 520 "p05lg_uni${T}_${MODE}_s$S" "${REL_LONG:-}"
wait
. "$HERE/env.sh"
log "PI05_ROUND19_DONE $MODE s$S"
