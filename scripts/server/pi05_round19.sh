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
#   pi05_round19.sh <mode> <seed> <gpu object> <gpu spatial> <gpu goal> <gpu long>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="$1"; S="$2"
UNI="--cf_mode $MODE --cf_frac 0.15 --cf_unique --cf_agree 0.5 --obj_tint 0.5 --lambda_cf 0 --cf_coshift_post 0.15,0.4 --seed $S"
REL_OBJECT="${REL_OBJECT:?}"; REL_SPATIAL="${REL_SPATIAL:?}"; REL_GOAL="${REL_GOAL:?}"; REL_LONG="${REL_LONG:?}"
bash "$HERE/pi05_arm.sh" "$3" "p05_uni_${MODE}_s$S" $UNI --relocate_tasks "$REL_OBJECT" &
SUITE=libero_spatial MAX_STEPS=220 bash "$HERE/pi05_arm.sh" "$4" "p05sp_uni_${MODE}_s$S" $UNI --relocate_tasks "$REL_SPATIAL" &
SUITE=libero_goal MAX_STEPS=300 bash "$HERE/pi05_arm.sh" "$5" "p05gl_uni_${MODE}_s$S" $UNI --relocate_tasks "$REL_GOAL" &
SUITE=libero_10 MAX_STEPS=520 bash "$HERE/pi05_arm.sh" "$6" "p05lg_uni_${MODE}_s$S" $UNI --relocate_tasks "$REL_LONG" &
wait
. "$HERE/env.sh"
log "PI05_ROUND19_DONE $MODE s$S"
