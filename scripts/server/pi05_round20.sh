#!/usr/bin/env bash
# pi0.5 round 20: the ECT baseline (Equivariant Counterfactual Training, arXiv 2609.39971) in our harness, same teacher
# and budget as our runs (20 iterations x 1024 states, pairs on at most 15% of the states, no consistency term; pairs share
# the flow noise and time, as the ECT loss). Each iteration, 16 successful episodes are replayed in a transformed scene
# (robot and start pose unchanged; tracking controller on the transformed end-effector path, kept if the task succeeds)
# and paired with the original queries. Transforms per suite as in the paper's Table 20: Spatial and Goal the y-mirror,
# Object the y-mirror and the x / xy mirrors with a shift, Long a shift (sizes not given in the paper; ours).
#   pi05_round20.sh <seed> <gpu object> <gpu spatial> <gpu goal> <gpu long>      (a gpu of "-" skips that suite)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="$1"
# FRAC (default 0.15) changes the share of states with a pair, TINT recolours objects as our runs do, and TAG the run
# names (e.g. FRAC=0.5 TAG=50 -> p05_ect50_s7; TINT=0.5 TAG=tint -> p05_ecttint_s7)
FRAC="${FRAC:-0.15}"; T="${TAG:-}"
ECT="--cf_mode ect --ect_episodes ${EPIS:-16} --cf_frac $FRAC --lambda_cf 0 --seed $S"
[ -n "${TINT:-}" ] && ECT="$ECT --obj_tint $TINT"  # same object recolouring as our runs (round 19 / 21: TINT=0.5)
[ -n "${SHIFT:-}" ] && ECT="$ECT --ect_shift_scale $SHIFT"  # sensitivity to the shift sizes of the transforms
arm() {  # arm <gpu, "-" = skip> <suite> <max steps> <run> <transforms>
  [ "$1" = - ] && return 0
  SUITE="$2" MAX_STEPS="$3" bash "$HERE/pi05_arm.sh" "$1" "$4" $ECT --ect_transforms "$5" &
}
arm "$2" libero_object 280 "p05_ect${T}_s$S" ymirror,xmirror_shift,xymirror_shift
arm "$3" libero_spatial 220 "p05sp_ect${T}_s$S" ymirror
arm "$4" libero_goal 300 "p05gl_ect${T}_s$S" ymirror
arm "$5" libero_10 520 "p05lg_ect${T}_s$S" shift
wait
. "$HERE/env.sh"
log "PI05_ROUND20_DONE s$S"
