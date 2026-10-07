#!/usr/bin/env bash
# Headroom of pi0.5 outside LIBERO-Object (is the position-memorisation failure general?): the released checkpoint on
# the standard Spatial / Goal / Long suites and on their LIBERO-PRO swap cells (swap keeps per-query states for the
# failure taxonomy). Max steps as in the openpi LIBERO evaluation.   pi05_headroom_suites.sh <gpu>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
G="$1"; O=outputs/week1; CK="$W/checkpoints/lerobot__pi05_libero_finetuned"
for spec in libero_spatial:220 libero_goal:300 libero_10:520; do
  s="${spec%%:*}"; ms="${spec##*:}"
  out="$O/pi05_${s#libero_}_std"
  [ -f "$out/summary.json" ] || LIBERO_VARIANT=pro bash "$HERE/run_py.sh" "$G" scripts/eval_libero.py --ckpt "pi05:$CK" --suite "$s" \
    --trials 20 --num_envs 8 --max_steps "$ms" --no_steps --out "$out" || log "FAILED $out"
  out="$O/pi05_${s#libero_}_pro_swap_steps"
  [ -f "$out/summary.json" ] || LIBERO_VARIANT=pro bash "$HERE/run_py.sh" "$G" scripts/eval_libero.py --ckpt "pi05:$CK" --suite "${s}_swap" \
    --trials 20 --num_envs 8 --max_steps "$ms" --out "$out" || log "FAILED $out"
done
log "HEADROOM_SUITES_DONE"
