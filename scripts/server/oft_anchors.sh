#!/usr/bin/env bash
# Standard OpenVLA-OFT track on LIBERO-Object: gate the policy wrapper, reproduce the published LIBERO number,
# then measure LIBERO-Plus zero-shot for the official checkpoint and for Spatial Forcing (same backbone, 3D-aligned).
# One job at a time. Launch detached: setsid nohup bash repo/scripts/server/oft_anchors.sh > logs/oft_anchors.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
C="$W/checkpoints"; O="outputs/week1"; N="${NUM_ENVS:-7}"
OFT="oft:$C/moojink__openvla-7b-oft-finetuned-libero-object"
SF="oft:$C/haofuly__spatial-forcing-7b-finetuned-libero-object"
# openvla-oft's horizon for LIBERO-Object is 280 steps
step() { local out="$1"; shift; [ -f "$W/$O/$out/summary.json" ] && { log "skip $out"; return 0; }; "$@" --out "$O/$out" || { log "STOPPED at $out"; exit 1; }; }
bash "$HERE/run_py.sh" 0 scripts/check_policy.py --oft --ckpt "${OFT#oft:}" --suite libero_object || { log "GATE_FAILED"; exit 1; }
step oft_object bash "$HERE/run_py.sh" 0 scripts/eval_libero.py --ckpt "$OFT" --suite libero_object --num_envs "$N" --max_steps 280 --no_steps
export LIBERO_VARIANT=plus
step plus_oft bash "$HERE/run_py.sh" 0 scripts/eval_libero_plus.py --ckpt "$OFT" --suite libero_object --per_category 60 --num_envs "$N" --max_steps 280
until [ -f "$C/haofuly__spatial-forcing-7b-finetuned-libero-object/.hf_fetch.json" ]; do sleep 60; done
step plus_spatial_forcing bash "$HERE/run_py.sh" 0 scripts/eval_libero_plus.py --ckpt "$SF" --suite libero_object --per_category 60 --num_envs "$N" --max_steps 280
log "OFT_ANCHORS_DONE"
