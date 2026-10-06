#!/usr/bin/env bash
# Verification of a fresh workspace on the 8x H200 node: correctness gates on GPU 0, then anchor evaluations already
# measured on the old machines, in parallel on GPUs 1-5 (also tests concurrent EGL rendering next to CUDA contexts).
# Launch detached: setsid nohup bash repo/scripts/server/verify_tensara.sh > logs/verify_tensara.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
C="$W/checkpoints"; O=outputs/verify
mkdir -p "$O"
export MIN_FREE_GB=24
gate() { local name="$1"; shift; if "$@" > "logs/verify_$name.log" 2>&1; then log "GATE_OK $name"; else log "GATE_FAILED $name"; fi; }
gate env_stepping bash "$HERE/run_py.sh" 0 scripts/check_env_stepping.py --suite libero_object --task 3
gate policy_token bash "$HERE/run_py.sh" 0 scripts/check_policy.py --ckpt "$C/Haozhan72__Openvla-oft-SFT-libero-object-traj1" --suite libero_object
gate policy_oft bash "$HERE/run_py.sh" 0 scripts/check_policy.py --oft --ckpt "$C/moojink__openvla-7b-oft-finetuned-libero-object" --suite libero_object
gate view_aug bash "$HERE/run_py.sh" 0 scripts/check_view_aug.py --suite libero_object --task 3 --out "$O/view_aug_check"
gate counterfactual bash "$HERE/run_py.sh" 0 scripts/check_counterfactual.py --suite libero_object --task 3 --out "$O/cf_check"
# anchors (old machines: student 51.4 / 52.6, full-SFT 95.8 / 95.2, OFT 96.8, pi0.5 100 on these 100 episodes or 500)
ev() { local gpu="$1" name="$2"; shift 2
  bash "$HERE/run_py.sh" "$gpu" scripts/eval_libero.py --suite libero_object --trials 10 --num_envs 8 --no_steps --out "$O/$name" "$@" \
    > "logs/verify_$name.log" 2>&1 && log "EVAL_OK $name" || log "EVAL_FAILED $name"; }
ev 1 student --ckpt "$C/Haozhan72__Openvla-oft-SFT-libero-object-traj1" &
ev 2 fullsft --ckpt "$C/Haozhan72__Openvla-oft-SFT-libero-object-trajall" &
ev 3 oft --ckpt "oft:$C/moojink__openvla-7b-oft-finetuned-libero-object" --max_steps 280 &
PY_ENV=pi05 ev 4 pi05 --ckpt "pi05:$C/lerobot__pi05_libero_finetuned" --max_steps 280 &
( LIBERO_VARIANT=plus bash "$HERE/run_py.sh" 5 scripts/eval_libero_plus.py --ckpt "oft:$C/moojink__openvla-7b-oft-finetuned-libero-object" \
    --suite libero_object --per_category 60 --num_envs 8 --max_steps 280 --out "$O/plus_oft" > logs/verify_plus_oft.log 2>&1 \
    && log "EVAL_OK plus_oft" || log "EVAL_FAILED plus_oft" ) &
wait
log "VERIFY_DONE"
