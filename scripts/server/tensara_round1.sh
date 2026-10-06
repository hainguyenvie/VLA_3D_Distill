#!/usr/bin/env bash
# First runs on the 8x H200 node: the jobs that were still running when the old machines were retired, rerun from
# scratch so that every arm of a comparison shares one machine / renderer (EGL here). One card each, all at once.
# Launch detached: setsid nohup bash repo/scripts/server/tensara_round1.sh > logs/tensara_round1.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
COMMON="--mode rkl --iters 20 --states_per_iter 2048 --batch_size 8 --grad_accum 1 --lr 1e-4 --lr_min 1e-5 --grad_checkpointing --eval_every 2 --eval_trials 10 --seed 7"
run_train() {  # run_train <gpu> <name> <args...>   then the final evals of that run on the same card
  local gpu="$1" name="$2"; shift 2
  GPU="$gpu" NUM_ENVS=8 bash "$HERE/train_opd.sh" "$name" "$@" > "logs/$name.log" 2>&1
  if [ "${SUITE:-libero_object}" = libero_object ]; then
    GPU="$gpu" NUM_ENVS=8 bash "$HERE/final_evals.sh" "$name" > "logs/final_evals_$name.log" 2>&1
  else  # other suites: the last adapter on 20 init states per task
    last="$(ls -d "outputs/week1/$name"/adapter_iter* | sort | tail -1)"
    bash "$HERE/run_py.sh" "$gpu" scripts/eval_libero.py --ckpt "$W/checkpoints/$STUDENT" --lora "$last" --suite "$SUITE" \
      --trials 20 --num_envs 8 --no_steps --out "outputs/week1/${name}_$(basename "$last")_eval200" > "logs/final_evals_$name.log" 2>&1
  fi
  log "$name done"
}
# OpenVLA-OFT track: method (A2) and its control (A1)
(GPU=0 NUM_ENVS=8 bash "$HERE/oft_track.sh" 7 a2; GPU=0 NUM_ENVS=8 bash "$HERE/oft_final_evals.sh" oft_a2_viewaug_s7 > logs/oft_final_evals_a2.log 2>&1) &
(GPU=1 NUM_ENVS=8 bash "$HERE/oft_track.sh" 7 a1; GPU=1 NUM_ENVS=8 bash "$HERE/oft_final_evals.sh" oft_a1_clean_s7 > logs/oft_final_evals_a1.log 2>&1) &
# token track, stable lr: the teacher-state control of the 2x2 and the baseline it is compared with
run_train 2 cos_b2p_teacher_states_rkl_s7 $COMMON --state_source teacher &
run_train 3 cos_b2_student_states_rkl_s7 $COMMON --state_source student &
# LIBERO-Long baseline (student = 1-traj SFT, teacher = SimpleVLA-RL checkpoint, same bins), stable lr
SUITE=libero_10 STUDENT=Haozhan72__Openvla-oft-SFT-libero10-traj1 TEACHER=Haozhan72__openvla-oft-libero10-traj1-rl \
  run_train 4 long_cos_b2_student_states_rkl_s7 --mode rkl --iters 40 --states_per_iter 2048 --batch_size 8 --grad_accum 1 \
  --lr 1e-4 --lr_min 1e-5 --grad_checkpointing --eval_every 4 --eval_trials 10 --seed 7 --state_source student &
wait
log "TENSARA_ROUND1_DONE"
