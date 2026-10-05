#!/usr/bin/env bash
# Run a repo python script on one GPU with the workspace environment.
#   run_py.sh <gpu> <script relative to repo> [args...]      (relative --out paths resolve inside the workspace)
# Launch detached: setsid nohup bash repo/scripts/server/run_py.sh 0 scripts/eval_libero.py ... > logs/x.log 2>&1 < /dev/null &
set -uo pipefail
. "$(dirname "$0")/env.sh"
GPU="$1"; SCRIPT="$2"; shift 2
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="${GPU%%,*}"  # <gpu> may be a list; render on the first
cd "$W"
# The machine is shared, has no swap, and other users' jobs grow: wait for RAM instead of pushing it into thrashing.
# A job with two 7B policies and 8 env workers needs about 15 GB; rollouts abort themselves below MIN_FREE_GB_RUN.
MIN_FREE_GB="${MIN_FREE_GB:-24}"
avail() { awk '/MemAvailable/ {printf "%d", $2 / 1e6}' /proc/meminfo; }
until [ "$(avail)" -ge "$MIN_FREE_GB" ]; do log "waiting for RAM: $(avail) GB available, need $MIN_FREE_GB"; sleep 120; done
# The card is shared (other projects, our own queues): a job starts only when its peak GPU memory is free. The
# check runs under a per-card lock that is kept until the job has had time to reach its peak (a training holds
# about 17 GiB while it loads and collects, 38 GiB once it trains), so two waiting jobs cannot take the same room.
case "$SCRIPT" in
  scripts/train_opd.py) NEED="${MIN_FREE_GPU_GB:-42}"; HOLD=1200 ;;
  scripts/eval_*.py|scripts/takeover_eval.py|scripts/probe_*.py|scripts/relabel_rollouts.py|scripts/check_policy.py) NEED="${MIN_FREE_GPU_GB:-22}"; HOLD=240 ;;
  *) NEED="${MIN_FREE_GPU_GB:-0}"; HOLD=0 ;;
esac
gpu_free() { nvidia-smi -i "${GPU%%,*}" --query-gpu=memory.free --format=csv,noheader,nounits | awk '{printf "%d", $1 / 1024}'; }
if [ "$NEED" -gt 0 ]; then
  exec 9> "$W/.gpu${GPU%%,*}_gate.lock"
  while :; do
    flock 9
    [ "$(gpu_free)" -ge "$NEED" ] && break
    flock -u 9
    log "waiting for GPU memory: $(gpu_free) GiB free, need $NEED"; sleep $((90 + RANDOM % 60))
  done
  (sleep "$HOLD"; flock -u 9) &
fi
log "start $SCRIPT on gpu $GPU (rev $(cat "$REPO/REVISION" 2>/dev/null), $(avail) GB RAM available)"
nice -n 10 "$PY" -u "$REPO/$SCRIPT" "$@" 9>&-
RC=$?
log "exit $RC"
exit $RC
