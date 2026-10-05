#!/usr/bin/env bash
# Run a repo python script on one GPU with the workspace environment.
#   run_py.sh <gpu> <script relative to repo> [args...]      (relative --out paths resolve inside the workspace)
# Launch detached: setsid nohup bash repo/scripts/server/run_py.sh 0 scripts/eval_libero.py ... > logs/x.log 2>&1 < /dev/null &
set -uo pipefail
. "$(dirname "$0")/env.sh"
GPU="$1"; SCRIPT="$2"; shift 2
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU"
cd "$W"
# The machine is shared, has no swap, and other users' jobs grow: wait for RAM instead of pushing it into thrashing.
# A job with two 7B policies and 8 env workers needs about 15 GB; rollouts abort themselves below MIN_FREE_GB_RUN.
MIN_FREE_GB="${MIN_FREE_GB:-24}"
avail() { awk '/MemAvailable/ {printf "%d", $2 / 1e6}' /proc/meminfo; }
until [ "$(avail)" -ge "$MIN_FREE_GB" ]; do log "waiting for RAM: $(avail) GB available, need $MIN_FREE_GB"; sleep 120; done
log "start $SCRIPT on gpu $GPU (rev $(cat "$REPO/REVISION" 2>/dev/null), $(avail) GB RAM available)"
nice -n 10 "$PY" -u "$REPO/$SCRIPT" "$@"
RC=$?
log "exit $RC"
exit $RC
