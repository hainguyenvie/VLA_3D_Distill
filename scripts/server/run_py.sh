#!/usr/bin/env bash
# Run a repo python script on one GPU with the workspace environment.
#   run_py.sh <gpu> <script relative to repo> [args...]      (relative --out paths resolve inside the workspace)
# Launch detached: setsid nohup bash repo/scripts/server/run_py.sh 0 scripts/eval_libero.py ... > logs/x.log 2>&1 < /dev/null &
set -uo pipefail
. "$(dirname "$0")/env.sh"
GPU="$1"; SCRIPT="$2"; shift 2
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU"
cd "$W"
log "start $SCRIPT on gpu $GPU (rev $(cat "$REPO/REVISION" 2>/dev/null))"
"$PY" -u "$REPO/$SCRIPT" "$@"
RC=$?
log "exit $RC"
exit $RC
