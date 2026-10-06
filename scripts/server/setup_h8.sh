#!/usr/bin/env bash
# Rest of the workspace setup on the 8x H200 node after build_env.sh: LIBERO-Plus, LIBERO-PRO, the pi0.5 env, then
# the gates (env stepping, renderer). Launch detached:
#   setsid nohup bash repo/scripts/server/setup_h8.sh > logs/setup_h8.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
until grep -qE "BUILD_DONE|_FAILED" logs/build_env.log 2>/dev/null; do sleep 30; done
grep -q BUILD_DONE logs/build_env.log || { log "SETUP_STOPPED: build_env failed"; exit 1; }
bash "$HERE/setup_libero_plus.sh" > logs/setup_libero_plus.log 2>&1 || log "libero-plus setup failed"
bash "$HERE/setup_libero_pro.sh" > logs/setup_libero_pro.log 2>&1 || log "libero-pro setup failed"
bash "$HERE/build_env_pi05.sh" > logs/build_env_pi05.log 2>&1 || log "pi05 env failed"
log "SETUP_H8_DONE"
