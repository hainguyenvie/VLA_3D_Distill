#!/usr/bin/env bash
# First step on the 8x H200 node, run from the laptop (the node has no system rsync, so the repo cannot be synced
# before this):  scripts/server/bootstrap_h8.sh
# Creates the workspace and a small micromamba env envs/tools with rsync, then hands over to scripts/sync.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
. ./infra.env
ssh -n "$H8_SSH" "mkdir -p $H8_REMOTE_ROOT/{envs,logs,outputs,checkpoints,third_party,data,scratch} && \
  { [ -x $H8_REMOTE_ROOT/envs/tools/bin/rsync ] || ~/.local/bin/micromamba create -y -q -p $H8_REMOTE_ROOT/envs/tools -c conda-forge rsync > /dev/null; } && \
  $H8_REMOTE_ROOT/envs/tools/bin/rsync --version | head -1"
scripts/sync.sh h8
