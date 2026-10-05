#!/usr/bin/env bash
# Push the local repo (source of truth) to <workspace>/repo on the server. Usage: scripts/sync.sh
set -euo pipefail
cd "$(dirname "$0")/.."
. ./infra.env
rsync -az --delete --exclude .git --exclude REVISION --exclude-from=.gitignore ./ "$L40_SSH:$L40_REMOTE_ROOT/repo/"
# results record which code produced them: commit hash, plus "-dirty" when the tree has uncommitted changes
REV="$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo -dirty)"
ssh -n "$L40_SSH" "echo $REV > $L40_REMOTE_ROOT/repo/REVISION"
echo "synced -> \$L40_REMOTE_ROOT/repo"
