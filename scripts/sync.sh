#!/usr/bin/env bash
# Push the local repo (source of truth) to <workspace>/repo on the server. Usage: scripts/sync.sh
set -euo pipefail
cd "$(dirname "$0")/.."
. ./infra.env
rsync -az --delete --exclude .git --exclude-from=.gitignore ./ "$L40_SSH:$L40_REMOTE_ROOT/repo/"
echo "synced -> \$L40_REMOTE_ROOT/repo"
