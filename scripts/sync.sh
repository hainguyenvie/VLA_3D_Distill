#!/usr/bin/env bash
# Push the local repo (source of truth) to <workspace>/repo on a server.
#   scripts/sync.sh [l40|h200|all]      default: every machine configured in infra.env
set -euo pipefail
cd "$(dirname "$0")/.."
. ./infra.env
# results record which code produced them: commit hash, plus "-dirty" when the tree has uncommitted changes
REV="$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo -dirty || true)"
push() {  # push <ssh alias> <remote root, relative to the remote home>
  [ -n "$1" ] && [ -n "$2" ] || return 0
  ssh -n "$1" "mkdir -p $2/repo"
  rsync -az --delete --exclude .git --exclude REVISION --exclude-from=.gitignore ./ "$1:$2/repo/"
  ssh -n "$1" "echo $REV > $2/repo/REVISION"
  echo "synced $REV -> $3"
}
case "${1:-all}" in
  l40) push "$L40_SSH" "$L40_REMOTE_ROOT" l40 ;;
  h200) push "${H200_SSH:-}" "${H200_REMOTE_ROOT:-}" h200 ;;
  all) push "$L40_SSH" "$L40_REMOTE_ROOT" l40; push "${H200_SSH:-}" "${H200_REMOTE_ROOT:-}" h200 ;;
  *) echo "usage: scripts/sync.sh [l40|h200|all]" >&2; exit 2 ;;
esac
