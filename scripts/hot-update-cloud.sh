#!/usr/bin/env bash
# Fetch a pushed commit on the cloud server and apply a safe incremental update.

set -euo pipefail

HOST="${DEPLOY_HOST:-your-server.example.com}"
PORT="${DEPLOY_PORT:-22}"
REMOTE_USER="${DEPLOY_USER:-opsadmin}"
REMOTE_DIR="${DEPLOY_DIR:-/home/opsadmin/ml_platform}"
BRANCH="${DEPLOY_BRANCH:-$(git branch --show-current)}"
FORCE_BACKEND=false

if [[ "${1:-}" == "--force-backend" ]]; then
  FORCE_BACKEND=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--force-backend]" >&2
  exit 2
fi

if [[ "$HOST" == "your-server.example.com" ]]; then
  echo "Set DEPLOY_HOST before running this script." >&2
  exit 2
fi

LOCAL_HEAD="$(git rev-parse HEAD)"
git fetch --quiet origin "$BRANCH"
REMOTE_BRANCH_HEAD="$(git rev-parse FETCH_HEAD)"
if [[ "$LOCAL_HEAD" != "$REMOTE_BRANCH_HEAD" ]]; then
  echo "Local HEAD is not the pushed origin/$BRANCH commit." >&2
  echo "local=$LOCAL_HEAD remote=$REMOTE_BRANCH_HEAD" >&2
  exit 3
fi

FROM_HEAD="$(ssh -p "$PORT" "${REMOTE_USER}@${HOST}" bash -s -- "$REMOTE_DIR" <<'REMOTE_STATE'
set -euo pipefail
APP_DIR="$1"
cd "$APP_DIR"
if [[ ! -d .git ]]; then
  echo "Cloud directory is not a git checkout; run the full deploy once first." >&2
  exit 4
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Cloud checkout has tracked modifications; refusing to overwrite them." >&2
  git status --short >&2
  exit 5
fi

RUNTIME_HEAD_FILE="$(git rev-parse --git-path ml-platform-runtime-head)"
if [[ -s "$RUNTIME_HEAD_FILE" ]]; then
  FROM_HEAD="$(cat "$RUNTIME_HEAD_FILE")"
else
  FROM_HEAD="$(git rev-parse HEAD)"
fi
printf '%s\n' "$FROM_HEAD"
REMOTE_STATE
)"

git cat-file -e "${FROM_HEAD}^{commit}"
if ! git merge-base --is-ancestor "$FROM_HEAD" "$LOCAL_HEAD"; then
  echo "Cloud runtime is not an ancestor of the target commit; full deployment is required." >&2
  exit 6
fi

BUNDLE_PATH=""
BUNDLE_DIR=""
if [[ "$FROM_HEAD" != "$LOCAL_HEAD" ]]; then
  BUNDLE_DIR="$(mktemp -d)"
  trap 'rm -rf "$BUNDLE_DIR"' EXIT
  BUNDLE_PATH="$BUNDLE_DIR/ml-platform.bundle"
  git bundle create "$BUNDLE_PATH" "$BRANCH" "^$FROM_HEAD"
  REMOTE_BUNDLE="/tmp/ml-platform-${LOCAL_HEAD}.bundle"
  scp -q -P "$PORT" "$BUNDLE_PATH" "${REMOTE_USER}@${HOST}:$REMOTE_BUNDLE"
else
  REMOTE_BUNDLE=""
fi

ssh -p "$PORT" "${REMOTE_USER}@${HOST}" bash -s -- \
  "$BRANCH" "$FROM_HEAD" "$LOCAL_HEAD" "$REMOTE_DIR" "$FORCE_BACKEND" "$REMOTE_BUNDLE" <<'REMOTE_SCRIPT'
set -euo pipefail

BRANCH="$1"
FROM_HEAD="$2"
EXPECTED_HEAD="$3"
APP_DIR="$4"
FORCE_BACKEND="$5"
BUNDLE_PATH="$6"

if [[ -n "$BUNDLE_PATH" ]]; then
  trap 'rm -f "$BUNDLE_PATH"' EXIT
fi
cd "$APP_DIR"
if [[ -n "$BUNDLE_PATH" ]]; then
  git fetch "$BUNDLE_PATH" "refs/heads/$BRANCH"
  ACTUAL_HEAD="$(git rev-parse FETCH_HEAD)"
else
  ACTUAL_HEAD="$(git rev-parse HEAD)"
fi
if [[ "$ACTUAL_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "Cloud bundle mismatch: expected $EXPECTED_HEAD, got $ACTUAL_HEAD" >&2
  exit 7
fi
git checkout -B "$BRANCH" "$ACTUAL_HEAD"

ARGS=(--from "$FROM_HEAD" --to "$ACTUAL_HEAD")
if [[ "$FORCE_BACKEND" == true ]]; then
  ARGS+=(--force-backend)
fi
bash scripts/hot-update-runtime.sh "${ARGS[@]}"
REMOTE_SCRIPT
