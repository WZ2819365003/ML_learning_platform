#!/usr/bin/env bash
# Deploy the current Git HEAD without storing credentials in the repository.
# Override DEPLOY_HOST, DEPLOY_PORT, DEPLOY_USER, and DEPLOY_DIR in the caller
# environment. Runtime and smoke-test credentials must live in the server's
# docker/.deploy_secrets file.

set -euo pipefail

# SSH target placeholders. A real host must be supplied by the caller.
HOST="${DEPLOY_HOST:-your-server.example.com}"
PORT="${DEPLOY_PORT:-22}"
REMOTE_USER="${DEPLOY_USER:-opsadmin}"
REMOTE_DIR="${DEPLOY_DIR:-/home/opsadmin/ml_platform}"

if [[ "$HOST" == "your-server.example.com" ]]; then
  echo "Set DEPLOY_HOST before running this script." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REVISION="$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)"
ARCHIVE="$(mktemp "/tmp/ml-platform-${REVISION}.XXXXXX.tar.gz")"
REMOTE_ARCHIVE="/tmp/ml-platform-${REVISION}.tar.gz"

cleanup_local_archive() {
  rm -f "$ARCHIVE"
}
trap cleanup_local_archive EXIT

# Step 1: package the exact local HEAD; uncommitted files are intentionally
# excluded so the deployed source is reproducible.
git -C "$REPO_ROOT" archive --format=tar.gz --output="$ARCHIVE" HEAD

# Step 2: upload the archive to the server's temporary directory.
scp -P "$PORT" "$ARCHIVE" "${REMOTE_USER}@${HOST}:${REMOTE_ARCHIVE}"

# Steps 3-6 run atomically on the server (set -e): a failed migration, build,
# health check, or authenticated smoke test stops the deployment immediately.
ssh -p "$PORT" "${REMOTE_USER}@${HOST}" bash -s -- \
  "$REMOTE_ARCHIVE" "$REMOTE_DIR" <<'REMOTE_SCRIPT'
set -euo pipefail

ARCHIVE="$1"
APP_DIR="$2"
SECRETS_FILE="$APP_DIR/docker/.deploy_secrets"
COMPOSE_FILE="$APP_DIR/docker/docker-compose.yml"

cleanup_remote_archive() {
  rm -f "$ARCHIVE"
}
trap cleanup_remote_archive EXIT

# Step 2 (server): unpack into the stable application directory. The ignored
# docker/.deploy_secrets file is retained across releases.
mkdir -p "$APP_DIR"
tar -xzf "$ARCHIVE" -C "$APP_DIR"
cd "$APP_DIR"

if [[ ! -f "$SECRETS_FILE" ]]; then
  echo "Missing server credential file: $SECRETS_FILE" >&2
  exit 3
fi

# Export server-only values for Compose interpolation and the smoke test.
set -a
# shellcheck disable=SC1090
source "$SECRETS_FILE"
set +a

COMPOSE=(docker compose -f "$COMPOSE_FILE")
# The previous release image already contains Alembic, but not this release's
# migration files. Mount the freshly unpacked backend source into the one-shot
# container so schema migration still happens before the new image is built.
MIGRATION_RUN=("${COMPOSE[@]}" run --rm -T -v "$APP_DIR/ml_platform:/app")

# Step 3: adopt an existing legacy schema once, then apply every revision. The
# bootstrap script is conservative: it only stamps when business tables exist
# but Alembic has no version row.
"${MIGRATION_RUN[@]}" backend python scripts/ensure_alembic_baseline.py </dev/null
"${MIGRATION_RUN[@]}" backend alembic upgrade head </dev/null

# Step 4: build both application images only after the schema is ready.
"${COMPOSE[@]}" build backend frontend

# Step 5: reconcile the stack to the newly built images.
"${COMPOSE[@]}" up -d

# Step 6a: wait for the public backend health endpoint.
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
for attempt in $(seq 1 60); do
  if curl -fsS --max-time 3 "$HEALTH_URL" >/dev/null; then
    break
  fi
  if [[ "$attempt" -eq 60 ]]; then
    echo "Health check failed after 120 seconds: $HEALTH_URL" >&2
    exit 4
  fi
  sleep 2
done

# Step 6b: authenticate with server-only credentials, then prove protected
# endpoints accept the issued bearer token. Multi-account deployments can set
# AUTH_SMOKE_USERNAME/AUTH_SMOKE_PASSWORD while keeping AUTH_USERS_JSON as the
# application credential source.
SMOKE_USERNAME="${AUTH_SMOKE_USERNAME:-${AUTH_USERNAME:-}}"
SMOKE_PASSWORD="${AUTH_SMOKE_PASSWORD:-${AUTH_PASSWORD:-}}"
: "${SMOKE_USERNAME:?AUTH_SMOKE_USERNAME or AUTH_USERNAME must be set in docker/.deploy_secrets}"
: "${SMOKE_PASSWORD:?AUTH_SMOKE_PASSWORD or AUTH_PASSWORD must be set in docker/.deploy_secrets}"
LOGIN_PAYLOAD="$(python3 -c \
  'import json, os; print(json.dumps({"username": os.environ["SMOKE_USERNAME"], "password": os.environ["SMOKE_PASSWORD"]}))')"
LOGIN_RESPONSE="$(curl -fsS \
  -H 'Content-Type: application/json' \
  --data "$LOGIN_PAYLOAD" \
  "$API_BASE_URL/api/auth/login")"
TOKEN="$(printf '%s' "$LOGIN_RESPONSE" | python3 -c \
  'import json, sys; print(json.load(sys.stdin)["token"])')"
curl -fsS \
  -H "Authorization: Bearer $TOKEN" \
  "$API_BASE_URL/api/auth/me" >/dev/null
curl -fsS \
  -H "Authorization: Bearer $TOKEN" \
  "$API_BASE_URL/api/data/list" >/dev/null
curl -fsS \
  -H "Authorization: Bearer $TOKEN" \
  "$API_BASE_URL/api/v3/tasks/" >/dev/null

echo "Deployment and authenticated smoke test succeeded."
REMOTE_SCRIPT
