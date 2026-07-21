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
MIGRATION_RUN=("${COMPOSE[@]}" run --rm -v "$APP_DIR/ml_platform:/app")

# Step 3: adopt an existing production schema once, then apply every revision.
# The application tables already match 0001; stamping records that fact without
# replaying baseline DDL against the live database.
HAS_ALEMBIC_VERSION="$(
  "${MIGRATION_RUN[@]}" backend python -c \
    'from sqlalchemy import create_engine, inspect; from app.config import get_settings; from app.models.database import _to_sync_database_url; engine = create_engine(_to_sync_database_url(get_settings().database_url)); print("yes" if inspect(engine).has_table("alembic_version") else "no")' \
    | tail -n 1
)"
if [[ "$HAS_ALEMBIC_VERSION" == "no" ]]; then
  "${MIGRATION_RUN[@]}" backend alembic stamp 0001
fi
"${MIGRATION_RUN[@]}" backend alembic upgrade head

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

# Step 6b: authenticate with server-only credentials, then prove a protected
# endpoint accepts the issued bearer token.
: "${AUTH_USERNAME:?AUTH_USERNAME must be set in docker/.deploy_secrets}"
: "${AUTH_PASSWORD:?AUTH_PASSWORD must be set in docker/.deploy_secrets}"
LOGIN_PAYLOAD="$(python3 -c \
  'import json, os; print(json.dumps({"username": os.environ["AUTH_USERNAME"], "password": os.environ["AUTH_PASSWORD"]}))')"
LOGIN_RESPONSE="$(curl -fsS \
  -H 'Content-Type: application/json' \
  --data "$LOGIN_PAYLOAD" \
  "$API_BASE_URL/api/auth/login")"
TOKEN="$(printf '%s' "$LOGIN_RESPONSE" | python3 -c \
  'import json, sys; print(json.load(sys.stdin)["token"])')"
curl -fsS \
  -H "Authorization: Bearer $TOKEN" \
  "$API_BASE_URL/api/auth/me" >/dev/null

echo "Deployment and authenticated smoke test succeeded."
REMOTE_SCRIPT
