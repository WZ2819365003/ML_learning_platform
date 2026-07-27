#!/usr/bin/env bash
# Deploy by updating the source checkout on the cloud server, then building
# there. Override DEPLOY_HOST, DEPLOY_PORT, DEPLOY_USER, DEPLOY_DIR, DEPLOY_REPO,
# and DEPLOY_BRANCH in the caller environment. Runtime and smoke-test
# credentials must live in the server's docker/.deploy_secrets file.

set -euo pipefail

# SSH target placeholders. A real host must be supplied by the caller.
HOST="${DEPLOY_HOST:-your-server.example.com}"
PORT="${DEPLOY_PORT:-22}"
REMOTE_USER="${DEPLOY_USER:-opsadmin}"
REMOTE_DIR="${DEPLOY_DIR:-/home/opsadmin/ml_platform}"
REPO_URL="${DEPLOY_REPO:-https://github.com/WZ2819365003/ML_learning_platform.git}"
BRANCH="${DEPLOY_BRANCH:-$(git branch --show-current)}"

if [[ "$HOST" == "your-server.example.com" ]]; then
  echo "Set DEPLOY_HOST before running this script." >&2
  exit 2
fi

LOCAL_HEAD="$(git rev-parse --short=12 HEAD)"

# The server performs the source update and image build. A failed fetch,
# migration, build, health check, or authenticated smoke test stops the
# deployment immediately.
ssh -p "$PORT" "${REMOTE_USER}@${HOST}" bash -s -- \
  "$REPO_URL" "$BRANCH" "$LOCAL_HEAD" "$REMOTE_DIR" <<'REMOTE_SCRIPT'
set -euo pipefail

REPO_URL="$1"
BRANCH="$2"
EXPECTED_HEAD="$3"
APP_DIR="$4"
SECRETS_FILE="$APP_DIR/docker/.deploy_secrets"
COMPOSE_FILE="$APP_DIR/docker/docker-compose.yml"

# Step 1 (server): make the deployment directory a real git checkout, then
# update it to the requested branch. Ignored runtime files and
# docker/.deploy_secrets are retained.
mkdir -p "$APP_DIR"
cd "$APP_DIR"
if [[ ! -d .git ]]; then
  TMP_SECRETS=""
  if [[ -f "$SECRETS_FILE" ]]; then
    TMP_SECRETS="$(mktemp)"
    cp "$SECRETS_FILE" "$TMP_SECRETS"
  fi
  find "$APP_DIR" -mindepth 1 -maxdepth 1 ! -name docker -exec rm -rf {} +
  if [[ -d "$APP_DIR/docker" ]]; then
    find "$APP_DIR/docker" -mindepth 1 ! -name .deploy_secrets -exec rm -rf {} +
  fi
  if [[ -n "$TMP_SECRETS" ]]; then
    mkdir -p "$APP_DIR/docker"
    cp "$TMP_SECRETS" "$SECRETS_FILE"
    rm -f "$TMP_SECRETS"
  fi
  git init
fi
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi
git fetch --prune origin "$BRANCH"
git checkout -B "$BRANCH" FETCH_HEAD
git reset --hard FETCH_HEAD
git clean -fd -e docker/.deploy_secrets

if [[ ! -f "$SECRETS_FILE" ]]; then
  echo "Missing server credential file: $SECRETS_FILE" >&2
  exit 3
fi

ACTUAL_HEAD="$(git rev-parse --short=12 HEAD)"
if [[ "$ACTUAL_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "Cloud checkout mismatch: expected $EXPECTED_HEAD, got $ACTUAL_HEAD" >&2
  exit 5
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

# Step 4: build every application image only after the schema is ready.
"${COMPOSE[@]}" build backend worker frontend

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
LOGIN_PAYLOAD="$(SMOKE_USERNAME="$SMOKE_USERNAME" SMOKE_PASSWORD="$SMOKE_PASSWORD" python3 -c \
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

echo "Deployment and authenticated smoke test succeeded at $ACTUAL_HEAD."
REMOTE_SCRIPT
