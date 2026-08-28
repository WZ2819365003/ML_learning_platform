#!/usr/bin/env bash
# Incrementally update the already-running production containers.
#
# This script is intentionally conservative. Application source and compiled
# frontend assets can be replaced without rebuilding the large ML images, but
# dependency, schema, Docker, or Nginx changes must use deploy-cloud.sh.

set -euo pipefail

FROM_HEAD=""
TO_HEAD=""
FORCE_BACKEND=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      FROM_HEAD="${2:-}"
      shift 2
      ;;
    --to)
      TO_HEAD="${2:-}"
      shift 2
      ;;
    --force-backend)
      FORCE_BACKEND=true
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

APP_DIR="$(git rev-parse --show-toplevel)"
cd "$APP_DIR"

FROM_HEAD="${FROM_HEAD:-$(git rev-parse HEAD)}"
TO_HEAD="${TO_HEAD:-$(git rev-parse HEAD)}"
RUNTIME_HEAD_FILE="$(git rev-parse --git-path ml-platform-runtime-head)"
SECRETS_FILE="$APP_DIR/docker/.deploy_secrets"

git cat-file -e "${FROM_HEAD}^{commit}"
git cat-file -e "${TO_HEAD}^{commit}"
if ! git merge-base --is-ancestor "$FROM_HEAD" "$TO_HEAD"; then
  echo "Hot update requires a fast-forward history: $FROM_HEAD is not an ancestor of $TO_HEAD." >&2
  exit 3
fi

mapfile -t CHANGED_FILES < <(git diff --name-only "$FROM_HEAD" "$TO_HEAD")
BACKEND_CHANGED=false
FRONTEND_CHANGED=false
declare -a FULL_DEPLOY_FILES=()
declare -a NON_RUNTIME_FILES=()

for path in "${CHANGED_FILES[@]}"; do
  case "$path" in
    ml_platform/app/*)
      BACKEND_CHANGED=true
      ;;
    ml_platform_web/src/*|ml_platform_web/public/*|ml_platform_web/index.html)
      FRONTEND_CHANGED=true
      ;;
    ml_platform/requirements.txt|ml_platform/alembic.ini|ml_platform/alembic/*|\
    ml_platform_web/package.json|ml_platform_web/package-lock.json|\
    ml_platform_web/vite.config.*|docker/*)
      FULL_DEPLOY_FILES+=("$path")
      ;;
    ml_platform/*)
      # Runtime files outside app/ may be copied or invoked by training jobs.
      # Rebuild the image rather than guessing whether a live process imports it.
      case "$path" in
        ml_platform/tests/*)
          NON_RUNTIME_FILES+=("$path")
          ;;
        *)
          FULL_DEPLOY_FILES+=("$path")
          ;;
      esac
      ;;
    ml_platform_web/*)
      NON_RUNTIME_FILES+=("$path")
      ;;
    *)
      NON_RUNTIME_FILES+=("$path")
      ;;
  esac
done

if [[ "$FORCE_BACKEND" == true ]]; then
  BACKEND_CHANGED=true
fi

if (( ${#FULL_DEPLOY_FILES[@]} > 0 )); then
  echo "Hot update refused: the following files require a full deployment:" >&2
  printf '  - %s\n' "${FULL_DEPLOY_FILES[@]}" >&2
  echo "Run scripts/deploy-cloud.sh so dependencies, schema, and images stay consistent." >&2
  exit 20
fi

echo "Hot update plan: ${FROM_HEAD:0:12} -> ${TO_HEAD:0:12}"
echo "  backend=$BACKEND_CHANGED frontend=$FRONTEND_CHANGED non_runtime=${#NON_RUNTIME_FILES[@]}"

if [[ "$BACKEND_CHANGED" == false && "$FRONTEND_CHANGED" == false ]]; then
  printf '%s\n' "$TO_HEAD" > "$RUNTIME_HEAD_FILE"
  echo "No runtime files changed; deployment marker advanced without restarting services."
  exit 0
fi

for container in ml_platform_backend ml_platform_worker ml_platform_frontend ml_platform_nginx; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" != "true" ]]; then
    echo "Required container is not running: $container" >&2
    exit 4
  fi
done

BACKEND_SWAPPED=false
FRONTEND_SWAPPED=false

rollback() {
  local exit_code="$1"
  trap - ERR
  if [[ "$FRONTEND_SWAPPED" == true ]]; then
    docker exec ml_platform_frontend sh -c '
      if [ -d /app/dist.previous ]; then
        rm -rf /app/dist.failed
        mv /app/dist /app/dist.failed
        mv /app/dist.previous /app/dist
        rm -rf /app/dist.failed
      fi
    ' || true
    docker restart ml_platform_frontend >/dev/null || true
  fi
  if [[ "$BACKEND_SWAPPED" == true ]]; then
    for container in ml_platform_backend ml_platform_worker; do
      docker exec "$container" sh -c '
        if [ -d /app/app.previous ]; then
          rm -rf /app/app.failed
          mv /app/app /app/app.failed
          mv /app/app.previous /app/app
          rm -rf /app/app.failed
        fi
      ' || true
    done
    docker restart ml_platform_backend ml_platform_worker >/dev/null || true
  fi
  echo "Hot update failed; previous runtime files were restored." >&2
  exit "$exit_code"
}
trap 'rollback $?' ERR

wait_for_http() {
  local url="$1"
  local attempts="${2:-60}"
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_worker() {
  local attempts="${1:-30}"
  for _ in $(seq 1 "$attempts"); do
    if docker exec ml_platform_worker \
      celery -A app.scheduler.celery_app inspect ping --timeout=2 2>/dev/null \
      | grep -q pong; then
      return 0
    fi
    sleep 1
  done
  docker logs --tail=80 ml_platform_worker >&2 || true
  return 1
}

if [[ "$BACKEND_CHANGED" == true ]]; then
  # Mark the backend stage before the first swap. The rollback function checks
  # for app.previous per container, so a partial two-container update is safe.
  BACKEND_SWAPPED=true
  for container in ml_platform_backend ml_platform_worker; do
    docker exec "$container" sh -c 'rm -rf /app/app.next /app/app.previous && mkdir -p /app/app.next'
    tar -C "$APP_DIR/ml_platform/app" -cf - . \
      | docker exec -i "$container" tar -xf - -C /app/app.next
    docker exec "$container" sh -c 'mv /app/app /app/app.previous && mv /app/app.next /app/app'
  done
  docker restart ml_platform_backend ml_platform_worker >/dev/null
  wait_for_http http://127.0.0.1:8000/health 60
  wait_for_worker 30
fi

if [[ "$FRONTEND_CHANGED" == true ]]; then
  WEB_UID="$(id -u)"
  WEB_GID="$(id -g)"
  docker run --rm \
    --user "$WEB_UID:$WEB_GID" \
    -e HOME=/tmp \
    -v "$APP_DIR/ml_platform_web:/workspace" \
    -w /workspace \
    node:20-alpine sh -c '
      if [ ! -x node_modules/.bin/vite ]; then
        npm install --prefer-offline --no-audit --no-package-lock
      fi
      npm run build
    '
  docker exec ml_platform_frontend sh -c 'rm -rf /app/dist.next /app/dist.previous && mkdir -p /app/dist.next'
  tar -C "$APP_DIR/ml_platform_web/dist" -cf - . \
    | docker exec -i ml_platform_frontend tar -xf - -C /app/dist.next
  docker exec ml_platform_frontend sh -c 'mv /app/dist /app/dist.previous && mv /app/dist.next /app/dist'
  FRONTEND_SWAPPED=true
  docker restart ml_platform_frontend >/dev/null
  wait_for_http http://127.0.0.1:3000 60
fi

# Check both the direct backend and the real public path through Nginx. Docker
# restart keeps container IPs stable, but the gateway check protects the actual
# user path and catches proxy regressions.
curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:18081/health >/dev/null

if [[ ! -f "$SECRETS_FILE" ]]; then
  echo "Missing server credential file: $SECRETS_FILE" >&2
  exit 5
fi
set -a
# shellcheck disable=SC1090
source "$SECRETS_FILE"
set +a
SMOKE_USERNAME="${AUTH_SMOKE_USERNAME:-${AUTH_USERNAME:-}}"
SMOKE_PASSWORD="${AUTH_SMOKE_PASSWORD:-${AUTH_PASSWORD:-}}"
: "${SMOKE_USERNAME:?AUTH_SMOKE_USERNAME or AUTH_USERNAME must be configured}"
: "${SMOKE_PASSWORD:?AUTH_SMOKE_PASSWORD or AUTH_PASSWORD must be configured}"
LOGIN_PAYLOAD="$(SMOKE_USERNAME="$SMOKE_USERNAME" SMOKE_PASSWORD="$SMOKE_PASSWORD" python3 -c \
  'import json, os; print(json.dumps({"username": os.environ["SMOKE_USERNAME"], "password": os.environ["SMOKE_PASSWORD"]}))')"
LOGIN_RESPONSE="$(curl -fsS --max-time 10 \
  -H 'Content-Type: application/json' \
  --data "$LOGIN_PAYLOAD" \
  http://127.0.0.1:18081/api/auth/login)"
TOKEN="$(printf '%s' "$LOGIN_RESPONSE" | python3 -c \
  'import json, sys; print(json.load(sys.stdin)["token"])')"
curl -fsS --max-time 10 -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:18081/api/auth/me >/dev/null
curl -fsS --max-time 10 -H "Authorization: Bearer $TOKEN" \
  'http://127.0.0.1:18081/api/models/assets?runtime_type=ml&page=1&page_size=1' >/dev/null

if [[ "$BACKEND_SWAPPED" == true ]]; then
  docker exec ml_platform_backend rm -rf /app/app.previous
  docker exec ml_platform_worker rm -rf /app/app.previous
fi
if [[ "$FRONTEND_SWAPPED" == true ]]; then
  docker exec ml_platform_frontend rm -rf /app/dist.previous
fi

printf '%s\n' "$TO_HEAD" > "$RUNTIME_HEAD_FILE"
trap - ERR
echo "Hot update and authenticated smoke test succeeded at ${TO_HEAD:0:12}."
