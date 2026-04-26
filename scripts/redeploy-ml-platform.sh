#!/usr/bin/env bash
# scripts/redeploy-ml-platform.sh
#
# 用当前 main 分支代码重建 ml-platform 的 backend + frontend 镜像并重启。
# 保留 MySQL / MinIO / Redis 数据卷（不带 -v）。
#
# 使用：
#   bash scripts/redeploy-ml-platform.sh
#
# 阶段：
#   1) git 状态检查（需在 ml-platform 仓库根目录）
#   2) docker compose down（保留卷）
#   3) docker compose build --no-cache backend frontend
#   4) docker compose up -d
#   5) 等待 /health 健康，校验版本号

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_DIR="$REPO_ROOT/docker"
EXPECTED_VERSION="${EXPECTED_VERSION:-}"

step() { echo ""; echo "===> $*"; }
ok()   { echo "  ✓ $*"; }
fail() { echo "  ✗ $*" >&2; exit 1; }

step "Pre-flight: git state"
cd "$REPO_ROOT"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
HEAD=$(git rev-parse --short HEAD)
DIRTY=""
if ! git diff --quiet || ! git diff --cached --quiet; then
  DIRTY=" (dirty)"
fi
ok "branch=$BRANCH head=$HEAD$DIRTY"
if [ -z "$EXPECTED_VERSION" ]; then
  EXPECTED_VERSION=$(grep -oE '"3\.[0-9]+\.[0-9]+"' ml_platform/app/main.py | head -1 | tr -d '"')
  ok "auto-detected EXPECTED_VERSION=$EXPECTED_VERSION from ml_platform/app/main.py"
fi

step "Stopping ml-platform stack (volumes preserved)"
cd "$COMPOSE_DIR"
docker compose down --remove-orphans

step "Building backend + frontend (no cache)"
docker compose build --no-cache backend frontend

step "Starting full stack"
docker compose up -d

step "Waiting for backend /health (max 120s)"
START=$(date +%s)
for i in $(seq 1 60); do
  if BODY=$(curl -sf --max-time 3 http://127.0.0.1:8000/health 2>/dev/null); then
    ELAPSED=$(( $(date +%s) - START ))
    ok "/health 200 after ${ELAPSED}s -> $BODY"
    if echo "$BODY" | grep -q "\"version\":\"$EXPECTED_VERSION\""; then
      ok "version matches expected $EXPECTED_VERSION"
    else
      fail "version mismatch — got $BODY, expected $EXPECTED_VERSION"
    fi
    break
  fi
  sleep 2
done
if [ -z "${BODY:-}" ]; then
  fail "/health never returned 200"
fi

step "Waiting for frontend (max 60s)"
for i in $(seq 1 30); do
  if curl -sf --max-time 3 http://127.0.0.1:3000 >/dev/null 2>&1; then
    ok "frontend 200"
    break
  fi
  sleep 2
done

step "Final container state"
docker ps --filter "name=ml_platform" --format "  {{.Names}}: {{.Status}}"

echo ""
echo "Done. Open:"
echo "  - SPA:        http://127.0.0.1:3000"
echo "  - API docs:   http://127.0.0.1:8000/docs"
echo "  - Nginx:      http://127.0.0.1"
echo "  - MinIO:      http://127.0.0.1:9001 (mlplatform / mlplatform123)"
