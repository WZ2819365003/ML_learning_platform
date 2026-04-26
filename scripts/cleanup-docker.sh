#!/usr/bin/env bash
# scripts/cleanup-docker.sh
#
# 清理 Docker 中与本项目无关的镜像 / 容器 / 卷 / 构建缓存。
# 保留：
#   - 当前运行的 ml_platform_* 镜像（docker-backend:latest, docker-frontend:latest）
#   - 当前 ml-platform 用到的官方基础镜像（mysql, redis, minio, nginx）
#   - PVMS 镜像族（pvms-*:2.2.1）
#   - PVMS 数据卷（pvms-mysql-data）
#   - ml-platform 当前数据卷（docker_ml_mysql_data / docker_ml_minio_data / docker_ml_redis_data）
#
# 删除：
#   - 已退出的 cloudflared-pvms / ml_platform_minio_init 容器（minio_init 是一次性 seed 任务）
#   - cloudflare/cloudflared:latest 镜像（PVMS 已不再使用）
#   - 历史 compose project 残留卷（2_*, ml_learning_platform_*）
#   - 与本机无关的孤儿卷（vpp_timescaledb-data, docker_pvms-mysql-data）
#   - 全部 docker builder 缓存
#
# 使用：
#   bash scripts/cleanup-docker.sh

set -uo pipefail

step() { echo ""; echo "===> $*"; }
ok()   { echo "  ✓ $*"; }
warn() { echo "  ⚠ $*"; }

step "Before cleanup"
docker system df

step "Removing exited orphan containers"
for c in cloudflared-pvms ml_platform_minio_init; do
  if docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
    docker rm "$c" >/dev/null && ok "rm container $c"
  else
    warn "container $c already gone"
  fi
done

step "Removing legacy / orphan volumes"
ORPHAN_VOLS=(
  2_minio_data
  2_mysql_data
  ml_learning_platform_ml_minio_data
  ml_learning_platform_ml_mysql_data
  ml_learning_platform_ml_redis_data
  docker_pvms-mysql-data
  vpp_timescaledb-data
)
for v in "${ORPHAN_VOLS[@]}"; do
  if docker volume ls --format '{{.Name}}' | grep -qx "$v"; then
    docker volume rm "$v" >/dev/null 2>&1 && ok "rm volume $v" || warn "volume $v in use, kept"
  else
    warn "volume $v already gone"
  fi
done

step "Removing cloudflare/cloudflared image (no active container)"
if docker images --format '{{.Repository}}:{{.Tag}}' | grep -qx 'cloudflare/cloudflared:latest'; then
  docker rmi cloudflare/cloudflared:latest >/dev/null 2>&1 && ok "rm image cloudflare/cloudflared:latest" \
    || warn "image cloudflare/cloudflared in use, kept"
else
  warn "cloudflare/cloudflared:latest already gone"
fi

step "Pruning dangling images"
docker image prune -f | tail -3

step "Pruning ALL build cache"
docker builder prune -af | tail -3

step "After cleanup"
docker system df
echo ""
echo "Containers still running:"
docker ps --format "  {{.Names}}: {{.Status}}"
