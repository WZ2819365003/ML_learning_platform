#!/usr/bin/env bash
# ML Platform daily backup.
#
# Install the 03:00 cron job (run as opsadmin):
#   (crontab -l 2>/dev/null; echo '0 3 * * * /home/opsadmin/ML_learning_platform/docker/backup.sh') | crontab -
#
# Quarterly restore drill (reference consistency, not row-count-only):
#   1. Create a temporary database, for example `ml_platform_restore_YYYYMMDD`.
#   2. Restore one mysql/*.sql.gz into it:
#        gzip -dc BACKUP.sql.gz | mysql --defaults-extra-file=TEMP_CNF ml_platform_restore_YYYYMMDD
#   3. Query one Dataset row and one ModelDeployment joined to TrainingTask:
#        SELECT id,file_path FROM datasets WHERE file_path IS NOT NULL LIMIT 1;
#        SELECT d.id,t.model_path FROM model_deployments d
#          JOIN training_tasks t ON t.id=d.task_id WHERE t.model_path IS NOT NULL LIMIT 1;
#   4. Convert the Dataset row to datasets/<dataset-id>/original/<basename(file_path)>
#      and the model_path to models/<basename(model_path)>. From the matching
#      minio/<stamp>/ snapshot, run `mc cp` for both keys into a temporary
#      directory and verify both downloads are non-empty. Record the two DB
#      primary keys and downloaded object sizes, then drop the temporary DB.
#
# Optional off-host copy hook (configure host/key separately; never add secrets here):
#   rsync -a --delete /home/opsadmin/backups/ backup-host:/srv/ml-platform-backups/

set -euo pipefail

BACKUP_ROOT="/home/opsadmin/backups"
MYSQL_BACKUP_DIR="$BACKUP_ROOT/mysql"
MINIO_BACKUP_DIR="$BACKUP_ROOT/minio"
LOG_FILE="$BACKUP_ROOT/backup.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SECRETS_FILE="$SCRIPT_DIR/.deploy_secrets"
STAMP="$(date '+%Y%m%d_%H%M%S')"
MYSQL_OUTPUT="$MYSQL_BACKUP_DIR/ml_platform_${STAMP}.sql.gz"
MINIO_OUTPUT="$MINIO_BACKUP_DIR/ml_platform_${STAMP}"
MYSQL_CONTAINER="${MYSQL_CONTAINER:-ml_platform_mysql}"
MYSQL_TEMP_CNF=""
MYSQL_CONTAINER_CNF="/tmp/ml-platform-backup-${STAMP}.cnf"
MC_CONFIG_DIR=""

mkdir -p "$MYSQL_BACKUP_DIR" "$MINIO_BACKUP_DIR"
touch "$LOG_FILE"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" >>"$LOG_FILE"
}

cleanup() {
  if [[ -n "$MYSQL_TEMP_CNF" ]]; then
    rm -f -- "$MYSQL_TEMP_CNF"
  fi
  if [[ -n "$MC_CONFIG_DIR" ]]; then
    rm -rf -- "$MC_CONFIG_DIR"
  fi
  docker exec "$MYSQL_CONTAINER" rm -f -- "$MYSQL_CONTAINER_CNF" >/dev/null 2>&1 || true
}

on_error() {
  local exit_code=$?
  log "ERROR backup failed (exit=${exit_code}, line=${BASH_LINENO[0]})"
  exit "$exit_code"
}

trap cleanup EXIT
trap on_error ERR

if [[ ! -r "$SECRETS_FILE" ]]; then
  log "ERROR secrets file is not readable: $SECRETS_FILE"
  exit 2
fi

set -a
# shellcheck disable=SC1090 # deployment secrets intentionally live beside this script.
source "$SECRETS_FILE"
set +a

: "${MYSQL_DATABASE:?MYSQL_DATABASE must be set in docker/.deploy_secrets}"
: "${S3_ACCESS_KEY:?S3_ACCESS_KEY must be set in docker/.deploy_secrets}"
: "${S3_SECRET_KEY:?S3_SECRET_KEY must be set in docker/.deploy_secrets}"
: "${S3_BUCKET:?S3_BUCKET must be set in docker/.deploy_secrets}"

# Prefer a least-privilege dump account; retain compatibility with the current
# deployment secrets until that server-side account is provisioned.
MYSQL_BACKUP_USER="${MYSQL_BACKUP_USER:-root}"
MYSQL_BACKUP_PASSWORD="${MYSQL_BACKUP_PASSWORD:-${MYSQL_ROOT_PASSWORD:-}}"
: "${MYSQL_BACKUP_PASSWORD:?MYSQL_BACKUP_PASSWORD or MYSQL_ROOT_PASSWORD must be set in docker/.deploy_secrets}"

MYSQL_TEMP_CNF="$(mktemp "$BACKUP_ROOT/.mysql-backup.XXXXXX.cnf")"
chmod 600 "$MYSQL_TEMP_CNF"
{
  printf '[client]\n'
  printf 'user=%s\n' "$MYSQL_BACKUP_USER"
  printf 'password=%s\n' "$MYSQL_BACKUP_PASSWORD"
} >"$MYSQL_TEMP_CNF"

docker cp "$MYSQL_TEMP_CNF" "$MYSQL_CONTAINER:$MYSQL_CONTAINER_CNF" >/dev/null
docker exec "$MYSQL_CONTAINER" chmod 600 "$MYSQL_CONTAINER_CNF"
docker exec "$MYSQL_CONTAINER" \
  mysqldump \
  "--defaults-extra-file=$MYSQL_CONTAINER_CNF" \
  --single-transaction \
  --quick \
  --routines \
  --events \
  "$MYSQL_DATABASE" \
  | gzip -c >"$MYSQL_OUTPUT"

gzip -t "$MYSQL_OUTPUT"
test -s "$MYSQL_OUTPUT"

MC_CONFIG_DIR="$(mktemp -d "$BACKUP_ROOT/.mc-config.XXXXXX")"
chmod 700 "$MC_CONFIG_DIR"
export MC_CONFIG_DIR
mc alias set backup http://127.0.0.1:9000 "$S3_ACCESS_KEY" "$S3_SECRET_KEY" >/dev/null
mkdir -p "$MINIO_OUTPUT"
mc mirror --overwrite "backup/$S3_BUCKET" "$MINIO_OUTPUT"

if ! find "$MINIO_OUTPUT" -type f -size +0c -print -quit | grep -q .; then
  log "ERROR MinIO mirror is empty: $MINIO_OUTPUT"
  exit 3
fi

find "$MINIO_OUTPUT" -type f -printf '%P\t%s\n' \
  | sort \
  | gzip -c >"$MINIO_OUTPUT/manifest.txt.gz"
gzip -t "$MINIO_OUTPUT/manifest.txt.gz"
test -s "$MINIO_OUTPUT/manifest.txt.gz"

find "$MYSQL_BACKUP_DIR" -type f -name '*.sql.gz' -mtime +14 -delete
while IFS= read -r -d '' old_backup; do
  find "$old_backup" -depth -delete
done < <(find "$MINIO_BACKUP_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +14 -print0)

log "OK mysql=$MYSQL_OUTPUT minio=$MINIO_OUTPUT"
