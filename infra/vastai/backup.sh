#!/usr/bin/env bash
# Runs via cron inside backup container
# Usage: backup.sh          → hourly DB backup
#        backup.sh --full   → full sync (models + minio + config)
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d-%H%M)
DUMP_FILE="/tmp/db-${TIMESTAMP}.sql.gz"

# Always: DB dump (validate before upload)
echo "Starting DB backup..."
mysqldump -h "$DB_HOST" -u "$DB_USERNAME" -p"$DB_PASSWORD" "$DB_DATABASE" \
  | gzip > "$DUMP_FILE"

# Validate: gzip file must be > 100 bytes (empty dump is ~20 bytes compressed)
DUMP_SIZE=$(stat -c%s "$DUMP_FILE" 2>/dev/null || stat -f%z "$DUMP_FILE")
if [ "$DUMP_SIZE" -lt 100 ]; then
  echo "ERROR: DB dump too small (${DUMP_SIZE} bytes), likely failed"
  rm -f "$DUMP_FILE"
  exit 1
fi

rclone copy "$DUMP_FILE" "r2:${R2_BUCKET}/db/"
echo "  DB backed up: db-${TIMESTAMP}.sql.gz (${DUMP_SIZE} bytes)"

# Full sync mode (daily)
if [ "${1:-}" = "--full" ]; then
  # Models: use copy (additive, never deletes remote files)
  rclone copy /storage/models "r2:${R2_BUCKET}/models/" --checksum
  # MinIO data: additive copy
  rclone copy /minio-data "r2:${R2_BUCKET}/minio/" --checksum

  # Encrypt .env before uploading (uses compose .env mounted at /env-file)
  if [ -f /env-file ]; then
    openssl enc -aes-256-cbc -pbkdf2 -in /env-file \
      -out "/tmp/.env-${TIMESTAMP}.enc" -pass "env:SECRETS_ENCRYPTION_KEY"
    rclone copy "/tmp/.env-${TIMESTAMP}.enc" "r2:${R2_BUCKET}/configs/"
    rm -f "/tmp/.env-${TIMESTAMP}.enc"
  fi

  echo "  Full sync complete"
fi

# Retention: cleanup old DB dumps from R2
rclone delete "r2:${R2_BUCKET}/db/" --min-age 30d 2>/dev/null || true
# Cleanup old config backups (keep 7 days)
rclone delete "r2:${R2_BUCKET}/configs/" --min-age 7d 2>/dev/null || true

# Cleanup local temp files
find /tmp -name "db-*.sql.gz" -mtime +1 -delete 2>/dev/null || true
