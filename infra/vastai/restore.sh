#!/bin/sh
# Restore from R2 backups — runs in the 'restore' profile container (RW mounts)
# Usage: docker compose --profile restore run restore
set -e

echo "=== Restoring from Cloudflare R2 ==="

# Setup rclone if not already configured
/scripts/setup-rclone.sh 2>/dev/null || true

# Restore latest DB dump
echo "Restoring database..."
LATEST_DB=$(rclone lsf "r2:${R2_BUCKET}/db/" --files-only | sort | tail -1)
if [ -n "$LATEST_DB" ]; then
  rclone copy "r2:${R2_BUCKET}/db/${LATEST_DB}" /tmp/
  gunzip -c "/tmp/${LATEST_DB}" | mysql -h "$DB_HOST" -u "$DB_USERNAME" -p"$DB_PASSWORD" "$DB_DATABASE"
  echo "  DB restored from ${LATEST_DB}"
else
  echo "  WARNING: No DB backup found, starting fresh"
fi

# Restore models (additive copy, never deletes local files)
echo "Restoring models..."
rclone copy "r2:${R2_BUCKET}/models/" /storage/models/ --checksum
echo "  Models restored"

# Restore MinIO data
echo "Restoring MinIO data..."
rclone copy "r2:${R2_BUCKET}/minio/" /minio-data/ --checksum
echo "  MinIO data restored"

# Restore and decrypt latest config → compose .env
echo "Restoring config..."
LATEST_ENV=$(rclone lsf "r2:${R2_BUCKET}/configs/" --files-only | sort | tail -1)
if [ -n "$LATEST_ENV" ]; then
  rclone copy "r2:${R2_BUCKET}/configs/${LATEST_ENV}" /tmp/
  openssl enc -aes-256-cbc -pbkdf2 -d -in "/tmp/${LATEST_ENV}" \
    -out /env-restore/.env -pass "env:SECRETS_ENCRYPTION_KEY"
  echo "  Config restored and decrypted to infra/compose/.env"
else
  echo "  WARNING: No config backup found — create .env manually"
fi

echo "=== Restore complete ==="
