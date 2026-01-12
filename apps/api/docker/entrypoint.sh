#!/bin/sh
set -e

# Override .env settings with container environment (for Docker compatibility)
# This ensures Docker Compose environment variables take precedence over .env file
export DB_HOST="${DB_HOST:-db}"
export DB_PORT="${DB_PORT:-3306}"
export REDIS_HOST="${REDIS_HOST:-redis}"

# Wait for MariaDB to be ready (disable SSL for connection check)
until mysql -h "$DB_HOST" -u "$DB_USERNAME" -p"$DB_PASSWORD" "$DB_DATABASE" --skip-ssl -e "SELECT 1;" >/dev/null 2>&1; do
  echo "Waiting for MariaDB..."
  sleep 2
done

# Run migrations and seeders
php artisan migrate --force
php artisan db:seed --force

# Create storage symlink (remove if exists as file)
rm -f public/storage 2>/dev/null || true
php artisan storage:link 2>/dev/null || true


# Sync voice models from storage to database
STORAGE_TYPE=${VOICE_MODELS_STORAGE:-local}
LOCAL_PATH=${VOICE_MODELS_PATH:-/var/www/html/storage/models}

if [ "$STORAGE_TYPE" = "local" ]; then
  php artisan voice-models:sync --storage=local --path="$LOCAL_PATH" --prune
else
  php artisan voice-models:sync --storage=s3 --prune
fi

# Start supervisor
exec /usr/bin/supervisord -c /etc/supervisord.conf
