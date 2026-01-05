#!/bin/sh
set -e

# Wait for MariaDB to be ready
until mysql -h "$DB_HOST" -u "$DB_USERNAME" -p"$DB_PASSWORD" "$DB_DATABASE" -e "SELECT 1;" >/dev/null 2>&1; do
  echo "Waiting for MariaDB..."
  sleep 2
done

# Run migrations and seeders
php artisan migrate --force
php artisan db:seed --force


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
