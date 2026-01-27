# =============================================================================
# VoiceForge API - Laravel Dockerfile
# =============================================================================

FROM php:8.3-fpm-alpine AS base

# Install system dependencies
RUN apk add --no-cache \
    nginx \
    supervisor \
    curl \
    libpng-dev \
    libjpeg-turbo-dev \
    freetype-dev \
    libzip-dev \
    oniguruma-dev \
    mariadb-connector-c-dev \
    icu-dev \
    linux-headers \
    $PHPIZE_DEPS

# Install PHP extensions
RUN docker-php-ext-configure gd --with-freetype --with-jpeg \
    && docker-php-ext-install -j$(nproc) \
        pdo \
        pdo_mysql \
        mbstring \
        zip \
        bcmath \
        opcache \
        gd \
        intl \
        pcntl

# Install Redis extension
RUN pecl install redis && docker-php-ext-enable redis

# Install Composer
COPY --from=composer:2 /usr/bin/composer /usr/bin/composer

# Configure PHP
RUN mv "$PHP_INI_DIR/php.ini-production" "$PHP_INI_DIR/php.ini"
COPY docker/php.ini /usr/local/etc/php/conf.d/custom.ini

# Configure PHP-FPM with increased workers for production load
COPY docker/www.conf /usr/local/etc/php-fpm.d/www.conf

# Configure nginx
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Configure supervisor
RUN mkdir -p /var/log/supervisor
COPY docker/supervisord.conf /etc/supervisord.conf

WORKDIR /var/www/html

# =============================================================================
# Development Stage
# =============================================================================

FROM base AS development

# Install development dependencies and docker-cli for log access
RUN apk add --no-cache git mariadb-client docker-cli

# Copy application code
COPY . .

# Remove .env file if it exists - we'll use environment variables
RUN rm -f .env

# Install dependencies (including dev)
RUN composer install --no-interaction --no-progress

# Set permissions
RUN chown -R www-data:www-data /var/www/html \
    && chmod -R 755 /var/www/html/storage \
    && chmod -R 755 /var/www/html/bootstrap/cache

EXPOSE 80

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# =============================================================================
# Production Stage
# =============================================================================

FROM base AS production

# Install mariadb-client for entrypoint database check
# Install docker-cli for log access via socket
RUN apk add --no-cache mariadb-client docker-cli

# Copy application code
COPY --chown=www-data:www-data . .

# Remove .env file - production uses environment variables from docker-compose
RUN rm -f .env

# Clear bootstrap cache to prevent stale dev dependency references
RUN rm -f bootstrap/cache/packages.php bootstrap/cache/services.php

# Install production dependencies only
RUN composer install --no-dev --no-interaction --no-progress --optimize-autoloader

# Don't cache config in build - we need runtime environment variables
# RUN php artisan config:cache \
#     && php artisan route:cache \
#     && php artisan view:cache

# Set permissions
RUN chown -R www-data:www-data /var/www/html \
    && chmod -R 755 /var/www/html/storage \
    && chmod -R 755 /var/www/html/bootstrap/cache

EXPOSE 80

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
