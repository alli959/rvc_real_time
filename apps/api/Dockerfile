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
    postgresql-dev \
    icu-dev \
    linux-headers \
    $PHPIZE_DEPS

# Install PHP extensions
RUN docker-php-ext-configure gd --with-freetype --with-jpeg \
    && docker-php-ext-install -j$(nproc) \
        pdo \
        pdo_pgsql \
        pgsql \
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

# Configure nginx
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Configure supervisor
COPY docker/supervisord.conf /etc/supervisord.conf

WORKDIR /var/www/html

# =============================================================================
# Development Stage
# =============================================================================

FROM base AS development

# Install development dependencies
RUN apk add --no-cache git

# Copy application code
COPY . .

# Install dependencies (including dev)
RUN composer install --no-interaction --no-progress

# Set permissions
RUN chown -R www-data:www-data /var/www/html \
    && chmod -R 755 /var/www/html/storage \
    && chmod -R 755 /var/www/html/bootstrap/cache

EXPOSE 80

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]

# =============================================================================
# Production Stage
# =============================================================================

FROM base AS production

# Copy application code
COPY --chown=www-data:www-data . .

# Install production dependencies only
RUN composer install --no-dev --no-interaction --no-progress --optimize-autoloader

# Cache config and routes
RUN php artisan config:cache \
    && php artisan route:cache \
    && php artisan view:cache

# Set permissions
RUN chown -R www-data:www-data /var/www/html \
    && chmod -R 755 /var/www/html/storage \
    && chmod -R 755 /var/www/html/bootstrap/cache

EXPOSE 80

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
