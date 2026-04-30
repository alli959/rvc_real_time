#!/bin/bash

# =============================================================================
# MorphVox Platform - SSL Setup Script
# =============================================================================
# This script sets up SSL certificates using the HOST nginx and certbot.
#
# Architecture:
#   Host nginx (ports 80/443, SSL termination)
#     -> Docker nginx (port 9080:80, HTTP only, internal routing)
#
# SSL is handled by the host, NOT by Docker containers.
#
# Usage:
#   sudo ./setup-ssl.sh <domain> <email>
#
# Example:
#   sudo ./setup-ssl.sh morphvox.net admin@example.com
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Must run as root for certbot and nginx config
if [ "$EUID" -ne 0 ]; then
    error "Please run with sudo: sudo $0 <domain> <email>"
fi

# Validate arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sudo $0 <domain> <email>"
    echo "Example: sudo $0 morphvox.net admin@example.com"
    exit 1
fi

DOMAIN=$1
EMAIL=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="${SCRIPT_DIR}"
NGINX_SITES="/etc/nginx/sites-available"
NGINX_ENABLED="/etc/nginx/sites-enabled"
DOCKER_PROXY_PORT=9080

info "Setting up SSL for domain: ${DOMAIN}"
info "Email for Let's Encrypt notifications: ${EMAIL}"

# Check if .env exists
if [ ! -f "${COMPOSE_DIR}/.env" ]; then
    if [ -f "${COMPOSE_DIR}/.env.example" ]; then
        info "Creating .env from .env.example..."
        cp "${COMPOSE_DIR}/.env.example" "${COMPOSE_DIR}/.env"
        warn "Please edit ${COMPOSE_DIR}/.env with your configuration before continuing"
        warn "At minimum, set: DB_PASSWORD, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, APP_KEY"
        exit 1
    else
        error ".env file not found. Please create one first."
    fi
fi

# Add DOMAIN to .env if not present
if ! grep -q "^DOMAIN=" "${COMPOSE_DIR}/.env"; then
    echo "DOMAIN=${DOMAIN}" >> "${COMPOSE_DIR}/.env"
    info "Added DOMAIN=${DOMAIN} to .env"
else
    sed -i "s/^DOMAIN=.*/DOMAIN=${DOMAIN}/" "${COMPOSE_DIR}/.env"
    info "Updated DOMAIN in .env"
fi

# Step 1: Ensure certbot is installed on the host
if ! command -v certbot &> /dev/null; then
    info "Installing certbot..."
    apt-get update && apt-get install -y certbot python3-certbot-nginx
fi

# Step 2: Ensure host nginx is installed
if ! command -v nginx &> /dev/null; then
    error "Host nginx is not installed. Install with: apt-get install nginx"
fi

# Step 3: Create initial HTTP-only host nginx config (for ACME challenge)
info "Creating host nginx config for ${DOMAIN}..."
SITE_NAME=$(echo "${DOMAIN}" | sed 's/\./-/g')
cat > "${NGINX_SITES}/${SITE_NAME}" <<EOF
# ${DOMAIN} - proxies to Docker nginx on port ${DOCKER_PROXY_PORT}
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN};

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        proxy_pass http://127.0.0.1:${DOCKER_PROXY_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable the site
ln -sf "${NGINX_SITES}/${SITE_NAME}" "${NGINX_ENABLED}/${SITE_NAME}"
nginx -t && systemctl reload nginx

# Step 4: Start Docker services (HTTP only, no SSL needed in Docker)
info "Starting Docker services..."
cd "${COMPOSE_DIR}"
docker-compose -f docker-compose.prod.yml up -d

info "Waiting for services to start..."
sleep 5

# Step 5: Obtain SSL certificate using host certbot
info "Obtaining SSL certificate from Let's Encrypt..."
certbot certonly \
    --webroot \
    --webroot-path=/var/www/html \
    --email "${EMAIL}" \
    --agree-tos \
    --no-eff-email \
    -d "${DOMAIN}"

if [ $? -ne 0 ]; then
    error "Failed to obtain SSL certificate. Check that:"
    echo "  1. Your domain (${DOMAIN}) points to this server's IP"
    echo "  2. Port 80 is accessible from the internet"
    echo "  3. Host nginx is running"
    exit 1
fi

success "SSL certificate obtained successfully!"

# Step 6: Update host nginx config with SSL
info "Enabling HTTPS in host nginx config..."
cat > "${NGINX_SITES}/${SITE_NAME}" <<EOF
# ${DOMAIN} - proxies to Docker nginx on port ${DOCKER_PROXY_PORT}
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN};

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name ${DOMAIN};

    ssl_certificate /etc/letsencrypt/live/${DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${DOMAIN}/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    client_max_body_size 2G;

    location / {
        proxy_pass http://127.0.0.1:${DOCKER_PROXY_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_connect_timeout 120s;
        proxy_read_timeout 1800s;
        proxy_send_timeout 1800s;
        proxy_request_buffering off;
        proxy_buffering off;
    }
}
EOF

nginx -t && systemctl reload nginx

success "=============================================="
success "MorphVox Platform is now running with SSL!"
success "=============================================="
echo ""
echo "Architecture:"
echo "  Host nginx (:80/:443, SSL) -> Docker nginx (:${DOCKER_PROXY_PORT}, HTTP) -> services"
echo ""
echo "Access your application at: https://${DOMAIN}"
echo ""
echo "Services:"
echo "  - Web UI:       https://${DOMAIN}"
echo "  - API:          https://${DOMAIN}/api"
echo "  - WebSocket:    wss://${DOMAIN}/ws"
echo ""
echo "SSL certificates auto-renew via host certbot timer."
echo "Check timer: systemctl list-timers certbot.timer"
echo ""
echo "Useful commands:"
echo "  - View logs:    docker-compose -f docker-compose.prod.yml logs -f"
echo "  - Stop:         docker-compose -f docker-compose.prod.yml down"
echo "  - Restart:      docker-compose -f docker-compose.prod.yml restart"
echo ""
