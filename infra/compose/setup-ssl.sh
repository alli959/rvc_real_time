#!/bin/bash

# =============================================================================
# MorphVox Platform - SSL Setup Script
# =============================================================================
# This script automates the process of obtaining SSL certificates from Let's
# Encrypt using Certbot with the webroot challenge.
#
# Usage:
#   ./setup-ssl.sh <domain> <email>
#
# Example:
#   ./setup-ssl.sh morphvox.example.com admin@example.com
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

# Validate arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <domain> <email>"
    echo "Example: $0 morphvox.example.com admin@example.com"
    exit 1
fi

DOMAIN=$1
EMAIL=$2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="${SCRIPT_DIR}"
NGINX_DIR="${SCRIPT_DIR}/../nginx"

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

# Step 1: Create initial nginx config (HTTP only, for certificate challenge)
info "Creating initial Nginx configuration..."
mkdir -p "${NGINX_DIR}/conf.d"
sed "s/YOUR_DOMAIN/${DOMAIN}/g" "${NGINX_DIR}/conf.d/morphvox-init.conf.template" > "${NGINX_DIR}/conf.d/default.conf"

# Step 2: Create required directories
info "Creating directories..."
docker volume create --name=morphvox_certbot_www || true
docker volume create --name=morphvox_certbot_conf || true

# Step 3: Start services with HTTP-only config
info "Starting services (HTTP only)..."
cd "${COMPOSE_DIR}"
docker-compose -f docker-compose.prod.yml up -d nginx

# Wait for nginx to be ready
info "Waiting for Nginx to start..."
sleep 5

# Step 4: Obtain SSL certificate
info "Obtaining SSL certificate from Let's Encrypt..."
docker-compose -f docker-compose.prod.yml run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email "${EMAIL}" \
    --agree-tos \
    --no-eff-email \
    -d "${DOMAIN}"

if [ $? -ne 0 ]; then
    error "Failed to obtain SSL certificate. Check that:"
    echo "  1. Your domain (${DOMAIN}) points to this server's IP"
    echo "  2. Port 80 is accessible from the internet"
    echo "  3. No firewall is blocking incoming connections"
    exit 1
fi

success "SSL certificate obtained successfully!"

# Step 5: Switch to full HTTPS config
info "Switching to HTTPS configuration..."
sed "s/YOUR_DOMAIN/${DOMAIN}/g" "${NGINX_DIR}/conf.d/morphvox.conf.template" > "${NGINX_DIR}/conf.d/default.conf"

# Step 6: Reload nginx with new config
info "Reloading Nginx with SSL configuration..."
docker-compose -f docker-compose.prod.yml exec nginx nginx -s reload

# Step 7: Start all services
info "Starting all services..."
docker-compose -f docker-compose.prod.yml up -d

success "=============================================="
success "MorphVox Platform is now running with SSL!"
success "=============================================="
echo ""
echo "Access your application at: https://${DOMAIN}"
echo ""
echo "Services:"
echo "  - Web UI:       https://${DOMAIN}"
echo "  - API:          https://${DOMAIN}/api"
echo "  - WebSocket:    wss://${DOMAIN}/ws"
echo ""
echo "SSL certificates will auto-renew via the certbot container."
echo ""
echo "Useful commands:"
echo "  - View logs:    docker-compose -f docker-compose.prod.yml logs -f"
echo "  - Stop:         docker-compose -f docker-compose.prod.yml down"
echo "  - Restart:      docker-compose -f docker-compose.prod.yml restart"
echo ""
