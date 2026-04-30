#!/bin/bash
# =============================================================================
# Setup SSL certificate for admin.morphvox.net subdomain
# =============================================================================
# Uses HOST nginx and certbot (not Docker containers).
#
# Architecture:
#   Host nginx (ports 80/443, SSL termination)
#     -> Docker nginx (port 9080:80, HTTP only, internal routing)
#
# Usage:
#   sudo ./setup-admin-ssl.sh <email>
# =============================================================================

set -e

DOMAIN="admin.morphvox.net"
EMAIL="${1:-your-email@example.com}"
NGINX_SITES="/etc/nginx/sites-available"
NGINX_ENABLED="/etc/nginx/sites-enabled"
DOCKER_PROXY_PORT=9080
SITE_NAME="admin-morphvox"

echo "=================================================="
echo "SSL Certificate Setup for Admin Subdomain"
echo "=================================================="
echo "Domain: $DOMAIN"
echo "Email: $EMAIL"
echo ""

# Must run as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: Please run with sudo: sudo $0 <email>"
    exit 1
fi

# Ensure certbot is installed
if ! command -v certbot &> /dev/null; then
    echo "Installing certbot..."
    apt-get update && apt-get install -y certbot python3-certbot-nginx
fi

# Create initial HTTP-only config for ACME challenge
echo "Creating host nginx config for $DOMAIN..."
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

ln -sf "${NGINX_SITES}/${SITE_NAME}" "${NGINX_ENABLED}/${SITE_NAME}"
nginx -t && systemctl reload nginx

# Request certificate using host certbot
echo ""
echo "Requesting SSL certificate for $DOMAIN..."
certbot certonly \
    --webroot \
    --webroot-path=/var/www/html \
    --email "$EMAIL" \
    --agree-tos \
    --no-eff-email \
    --force-renewal \
    -d "$DOMAIN"

if [ $? -eq 0 ]; then
    echo ""
    echo "Certificate obtained successfully!"
    
    # Update config with SSL
    echo "Enabling HTTPS in host nginx config..."
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
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Forwarded-Host \$host;
        proxy_connect_timeout 60s;
        proxy_read_timeout 300s;
    }
}
EOF

    nginx -t && systemctl reload nginx
    
    echo ""
    echo "=================================================="
    echo "Admin panel is now accessible at:"
    echo "   https://admin.morphvox.net"
    echo "=================================================="
else
    echo ""
    echo "Certificate request failed."
    echo "Make sure:"
    echo "1. DNS A record for admin.morphvox.net points to your server IP"
    echo "2. Port 80 is accessible from the internet"
    echo "3. Host nginx is running"
fi
