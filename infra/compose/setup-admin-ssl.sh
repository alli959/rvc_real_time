#!/bin/bash
# Setup SSL certificate for admin.morphvox.net subdomain

DOMAIN="admin.morphvox.net"
EMAIL="${1:-your-email@example.com}"

echo "=================================================="
echo "SSL Certificate Setup for Admin Subdomain"
echo "=================================================="
echo "Domain: $DOMAIN"
echo "Email: $EMAIL"
echo ""

# Check if running from correct directory
if [ ! -f "docker-compose.prod.yml" ]; then
    echo "Error: Must run from infra/compose directory"
    exit 1
fi

# Reload nginx to pick up the new HTTP config for ACME challenge
echo "Reloading nginx..."
docker-compose -f docker-compose.prod.yml exec nginx nginx -s reload

# Run certbot to get certificate for admin subdomain
echo ""
echo "Requesting SSL certificate for $DOMAIN..."
docker-compose -f docker-compose.prod.yml run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    --force-renewal \
    -d $DOMAIN

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Certificate obtained successfully!"
    echo ""
    echo "Enabling HTTPS configuration in nginx..."
    
    # Uncomment the HTTPS server block for admin subdomain
    cd ../nginx/conf.d
    sed -i '/# Uncomment this block AFTER obtaining SSL certificate/,/# }$/ s/^# //' default.conf
    cd -
    
    echo "Reloading nginx to use the new certificate..."
    docker-compose -f docker-compose.prod.yml exec nginx nginx -s reload
    
    echo ""
    echo "=================================================="
    echo "✅ Admin panel is now accessible at:"
    echo "   https://admin.morphvox.net"
    echo "=================================================="
else
    echo ""
    echo "❌ Certificate request failed."
    echo "Make sure:"
    echo "1. DNS A record for admin.morphvox.net points to your server IP"
    echo "2. Port 80 is accessible from the internet"
    echo "3. Docker containers are running"
fi
