# Admin Panel Setup Guide

## Access the Admin Panel at admin.morphvox.net

### Prerequisites
1. **DNS Configuration**: Create an A record for `admin.morphvox.net` pointing to your server's IP address
2. **Port Forwarding**: Ensure ports 80 and 443 are forwarded to your server (already done for main domain)

### Setup Steps

#### 1. Add DNS Record
In your domain registrar or DNS provider:
- **Type**: A
- **Name**: admin
- **Value**: Your server's IP address
- **TTL**: 300 (or default)

#### 2. Get SSL Certificate
Once DNS is propagated (can take 5-60 minutes):

```bash
cd /home/alexanderg/rvc_real_time/infra/compose
./setup-admin-ssl.sh your-email@example.com
```

This will:
- Request an SSL certificate from Let's Encrypt for admin.morphvox.net
- Configure nginx to use the certificate
- Reload nginx

#### 3. Access Admin Panel
Visit: **https://admin.morphvox.net**

Default admin login:
- Email: admin@example.com
- Password: (the one you set up)

### Manual SSL Setup (Alternative)

If the script doesn't work, run certbot manually:

```bash
cd /home/alexanderg/rvc_real_time/infra/compose

# Reload nginx first
docker-compose -f docker-compose.prod.yml exec nginx nginx -s reload

# Request certificate
docker-compose -f docker-compose.prod.yml run --rm certbot certonly \
  --webroot \
  --webroot-path=/var/www/certbot \
  --email your-email@example.com \
  --agree-tos \
  --no-eff-email \
  -d admin.morphvox.net

# Reload nginx again
docker-compose -f docker-compose.prod.yml exec nginx nginx -s reload
```

### Temporary Access (Without SSL)

For testing before SSL is set up, you can temporarily access via:
`http://admin.morphvox.net` (will redirect to HTTPS once SSL is configured)

Or modify your `/etc/hosts` file to test locally:
```
127.0.0.1 admin.morphvox.net
```

Then access: `http://localhost` with Host header set to admin.morphvox.net

### Troubleshooting

**DNS not propagating?**
Check with: `nslookup admin.morphvox.net` or `dig admin.morphvox.net`

**Certificate request failing?**
- Verify DNS points to your server
- Check ports 80/443 are open: `nc -zv your-ip 80`
- View certbot logs: `docker-compose -f docker-compose.prod.yml logs certbot`

**403/404 errors?**
- Check nginx logs: `docker-compose -f docker-compose.prod.yml logs nginx`
- Verify API container is running: `docker-compose -f docker-compose.prod.yml ps`
- Check Laravel logs in API container: `/var/www/html/storage/logs/`

### Configuration Files Updated
- `/infra/nginx/conf.d/default.conf` - Added admin subdomain server blocks
- `/apps/api/routes/web.php` - Already configured for admin subdomain in production

### Architecture
- **Main domain** (morphvox.net): Next.js frontend + API endpoints
- **Admin subdomain** (admin.morphvox.net): Laravel Blade admin panel
- Both route through the same nginx reverse proxy
- Admin panel uses Laravel's auth middleware with 'admin' role requirement
