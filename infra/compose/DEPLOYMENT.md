# MorphVox Production Deployment Guide

## Prerequisites

- A server with Docker and Docker Compose installed
- Host nginx installed on the server
- A domain name pointing to your server's IP address
- Ports 80 and 443 open on your firewall

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Internet                    │
                    └─────────────────┬───────────────────────┘
                                      │
                              ┌───────▼───────┐
                              │  Host Nginx   │
                              │ (SSL termin.) │
                              │  :80 / :443   │
                              └───────┬───────┘
                                      │ proxy_pass
                                      │ 127.0.0.1:9080
                              ┌───────▼───────┐
                              │ Docker Nginx  │
                              │  (HTTP only)  │
                              │  9080:80      │
                              └───────┬───────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
      ┌───────▼───────┐       ┌───────▼───────┐       ┌───────▼───────┐
      │   Next.js     │       │  Laravel API  │       │ Voice Engine  │
      │   Frontend    │       │   Backend     │       │   (Python)    │
      │    :3000      │       │     :80       │       │ :8001 / :8765 │
      └───────────────┘       └───────┬───────┘       └───────┬───────┘
                                      │                       │
                              ┌───────┴───────────────────────┤
                              │                               │
              ┌───────────────┼───────────────────────┐       │
              │               │                       │       │
      ┌───────▼───────┐ ┌─────▼─────┐         ┌───────▼───────┐
      │   MariaDB     │ │   Redis   │         │     MinIO     │
      │    :3306      │ │   :6379   │         │  :9000/:9001  │
      └───────────────┘ └───────────┘         └───────────────┘
```

**Key points:**
- Host nginx (ports 80/443) handles SSL termination and certbot
- Docker nginx (port 9080:80) is HTTP-only, handles internal routing
- SSL configs live in `/etc/nginx/sites-available/` on the host
- SSL certs are managed by host certbot in `/etc/letsencrypt/`

## Quick Start

### 1. Configure Environment

```bash
cd infra/compose

# Copy and edit the environment file
cp .env.example .env
nano .env
```

**Required settings:**
```env
# Your domain
DOMAIN=morphvox.yourdomain.com
LETSENCRYPT_EMAIL=your-email@example.com

# Strong passwords (generate with: openssl rand -base64 24)
DB_PASSWORD=<strong-password>
MINIO_ACCESS_KEY=<strong-access-key>
MINIO_SECRET_KEY=<strong-secret-key>

# Laravel app key (generate with: echo "base64:$(openssl rand -base64 32)")
APP_KEY=base64:<your-key>

# GPU support (optional)
VOICE_ENGINE_DEVICE=cuda  # or 'cpu'
```

### 2. Run SSL Setup

```bash
sudo ./setup-ssl.sh morphvox.net your-email@example.com
```

This script will:
1. Install certbot on the host if needed
2. Create a host nginx config for the domain
3. Obtain SSL certificate from Let's Encrypt via host certbot
4. Enable HTTPS in the host nginx config
5. Start all Docker services

For the admin subdomain:
```bash
sudo ./setup-admin-ssl.sh your-email@example.com
```

### 3. Access Your Application

- **Web UI**: https://your-domain.com
- **API**: https://your-domain.com/api
- **WebSocket**: wss://your-domain.com/ws

## Manual Deployment

If you prefer manual control:

```bash
cd infra/compose

# Start services
docker compose -f docker-compose.prod.yml up -d

# View logs
docker compose -f docker-compose.prod.yml logs -f

# Stop services
docker compose -f docker-compose.prod.yml down
```

## URL Routing

| Path | Service | Description |
|------|---------|-------------|
| `/` | Next.js | Web frontend |
| `/api/*` | Laravel | REST API |
| `/ws` | Voice Engine | WebSocket for real-time voice |
| `/voice-api/*` | Voice Engine | HTTP API for voice processing |

## SSL Certificate Renewal

Certificates auto-renew via the host certbot timer. To check or manually renew:

```bash
# Check timer status
systemctl list-timers certbot.timer

# Manually renew all certificates
sudo certbot renew

# Reload host nginx after renewal
sudo systemctl reload nginx
```

## Troubleshooting

### Check service status
```bash
docker compose -f docker-compose.prod.yml ps
```

### View logs
```bash
# All services
docker compose -f docker-compose.prod.yml logs -f

# Specific service
docker compose -f docker-compose.prod.yml logs -f nginx
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml logs -f voice-engine
```

### Test SSL
```bash
curl -I https://your-domain.com
```

### Check certificate status
```bash
sudo certbot certificates
```

## Security Checklist

- [ ] Strong passwords in `.env`
- [ ] Firewall configured (only 80/443 exposed to internet)
- [ ] Docker nginx only on port 9080 (not exposed externally)
- [ ] SSH key-only authentication
- [ ] Regular backups of database volume
- [ ] Monitoring set up (optional)

## Backup & Restore

### Backup database
```bash
docker compose -f docker-compose.prod.yml exec db \
  pg_dump -U morphvox morphvox > backup.sql
```

### Restore database
```bash
cat backup.sql | docker compose -f docker-compose.prod.yml exec -T db \
  psql -U morphvox morphvox
```
