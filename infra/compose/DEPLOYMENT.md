# MorphVox Production Deployment Guide

## Prerequisites

- A server with Docker and Docker Compose installed
- A domain name pointing to your server's IP address
- Ports 80 and 443 open on your firewall

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
./setup-ssl.sh your-domain.com your-email@example.com
```

This script will:
1. Start Nginx with HTTP-only config
2. Obtain SSL certificate from Let's Encrypt
3. Switch to HTTPS config
4. Start all services

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

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Internet                    │
                    └─────────────────┬───────────────────────┘
                                      │
                              ┌───────▼───────┐
                              │    Nginx      │
                              │  (SSL/Proxy)  │
                              │  :80 / :443   │
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
                              ┌───────┴───────────────────────┘
                              │
              ┌───────────────┼───────────────────────┐
              │               │                       │
      ┌───────▼───────┐ ┌─────▼─────┐         ┌───────▼───────┐
      │  PostgreSQL   │ │   Redis   │         │     MinIO     │
      │    :5432      │ │   :6379   │         │ (S3 Storage)  │
      └───────────────┘ └───────────┘         └───────────────┘
```

## URL Routing

| Path | Service | Description |
|------|---------|-------------|
| `/` | Next.js | Web frontend |
| `/api/*` | Laravel | REST API |
| `/ws` | Voice Engine | WebSocket for real-time voice |
| `/voice-api/*` | Voice Engine | HTTP API for voice processing |

## SSL Certificate Renewal

Certificates auto-renew via the certbot container. To manually renew:

```bash
docker compose -f docker-compose.prod.yml run --rm certbot renew
docker compose -f docker-compose.prod.yml exec nginx nginx -s reload
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
docker compose -f docker-compose.prod.yml run --rm certbot certificates
```

## Security Checklist

- [ ] Strong passwords in `.env`
- [ ] Firewall configured (only 80/443 exposed)
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
