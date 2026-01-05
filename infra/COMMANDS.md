# MorphVox Deployment Commands Cheatsheet

## Quick Reference

```bash
cd /home/<User>/rvc_real_time/infra/compose
```

---

## 🚀 Basic Operations

### Start all services
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Stop all services
```bash
docker-compose -f docker-compose.prod.yml down
```

### Restart all services
```bash
docker-compose -f docker-compose.prod.yml restart
```

### View running services
```bash
docker-compose -f docker-compose.prod.yml ps
```

---

## 📋 Logs

### View all logs (live)
```bash
docker-compose -f docker-compose.prod.yml logs -f
```

### View specific service logs
```bash
docker-compose -f docker-compose.prod.yml logs -f nginx
docker-compose -f docker-compose.prod.yml logs -f api
docker-compose -f docker-compose.prod.yml logs -f web
docker-compose -f docker-compose.prod.yml logs -f voice-engine
docker-compose -f docker-compose.prod.yml logs -f db
```

### View last N lines
```bash
docker-compose -f docker-compose.prod.yml logs --tail 100 api
```

---

## 🔄 After Code Changes

### Rebuild and restart a specific service
```bash
# API changes
docker-compose -f docker-compose.prod.yml build api --no-cache
docker-compose -f docker-compose.prod.yml up -d api

# Web frontend changes
docker-compose -f docker-compose.prod.yml build web --no-cache
docker-compose -f docker-compose.prod.yml up -d web

# Voice engine changes
docker-compose -f docker-compose.prod.yml build voice-engine --no-cache
docker-compose -f docker-compose.prod.yml up -d voice-engine
```

### Rebuild ALL services
```bash
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up -d
```

### Quick restart (no rebuild, just restart container)
```bash
docker-compose -f docker-compose.prod.yml restart api
docker-compose -f docker-compose.prod.yml restart web
docker-compose -f docker-compose.prod.yml restart nginx
```

---

## 🔧 Nginx (Reverse Proxy)

### Reload nginx config (no downtime)
```bash
docker exec morphvox-nginx nginx -s reload
```

### Test nginx config
```bash
docker exec morphvox-nginx nginx -t
```

### View nginx config
```bash
cat ../nginx/conf.d/default.conf
```

---

## 🔐 SSL Certificates

### Check certificate status
```bash
docker run --rm -v compose_certbot_conf:/etc/letsencrypt certbot/certbot certificates
```

### Manually renew certificate
```bash
docker run --rm -v compose_certbot_www:/var/www/certbot -v compose_certbot_conf:/etc/letsencrypt --network compose_morphvox certbot/certbot renew
docker exec morphvox-nginx nginx -s reload
```

### Force renew certificate
```bash
docker run --rm -v compose_certbot_www:/var/www/certbot -v compose_certbot_conf:/etc/letsencrypt --network compose_morphvox certbot/certbot renew --force-renewal
docker exec morphvox-nginx nginx -s reload
```

---

## 🗄️ Database

### Access MariaDB/MySQL shell
```bash
docker exec -it morphvox-db mysql -u morphvox -p morphvox
```

### Backup database
```bash
docker exec morphvox-db sh -c 'exec mysqldump -u morphvox -p"$DB_PASSWORD" morphvox' > backup_$(date +%Y%m%d_%H%M%S).sql
```

### Restore database
```bash
cat backup.sql | docker exec -i morphvox-db mysql -u morphvox -p"$DB_PASSWORD" morphvox
```

### Run Laravel migrations
```bash
docker exec morphvox-api php artisan migrate
```

### Run Laravel migrations (fresh, DESTROYS DATA)
```bash
docker exec morphvox-api php artisan migrate:fresh --seed
```

---

## 🧹 Cleanup

### Remove stopped containers
```bash
docker-compose -f docker-compose.prod.yml down --remove-orphans
```

### Remove unused images
```bash
docker image prune -f
```

### Remove ALL unused Docker resources (careful!)
```bash
docker system prune -f
```

### Remove unused volumes (DESTROYS DATA)
```bash
docker volume prune -f
```

### Full cleanup (DESTROYS ALL DATA)
```bash
docker-compose -f docker-compose.prod.yml down -v
docker system prune -af
```

---

## 🔍 Debugging

### Shell into a container
```bash
docker exec -it morphvox-api sh
docker exec -it morphvox-web sh
docker exec -it morphvox-nginx sh
docker exec -it morphvox-voice-engine bash
```

### Check container resource usage
```bash
docker stats
```

### Check container health
```bash
docker inspect --format='{{.State.Health.Status}}' morphvox-db
docker inspect --format='{{.State.Health.Status}}' morphvox-redis
docker inspect --format='{{.State.Health.Status}}' morphvox-minio
```

### View container details
```bash
docker inspect morphvox-api
```

---

## 🌐 Network

### Test internal connectivity
```bash
# From nginx to api
docker exec morphvox-nginx ping -c 3 api

# From api to database
docker exec morphvox-api ping -c 3 db
```

### List networks
```bash
docker network ls
```

---

## 📦 Git Workflow (Deploy Updates)

### Pull latest code and redeploy
```bash
cd /home/<User>/rvc_real_time
git pull origin main

cd infra/compose
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up -d
```

### Quick deploy (if no Dockerfile changes)
```bash
git pull origin main
docker-compose -f docker-compose.prod.yml restart api web voice-engine
```

---

## 🔥 Emergency Commands

### Force stop everything
```bash
docker-compose -f docker-compose.prod.yml kill
```

### View what's using port 80/443
```bash
sudo lsof -i :80
sudo lsof -i :443
```

### Kill process on port (if needed)
```bash
sudo fuser -k 80/tcp
sudo fuser -k 443/tcp
```

### Restart Docker daemon
```bash
sudo systemctl restart docker
```

---

## 📊 Monitoring

### Watch logs in real-time (all services)
```bash
docker-compose -f docker-compose.prod.yml logs -f --tail 50
```

### Check disk space
```bash
df -h
docker system df
```

### Check memory usage
```bash
free -h
docker stats --no-stream
```

---

## 🔗 URLs

- **Website**: https://morphvox.net
- **API**: https://morphvox.net/api
- **Health Check**: https://morphvox.net/health

---

## 📝 Common Scenarios

### Scenario: Updated frontend code
```bash
cd /home/<User>/rvc_real_time
git pull
cd infra/compose
docker-compose -f docker-compose.prod.yml build web --no-cache
docker-compose -f docker-compose.prod.yml up -d web
```

### Scenario: Updated API code
```bash
cd /home/<User>/rvc_real_time
git pull
cd infra/compose
docker-compose -f docker-compose.prod.yml build api --no-cache
docker-compose -f docker-compose.prod.yml up -d api
docker exec morphvox-api php artisan migrate
docker exec morphvox-api php artisan config:cache
docker exec morphvox-api php artisan route:cache
```

### Scenario: Changed nginx config
```bash
# Edit the config
nano ../nginx/conf.d/default.conf

# Test it
docker exec morphvox-nginx nginx -t

# Reload (no downtime)
docker exec morphvox-nginx nginx -s reload
```

### Scenario: Service won't start
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs api --tail 50

# Try rebuilding
docker-compose -f docker-compose.prod.yml build api --no-cache
docker-compose -f docker-compose.prod.yml up -d api

# Check if dependencies are healthy
docker-compose -f docker-compose.prod.yml ps
```

### Scenario: Full restart after server reboot
```bash
cd /home/<User>/rvc_real_time/infra/compose
docker-compose -f docker-compose.prod.yml up -d
```
