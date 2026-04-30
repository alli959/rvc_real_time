# Cloud Hosting Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy MorphVox to a Vast.ai persistent pod with Cloudflare Tunnel, automated backups to R2, cost monitoring, and CI/CD via GitHub Actions.

**Architecture:** Full Docker Compose stack on Vast.ai RTX 3090 pod. Cloudflare Tunnel provides HTTPS with zero exposed ports. Alpine-based sidecar container handles backups (hourly DB + daily full sync to R2) and cost/health monitoring via cron. GitHub Actions deploys on push to main with automatic rollback.

**Tech Stack:** Docker Compose, Cloudflare Tunnel (cloudflared), Cloudflare R2 (via rclone), Vast.ai API, Alpine Linux (cron + shell scripts), GitHub Actions (appleboy/ssh-action)

**Spec:** `docs/superpowers/specs/2026-04-30-cloud-hosting-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `infra/vastai/setup-rclone.sh` | Generates rclone config from env vars |
| `infra/vastai/backup.sh` | Hourly DB backup + daily full sync to R2 |
| `infra/vastai/restore.sh` | Restores DB, models, MinIO, config from R2 |
| `infra/vastai/cost-monitor.sh` | Queries Vast.ai API, alerts on thresholds, stops at cap |
| `infra/vastai/health-check.sh` | Pings app health endpoint, alerts on failure |
| `infra/vastai/cloud-bootstrap.sh` | Full instance setup from scratch (disaster recovery) |
| `infra/vastai/crontab` | Cron schedule for all periodic jobs |
| `infra/vastai/instance-template.json` | Vast.ai instance requirements |
| `infra/vastai/README.md` | Cloud deployment guide |
| `.github/workflows/deploy.yml` | CI/CD: deploy on push to main with rollback |

### Modified Files

| File | Change |
|------|--------|
| `infra/compose/docker-compose.prod.yml` | Add cloudflared, backup, restore services; remove host ports |
| `infra/compose/.env.example` | Add cloud-specific env vars |

---

## Chunk 1: Infrastructure Scripts

### Task 1: Create setup-rclone.sh

**Files:**
- Create: `infra/vastai/setup-rclone.sh`

- [ ] **Step 1: Create the script**

```sh
#!/bin/sh
# Generate rclone config from environment variables
mkdir -p /root/.config/rclone
cat > /root/.config/rclone/rclone.conf <<EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY}
secret_access_key = ${R2_SECRET_KEY}
endpoint = https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
EOF
```

- [ ] **Step 2: Make executable**

Run: `chmod +x infra/vastai/setup-rclone.sh`

- [ ] **Step 3: Commit**

```bash
git add infra/vastai/setup-rclone.sh
git commit -m "infra(vastai): add rclone setup script for R2"
```

---

### Task 2: Create backup.sh

**Files:**
- Create: `infra/vastai/backup.sh`

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
# Runs via cron inside backup container
# Usage: backup.sh          → hourly DB backup
#        backup.sh --full   → full sync (models + minio + config)
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d-%H%M)
DUMP_FILE="/tmp/db-${TIMESTAMP}.sql.gz"

# Always: DB dump (validate before upload)
echo "Starting DB backup..."
mysqldump -h "$DB_HOST" -u "$DB_USERNAME" -p"$DB_PASSWORD" "$DB_DATABASE" \
  | gzip > "$DUMP_FILE"

# Validate: gzip file must be > 100 bytes (empty dump is ~20 bytes compressed)
DUMP_SIZE=$(stat -c%s "$DUMP_FILE" 2>/dev/null || stat -f%z "$DUMP_FILE")
if [ "$DUMP_SIZE" -lt 100 ]; then
  echo "ERROR: DB dump too small (${DUMP_SIZE} bytes), likely failed"
  rm -f "$DUMP_FILE"
  exit 1
fi

rclone copy "$DUMP_FILE" "r2:${R2_BUCKET}/db/"
echo "  DB backed up: db-${TIMESTAMP}.sql.gz (${DUMP_SIZE} bytes)"

# Full sync mode (daily)
if [ "${1:-}" = "--full" ]; then
  # Models: use copy (additive, never deletes remote files)
  rclone copy /storage/models "r2:${R2_BUCKET}/models/" --checksum
  # MinIO data: additive copy
  rclone copy /minio-data "r2:${R2_BUCKET}/minio/" --checksum

  # Encrypt .env before uploading (uses compose .env mounted at /env-file)
  if [ -f /env-file ]; then
    openssl enc -aes-256-cbc -pbkdf2 -in /env-file \
      -out "/tmp/.env-${TIMESTAMP}.enc" -pass "env:SECRETS_ENCRYPTION_KEY"
    rclone copy "/tmp/.env-${TIMESTAMP}.enc" "r2:${R2_BUCKET}/configs/"
    rm -f "/tmp/.env-${TIMESTAMP}.enc"
  fi

  echo "  Full sync complete"
fi

# Retention: cleanup old DB dumps from R2
rclone delete "r2:${R2_BUCKET}/db/" --min-age 30d 2>/dev/null || true
# Cleanup old config backups (keep 7 days)
rclone delete "r2:${R2_BUCKET}/configs/" --min-age 7d 2>/dev/null || true

# Cleanup local temp files
find /tmp -name "db-*.sql.gz" -mtime +1 -delete 2>/dev/null || true
```

- [ ] **Step 2: Make executable**

Run: `chmod +x infra/vastai/backup.sh`

- [ ] **Step 3: Commit**

```bash
git add infra/vastai/backup.sh
git commit -m "infra(vastai): add R2 backup script with validation and retention"
```

---

### Task 3: Create restore.sh

**Files:**
- Create: `infra/vastai/restore.sh`

- [ ] **Step 1: Create the script**

```sh
#!/bin/sh
# Restore from R2 backups — runs in the 'restore' profile container (RW mounts)
# Usage: docker compose --profile restore run restore
set -e

echo "=== Restoring from Cloudflare R2 ==="

# Setup rclone if not already configured
/scripts/setup-rclone.sh 2>/dev/null || true

# Restore latest DB dump
echo "Restoring database..."
LATEST_DB=$(rclone lsf "r2:${R2_BUCKET}/db/" --files-only | sort | tail -1)
if [ -n "$LATEST_DB" ]; then
  rclone copy "r2:${R2_BUCKET}/db/${LATEST_DB}" /tmp/
  gunzip -c "/tmp/${LATEST_DB}" | mysql -h "$DB_HOST" -u "$DB_USERNAME" -p"$DB_PASSWORD" "$DB_DATABASE"
  echo "  DB restored from ${LATEST_DB}"
else
  echo "  WARNING: No DB backup found, starting fresh"
fi

# Restore models (additive copy, never deletes local files)
echo "Restoring models..."
rclone copy "r2:${R2_BUCKET}/models/" /storage/models/ --checksum
echo "  Models restored"

# Restore MinIO data
echo "Restoring MinIO data..."
rclone copy "r2:${R2_BUCKET}/minio/" /minio-data/ --checksum
echo "  MinIO data restored"

# Restore and decrypt latest config → compose .env
echo "Restoring config..."
LATEST_ENV=$(rclone lsf "r2:${R2_BUCKET}/configs/" --files-only | sort | tail -1)
if [ -n "$LATEST_ENV" ]; then
  rclone copy "r2:${R2_BUCKET}/configs/${LATEST_ENV}" /tmp/
  openssl enc -aes-256-cbc -pbkdf2 -d -in "/tmp/${LATEST_ENV}" \
    -out /env-restore/.env -pass "env:SECRETS_ENCRYPTION_KEY"
  echo "  Config restored and decrypted to infra/compose/.env"
else
  echo "  WARNING: No config backup found — create .env manually"
fi

echo "=== Restore complete ==="
```

- [ ] **Step 2: Make executable**

Run: `chmod +x infra/vastai/restore.sh`

- [ ] **Step 3: Commit**

```bash
git add infra/vastai/restore.sh
git commit -m "infra(vastai): add R2 restore script for disaster recovery"
```

---

### Task 4: Create cost-monitor.sh

**Files:**
- Create: `infra/vastai/cost-monitor.sh`

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/env bash
set -uo pipefail
# Note: not using -e so we can handle curl failures gracefully

notify() {
  local message="$1"
  if [ -n "${COST_ALERT_WEBHOOK:-}" ]; then
    curl -s -H "Content-Type: application/json" \
      -d "{\"content\": \"$message\"}" \
      "$COST_ALERT_WEBHOOK"
  fi
  echo "[$(date)] $message"
}

# Query our specific instance by ID (not [0] which could be wrong)
if ! RESPONSE=$(curl -sf -H "Authorization: Bearer $VASTAI_API_KEY" \
  "https://console.vast.ai/api/v0/instances/${VASTAI_INSTANCE_ID}/"); then
  notify "⚠️ MorphVox: Failed to query Vast.ai API for instance ${VASTAI_INSTANCE_ID}"
  exit 1
fi

if [ -z "$RESPONSE" ]; then
  notify "⚠️ MorphVox: Empty response from Vast.ai API"
  exit 1
fi

SPENDING=$(echo "$RESPONSE" | jq '.total_cost // 0')

if [ -z "$SPENDING" ] || [ "$SPENDING" = "null" ]; then
  notify "⚠️ MorphVox: Could not parse spending from API response"
  exit 1
fi

if (( $(echo "$SPENDING > $COST_HARD_CAP" | bc -l) )); then
  # STOP (not delete!) — preserves disk and data
  curl -sf -H "Authorization: Bearer $VASTAI_API_KEY" \
    -X PUT "https://console.vast.ai/api/v0/instances/${VASTAI_INSTANCE_ID}/stop/"
  notify "🚨 MorphVox STOPPED (not deleted): \$${SPENDING} exceeds hard cap (\$$COST_HARD_CAP). Data preserved. Manual restart required."
elif (( $(echo "$SPENDING > $COST_URGENT_THRESHOLD" | bc -l) )); then
  notify "⚠️ MorphVox spending URGENT: \$${SPENDING}/mo (hard cap: \$$COST_HARD_CAP)"
elif (( $(echo "$SPENDING > $COST_WARN_THRESHOLD" | bc -l) )); then
  notify "📊 MorphVox spending notice: \$${SPENDING}/mo"
fi

# Disk space check (alert at 85%)
DISK_PCT=$(df /storage 2>/dev/null | tail -1 | awk '{print $5}' | tr -d '%')
if [ -n "$DISK_PCT" ] && [ "$DISK_PCT" -gt 85 ]; then
  notify "💾 MorphVox disk usage: ${DISK_PCT}% — consider cleanup"
fi
```

- [ ] **Step 2: Make executable**

Run: `chmod +x infra/vastai/cost-monitor.sh`

- [ ] **Step 3: Commit**

```bash
git add infra/vastai/cost-monitor.sh
git commit -m "infra(vastai): add cost monitor with safe stop and disk alerts"
```

---

### Task 5: Create health-check.sh

**Files:**
- Create: `infra/vastai/health-check.sh`

- [ ] **Step 1: Create the script**

```sh
#!/bin/sh
# Simple health check — alerts if app is down
if ! curl -sf http://nginx:80/api/health > /dev/null 2>&1; then
  if [ -n "${COST_ALERT_WEBHOOK:-}" ]; then
    curl -s -H "Content-Type: application/json" \
      -d '{"content": "🔴 MorphVox health check FAILED — app may be down"}' \
      "$COST_ALERT_WEBHOOK"
  fi
  echo "[$(date)] Health check failed"
fi
```

- [ ] **Step 2: Make executable**

Run: `chmod +x infra/vastai/health-check.sh`

- [ ] **Step 3: Commit**

```bash
git add infra/vastai/health-check.sh
git commit -m "infra(vastai): add health check script with Discord alerting"
```

---

### Task 6: Create crontab and instance-template.json

**Files:**
- Create: `infra/vastai/crontab`
- Create: `infra/vastai/instance-template.json`

- [ ] **Step 1: Create crontab**

```cron
# Backup DB every hour
0 * * * * /scripts/backup.sh >> /var/log/backup.log 2>&1
# Cost + disk check every 6 hours
0 */6 * * * /scripts/cost-monitor.sh >> /var/log/cost.log 2>&1
# Daily full storage sync (models + minio + encrypted config)
0 3 * * * /scripts/backup.sh --full >> /var/log/backup.log 2>&1
# App health check every 5 minutes
*/5 * * * * /scripts/health-check.sh >> /var/log/health.log 2>&1
```

- [ ] **Step 2: Create instance-template.json**

```json
{
  "gpu_name": "RTX 3090",
  "num_gpus": 1,
  "min_ram": 32,
  "min_disk": 100,
  "cuda_version": "12.2",
  "driver_version": "535",
  "docker": true,
  "reliability": 0.95,
  "sort_by": "dph_total",
  "type": "bid"
}
```

- [ ] **Step 3: Commit**

```bash
git add infra/vastai/crontab infra/vastai/instance-template.json
git commit -m "infra(vastai): add crontab and instance template"
```

---

## Chunk 2: Bootstrap, Docker Compose, and CI/CD

### Task 7: Create cloud-bootstrap.sh

**Files:**
- Create: `infra/vastai/cloud-bootstrap.sh`

- [ ] **Step 1: Create the bootstrap script**

```bash
#!/usr/bin/env bash
# Full instance bootstrap for Vast.ai pod — run after cloning the repo
# Usage: cd rvc_real_time && bash infra/vastai/cloud-bootstrap.sh
set -euo pipefail

echo "=== MorphVox Cloud Bootstrap ==="
echo "Instance: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"

# 1. Install dependencies
echo ""
echo "--- Installing dependencies ---"
apt-get update -qq
apt-get install -y -qq docker.io docker-compose-plugin curl jq rclone openssl

# Ensure NVIDIA Container Toolkit is available
if ! dpkg -l | grep -q nvidia-container-toolkit; then
  echo "Installing NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  apt-get install -y -qq nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

# 2. Create deploy user
echo ""
echo "--- Setting up deploy user ---"
if ! id deploy &>/dev/null; then
  useradd -m -s /bin/bash deploy
  usermod -aG docker deploy
  mkdir -p /home/deploy/.ssh
  cp /root/.ssh/authorized_keys /home/deploy/.ssh/ 2>/dev/null || true
  chown -R deploy:deploy /home/deploy/.ssh
  chmod 700 /home/deploy/.ssh
fi
chown -R deploy:deploy /workspace/rvc_real_time

# 3. Setup rclone for R2 access
echo ""
echo "--- Configuring rclone ---"
if [ -z "${R2_ACCESS_KEY:-}" ]; then
  echo "ERROR: R2_ACCESS_KEY not set. Source your .env first or export manually:"
  echo "  export R2_ACCESS_KEY=... R2_SECRET_KEY=... R2_ACCOUNT_ID=... R2_BUCKET=..."
  exit 1
fi
bash infra/vastai/setup-rclone.sh

# 4. Restore from R2 backups
echo ""
echo "--- Restoring from R2 backups ---"
cd infra/compose

if [ ! -f .env ]; then
  echo "No .env found. Attempting to restore from R2..."
  export SECRETS_ENCRYPTION_KEY="${SECRETS_ENCRYPTION_KEY:?SECRETS_ENCRYPTION_KEY required for config restore}"
  LATEST_ENV=$(rclone lsf "r2:${R2_BUCKET}/configs/" --files-only 2>/dev/null | sort | tail -1)
  if [ -n "$LATEST_ENV" ]; then
    rclone copy "r2:${R2_BUCKET}/configs/${LATEST_ENV}" /tmp/
    openssl enc -aes-256-cbc -pbkdf2 -d -in "/tmp/${LATEST_ENV}" \
      -out .env -pass "env:SECRETS_ENCRYPTION_KEY"
    echo "  .env restored from R2"
  else
    echo "  No config backup found. Copy .env.example to .env and configure manually."
    cp .env.example .env
    echo "  EDIT .env NOW before continuing!"
    exit 1
  fi
fi

# 5. Start the stack
echo ""
echo "--- Starting Docker Compose stack ---"
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml build --parallel
docker compose -f docker-compose.prod.yml up -d

# 6. Wait for DB to be ready, then restore data
echo "Waiting for database..."
for i in $(seq 1 30); do
  if docker compose -f docker-compose.prod.yml exec -T db mysqladmin ping -h localhost --silent 2>/dev/null; then
    echo "  Database ready"
    break
  fi
  sleep 2
done

echo "Running data restore..."
docker compose -f docker-compose.prod.yml --profile restore run --rm restore

# 7. Health check
echo ""
echo "--- Running health check ---"
sleep 15
for i in 1 2 3 4 5; do
  if docker compose -f docker-compose.prod.yml exec -T nginx curl -sf http://localhost:80/api/health > /dev/null 2>&1; then
    echo "✅ MorphVox is healthy!"
    break
  fi
  echo "  Attempt $i: not ready yet..."
  sleep 10
done

# 8. Print connection info
echo ""
echo "=== Bootstrap Complete ==="
echo ""
echo "Update GitHub Actions secrets with:"
echo "  VASTAI_HOST: $(curl -sf ifconfig.me)"
echo "  VASTAI_SSH_PORT: $(grep -oP '(?<=Port )\d+' /etc/ssh/sshd_config 2>/dev/null || echo '22')"
echo ""
echo "Verify tunnel is working: https://morphvox.net"
```

- [ ] **Step 2: Make executable**

Run: `chmod +x infra/vastai/cloud-bootstrap.sh`

- [ ] **Step 3: Commit**

```bash
git add infra/vastai/cloud-bootstrap.sh
git commit -m "infra(vastai): add cloud bootstrap script for disaster recovery"
```

---

### Task 8: Modify docker-compose.prod.yml

**Files:**
- Modify: `infra/compose/docker-compose.prod.yml`

Changes:
1. Remove `ports: - "9080:80"` from nginx service
2. Remove `ports: - "3307:3306"` from db service
3. Add `cloudflared` service
4. Add `backup` service
5. Add `restore` service (in `restore` profile)
6. Add `minio_data` volume reference for backup container

- [ ] **Step 1: Remove nginx host port binding**

In `infra/compose/docker-compose.prod.yml`, find the nginx service and remove:
```yaml
    ports:
      - "9080:80"
```

- [ ] **Step 2: Remove db host port binding**

In `infra/compose/docker-compose.prod.yml`, find the db service and remove:
```yaml
    ports:
      - "3307:3306"
```

- [ ] **Step 3: Add cloudflared service**

Add after the nginx service block:
```yaml
  # ===========================================================================
  # Cloudflare Tunnel (HTTPS ingress — zero exposed ports)
  # ===========================================================================

  cloudflared:
    image: cloudflare/cloudflared:latest
    container_name: morphvox-cloudflared
    restart: unless-stopped
    command: tunnel run
    environment:
      TUNNEL_TOKEN: ${CLOUDFLARE_TUNNEL_TOKEN}
    networks:
      - morphvox
    depends_on:
      - nginx
```

- [ ] **Step 4: Add backup service**

Add after the cloudflared service:
```yaml
  # ===========================================================================
  # Backup & Monitoring Sidecar
  # ===========================================================================

  backup:
    image: alpine:3.19
    container_name: morphvox-backup
    restart: unless-stopped
    volumes:
      - ../vastai/backup.sh:/scripts/backup.sh:ro
      - ../vastai/cost-monitor.sh:/scripts/cost-monitor.sh:ro
      - ../vastai/health-check.sh:/scripts/health-check.sh:ro
      - ../vastai/setup-rclone.sh:/scripts/setup-rclone.sh:ro
      - ../vastai/crontab:/etc/crontabs/root:ro
      - mariadb_data:/var/lib/mysql:ro
      - minio_data:/minio-data:ro
      - ../../storage:/storage:ro
      - ../../infra/compose/.env:/env-file:ro
    environment:
      DB_HOST: db
      DB_DATABASE: ${DB_DATABASE:-morphvox}
      DB_USERNAME: ${DB_USERNAME:-morphvox}
      DB_PASSWORD: ${DB_PASSWORD}
      R2_ACCOUNT_ID: ${R2_ACCOUNT_ID}
      R2_ACCESS_KEY: ${R2_ACCESS_KEY}
      R2_SECRET_KEY: ${R2_SECRET_KEY}
      R2_BUCKET: ${R2_BUCKET:-morphvox-backups}
      VASTAI_API_KEY: ${VASTAI_API_KEY}
      VASTAI_INSTANCE_ID: ${VASTAI_INSTANCE_ID}
      COST_HARD_CAP: ${COST_HARD_CAP:-150}
      COST_WARN_THRESHOLD: ${COST_WARN_THRESHOLD:-80}
      COST_URGENT_THRESHOLD: ${COST_URGENT_THRESHOLD:-100}
      COST_ALERT_WEBHOOK: ${COST_ALERT_WEBHOOK}
      SECRETS_ENCRYPTION_KEY: ${SECRETS_ENCRYPTION_KEY}
    entrypoint: /bin/sh -c "apk add --no-cache rclone curl jq mysql-client bc openssl bash && /scripts/setup-rclone.sh && crond -f -l 2"
    networks:
      - morphvox
    depends_on:
      db:
        condition: service_healthy
```

- [ ] **Step 5: Add restore service**

Add after the backup service:
```yaml
  # ===========================================================================
  # Restore (disaster recovery — only runs with --profile restore)
  # ===========================================================================

  restore:
    image: alpine:3.19
    container_name: morphvox-restore
    profiles: ["restore"]
    volumes:
      - ../vastai/restore.sh:/scripts/restore.sh:ro
      - ../vastai/setup-rclone.sh:/scripts/setup-rclone.sh:ro
      - ../../storage:/storage
      - minio_data:/minio-data
      - ../../infra/compose:/env-restore
    environment:
      DB_HOST: db
      DB_DATABASE: ${DB_DATABASE:-morphvox}
      DB_USERNAME: ${DB_USERNAME:-morphvox}
      DB_PASSWORD: ${DB_PASSWORD}
      R2_ACCOUNT_ID: ${R2_ACCOUNT_ID}
      R2_ACCESS_KEY: ${R2_ACCESS_KEY}
      R2_SECRET_KEY: ${R2_SECRET_KEY}
      R2_BUCKET: ${R2_BUCKET:-morphvox-backups}
      SECRETS_ENCRYPTION_KEY: ${SECRETS_ENCRYPTION_KEY}
    entrypoint: /bin/sh -c "apk add --no-cache rclone mysql-client openssl && /bin/sh /scripts/restore.sh"
    networks:
      - morphvox
    depends_on:
      db:
        condition: service_healthy
```

- [ ] **Step 6: Rename postgres_data volume to mariadb_data**

In the `volumes:` section at the bottom, rename:
```yaml
volumes:
  mariadb_data:    # was: postgres_data
  redis_data:
  minio_data:
  certbot_www:
  certbot_conf:
```

Also update the db service volume mount from `postgres_data:/var/lib/mysql` to `mariadb_data:/var/lib/mysql`.

**⚠️ Data migration note:** If running on an existing local instance with data in the `postgres_data` volume, rename it first:
```bash
docker volume create mariadb_data
docker run --rm -v postgres_data:/from -v mariadb_data:/to alpine sh -c "cp -a /from/. /to/"
```

- [ ] **Step 7: Verify compose file is valid**

Run: `cd infra/compose && docker compose -f docker-compose.prod.yml config --quiet`
Expected: No output (valid YAML)

- [ ] **Step 8: Commit**

```bash
git add infra/compose/docker-compose.prod.yml
git commit -m "infra(compose): add cloudflared, backup, restore; remove host ports"
```

---

### Task 9: Update .env.example

**Files:**
- Modify: `infra/compose/.env.example`

- [ ] **Step 1: Append cloud hosting variables**

Add to the end of `infra/compose/.env.example`:
```env

# -----------------------------------------------------------------------------
# Cloudflare Tunnel (cloud deployment)
# -----------------------------------------------------------------------------
CLOUDFLARE_TUNNEL_TOKEN=

# -----------------------------------------------------------------------------
# Cloudflare R2 Backup
# -----------------------------------------------------------------------------
R2_ACCOUNT_ID=
R2_ACCESS_KEY=
R2_SECRET_KEY=
R2_BUCKET=morphvox-backups

# -----------------------------------------------------------------------------
# Cost Monitoring (Vast.ai)
# -----------------------------------------------------------------------------
VASTAI_API_KEY=
VASTAI_INSTANCE_ID=
COST_ALERT_WEBHOOK=
COST_HARD_CAP=150
COST_WARN_THRESHOLD=80
COST_URGENT_THRESHOLD=100

# -----------------------------------------------------------------------------
# Backup Encryption
# -----------------------------------------------------------------------------
SECRETS_ENCRYPTION_KEY=
```

- [ ] **Step 2: Commit**

```bash
git add infra/compose/.env.example
git commit -m "infra(compose): add cloud hosting env vars to .env.example"
```

---

### Task 10: Create GitHub Actions deploy workflow

**Files:**
- Create: `.github/workflows/deploy.yml`

- [ ] **Step 1: Create the workflow**

```yaml
name: Deploy to Vast.ai

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Deploy via SSH
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.VASTAI_HOST }}
          username: deploy
          key: ${{ secrets.VASTAI_SSH_KEY }}
          port: ${{ secrets.VASTAI_SSH_PORT }}
          script: |
            set -e
            cd /workspace/rvc_real_time

            # Save current commit for rollback
            PREV_COMMIT=$(git rev-parse HEAD)
            echo "Previous commit: $PREV_COMMIT"

            # Pull and build
            git pull origin main
            NEW_COMMIT=$(git rev-parse HEAD)
            cd infra/compose
            docker compose -f docker-compose.prod.yml build --parallel
            docker compose -f docker-compose.prod.yml up -d

            # Health check with retry (3 attempts, 10s apart)
            HEALTHY=false
            for i in 1 2 3; do
              sleep 10
              if docker compose -f docker-compose.prod.yml exec -T nginx curl -sf http://localhost:80/api/health > /dev/null 2>&1; then
                HEALTHY=true
                break
              fi
              echo "Health check attempt $i failed, retrying..."
            done

            if [ "$HEALTHY" = "true" ]; then
              echo "✅ Deploy successful ($(git rev-parse --short HEAD))"
            else
              echo "❌ Health check failed — rolling back to $PREV_COMMIT"
              cd /workspace/rvc_real_time
              git reset --hard "$PREV_COMMIT"
              cd infra/compose
              docker compose -f docker-compose.prod.yml build --parallel
              docker compose -f docker-compose.prod.yml up -d
              sleep 15
              if docker compose -f docker-compose.prod.yml exec -T nginx curl -sf http://localhost:80/api/health > /dev/null 2>&1; then
                echo "⚠️ Rollback to $PREV_COMMIT successful"
              else
                echo "🚨 CRITICAL: Rollback also failed, manual intervention needed"
              fi
              exit 1
            fi
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/deploy.yml
git commit -m "ci: add deploy workflow with rollback for Vast.ai"
```

---

### Task 11: Create README.md for vastai directory

**Files:**
- Create: `infra/vastai/README.md`

- [ ] **Step 1: Create the README**

````markdown
# MorphVox Cloud Deployment (Vast.ai)

## Prerequisites

- **Vast.ai account** with API key and SSH key configured
- **Cloudflare account** with:
  - `morphvox.net` domain on Cloudflare DNS
  - Zero Trust tunnel created (get token from Dashboard → Zero Trust → Networks → Tunnels)
  - R2 bucket `morphvox-backups` created (Storage → R2)
  - Access policy on `admin.morphvox.net` (Zero Trust → Access → Applications)
- **Discord webhook URL** for alerts (Server Settings → Integrations → Webhooks)
- **GitHub repo secrets** configured (see CI/CD section below)

## First-Time Setup

### 1. Provision Vast.ai Instance

```bash
# Find suitable instances
vastai search offers --type bid --gpu-name "RTX 3090" --min-ram 32 \
  --min-disk 100 --min-cuda 12.2 --docker --sort-by dph_total --limit 5

# Create instance (replace OFFER_ID with cheapest suitable offer)
vastai create instance OFFER_ID --image nvidia/cuda:12.2.0-runtime-ubuntu22.04 \
  --disk 100 --ssh
```

### 2. Bootstrap the Instance

```bash
# SSH into the new instance (Vast.ai shows the SSH command)
ssh -p PORT root@HOST

# Clone and bootstrap
git clone git@github.com:alli959/rvc_real_time.git /workspace/rvc_real_time
cd /workspace/rvc_real_time

# Export required env vars for bootstrap (or create infra/compose/.env first)
export R2_ACCESS_KEY=... R2_SECRET_KEY=... R2_ACCOUNT_ID=... R2_BUCKET=morphvox-backups
export SECRETS_ENCRYPTION_KEY=...

bash infra/vastai/cloud-bootstrap.sh
```

### 3. Configure GitHub Secrets

After bootstrap completes, it prints connection info. Add to GitHub repo Settings → Secrets:
- `VASTAI_HOST` — pod IP address
- `VASTAI_SSH_PORT` — SSH port
- `VASTAI_SSH_KEY` — SSH private key for `deploy` user

## Routine Operations

### View Logs

```bash
# Inside the pod:
docker compose -f infra/compose/docker-compose.prod.yml logs -f --tail=100

# Backup/monitor logs:
docker exec morphvox-backup cat /var/log/backup.log
docker exec morphvox-backup cat /var/log/cost.log
docker exec morphvox-backup cat /var/log/health.log
```

### Manual Backup

```bash
docker exec morphvox-backup /scripts/backup.sh --full
```

### Manual Restore

```bash
cd infra/compose
docker compose -f docker-compose.prod.yml --profile restore run --rm restore
```

## Disaster Recovery

**Estimated recovery time: ~10 minutes**

1. Provision new Vast.ai instance (see First-Time Setup §1)
2. Run bootstrap script (see First-Time Setup §2)
3. Update GitHub secrets with new host/port
4. Verify at https://morphvox.net

## Cost Monitoring

| Alert | Threshold | Action |
|-------|-----------|--------|
| 📊 Notice | $80/mo | Informational — no action needed |
| ⚠️ Urgent | $100/mo | Consider reducing usage or switching plans |
| 🚨 Hard cap | $150/mo | Instance auto-stopped. Restart manually after review |

**To restart after hard cap stop:**
```bash
vastai start instance $VASTAI_INSTANCE_ID
```

## CI/CD

Push to `main` triggers automatic deployment via GitHub Actions. The workflow:
1. SSHs into the pod as `deploy` user
2. Pulls latest code and rebuilds containers
3. Runs health check (3 retries)
4. On failure: rolls back to previous commit and rebuilds

Failed deploys exit non-zero — check GitHub Actions logs for details.
````

- [ ] **Step 2: Commit**

```bash
git add infra/vastai/README.md
git commit -m "docs(vastai): add cloud deployment guide"
```

---

### Task 12: Final validation

- [ ] **Step 1: Verify all scripts are executable**

Run: `ls -la infra/vastai/*.sh`
Expected: All `.sh` files have `-rwxr-xr-x` permissions

- [ ] **Step 2: Verify compose config is valid**

Run: `cd infra/compose && docker compose -f docker-compose.prod.yml config --quiet`
Expected: No errors (note: will warn about missing env vars — that's fine)

- [ ] **Step 3: Verify GitHub Actions workflow syntax**

Run: `cat .github/workflows/deploy.yml | python3 -c "import sys,yaml; yaml.safe_load(sys.stdin)"`
Expected: No output (valid YAML)

- [ ] **Step 4: Final commit (if any fixes needed)**

```bash
git add -A
git status  # Should be clean
```

- [ ] **Step 5: Push to main**

Note: This is initial infrastructure setup — we push directly to main. Future feature work follows the normal branch/PR workflow.

```bash
git push origin main
```
