# MorphVox Cloud Hosting Design — Vast.ai + Cloudflare

**Date:** 2026-04-30
**Status:** Approved
**Author:** Alexander Guðmundsson + Copilot

## Problem Statement

MorphVox is currently hosted locally via `docker-compose.prod.yml`. The platform needs to be deployed to the cloud with:
- GPU availability (RTX 3090 24GB) for inference, training, and preprocessing
- Budget cap of $150/month (alerts at $80 and $100)
- Custom domain (morphvox.net) with HTTPS
- Automated deployment via GitHub Actions
- Resilience against instance loss (automated backup/restore)

## Solution Overview

Deploy the full Docker Compose stack on a **Vast.ai persistent pod** with an RTX 3090, using **Cloudflare Tunnel** for secure HTTPS access and **Cloudflare R2** for backup storage.

### Architecture

```
Internet → Cloudflare Edge (DNS + Tunnel) → Vast.ai Pod → Docker Compose Stack
```

**Why this approach:**
- Cheapest GPU hosting available ($60–100/mo for persistent RTX 3090)
- Existing `docker-compose.prod.yml` works with minimal modifications
- Cloudflare Tunnel provides free HTTPS, DDoS protection, and zero exposed ports
- R2 provides cheap, reliable backup storage (10GB free)

## Detailed Design

### 1. Vast.ai Instance Selection

**Requirements:**
- GPU: RTX 3090 (24GB VRAM) or equivalent
- RAM: ≥32GB
- Storage: ≥100GB SSD (persistent)
- Docker support: required (with NVIDIA Container Toolkit)
- NVIDIA driver: ≥535 (CUDA 12.2+)
- Network: ≥200 Mbps upload
- OS image: `nvidia/cuda:12.2.0-runtime-ubuntu22.04` (or any image with NVIDIA runtime)

**GPU runtime requirements:**
The Docker Compose stack uses `deploy.resources.reservations.devices` to claim the GPU. Vast.ai instances must have the NVIDIA Container Toolkit pre-installed (most do). The compose file specifies GPU reservations per-service:
- `voice-engine`: 1 GPU, 14GB VRAM
- `trainer`: 1 GPU, 14GB VRAM (time-shared with voice-engine)
- `preprocessor`: 1 GPU, 8GB VRAM (time-shared)

**Selection strategy:**
- Use Vast.ai API to find cheapest RTX 3090 instance meeting requirements
- Prefer instances with high reliability scores (>0.95)
- Store instance selection criteria in a config file for reproducibility

**Instance template (`infra/vastai/instance-template.json`):**
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

**Search command (vastai CLI):**
```bash
vastai search offers \
  --type bid \
  --gpu-name "RTX 3090" \
  --min-ram 32 \
  --min-disk 100 \
  --min-cuda 12.2 \
  --docker \
  --sort-by dph_total \
  --limit 5
```

### 2. Cloudflare Tunnel

**Setup:**
- `cloudflared` daemon runs as a Docker container alongside the app stack
- Tunnel routes traffic from morphvox.net → pod's internal Docker network
- Zero exposed ports on the pod (security benefit)

**Routing rules:**
| Hostname | Service | Internal Target |
|----------|---------|-----------------|
| morphvox.net | Web + API | http://nginx:80 |
| admin.morphvox.net | Admin panel | http://nginx:80 (behind Cloudflare Access) |

**Tunnel mode: Token-based (recommended for single-pod deployments)**

The tunnel is configured entirely via the Cloudflare Dashboard. No local config file is needed — the `TUNNEL_TOKEN` environment variable contains the encrypted tunnel configuration including ingress rules:

```yaml
# Dashboard ingress rules (configured via Cloudflare Zero Trust UI):
# morphvox.net → http://nginx:80
# admin.morphvox.net → http://nginx:80
# Catch-all → HTTP 404
```

**Admin access control (Cloudflare Access):**
- `admin.morphvox.net` is protected by a Cloudflare Access policy
- Authentication: email OTP or GitHub SSO (free for up to 50 users)
- Configured in Cloudflare Zero Trust Dashboard → Access → Applications
- No additional infrastructure needed (Cloudflare handles auth at the edge)

**DNS:**
- CNAME `morphvox.net` → `<tunnel-id>.cfargotunnel.com` (managed by cloudflared)
- CNAME `admin.morphvox.net` → same tunnel

### 3. Docker Compose Modifications

Add to `docker-compose.prod.yml`:

```yaml
services:
  # Cloudflare Tunnel
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

  # Backup & Cost Monitor (Alpine-based, POSIX sh only)
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
      R2_BUCKET: morphvox-backups
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

  # Restore container (runs once for disaster recovery, RW mounts)
  restore:
    image: alpine:3.19
    container_name: morphvox-restore
    profiles: ["restore"]  # Only started explicitly: docker compose --profile restore run restore
    volumes:
      - ../vastai/restore.sh:/scripts/restore.sh:ro
      - ../vastai/setup-rclone.sh:/scripts/setup-rclone.sh:ro
      - ../../storage:/storage                  # RW for model restore
      - minio_data:/minio-data                  # RW for MinIO restore
      - ../../infra/compose:/env-restore        # RW to write restored .env
    environment:
      DB_HOST: db
      DB_DATABASE: ${DB_DATABASE:-morphvox}
      DB_USERNAME: ${DB_USERNAME:-morphvox}
      DB_PASSWORD: ${DB_PASSWORD}
      R2_ACCOUNT_ID: ${R2_ACCOUNT_ID}
      R2_ACCESS_KEY: ${R2_ACCESS_KEY}
      R2_SECRET_KEY: ${R2_SECRET_KEY}
      R2_BUCKET: morphvox-backups
      SECRETS_ENCRYPTION_KEY: ${SECRETS_ENCRYPTION_KEY}
    entrypoint: /bin/sh -c "apk add --no-cache rclone mysql-client openssl bash && /scripts/restore.sh"
    networks:
      - morphvox
    depends_on:
      - db
```

**Key design decisions:**
- **Backup container** has `/storage:ro` — cannot corrupt production data
- **Restore container** is in a `restore` profile — only runs explicitly during disaster recovery with RW mounts
- **Scripts use `#!/usr/bin/env bash`** for `backup.sh` and `cost-monitor.sh` (need `pipefail`); bash installed via entrypoint `apk add`
- **Restore script uses `#!/bin/sh`** (POSIX — no pipefail needed, simpler flow)

**Cloud port changes (remove host port bindings):**
When deploying to Vast.ai, the following changes are made to existing services in `docker-compose.prod.yml`:
```yaml
# REMOVE these port mappings (traffic comes through Cloudflare Tunnel only):
# nginx:
#   ports:
#     - "9080:80"    ← REMOVE (was for host nginx proxy)
# db:
#   ports:
#     - "3307:3306"  ← REMOVE (was for local dev access)

# The nginx service keeps its internal port 80 — cloudflared connects to it
# via the Docker network (http://nginx:80). No host ports exposed.
```
This achieves "zero exposed ports" — all external traffic enters via Cloudflare Tunnel.

**Rclone setup script (`infra/vastai/setup-rclone.sh`):**
```bash
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

**Crontab file (`infra/vastai/crontab`):**
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

### 4. Backup Strategy (Cloudflare R2)

**What gets backed up:**

| Data | Frequency | Method | Size Estimate |
|------|-----------|--------|---------------|
| MariaDB dump | Every hour | mysqldump → gzip → R2 | ~10MB |
| Voice models (.pth) | Daily (3am full sync) | rclone copy → R2 | ~500MB total |
| MinIO object data | Daily (3am full sync) | rclone copy → R2 | ~1GB |
| Compose .env (encrypted) | Daily | openssl enc → R2 | <1KB |

**Note on MinIO:** In cloud deployment, MinIO's `minio_data` volume is backed up via the backup container's mount. Alternatively, MinIO can be replaced entirely by R2 (direct S3-compatible access) — this is a future optimization.

**Retention policy (implemented in backup.sh via `rclone delete --min-age`):**
- DB dumps: auto-deleted after 30 days (`rclone delete --min-age 30d`)
- Config backups: auto-deleted after 7 days
- Models/MinIO data: use `rclone copy` (additive — never deletes from R2, accumulates versions)
- Manual cleanup: if R2 grows too large, prune old model versions manually

**Backup script (`infra/vastai/backup.sh`):**
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

**Restore script (`infra/vastai/restore.sh`):**
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

### 5. Cost Monitoring & Alerting

**Mechanism:**
- A cron job (every 6 hours) queries Vast.ai API for current spending
- Sends webhook alerts (Discord/email) at configurable thresholds
- Auto-stops instance at hard cap

**Alert thresholds:**
- $80: Warning notification
- $100: Urgent notification
- $150: Auto-stop instance (hard cap)

**Cost monitor script (`infra/vastai/cost-monitor.sh`):**
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

**Important safety notes:**
- Uses `PUT .../stop/` (not `DELETE`) — stopping preserves the disk and all data
- Queries the specific instance by ID — never relies on array position
- Uses `set -uo pipefail` (not `-e`) so curl failures are caught gracefully
- Disk check uses `/storage` mount (the actual persistent data path)
- Manual restart required after hard cap stop (intentional friction)

### 6. CI/CD — GitHub Actions

**Workflow: `.github/workflows/deploy.yml`**

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

**Deploy user setup (in cloud-bootstrap.sh):**
```bash
# Create limited deploy user (not root)
useradd -m -s /bin/bash deploy
usermod -aG docker deploy
mkdir -p /home/deploy/.ssh
cp /root/.ssh/authorized_keys /home/deploy/.ssh/
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh

# Give deploy user access to workspace
chown -R deploy:deploy /workspace/rvc_real_time
```

**Secrets required in GitHub:**
- `VASTAI_HOST` — pod IP address
- `VASTAI_SSH_KEY` — SSH private key (for `deploy` user)
- `VASTAI_SSH_PORT` — SSH port (Vast.ai assigns random port)

**Note:** When pod IP/port changes (new instance), update secrets manually or via Vast.ai API in a setup step.

### 7. Disaster Recovery

**Scenario: Instance evicted or dies**

**Recovery procedure (automated via bootstrap script):**

1. Provision new Vast.ai instance (via API or CLI)
2. SSH into new instance
3. Clone repository: `git clone git@github.com:alli959/rvc_real_time.git`
4. Run bootstrap script:
   ```bash
   cd rvc_real_time && bash infra/vastai/cloud-bootstrap.sh
   ```
5. Bootstrap script:
   - Installs Docker, nvidia-container-toolkit, rclone, vastai CLI
   - Pulls `.env` and configs from R2
   - Restores latest DB dump from R2
   - Syncs models from R2
   - Starts Docker Compose stack (cloudflared starts as part of compose)
   - Runs health check
   - Prints new SSH host/port for GitHub Actions secret update

6. **Update GitHub Actions secrets** — After recovery, manually update `VASTAI_HOST` and `VASTAI_SSH_PORT` in GitHub repo settings (the bootstrap script outputs the values to copy).

**Recovery Time Objective (RTO):** ~10 minutes
**Recovery Point Objective (RPO):** 1 hour (last DB backup)

### 8. Cold Start UX

Since the pod may occasionally restart, the frontend needs to handle the voice engine being temporarily unavailable:

**Implementation:**
- API returns 503 with `{"status": "warming_up", "eta_seconds": 45}` when voice engine is loading
- Frontend shows a friendly loading indicator: "Voice engine is warming up... (~30s)"
- Auto-retry with exponential backoff until ready
- Cloudflare maintenance worker shows branded 503 if entire pod is down

### 8.1 Health Monitoring

Beyond cost monitoring, basic app health is checked:

**Health check endpoint:** `GET /api/health` (internal Docker network only — nginx:80, not exposed to host)

**Cron-based health ping (added to crontab):**
```cron
# App health check every 5 minutes
*/5 * * * * /scripts/health-check.sh >> /var/log/health.log 2>&1
```

**Health check script (`infra/vastai/health-check.sh`):**
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

**Disk space alert (appended to cost-monitor.sh):**
```bash
# Check disk usage on /storage mount (alert at 85%)
DISK_PCT=$(df /storage | tail -1 | awk '{print $5}' | tr -d '%')
if [ "$DISK_PCT" -gt 85 ]; then
  notify "💾 MorphVox disk usage: ${DISK_PCT}% — consider cleanup"
fi
```

**GPU health:** The voice engine already logs GPU errors. If health endpoint fails repeatedly, the CI/CD rollback (or manual intervention) addresses it.

### 9. Environment Variables (New)

Added to `.env.example`:
```env
# Cloudflare Tunnel
CLOUDFLARE_TUNNEL_TOKEN=<from-cloudflare-dashboard>

# Cloudflare R2 Backup
R2_ACCOUNT_ID=<cloudflare-account-id>
R2_ACCESS_KEY=<r2-access-key>
R2_SECRET_KEY=<r2-secret-key>
R2_BUCKET=morphvox-backups

# Cost Monitoring
VASTAI_API_KEY=<vast-ai-api-key>
VASTAI_INSTANCE_ID=<instance-id>
COST_ALERT_WEBHOOK=<discord-webhook-url>
COST_HARD_CAP=150
COST_WARN_THRESHOLD=80
COST_URGENT_THRESHOLD=100

# Backup Encryption
SECRETS_ENCRYPTION_KEY=<random-32-char-key-for-env-backup-encryption>

# Internal Service Token
INTERNAL_SERVICE_TOKEN=<random-secret>
```

### 10. File Structure (New Files)

```
infra/
├── vastai/
│   ├── cloud-bootstrap.sh       # Full instance bootstrap from scratch
│   ├── cost-monitor.sh          # Spending alerts + hard cap + disk alerts
│   ├── backup.sh                # R2 backup script (hourly DB + daily full)
│   ├── restore.sh               # R2 restore script (runs in 'restore' profile)
│   ├── health-check.sh          # App health ping + Discord alert
│   ├── setup-rclone.sh          # Generates rclone config from env vars
│   ├── crontab                  # Cron schedule for backup + cost + health
│   ├── instance-template.json   # Vast.ai instance requirements (GPU/RAM/disk)
│   └── README.md                # Cloud deployment guide
├── cloudflare/
│   ├── maintenance-worker.js    # (existing) 503 page when pod is down
│   └── wrangler.toml            # (existing)
└── compose/
    └── docker-compose.prod.yml  # (modified: + cloudflared, backup, restore profile)

.github/
└── workflows/
    └── deploy.yml               # CI/CD with rollback
```

## Security Considerations

- **No exposed ports** — Cloudflare Tunnel means the pod has no public IP exposure
- **Admin access** — `admin.morphvox.net` protected by Cloudflare Access (email OTP/GitHub SSO)
- **No wildcard routing** — Only explicit hostnames (`morphvox.net`, `admin.morphvox.net`) routed
- **Secrets management** — All secrets in `.env` (not committed); backed up **encrypted** (AES-256-CBC) to R2
- **Encryption key** — `SECRETS_ENCRYPTION_KEY` stored only in GitHub Secrets and the pod's `.env`; not backed up to R2 (chicken-and-egg is broken by keeping it in a password manager)
- **SSH access** — `deploy` user (not root), key-only auth, non-standard port (Vast.ai assigned)
- **Internal token** — Service-to-service auth already implemented
- **Cloudflare WAF** — Free tier provides basic bot/attack protection
- **Deploy user** — CI/CD uses a limited `deploy` user with Docker access, not root

## Cost Breakdown (Estimated)

| Item | Monthly Cost | Notes |
|------|-------------|-------|
| Vast.ai RTX 3090 pod (persistent) | $60–100 | Varies by availability; $0.30–0.45/hr |
| Cloudflare Tunnel | Free | |
| Cloudflare R2 (< 10GB) | Free | 10GB/mo included |
| Cloudflare Access (< 50 users) | Free | |
| Cloudflare DNS | Free | |
| GitHub Actions | Free | 2000 min/mo included |
| **Total** | **$60–100/mo** | |

**Pricing caveat:** Vast.ai pricing fluctuates with demand. The $60–100 estimate assumes winning bids at typical rates. During GPU shortages (new model releases, crypto booms), prices may spike. The cost monitor will alert if approaching cap. If sustained above $100, consider switching to a fixed-price RunPod pod ($0.44/hr = ~$316/mo) or scheduling off-hours shutdown.

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Instance eviction | Low | High | Automated R2 backup + bootstrap restore |
| GPU availability | Low | Medium | Multiple acceptable GPU types (3090, A5000) |
| Vast.ai outage | Very Low | High | Manual migration to RunPod (docker-compose portable) |
| Cost overrun | Medium | Low | Hard spending cap with auto-stop |
| Data loss | Low | Critical | Hourly backups to R2, tested restore procedure |

## Success Criteria

1. morphvox.net accessible via HTTPS from anywhere
2. Voice conversion completes in < 10s for typical audio
3. Monthly cost stays under $100 (alerting at $80)
4. Recovery from instance loss in < 15 minutes
5. Push to main auto-deploys in < 5 minutes
