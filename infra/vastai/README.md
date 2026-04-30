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
