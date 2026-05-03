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
apt-get install -y -qq ca-certificates curl jq rclone openssl gnupg

# Add Docker's official repository
if [ ! -f /etc/apt/sources.list.d/docker.list ]; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
  apt-get update -qq
fi
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin

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
