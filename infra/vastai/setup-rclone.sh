#!/bin/sh
# Generate rclone config from environment variables
mkdir -p /root/.config/rclone
chmod 700 /root/.config/rclone
cat > /root/.config/rclone/rclone.conf <<EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY}
secret_access_key = ${R2_SECRET_KEY}
endpoint = https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
EOF
chmod 600 /root/.config/rclone/rclone.conf
