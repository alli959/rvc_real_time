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
