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
