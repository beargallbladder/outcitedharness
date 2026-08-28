#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-spark}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="$(mktemp)"
ENV_FILE="$(mktemp)"
trap 'rm -f "$ARCHIVE" "$ENV_FILE"' EXIT

TOKEN="${HARNESS_GCI_TOKEN:-}"
if [[ -z "$TOKEN" && -f "$ROOT/.env" ]]; then
  TOKEN="$(awk -F= '$1 == "HARNESS_GCI_TOKEN" {sub(/^[^=]*=/, ""); print; exit}' "$ROOT/.env")"
fi
if [[ -z "$TOKEN" ]]; then
  TOKEN="$(openssl rand -hex 32)"
  printf '\nHARNESS_GCI_TOKEN=%s\n' "$TOKEN" >> "$ROOT/.env"
fi

printf '%s\n' \
  "HARNESS_GCI_TOKEN=$TOKEN" \
  "HARNESS_GCI_DB=/data/harness-gci/code-intel.sqlite" \
  "HARNESS_GCI_HOST=100.81.201.24" \
  "HARNESS_GCI_PORT=8810" > "$ENV_FILE"
chmod 600 "$ENV_FILE"

COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  -C "$ROOT" \
  -czf "$ARCHIVE" \
  harness config deploy pyproject.toml README.md

ssh "$TARGET" "mkdir -p /home/samkim/harness-gci/releases/$RELEASE"
scp -q "$ARCHIVE" "$TARGET:/tmp/harness-gci-$RELEASE.tar.gz"
scp -q "$ENV_FILE" "$TARGET:/tmp/harness-gci.env"
ssh "$TARGET" \
  "tar -xzf /tmp/harness-gci-$RELEASE.tar.gz -C /home/samkim/harness-gci/releases/$RELEASE && \
   rm /tmp/harness-gci-$RELEASE.tar.gz && \
   python3 -m venv /home/samkim/harness-gci/venv && \
   /home/samkim/harness-gci/venv/bin/pip install -q /home/samkim/harness-gci/releases/$RELEASE && \
   ln -sfn /home/samkim/harness-gci/releases/$RELEASE /home/samkim/harness-gci/app && \
   sudo install -o root -g root -m 0600 /tmp/harness-gci.env /etc/harness-gci.env && \
   rm /tmp/harness-gci.env && \
   sudo install -o root -g root -m 0644 /home/samkim/harness-gci/app/deploy/harness-gci.service /etc/systemd/system/harness-gci.service && \
   sudo install -d -o samkim -g samkim -m 0750 /data/harness-gci && \
   sudo systemctl daemon-reload && \
   sudo systemctl enable harness-gci.service && \
   sudo systemctl restart harness-gci.service"

printf 'Deployed GCI release %s to %s without changing bge-m3-embed.service\n' "$RELEASE" "$TARGET"
