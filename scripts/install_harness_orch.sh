#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.samkim.harness-orch"
LEGACY_LABEL="com.samkim.harness-cline"
DOMAIN="gui/$(id -u)"
SOURCE="$ROOT/scripts/$LABEL.plist"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LEGACY_PLIST="$HOME/Library/LaunchAgents/$LEGACY_LABEL.plist"

install_service() {
  mkdir -p "$HOME/Library/LaunchAgents" "$HOME/.harness/logs"
  cp "$SOURCE" "$PLIST"
  launchctl bootout "$DOMAIN/$LEGACY_LABEL" >/dev/null 2>&1 || true
  launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  rm -f "$LEGACY_PLIST"
  launchctl bootstrap "$DOMAIN" "$PLIST"
  launchctl kickstart -k "$DOMAIN/$LABEL"
  echo "Installed Harness orchestration on http://127.0.0.1:8787/v1"
}

case "$ACTION" in
  install) install_service ;;
  restart) launchctl kickstart -k "$DOMAIN/$LABEL" ;;
  status) launchctl print "$DOMAIN/$LABEL" ;;
  uninstall)
    launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
    rm -f "$PLIST"
    ;;
  *)
    echo "Usage: $0 {install|restart|status|uninstall}" >&2
    exit 2
    ;;
esac
