#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-install}"
LABEL="com.samkim.harness-gci-refresh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE="$ROOT/scripts/$LABEL.plist.template"
HARNESS_EXECUTABLE="$ROOT/.venv/bin/harness"
AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST="$AGENTS_DIR/$LABEL.plist"
LOG_ROOT="$HOME/.harness/logs"
DOMAIN="gui/$(id -u)"

render() {
  [[ -x "$HARNESS_EXECUTABLE" ]] || {
    printf 'Harness executable is missing: %s\n' "$HARNESS_EXECUTABLE" >&2
    exit 1
  }
  mkdir -p "$AGENTS_DIR" "$LOG_ROOT"
  /usr/bin/sed \
    -e "s|__HARNESS_EXECUTABLE__|$HARNESS_EXECUTABLE|g" \
    -e "s|__HARNESS_ROOT__|$ROOT|g" \
    -e "s|__LOG_ROOT__|$LOG_ROOT|g" \
    "$TEMPLATE" > "$PLIST"
  /usr/bin/plutil -lint "$PLIST" >/dev/null
}

case "$ACTION" in
  install)
    render
    /bin/launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    /bin/launchctl bootstrap "$DOMAIN" "$PLIST"
    /bin/launchctl enable "$DOMAIN/$LABEL"
    /bin/launchctl kickstart "$DOMAIN/$LABEL"
    printf 'Installed %s; refresh checks run every five minutes.\n' "$LABEL"
    ;;
  uninstall)
    /bin/launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    printf 'Uninstalled %s.\n' "$LABEL"
    ;;
  status)
    /bin/launchctl print "$DOMAIN/$LABEL"
    ;;
  *)
    printf 'Usage: %s [install|uninstall|status]\n' "$0" >&2
    exit 2
    ;;
esac
