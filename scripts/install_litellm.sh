#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.samkim.litellm"
LEGACY_LABEL="com.samkim.litellm-canary"
DOMAIN="gui/$(id -u)"
VENV="$ROOT/.litellm-venv"
TEMPLATE="$ROOT/scripts/$LABEL.plist.template"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LEGACY_PLIST="$HOME/Library/LaunchAgents/$LEGACY_LABEL.plist"

write_secret() {
  python3 - "$ROOT/.env" <<'PY'
from pathlib import Path
import secrets
import sys

path = Path(sys.argv[1])
text = path.read_text() if path.exists() else ""
if not any(line.startswith("LITELLM_MASTER_KEY=") for line in text.splitlines()):
    separator = "" if not text or text.endswith("\n") else "\n"
    path.write_text(
        text
        + separator
        + "LITELLM_MASTER_KEY=sk-"
        + secrets.token_urlsafe(32)
        + "\n"
    )
PY
}

render_plist() {
  python3 - "$TEMPLATE" "$PLIST" "$ROOT" "$HOME" <<'PY'
from pathlib import Path
import sys

source, destination, root, home = sys.argv[1:]
text = Path(source).read_text()
text = text.replace("__ROOT__", root).replace("__HOME__", home)
Path(destination).write_text(text)
PY
}

install_service() {
  command -v uv >/dev/null
  mkdir -p "$HOME/Library/LaunchAgents" "$HOME/.harness/logs"
  uv venv "$VENV"
  uv pip install --python "$VENV/bin/python" "litellm[proxy]" prisma
  write_secret
  chmod +x "$ROOT/scripts/serve_litellm.sh"
  render_plist
  launchctl bootout "$DOMAIN/$LEGACY_LABEL" >/dev/null 2>&1 || true
  launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  rm -f "$LEGACY_PLIST"
  launchctl bootstrap "$DOMAIN" "$PLIST"
  launchctl kickstart -k "$DOMAIN/$LABEL"
  echo "Installed LiteLLM on http://127.0.0.1:7410/v1"
}

uninstall_service() {
  launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  rm -f "$PLIST"
  echo "Uninstalled LiteLLM LaunchAgent; venv retained at $VENV"
}

case "$ACTION" in
  install) install_service ;;
  restart) launchctl kickstart -k "$DOMAIN/$LABEL" ;;
  uninstall) uninstall_service ;;
  status) launchctl print "$DOMAIN/$LABEL" ;;
  *)
    echo "Usage: $0 {install|restart|uninstall|status}" >&2
    exit 2
    ;;
esac
