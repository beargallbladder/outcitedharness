#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.samkim.litellm-travel"
DOMAIN="gui/$(id -u)"
VENV="$ROOT/.litellm-venv"
TEMPLATE="$ROOT/scripts/$LABEL.plist.template"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

write_secret() {
  python3 - "$ROOT/.env" <<'PY'
from pathlib import Path
import secrets
import sys

path = Path(sys.argv[1])
text = path.read_text() if path.exists() else ""
if not any(line.startswith("M4_CLINE_API_KEY=") for line in text.splitlines()):
    separator = "" if not text or text.endswith("\n") else "\n"
    path.write_text(
        text
        + separator
        + "M4_CLINE_API_KEY=sk-m4-"
        + secrets.token_urlsafe(32)
        + "\n"
    )
    text = path.read_text()
if not any(line.startswith("QWEN38_API_KEY=") for line in text.splitlines()):
    separator = "" if not text or text.endswith("\n") else "\n"
    path.write_text(
        text
        + separator
        + "QWEN38_API_KEY=sk-qwen38-"
        + secrets.token_urlsafe(32)
        + "\n"
    )
path.chmod(0o600)
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
  test -x "$VENV/bin/litellm" || {
    echo "Install the primary LiteLLM service first." >&2
    exit 1
  }
  mkdir -p "$HOME/Library/LaunchAgents" "$HOME/.harness/logs"
  write_secret
  chmod +x "$ROOT/scripts/serve_litellm_travel.sh"
  render_plist
  launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  launchctl bootstrap "$DOMAIN" "$PLIST"
  launchctl kickstart -k "$DOMAIN/$LABEL"
  echo "Installed travel LiteLLM on http://127.0.0.1:7411/v1"
}

uninstall_service() {
  launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
  rm -f "$PLIST"
  echo "Uninstalled travel LiteLLM; credential retained in .env"
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
