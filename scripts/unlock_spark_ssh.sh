#!/bin/zsh
# Run THIS in Terminal.app (not Cursor). Your agent / Touch ID works there.
# Installs this Mac's file key on the Spark so the harness agent can SSH
# without a tap. Does not restart :8900 / :8800.
set -euo pipefail

PUB="$HOME/.ssh/id_ed25519.pub"
HOST="${1:-100.81.201.24}"
USER="${2:-}"

if [[ ! -f "$PUB" ]]; then
  echo "missing $PUB" >&2
  exit 1
fi

if [[ -z "$USER" ]]; then
  echo "Usage: $0 [host] <user>"
  echo "Example: $0 100.81.201.24 spark"
  echo
  echo "This Mac's file key (rejected today as samkim@):"
  cat "$PUB"
  exit 2
fi

echo "copying $(ssh-keygen -l -f "$PUB") → ${USER}@${HOST}"
ssh-copy-id -i "$PUB" -o StrictHostKeyChecking=accept-new "${USER}@${HOST}"
echo "testing BatchMode (what Cursor uses)..."
ssh -o BatchMode=yes -o ConnectTimeout=8 "${USER}@${HOST}" 'echo OK; whoami; hostname'
echo "done. Tell Cursor the user is: $USER"
