#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LITELLM_VENV:-$ROOT/.litellm-venv}"
CONFIG="${LITELLM_TRAVEL_CONFIG:-$ROOT/config/litellm-travel.yaml}"
HOST="${LITELLM_TRAVEL_HOST:-127.0.0.1}"
PORT="${LITELLM_TRAVEL_PORT:-7411}"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

: "${M4_CLINE_API_KEY:?M4_CLINE_API_KEY must be set in $ROOT/.env}"
: "${QWEN38_API_KEY:?QWEN38_API_KEY must be set in $ROOT/.env}"
test -x "$VENV/bin/litellm" || {
  echo "LiteLLM is not installed; run scripts/install_litellm.sh install" >&2
  exit 1
}

exec "$VENV/bin/litellm" \
  --config "$CONFIG" \
  --host "$HOST" \
  --port "$PORT" \
  --num_workers 1
