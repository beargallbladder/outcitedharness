#!/bin/zsh
# M5 Max serve: Qwen3.8-27B MLX 8-bit — interactive / critic / vision
#
# Role split:
#   M5  :8082  Qwen3.8-27B 8-bit  → fallback foreman and screenshots
#   DGX :8900  Qwen3.6-27B        → factory extract until 3.8 A/B holds
#   M5  :8080  GLM-OCR            → scanned-page fallback (leave running)
#
# Thinking stays OFF unless the client sends enable_thinking=true.

set -euo pipefail

MODEL_DIR="/Volumes/M5_4TB/models/Qwen3.8-27B-8bit"
DRAFT_DIR="/Volumes/M5_4TB/models/Qwen3.8-27B-MTP-8bit"
VENV_PY="${HOME}/mlx-reasoner-venv/bin/python"
CRED="/Volumes/M5_4TB/.credentials/api_keys.env"
LOG="/tmp/mlx_qwen38.log"
HOST="0.0.0.0"
PORT="8082"

if [[ -r "$CRED" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CRED" 2>/dev/null || true
  set +a
fi
export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-/Volumes/M5_4TB/models/hf_home}"

if [[ ! -d "$MODEL_DIR" ]] || [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR — download incomplete?" >&2
  exit 1
fi
if [[ ! -d "$DRAFT_DIR" ]] || [[ ! -f "$DRAFT_DIR/config.json" ]]; then
  echo "MTP drafter not found at $DRAFT_DIR — download incomplete?" >&2
  exit 1
fi

exec >>"$LOG" 2>&1
echo "==== $(date -u +%Y-%m-%dT%H:%M:%SZ) starting Qwen3.8-27B-8bit + MTP-8bit on :$PORT ===="

exec "$VENV_PY" -m mlx_vlm.server \
  --model "$MODEL_DIR" \
  --draft-model "$DRAFT_DIR" \
  --draft-kind mtp \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --max-tokens 8192 \
  --prefill-step-size 2048 \
  --vision-cache-size 64 \
  --kv-bits 4 \
  --kv-quant-scheme turboquant \
  --max-kv-size 131072 \
  --log-level INFO
