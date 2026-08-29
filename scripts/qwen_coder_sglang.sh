#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
CONTAINER="${CONTAINER:-qwen3-coder-next-sglang}"
LEGACY_CONTAINER="${LEGACY_CONTAINER:-qwen3-coder-next}"
IMAGE="${IMAGE:-local/qwen3-coder-next-sglang:sm121-dflash}"
MODEL_PATH="${MODEL_PATH:-/home/$(id -un)/models/qwen3-coder-next-nvfp4-gb10}"
DRAFT_PATH="${DRAFT_PATH:-/home/$(id -un)/models/qwen3-coder-next-dflash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-coder-next}"
PORT="${PORT:-8900}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"

start() {
  test -d "${MODEL_PATH}" || {
    echo "Missing target model: ${MODEL_PATH}" >&2
    exit 1
  }
  test -d "${DRAFT_PATH}" || {
    echo "Missing DFlash draft model: ${DRAFT_PATH}" >&2
    exit 1
  }
  docker image inspect "${IMAGE}" >/dev/null

  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  if docker ps --format '{{.Names}}' | awk -v name="${LEGACY_CONTAINER}" '$0 == name { found=1 } END { exit !found }'; then
    docker stop "${LEGACY_CONTAINER}" >/dev/null
  fi

  sync
  sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

  if ! docker run -d \
    --name "${CONTAINER}" \
    --gpus all \
    --ipc host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --restart unless-stopped \
    -p "${PORT}:${PORT}" \
    -e HF_HUB_DISABLE_XET=1 \
    -e SGLANG_ENABLE_JIT_DEEPGEMM=0 \
    -e SGLANG_ENABLE_DEEP_GEMM=0 \
    -v "${MODEL_PATH}:/model:ro" \
    -v "${DRAFT_PATH}:/draft:ro" \
    "${IMAGE}" \
    python3 -m sglang.launch_server \
      --model-path /model \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --speculative-algorithm DFLASH \
      --speculative-draft-model-path /draft \
      --attention-backend flashinfer \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --context-length "${CONTEXT_LENGTH}" \
      --chunked-prefill-size 8192 \
      --max-running-requests "${MAX_RUNNING_REQUESTS}" \
      --disable-cuda-graph \
      --mamba-radix-cache-strategy extra_buffer \
      --tool-call-parser qwen3_coder \
      --trust-remote-code \
      --default-chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":false}' \
      --host 0.0.0.0 \
      --port "${PORT}" >/dev/null; then
    docker start "${LEGACY_CONTAINER}" >/dev/null 2>&1 || true
    exit 1
  fi

  echo "Started ${CONTAINER}; logs: docker logs -f ${CONTAINER}"
}

rollback() {
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  docker start "${LEGACY_CONTAINER}" >/dev/null
  echo "Restored ${LEGACY_CONTAINER}"
}

status() {
  docker ps -a \
    --filter "name=^/${CONTAINER}$" \
    --filter "name=^/${LEGACY_CONTAINER}$" \
    --format '{{.Names}} {{.Status}} {{.Image}}'
  curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/v1/models" || true
  echo
}

case "${ACTION}" in
  start) start ;;
  stop) docker rm -f "${CONTAINER}" ;;
  rollback) rollback ;;
  status) status ;;
  *)
    echo "Usage: $0 {start|stop|rollback|status}" >&2
    exit 2
    ;;
esac
