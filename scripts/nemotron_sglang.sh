#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
CONTAINER="${CONTAINER:-nemotron-lightning-sglang}"
LEGACY_CONTAINER="${LEGACY_CONTAINER:-nemotron-lightning}"
IMAGE="${IMAGE:-lmsysorg/sglang@sha256:ee63304703432429d55cf2e4579b70c5a334ed4cc49194fb5290c077fcbbd4a0}"
MODEL_PATH="${MODEL_PATH:-/home/$(id -un)/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nemotron-lightning}"
PORT="${PORT:-8900}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.78}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"

start() {
  test -d "${MODEL_PATH}" || {
    echo "Missing Nemotron checkpoint: ${MODEL_PATH}" >&2
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
    -v "${MODEL_PATH}:/model:ro" \
    "${IMAGE}" \
    sglang serve \
      --model-path /model \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --mamba-ssm-dtype float16 \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --context-length "${CONTEXT_LENGTH}" \
      --max-running-requests "${MAX_RUNNING_REQUESTS}" \
      --cuda-graph-max-bs-decode 4 \
      --speculative-algorithm EAGLE \
      --speculative-draft-model-path /model \
      --speculative-num-steps 5 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 6 \
      --reasoning-parser nemotron_3 \
      --tool-call-parser qwen3_coder \
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
