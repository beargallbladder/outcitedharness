#!/usr/bin/env bash
set -euo pipefail

root="${DGX3_VISION_ROOT:-$HOME/datasheet-vision}"
runtime_image_id="sha256:46591c6e4a018d8d197fa246b1e3d682c907654aab4e9402302abb3e6a7dd916"
coder_container="qwen3-coder-next-sglang"
port=8902
profile="${DGX3_VISION_PROFILE:-production}"

case "$profile" in
  production)
    model="$HOME/models/Qwen3-VL-30B-A3B-Instruct-FP8"
    model_manifest="$root/manifests/qwen3-vl-30b-a3b-instruct-fp8.sha256.json"
    vision_container="datasheet-qwen3-vl"
    served_model="qwen3-vl-30b-a3b-instruct-fp8"
    max_model_len=32768
    max_num_seqs=4
    gpu_memory_utilization=0.70
    ;;
  bootstrap)
    model="$HOME/models/Qwen3-VL-8B-Instruct"
    model_manifest="$root/manifests/qwen3-vl-8b-instruct.sha256.json"
    vision_container="datasheet-qwen3-vl-bootstrap"
    served_model="qwen3-vl-8b-instruct"
    max_model_len=16384
    max_num_seqs=1
    gpu_memory_utilization=0.20
    ;;
  *) printf 'ERROR: unsupported DGX3_VISION_PROFILE=%s\n' "$profile" >&2; exit 2 ;;
esac

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

container_running() {
  [[ "$(docker inspect --format '{{.State.Running}}' "$1" 2>/dev/null || true)" == "true" ]]
}

verify_preflight() {
  test -f "$root/.harness-datasheet-vision-owner-v1"
  test -x "$root/scripts/training_manifest.py"
  test -d "$model"
  test ! -L "$model"
  test -f "$model_manifest"
  test ! -L "$model_manifest"
  [[ "$(docker image inspect --format '{{.Id}}' "$runtime_image_id")" == "$runtime_image_id" ]]
  python3 "$root/scripts/training_manifest.py" verify "$model" "$model_manifest"
  [[ "$(docker inspect --format '{{.Image}}' "$coder_container")" \
    == "sha256:11dda08f2a5270c3afe6cd0461f6a42b9553b0bcaadc90ac39cbf849b7f4782c" ]]
  if container_running "$coder_container"; then
    python3 - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8900/v1/models", timeout=5) as response:
    payload = json.load(response)
models = payload.get("data") if isinstance(payload, dict) else None
if not isinstance(models, list) or [row.get("id") for row in models] != ["qwen3-coder-next"]:
    raise SystemExit("coder rollback endpoint is not healthy")
PY
  fi
  python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as probe:
    try:
        probe.bind(("0.0.0.0", port))
    except OSError as error:
        raise SystemExit(f"vision port {port} is not free: {error}")
PY
}

wait_ready() {
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if ! container_running "$vision_container"; then
      docker logs "$vision_container" >&2 || true
      return 1
    fi
    if python3 - "$port" "$served_model" <<'PY'
import json
import sys
import urllib.request

port, expected = sys.argv[1:]
try:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/v1/models",
        timeout=3,
    ) as response:
        payload = json.load(response)
except Exception:
    raise SystemExit(1)
models = payload.get("data") if isinstance(payload, dict) else None
if not isinstance(models, list):
    raise SystemExit(1)
raise SystemExit(0 if [row.get("id") for row in models] == [expected] else 1)
PY
    then
      return 0
    fi
    sleep 5
  done
  docker logs "$vision_container" >&2 || true
  return 1
}

start_vision() {
  [[ "$profile" == "production" ]] &&
    container_running "$coder_container" &&
    die "$coder_container is still running; use activate for an atomic handoff"
  if docker inspect "$vision_container" >/dev/null 2>&1; then
    die "$vision_container already exists; stop it before replacing it"
  fi
  mkdir -p "$HOME/.cache/vllm-datasheet-vision"
  docker run -d \
    --name "$vision_container" \
    --restart unless-stopped \
    --gpus all \
    --ipc host \
    --shm-size 16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "${port}:${port}" \
    -e HOME=/tmp \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -v "$HOME/.cache/vllm-datasheet-vision:/tmp/.cache" \
    -v "$model:/model:ro" \
    --entrypoint vllm \
    "$runtime_image_id" \
    serve /model \
    --served-model-name "$served_model" \
    --host 0.0.0.0 \
    --port "$port" \
    --trust-remote-code \
    --dtype auto \
    --max-model-len "$max_model_len" \
    --max-num-seqs "$max_num_seqs" \
    --max-num-batched-tokens "$max_model_len" \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    --limit-mm-per-prompt '{"image":1}' \
    --generation-config vllm \
    --enable-prefix-caching \
    --enforce-eager >/dev/null
  wait_ready
}

activate() {
  [[ "$profile" == "production" ]] ||
    die "activate is valid only for the production profile"
  verify_preflight
  container_running "$vision_container" &&
    die "$vision_container is already running"
  container_running "$coder_container" ||
    die "$coder_container is not running; rollback baseline is not healthy"
  docker stop --time 60 "$coder_container" >/dev/null
  if start_vision; then
    printf 'DGX3_DATASHEET_VISION_READY model=%s port=%s\n' "$served_model" "$port"
    return 0
  fi
  docker rm -f "$vision_container" >/dev/null 2>&1 || true
  docker start "$coder_container" >/dev/null
  printf 'Vision activation failed; coder rollback restored.\n' >&2
  return 1
}

stop_vision() {
  if docker inspect "$vision_container" >/dev/null 2>&1; then
    docker rm -f "$vision_container" >/dev/null
  fi
}

rollback_coder() {
  stop_vision
  docker start "$coder_container" >/dev/null
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if python3 - <<'PY'
import json
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8900/v1/models", timeout=3) as response:
        payload = json.load(response)
except Exception:
    raise SystemExit(1)
models = payload.get("data") if isinstance(payload, dict) else None
raise SystemExit(
    0 if isinstance(models, list) and [row.get("id") for row in models] == ["qwen3-coder-next"] else 1
)
PY
    then
      printf 'DGX3_CODER_ROLLBACK_READY\n'
      return 0
    fi
    sleep 5
  done
  die "coder rollback did not become ready"
}

status() {
  docker ps -a \
    --filter "name=^/${vision_container}$" \
    --filter "name=^/${coder_container}$" \
    --format '{{.Names}} {{.Status}}'
}

case "${1:-}" in
  preflight) verify_preflight ;;
  activate) activate ;;
  start) verify_preflight; start_vision ;;
  stop) stop_vision ;;
  rollback-coder) rollback_coder ;;
  status) status ;;
  *) die "usage: $0 preflight|activate|start|stop|rollback-coder|status" ;;
esac
