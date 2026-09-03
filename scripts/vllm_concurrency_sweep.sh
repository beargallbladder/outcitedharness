#!/usr/bin/env bash
# Concurrency sweep on dgx3 after its v3 share seals: restart vLLM with
# max-num-seqs 16 + larger chunked-prefill budget, then replay the
# already-completed t07 range at 3, 6, and 12 client streams and report
# pages/minute per config. Throwaway outputs; never sealed into a run.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V3R="$ROOT/results/datasheet-six-node-local-pin-volume-v3-overflow-20260902"
Q="$ROOT/results/datasheet-structural-pin-volume-v3-overflow-20260902.json"
EV="$ROOT/results/datasheet-page-evidence-20260901"
OUT="$ROOT/results/vllm-concurrency-sweep-20260903"
HOST=dgx3
IP=192.168.4.49
BASE_OFF=2138 # dgx3 t07 range, already completed and sealed
mkdir -p "$OUT/logs"

# Gate: dgx3's whole v3 share must be sealed so nothing in flight dies.
for d in t07 t08 t09; do
  while [ ! -f "$V3R/$d/manifest.json" ]; do sleep 120; done
done

ssh -o ConnectTimeout=10 "$HOST" '
  set -e
  docker rm -f qwen3-vl-30b-bf16-vllm-v1
  docker run -d \
    --name qwen3-vl-30b-bf16-vllm-v1 \
    --restart unless-stopped \
    --gpus all --ipc host --shm-size 16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8912:8912 \
    -e HOME=/tmp \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -v "$HOME/.cache/vllm-datasheet-vision:/tmp/.cache" \
    -v "$HOME/harness-training/models/Qwen3-VL-30B-A3B-Instruct-BF16:/model:ro" \
    --entrypoint vllm \
    46591c6e4a01 \
    serve /model \
    --served-model-name qwen3-vl-30b-bf16-base \
    --host 0.0.0.0 --port 8912 \
    --trust-remote-code --dtype auto \
    --max-model-len 32768 --max-num-seqs 16 \
    --max-num-batched-tokens 65536 \
    --gpu-memory-utilization 0.70 \
    --limit-mm-per-prompt "{\"image\":1}" \
    --generation-config vllm \
    --enable-prefix-caching --enforce-eager >/dev/null
' || { echo "sweep: server restart failed" >&2; exit 1; }

deadline=$((SECONDS + 1500))
until curl -fsS --max-time 4 "http://$IP:8912/v1/models" 2>/dev/null |
  grep -q qwen3-vl-30b-bf16-base; do
  ((SECONDS < deadline)) || { echo "sweep: server not ready" >&2; exit 1; }
  sleep 10
done

run_config() {
  local label="$1" streams="$2" per="$3"
  local start=$SECONDS pids=() i off
  for ((i = 0; i < streams; i++)); do
    off=$((BASE_OFF + i * per))
    uv run --python 3.11 --extra vision python \
      "$ROOT/scripts/run_datasheet_structural_extraction.py" \
      --structural-queue "$Q" \
      --page-evidence "$EV" \
      --base-url "http://$IP:8912/v1" \
      --model qwen3-vl-30b-bf16-base \
      --offset "$off" --limit "$per" \
      --render-dpi 120 --timeout-seconds 1800 --vision-policy always \
      --output-directory "$OUT/$label-s$i" \
      >"$OUT/logs/$label-s$i.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  local elapsed=$((SECONDS - start)) total=$((streams * per))
  echo "SWEEP_RESULT label=$label streams=$streams items=$total" \
    "seconds=$elapsed pages_per_hour=$((total * 3600 / elapsed))" |
    tee -a "$OUT/results.txt"
}

run_config c3 3 60
run_config c6 6 30
run_config c12 12 15
echo sweep-complete
