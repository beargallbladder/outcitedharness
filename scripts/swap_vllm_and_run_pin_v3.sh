#!/usr/bin/env bash
# Per box: when its v2 primary shard seals, replace the LLaMA Factory
# huggingface-backend server with a real vLLM server (same port, same
# served model name), then run that box's share of the pin v3 overflow
# queue as three concurrent client streams (vLLM continuous batching).
# Falls back to restarting the old server single-stream if vLLM fails.
# Finally seals candidates from all 18 bundles and submits the teacher
# batch under the approved cap.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V2R="$ROOT/results/datasheet-six-node-local-pin-volume-v2-locator-20260902"
V3R="$ROOT/results/datasheet-six-node-local-pin-volume-v3-overflow-20260902"
Q="$ROOT/results/datasheet-structural-pin-volume-v3-overflow-20260902.json"
EV="$ROOT/results/datasheet-page-evidence-20260901"
VLLM_IMAGE="46591c6e4a01"
VLLM_NAME="qwen3-vl-30b-bf16-vllm-v1"
SERVED="qwen3-vl-30b-bf16-base"
mkdir -p "$V3R/logs"

wait_ready() {
  local ip="$1" seconds="$2" deadline=$((SECONDS + $2))
  while ((SECONDS < deadline)); do
    if curl -fsS --max-time 4 "http://$ip:8912/v1/models" 2>/dev/null |
      grep -q "$SERVED"; then
      return 0
    fi
    sleep 10
  done
  return 1
}

swap_to_vllm() {
  local host="$1" ip="$2"
  # Idempotent: skip if this box already serves via vLLM.
  if ssh -o ConnectTimeout=10 "$host" \
    'docker ps --format "{{.Names}}" | grep -q "^'"$VLLM_NAME"'$"' &&
    wait_ready "$ip" 60; then
    return 0
  fi
  ssh -o ConnectTimeout=10 "$host" '
    set -e
    OLD="$(docker ps --format "{{.Names}}" | grep "^electronics" | head -1)"
    echo "$OLD" > /tmp/pin-v3-old-server-name
    [ -n "$OLD" ] && docker stop --time 60 "$OLD"
    docker rm -f '"$VLLM_NAME"' 2>/dev/null || true
    mkdir -p "$HOME/.cache/vllm-datasheet-vision"
    docker run -d \
      --name '"$VLLM_NAME"' \
      --restart unless-stopped \
      --gpus all \
      --ipc host \
      --shm-size 16g \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      -p 8912:8912 \
      -e HOME=/tmp \
      -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
      -v "$HOME/.cache/vllm-datasheet-vision:/tmp/.cache" \
      -v "$HOME/harness-training/models/Qwen3-VL-30B-A3B-Instruct-BF16:/model:ro" \
      --entrypoint vllm \
      '"$VLLM_IMAGE"' \
      serve /model \
      --served-model-name '"$SERVED"' \
      --host 0.0.0.0 \
      --port 8912 \
      --trust-remote-code \
      --dtype auto \
      --max-model-len 32768 \
      --max-num-seqs 4 \
      --max-num-batched-tokens 32768 \
      --gpu-memory-utilization 0.70 \
      --limit-mm-per-prompt "{\"image\":1}" \
      --generation-config vllm \
      --enable-prefix-caching \
      --enforce-eager >/dev/null
  ' || return 1
  wait_ready "$ip" 1500
}

restore_old_server() {
  local host="$1" ip="$2"
  ssh -o ConnectTimeout=10 "$host" '
    docker rm -f '"$VLLM_NAME"' 2>/dev/null || true
    docker start "$(cat /tmp/pin-v3-old-server-name)"
  '
  wait_ready "$ip" 900
}

run_shard() {
  local name="$1" off="$2" lim="$3" ip="$4"
  [ -f "$V3R/$name/manifest.json" ] && return 0
  uv run --python 3.11 --extra vision python \
    "$ROOT/scripts/run_datasheet_structural_extraction.py" \
    --structural-queue "$Q" \
    --page-evidence "$EV" \
    --base-url "http://$ip:8912/v1" \
    --model "$SERVED" \
    --offset "$off" --limit "$lim" \
    --render-dpi 120 --timeout-seconds 1800 --vision-policy always \
    --output-directory "$V3R/$name" \
    >"$V3R/logs/$name.log" 2>&1
}

gate_sealed() {
  # $1 is a comma-separated list of v2 bundle names that must all be sealed.
  local names="$1" name
  for name in ${names//,/ }; do
    [ -f "$V2R/$name/manifest.json" ] || return 1
  done
  return 0
}

box_chain() {
  local box="$1" host="$2" ip="$3"
  local an="$4" ao="$5" al="$6" bn="$7" bo="$8" bl="$9"
  local cn="${10}" co="${11}" cl="${12}" gates="${13:-$1}"
  while ! gate_sealed "$gates"; do sleep 120; done
  if swap_to_vllm "$host" "$ip"; then
    echo "$box swapped to vllm; running 3 concurrent streams"
    run_shard "$an" "$ao" "$al" "$ip" &
    run_shard "$bn" "$bo" "$bl" "$ip" &
    run_shard "$cn" "$co" "$cl" "$ip" &
    wait
  else
    echo "$box vllm swap FAILED; restoring old server, single stream" >&2
    restore_old_server "$host" "$ip" || {
      echo "$box has NO healthy server" >&2
      return 1
    }
    run_shard "$an" "$ao" "$al" "$ip"
    run_shard "$bn" "$bo" "$bl" "$ip"
    run_shard "$cn" "$co" "$cl" "$ip"
  fi
  echo "$box v3 chain complete"
}

box_chain dgx2  dgx2  192.168.4.45 t01 0    357 t02 357  357 t03 714  356 dgx2-r1,dgx2-r2,dgx2-r3 &
box_chain asus1 asus1 192.168.4.58 t04 1070 356 t05 1426 356 t06 1782 356 asus1-r1,asus1-r2,asus1-r3 &
box_chain dgx3  dgx3  192.168.4.49 t07 2138 356 t08 2494 356 t09 2850 356 dgx3-r1,dgx3-r2,dgx3-r3 &
box_chain asus3 asus3 192.168.4.32 t10 3206 356 t11 3562 356 t12 3918 356 asus3-r1,asus3-r2,asus3-r3 &
box_chain asus2 asus2 192.168.4.39 t13 4274 356 t14 4630 356 t15 4986 356 asus2-r1,asus2-r2,asus2-r3 &
box_chain asus4 asus4 192.168.4.56 t16 5342 356 t17 5698 356 t18 6054 356 asus4-r1,asus4-r2,asus4-r3 &
wait

BUNDLES=()
for d in t01 t02 t03 t04 t05 t06 t07 t08 t09 t10 t11 t12 t13 t14 t15 t16 t17 t18; do
  [ -f "$V3R/$d/manifest.json" ] || { echo "missing bundle $d; not sealing" >&2; exit 1; }
  BUNDLES+=(--local-bundle "$V3R/$d")
done

uv run --python 3.11 python "$ROOT/scripts/build_datasheet_frontier_candidates.py" \
  --work-queue "$Q" \
  "${BUNDLES[@]}" \
  --require-complete \
  --output-directory "$V3R/frontier-candidates" \
  >"$V3R/logs/frontier-candidates.log" 2>&1 || exit 1

uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" prepare \
  --candidates "$V3R/frontier-candidates/candidates.jsonl" \
  --allowed-root "$V3R" \
  --model claude-sonnet-5 \
  --input-price-per-million 3.0 --output-price-per-million 15.0 \
  --batch-discount 0.5 --spend-cap-usd 420 \
  --output "$V3R/frontier-prepared" \
  >>"$V3R/logs/frontier-candidates.log" 2>&1 || exit 1

uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" submit \
  --bundle "$V3R/frontier-prepared" \
  --state-directory "$V3R/frontier-submission" \
  --approved-spend-cap-usd 420 --resume \
  >>"$V3R/logs/frontier-candidates.log" 2>&1 || exit 1

echo v3-teacher-batch-submitted
