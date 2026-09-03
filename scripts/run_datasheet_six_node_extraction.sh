#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
QUEUE="${QUEUE:-$ROOT/results/datasheet-structural-local-work-v13-pin-semantic-balanced-20260901.json}"
PAGE_EVIDENCE="${PAGE_EVIDENCE:-$ROOT/results/datasheet-page-evidence-20260901}"
TAG="${TAG:-semantic-v6-balanced-20260901}"
RUN_DIR="${RUN_DIR:-$ROOT/results/datasheet-six-node-local-$TAG}"
MODEL="${MODEL:-qwen3-vl-30b-bf16-base}"
RENDER_DPI="${RENDER_DPI:-220}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"

nodes=(dgx2 asus1 dgx3 asus3 asus2 asus4)
weights=(${NODE_WEIGHTS:-9 7 14 8 8 9})
endpoints=(
  "http://192.168.4.45:8912/v1"
  "http://192.168.4.58:8912/v1"
  "http://192.168.4.49:8912/v1"
  "http://192.168.4.32:8912/v1"
  "http://192.168.4.39:8912/v1"
  "http://192.168.4.56:8912/v1"
)
if ((${#weights[@]} != ${#nodes[@]})); then
  printf 'NODE_WEIGHTS must provide one positive integer per node\n' >&2
  exit 2
fi
weight_total=0
for weight in "${weights[@]}"; do
  [[ "$weight" =~ ^[1-9][0-9]*$ ]] || {
    printf 'invalid node weight: %s\n' "$weight" >&2
    exit 2
  }
  weight_total=$((weight_total + weight))
done

if [[ -e "$RUN_DIR" || -L "$RUN_DIR" ]]; then
  printf 'immutable six-node extraction run already exists: %s\n' "$RUN_DIR" >&2
  exit 2
fi
[[ -f "$QUEUE" && ! -L "$QUEUE" ]] || {
  printf 'work queue is absent or unsafe: %s\n' "$QUEUE" >&2
  exit 2
}
[[ -d "$PAGE_EVIDENCE" && ! -L "$PAGE_EVIDENCE" ]] || {
  printf 'page evidence is absent or unsafe: %s\n' "$PAGE_EVIDENCE" >&2
  exit 2
}
mkdir -p "$RUN_DIR/logs"

wait_for_endpoint() {
  local base_url="$1"
  local deadline=$((SECONDS + 1200))
  local payload
  while ((SECONDS < deadline)); do
    if payload="$(curl -fsS --max-time 5 "$base_url/models" 2>/dev/null)" \
      && [[ "$payload" == *"$MODEL"* ]]; then
      printf 'ready %s %s\n' "$base_url" "$MODEL"
      return 0
    fi
    sleep 10
  done
  printf 'timed out waiting for %s (%s)\n' "$base_url" "$MODEL" >&2
  return 1
}

readiness_pids=()
for endpoint in "${endpoints[@]}"; do
  wait_for_endpoint "$endpoint" &
  readiness_pids+=("$!")
done
for pid in "${readiness_pids[@]}"; do
  wait "$pid"
done

total="$(
  python3 - "$QUEUE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(len(json.load(handle)["work"]))
PY
)"
[[ "$total" =~ ^[1-9][0-9]*$ ]] || {
  printf 'queue has an invalid work count: %s\n' "$total" >&2
  exit 2
}

counts=()
assigned=0
for weight in "${weights[@]}"; do
  count=$((total * weight / weight_total))
  counts+=("$count")
  assigned=$((assigned + count))
done
remainder=$((total - assigned))
for ((index = 0; index < remainder; index++)); do
  slot=$((index % ${#counts[@]}))
  counts[$slot]=$((counts[$slot] + 1))
done
offset=0
shard_pids=()
bundle_args=()
for index in "${!nodes[@]}"; do
  count="${counts[$index]}"
  ((count > 0)) || {
    printf 'queue is too small for the configured six-node weights\n' >&2
    exit 2
  }
  node="${nodes[$index]}"
  output="$RUN_DIR/$node"
  bundle_args+=(--local-bundle "$output")
  uv run --python 3.11 --extra vision python \
    "$ROOT/scripts/run_datasheet_structural_extraction.py" \
    --structural-queue "$QUEUE" \
    --page-evidence "$PAGE_EVIDENCE" \
    --base-url "${endpoints[$index]}" \
    --model "$MODEL" \
    --offset "$offset" \
    --limit "$count" \
    --render-dpi "$RENDER_DPI" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --vision-policy always \
    --output-directory "$output" \
    >"$RUN_DIR/logs/$node.log" 2>&1 &
  shard_pids+=("$!")
  offset=$((offset + count))
done

failed=0
for pid in "${shard_pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if ((failed)); then
  printf 'one or more extraction shards failed; inspect %s\n' \
    "$RUN_DIR/logs" >&2
  exit 1
fi

uv run --python 3.11 python \
  "$ROOT/scripts/build_datasheet_frontier_candidates.py" \
  --work-queue "$QUEUE" \
  "${bundle_args[@]}" \
  --require-complete \
  --output-directory "$RUN_DIR/frontier-candidates" \
  >"$RUN_DIR/logs/frontier-candidates.log"

printf 'completed %s work items in %s\n' "$total" "$RUN_DIR"
