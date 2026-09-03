#!/usr/bin/env bash
# dgx3's MCU share (queue offsets 1336-2001, 666 items) as 12 concurrent
# streams, per the concurrency sweep (12 streams = 278 pages/hour vs 148
# at 3). dgx3's vLLM already runs with max-num-seqs 16 from the sweep.
# After all fleet bundles seal (15 m-shards + these 12 d-shards), builds
# frontier candidates and submits the MCU teacher batch at the $300 cap.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MR="$ROOT/results/datasheet-six-node-mcu-v1-20260903"
Q="$ROOT/results/datasheet-mcu-factory-v1b-20260903/structural-queue.json"
EV="$ROOT/results/datasheet-mcu-factory-v1b-20260903/page-evidence"
IP=192.168.4.49
SERVED="qwen3-vl-30b-bf16-base"
SPEND_CAP=300
mkdir -p "$MR/logs"

run_shard() {
  local name="$1" off="$2" lim="$3"
  [ -f "$MR/$name/manifest.json" ] && return 0
  uv run --python 3.11 --extra vision python \
    "$ROOT/scripts/run_datasheet_structural_extraction.py" \
    --structural-queue "$Q" \
    --page-evidence "$EV" \
    --base-url "http://$IP:8912/v1" \
    --model "$SERVED" \
    --offset "$off" --limit "$lim" \
    --render-dpi 220 --timeout-seconds 1800 --vision-policy always \
    --output-directory "$MR/$name" \
    >"$MR/logs/$name.log" 2>&1
}

run_shard d01 1336 56 &
run_shard d02 1392 56 &
run_shard d03 1448 56 &
run_shard d04 1504 56 &
run_shard d05 1560 56 &
run_shard d06 1616 56 &
run_shard d07 1672 55 &
run_shard d08 1727 55 &
run_shard d09 1782 55 &
run_shard d10 1837 55 &
run_shard d11 1892 55 &
run_shard d12 1947 55 &
wait
echo dgx3-12stream-complete

# Wait for the rest of the fleet's m-shards, then seal and submit.
ALL=(m01 m02 m03 m04 m05 m06 m10 m11 m12 m13 m14 m15 m16 m17 m18
     d01 d02 d03 d04 d05 d06 d07 d08 d09 d10 d11 d12)
while :; do
  missing=0
  for d in "${ALL[@]}"; do
    [ -f "$MR/$d/manifest.json" ] || { missing=1; break; }
  done
  [ "$missing" = 0 ] && break
  sleep 120
done

BUNDLES=()
for d in "${ALL[@]}"; do BUNDLES+=(--local-bundle "$MR/$d"); done

uv run --python 3.11 python "$ROOT/scripts/build_datasheet_frontier_candidates.py" \
  --work-queue "$Q" \
  "${BUNDLES[@]}" \
  --require-complete \
  --output-directory "$MR/frontier-candidates" \
  >"$MR/logs/frontier-candidates.log" 2>&1 || exit 1

uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" prepare \
  --candidates "$MR/frontier-candidates/candidates.jsonl" \
  --allowed-root "$MR" \
  --model claude-sonnet-5 \
  --input-price-per-million 3.0 --output-price-per-million 15.0 \
  --batch-discount 0.5 --spend-cap-usd "$SPEND_CAP" \
  --output "$MR/frontier-prepared" \
  >>"$MR/logs/frontier-candidates.log" 2>&1 || exit 1

uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" submit \
  --bundle "$MR/frontier-prepared" \
  --state-directory "$MR/frontier-submission" \
  --approved-spend-cap-usd "$SPEND_CAP" --resume \
  >>"$MR/logs/frontier-candidates.log" 2>&1 || exit 1

echo mcu-v1-teacher-batch-submitted
