#!/usr/bin/env bash
# Per box: when its three pin-v3 shards seal, run that box's share of the
# MCU intake structural queue (4,000 items: parametrics, series summaries,
# OPN decodes, pin semantics) as three concurrent client streams against
# the box's vLLM server. dgx3 additionally waits for the concurrency sweep
# to finish so the two workloads never share the box. Finally seals
# frontier candidates from all 18 bundles and submits the teacher batch
# under the approved cap.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V3R="$ROOT/results/datasheet-six-node-local-pin-volume-v3-overflow-20260902"
MR="$ROOT/results/datasheet-six-node-mcu-v1-20260903"
Q="$ROOT/results/datasheet-mcu-factory-v1b-20260903/structural-queue.json"
EV="$ROOT/results/datasheet-mcu-factory-v1b-20260903/page-evidence"
SWEEP="$ROOT/results/vllm-concurrency-sweep-20260903/results.txt"
SERVED="qwen3-vl-30b-bf16-base"
SPEND_CAP=300
mkdir -p "$MR/logs"

wait_ready() {
  local ip="$1" deadline=$((SECONDS + $2))
  while ((SECONDS < deadline)); do
    if curl -fsS --max-time 4 "http://$ip:8912/v1/models" 2>/dev/null |
      grep -q "$SERVED"; then
      return 0
    fi
    sleep 10
  done
  return 1
}

run_shard() {
  local name="$1" off="$2" lim="$3" ip="$4"
  [ -f "$MR/$name/manifest.json" ] && return 0
  uv run --python 3.11 --extra vision python \
    "$ROOT/scripts/run_datasheet_structural_extraction.py" \
    --structural-queue "$Q" \
    --page-evidence "$EV" \
    --base-url "http://$ip:8912/v1" \
    --model "$SERVED" \
    --offset "$off" --limit "$lim" \
    --render-dpi 220 --timeout-seconds 1800 --vision-policy always \
    --output-directory "$MR/$name" \
    >"$MR/logs/$name.log" 2>&1
}

gate_sealed() {
  local names="$1" name
  for name in ${names//,/ }; do
    [ -f "$V3R/$name/manifest.json" ] || return 1
  done
  return 0
}

box_chain() {
  local box="$1" ip="$2"
  local an="$3" ao="$4" al="$5" bn="$6" bo="$7" bl="$8"
  local cn="$9" co="${10}" cl="${11}" gates="${12}" extra_gate="${13:-}"
  while ! gate_sealed "$gates"; do sleep 120; done
  if [ -n "$extra_gate" ]; then
    # dgx3: wait for the concurrency sweep's final configuration to land.
    while ! grep -qs "label=c12" "$extra_gate"; do sleep 120; done
  fi
  if ! wait_ready "$ip" 1800; then
    echo "$box server never became ready; skipping its MCU share" >&2
    return 1
  fi
  echo "$box starting MCU shards"
  run_shard "$an" "$ao" "$al" "$ip" &
  run_shard "$bn" "$bo" "$bl" "$ip" &
  run_shard "$cn" "$co" "$cl" "$ip" &
  wait
  echo "$box mcu chain complete"
}

box_chain dgx2  192.168.4.45 m01 0    223 m02 223  223 m03 446  223 t01,t02,t03 &
box_chain asus1 192.168.4.58 m04 669  223 m05 892  222 m06 1114 222 t04,t05,t06 &
box_chain dgx3  192.168.4.49 m07 1336 222 m08 1558 222 m09 1780 222 t07,t08,t09 "$SWEEP" &
box_chain asus3 192.168.4.32 m10 2002 222 m11 2224 222 m12 2446 222 t10,t11,t12 &
box_chain asus2 192.168.4.39 m13 2668 222 m14 2890 222 m15 3112 222 t13,t14,t15 &
box_chain asus4 192.168.4.56 m16 3334 222 m17 3556 222 m18 3778 222 t16,t17,t18 &
wait

BUNDLES=()
for d in m01 m02 m03 m04 m05 m06 m07 m08 m09 m10 m11 m12 m13 m14 m15 m16 m17 m18; do
  [ -f "$MR/$d/manifest.json" ] || { echo "missing bundle $d; not sealing" >&2; exit 1; }
  BUNDLES+=(--local-bundle "$MR/$d")
done

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
