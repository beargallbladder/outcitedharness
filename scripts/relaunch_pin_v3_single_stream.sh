#!/usr/bin/env bash
# Relaunch pin-volume-v3 shards single-stream per box after each box's
# v2 work seals, then seal candidates and submit the teacher batch.
# Rationale: measured fleet throughput fell as streams per box rose
# (200/hr @1, 192/hr @2, 162/hr @4) on GB10-class boxes.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V2R="$ROOT/results/datasheet-six-node-local-pin-volume-v2-locator-20260902"
V3R="$ROOT/results/datasheet-six-node-local-pin-volume-v3-overflow-20260902"
Q="$ROOT/results/datasheet-structural-pin-volume-v3-overflow-20260902.json"
EV="$ROOT/results/datasheet-page-evidence-20260901"
mkdir -p "$V3R/logs"

run_shard() {
  local name="$1" off="$2" lim="$3" ip="$4"
  [ -f "$V3R/$name/manifest.json" ] && return 0
  uv run --python 3.11 --extra vision python \
    "$ROOT/scripts/run_datasheet_structural_extraction.py" \
    --structural-queue "$Q" \
    --page-evidence "$EV" \
    --base-url "http://$ip:8912/v1" \
    --model qwen3-vl-30b-bf16-base \
    --offset "$off" --limit "$lim" \
    --render-dpi 120 --timeout-seconds 1800 --vision-policy always \
    --output-directory "$V3R/$name" \
    >"$V3R/logs/$name.log" 2>&1
}

box_chain() {
  local box="$1" bshard="$2" ip="$3" an="$4" ao="$5" al="$6" bn="$7" bo="$8" bl="$9"
  while [ ! -f "$V2R/$box/manifest.json" ] || [ ! -f "$V2R/$bshard/manifest.json" ]; do
    sleep 120
  done
  run_shard "$an" "$ao" "$al" "$ip" || { echo "FAILED $an" >&2; return 1; }
  run_shard "$bn" "$bo" "$bl" "$ip" || { echo "FAILED $bn" >&2; return 1; }
  echo "$box v3 chain complete"
}

box_chain dgx2  dgx3-b2 192.168.4.45 s01 0    535 s07 3206 534 &
box_chain asus1 dgx3-b3 192.168.4.58 s02 535  535 s08 3740 534 &
box_chain dgx3  dgx3-b4 192.168.4.49 s03 1070 534 s09 4274 534 &
box_chain asus3 dgx3-b5 192.168.4.32 s04 1604 534 s10 4808 534 &
box_chain asus2 dgx3-b6 192.168.4.39 s05 2138 534 s11 5342 534 &
box_chain asus4 dgx3-b1 192.168.4.56 s06 2672 534 s12 5876 534 &
wait

for d in s01 s02 s03 s04 s05 s06 s07 s08 s09 s10 s11 s12; do
  [ -f "$V3R/$d/manifest.json" ] || { echo "missing bundle $d; not sealing" >&2; exit 1; }
done

uv run --python 3.11 python "$ROOT/scripts/build_datasheet_frontier_candidates.py" \
  --work-queue "$Q" \
  --local-bundle "$V3R/s01" --local-bundle "$V3R/s02" --local-bundle "$V3R/s03" \
  --local-bundle "$V3R/s04" --local-bundle "$V3R/s05" --local-bundle "$V3R/s06" \
  --local-bundle "$V3R/s07" --local-bundle "$V3R/s08" --local-bundle "$V3R/s09" \
  --local-bundle "$V3R/s10" --local-bundle "$V3R/s11" --local-bundle "$V3R/s12" \
  --require-complete \
  --output-directory "$V3R/frontier-candidates" \
  >"$V3R/logs/frontier-candidates.log" 2>&1 || exit 1

uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" prepare \
  --candidates "$V3R/frontier-candidates/candidates.jsonl" \
  --allowed-root "$V3R" \
  --model claude-sonnet-5 \
  --input-price-per-million 3.0 --output-price-per-million 15.0 \
  --batch-discount 0.5 --spend-cap-usd 260 \
  --output "$V3R/frontier-prepared" \
  >>"$V3R/logs/frontier-candidates.log" 2>&1 || exit 1

uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" submit \
  --bundle "$V3R/frontier-prepared" \
  --state-directory "$V3R/frontier-submission" \
  --approved-spend-cap-usd 260 --resume \
  >>"$V3R/logs/frontier-candidates.log" 2>&1 || exit 1

echo v3-teacher-batch-submitted
