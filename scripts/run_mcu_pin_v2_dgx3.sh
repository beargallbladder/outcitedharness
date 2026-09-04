#!/usr/bin/env bash
# Second-pass MCU pin queue (69 items unlocked by OPN-minted package
# bindings) on dgx3 as six concurrent streams, then seal, build frontier
# candidates, and submit the teacher batch under a $25 cap.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PR="$ROOT/results/datasheet-mcu-pin-v2-local-20260903"
Q="$ROOT/results/datasheet-mcu-structural-pin-v2-bindings-20260903.json"
EV="$ROOT/results/datasheet-mcu-factory-v1b-20260903/page-evidence"
IP=192.168.4.49
SERVED="qwen3-vl-30b-bf16-base"
mkdir -p "$PR/logs"

run_shard() {
  local name="$1" off="$2" lim="$3"
  [ -f "$PR/$name/manifest.json" ] && return 0
  uv run --python 3.11 --extra vision python \
    "$ROOT/scripts/run_datasheet_structural_extraction.py" \
    --structural-queue "$Q" \
    --page-evidence "$EV" \
    --base-url "http://$IP:8912/v1" \
    --model "$SERVED" \
    --offset "$off" --limit "$lim" \
    --render-dpi 120 --timeout-seconds 1800 --vision-policy always \
    --output-directory "$PR/$name" \
    >"$PR/logs/$name.log" 2>&1
}

run_shard p01 0  12 &
run_shard p02 12 12 &
run_shard p03 24 12 &
run_shard p04 36 12 &
run_shard p05 48 12 &
run_shard p06 60 9 &
wait

BUNDLES=()
for d in p01 p02 p03 p04 p05 p06; do
  [ -f "$PR/$d/manifest.json" ] || { echo "missing bundle $d" >&2; exit 1; }
  BUNDLES+=(--local-bundle "$PR/$d")
done

uv run --python 3.11 python "$ROOT/scripts/build_datasheet_frontier_candidates.py" \
  --work-queue "$Q" \
  "${BUNDLES[@]}" \
  --require-complete \
  --output-directory "$PR/frontier-candidates" \
  >"$PR/logs/frontier.log" 2>&1 || exit 1

uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" prepare \
  --candidates "$PR/frontier-candidates/candidates.jsonl" \
  --allowed-root "$PR" \
  --model claude-sonnet-5 \
  --input-price-per-million 3.0 --output-price-per-million 15.0 \
  --batch-discount 0.5 --spend-cap-usd 25 \
  --output "$PR/frontier-prepared" \
  >>"$PR/logs/frontier.log" 2>&1 || exit 1

uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" submit \
  --bundle "$PR/frontier-prepared" \
  --state-directory "$PR/frontier-submission" \
  --approved-spend-cap-usd 25 --resume \
  >>"$PR/logs/frontier.log" 2>&1 || exit 1

echo mcu-pin-v2-teacher-batch-submitted
