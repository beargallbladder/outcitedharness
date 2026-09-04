#!/usr/bin/env bash
# Third-pass MCU pin queue (211 items unlocked by CR pin-count receipts
# corroborating printed package tokens) across the idle fleet: 6 streams
# on five boxes, 12 on dgx3. Seals, builds candidates, submits the
# teacher batch under a $40 cap.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PR="$ROOT/results/datasheet-mcu-pin-v3-local-20260903"
Q="$ROOT/results/datasheet-mcu-structural-pin-v3-receipts-20260903.json"
EV="$ROOT/results/datasheet-mcu-factory-v1b-20260903/page-evidence"
SERVED="qwen3-vl-30b-bf16-base"
mkdir -p "$PR/logs"

run_shard() {
  local name="$1" off="$2" lim="$3" ip="$4"
  [ -f "$PR/$name/manifest.json" ] && return 0
  [ "$lim" -lt 1 ] && return 0
  uv run --python 3.11 --extra vision python \
    "$ROOT/scripts/run_datasheet_structural_extraction.py" \
    --structural-queue "$Q" \
    --page-evidence "$EV" \
    --base-url "http://$ip:8912/v1" \
    --model "$SERVED" \
    --offset "$off" --limit "$lim" \
    --render-dpi 120 --timeout-seconds 1800 --vision-policy always \
    --output-directory "$PR/$name" \
    >"$PR/logs/$name.log" 2>&1
}

# 211 items: five boxes x 6 streams x 6 items = 180; dgx3 takes 31 as
# 12 streams (7x3 + 5x2).
launch_box6() {
  local ip="$1" base="$2" p="$3"
  for i in 1 2 3 4 5 6; do
    run_shard "${p}${i}" $((base + (i - 1) * 6)) 6 "$ip" &
  done
}

launch_box6 192.168.4.45 0   ga # dgx2
launch_box6 192.168.4.58 36  gb # asus1
launch_box6 192.168.4.32 72  gc # asus3
launch_box6 192.168.4.39 108 gd # asus2
launch_box6 192.168.4.56 144 ge # asus4

IP3=192.168.4.49
off=180
for i in 01 02 03 04 05 06 07; do
  run_shard "gf$i" $off 3 "$IP3" &
  off=$((off + 3))
done
for i in 08 09 10 11 12; do
  run_shard "gf$i" $off 2 "$IP3" &
  off=$((off + 2))
done
wait

BUNDLES=()
for d in ga1 ga2 ga3 ga4 ga5 ga6 gb1 gb2 gb3 gb4 gb5 gb6 \
         gc1 gc2 gc3 gc4 gc5 gc6 gd1 gd2 gd3 gd4 gd5 gd6 \
         ge1 ge2 ge3 ge4 ge5 ge6 \
         gf01 gf02 gf03 gf04 gf05 gf06 gf07 gf08 gf09 gf10 gf11 gf12; do
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
  --batch-discount 0.5 --spend-cap-usd 40 \
  --output "$PR/frontier-prepared" \
  >>"$PR/logs/frontier.log" 2>&1 || exit 1

uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" submit \
  --bundle "$PR/frontier-prepared" \
  --state-directory "$PR/frontier-submission" \
  --approved-spend-cap-usd 40 --resume \
  >>"$PR/logs/frontier.log" 2>&1 || exit 1

echo mcu-pin-v3-teacher-batch-submitted
