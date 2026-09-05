#!/usr/bin/env bash
# Fourth-pass MCU pin queue: 574 items unlocked by the borderless-table
# locator relaxation (single-package docs, TOC-located pin sections where
# find_tables sees nothing). Six boxes; seals, builds candidates, submits
# the teacher batch under a $60 cap.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PR="$ROOT/results/datasheet-mcu-pin-v4-local-20260905"
Q="$ROOT/results/datasheet-mcu-structural-pin-v4-borderless-20260905.json"
EV="$ROOT/results/datasheet-mcu-page-evidence-v4-borderless-20260905"
SERVED="qwen3-vl-30b-pin-gate-457"
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

# 574 items: five boxes x 6 streams x 16 = 480; dgx3 takes 94 as
# 12 streams (10x8 + 2x7).
launch_box6() {
  local ip="$1" base="$2" p="$3"
  for i in 1 2 3 4 5 6; do
    run_shard "${p}${i}" $((base + (i - 1) * 16)) 16 "$ip" &
  done
}

launch_box6 192.168.4.45 0   ma # dgx2
launch_box6 192.168.4.58 96  mb # asus1
launch_box6 192.168.4.32 192 mc # asus3
launch_box6 192.168.4.39 288 md # asus2
launch_box6 192.168.4.56 384 me # asus4

IP3=192.168.4.49
off=480
for i in 01 02 03 04 05 06 07 08 09 10; do
  run_shard "mf$i" $off 8 "$IP3" &
  off=$((off + 8))
done
for i in 11 12; do
  run_shard "mf$i" $off 7 "$IP3" &
  off=$((off + 7))
done
wait

BUNDLES=()
for d in ma1 ma2 ma3 ma4 ma5 ma6 mb1 mb2 mb3 mb4 mb5 mb6 \
         mc1 mc2 mc3 mc4 mc5 mc6 md1 md2 md3 md4 md5 md6 \
         me1 me2 me3 me4 me5 me6 \
         mf01 mf02 mf03 mf04 mf05 mf06 mf07 mf08 mf09 mf10 mf11 mf12; do
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
  --batch-discount 0.5 --spend-cap-usd 60 \
  --output "$PR/frontier-prepared" \
  >>"$PR/logs/frontier.log" 2>&1 || exit 1

uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" submit \
  --bundle "$PR/frontier-prepared" \
  --state-directory "$PR/frontier-submission" \
  --approved-spend-cap-usd 60 --resume \
  >>"$PR/logs/frontier.log" 2>&1 || exit 1

echo mcu-pin-v4-teacher-batch-submitted
