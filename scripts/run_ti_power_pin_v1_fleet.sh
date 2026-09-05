#!/usr/bin/env bash
# First TI-power pin run: 163 items unlocked by printed pin-count bindings
# plus the borderless-table locator relaxation (single-package docs only).
# Spreads work across the six serving boxes, seals, builds candidates, and
# submits the teacher batch under a $40 cap.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PR="$ROOT/results/datasheet-ti-power-pin-v1-local-20260905"
Q="$ROOT/results/datasheet-ti-power-structural-pin-v1-20260905.json"
EV="$ROOT/results/datasheet-ti-power-page-evidence-pin-v3-20260905"
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

# 163 items: five boxes x 6 streams x 5 items = 150; dgx3 takes 13 as
# 6 streams (1x3 + 5x2).
launch_box6() {
  local ip="$1" base="$2" p="$3"
  for i in 1 2 3 4 5 6; do
    run_shard "${p}${i}" $((base + (i - 1) * 5)) 5 "$ip" &
  done
}

launch_box6 192.168.4.45 0   ta # dgx2
launch_box6 192.168.4.58 30  tb # asus1
launch_box6 192.168.4.32 60  tc # asus3
launch_box6 192.168.4.39 90  td # asus2
launch_box6 192.168.4.56 120 te # asus4

IP3=192.168.4.49
run_shard "tf1" 150 3 "$IP3" &
off=153
for i in 2 3 4 5 6; do
  run_shard "tf$i" $off 2 "$IP3" &
  off=$((off + 2))
done
wait

BUNDLES=()
for d in ta1 ta2 ta3 ta4 ta5 ta6 tb1 tb2 tb3 tb4 tb5 tb6 \
         tc1 tc2 tc3 tc4 tc5 tc6 td1 td2 td3 td4 td5 td6 \
         te1 te2 te3 te4 te5 te6 tf1 tf2 tf3 tf4 tf5 tf6; do
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

echo ti-power-pin-v1-teacher-batch-submitted
