#!/usr/bin/env bash
# CR apps-verbatim run: 461 series_summary items over the 179-doc apps
# cohort (factory half of cr_apps_extraction_candidates: TI/NXP/Microchip/
# SiLabs/Renesas). Local-first on all six boxes, then teacher batch.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PR="$ROOT/results/datasheet-apps-summary-v1-local-20260905"
Q="$ROOT/results/datasheet-apps-structural-summary-v1-20260905.json"
EV="$ROOT/results/datasheet-apps-page-evidence-v1-20260905"
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
    --render-dpi 150 --timeout-seconds 1800 --vision-policy always \
    --output-directory "$PR/$name" \
    >"$PR/logs/$name.log" 2>&1
}

# 461 items: five boxes x 6 streams x 13 = 390; dgx3 takes 71 as
# 12 streams (11x6 + 1x5).
launch_box6() {
  local ip="$1" base="$2" p="$3"
  for i in 1 2 3 4 5 6; do
    run_shard "${p}${i}" $((base + (i - 1) * 13)) 13 "$ip" &
  done
}

launch_box6 192.168.4.45 0   aa # dgx2
launch_box6 192.168.4.58 78  ab # asus1
launch_box6 192.168.4.32 156 ac # asus3
launch_box6 192.168.4.39 234 ad # asus2
launch_box6 192.168.4.56 312 ae # asus4

IP3=192.168.4.49
off=390
for i in 01 02 03 04 05 06 07 08 09 10 11; do
  run_shard "af$i" $off 6 "$IP3" &
  off=$((off + 6))
done
run_shard "af12" $off 5 "$IP3" &
wait

BUNDLES=()
for d in aa1 aa2 aa3 aa4 aa5 aa6 ab1 ab2 ab3 ab4 ab5 ab6 \
         ac1 ac2 ac3 ac4 ac5 ac6 ad1 ad2 ad3 ad4 ad5 ad6 \
         ae1 ae2 ae3 ae4 ae5 ae6 \
         af01 af02 af03 af04 af05 af06 af07 af08 af09 af10 af11 af12; do
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
  --batch-discount 0.5 --spend-cap-usd 45 \
  --output "$PR/frontier-prepared" \
  >>"$PR/logs/frontier.log" 2>&1 || exit 1

uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" submit \
  --bundle "$PR/frontier-prepared" \
  --state-directory "$PR/frontier-submission" \
  --approved-spend-cap-usd 45 --resume \
  >>"$PR/logs/frontier.log" 2>&1 || exit 1

echo apps-summary-v1-teacher-batch-submitted
