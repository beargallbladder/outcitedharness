#!/usr/bin/env bash
# TI power summary/OPN queue (3,179 items, built 2026-09-02 but never
# locally extracted) across all six boxes overnight. Stream counts per
# the concurrency sweep: 6 client streams per box, 12 on dgx3 whose
# vLLM already runs with raised max-num-seqs. Seals all bundles, builds
# frontier candidates, submits the teacher batch under a $250 cap.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V="$ROOT/results/datasheet-ti-power-factory-v1-20260902"
SR="$ROOT/results/datasheet-ti-power-summary-local-20260903"
Q="$V/structural-summary-opn-v1.json"
EV="$V/page-evidence"
SERVED="qwen3-vl-30b-bf16-base"
SPEND_CAP=250
mkdir -p "$SR/logs"

run_shard() {
  local name="$1" off="$2" lim="$3" ip="$4"
  [ -f "$SR/$name/manifest.json" ] && return 0
  uv run --python 3.11 --extra vision python \
    "$ROOT/scripts/run_datasheet_structural_extraction.py" \
    --structural-queue "$Q" \
    --page-evidence "$EV" \
    --base-url "http://$ip:8912/v1" \
    --model "$SERVED" \
    --offset "$off" --limit "$lim" \
    --render-dpi 220 --timeout-seconds 1800 --vision-policy always \
    --output-directory "$SR/$name" \
    >"$SR/logs/$name.log" 2>&1
}

box6() {
  # Six concurrent streams over one box's 520-item share.
  local ip="$1" base="$2" p="$3"
  run_shard "${p}1" "$base" 87 "$ip" &
  run_shard "${p}2" $((base + 87)) 87 "$ip" &
  run_shard "${p}3" $((base + 174)) 87 "$ip" &
  run_shard "${p}4" $((base + 261)) 87 "$ip" &
  run_shard "${p}5" $((base + 348)) 86 "$ip" &
  run_shard "${p}6" $((base + 434)) 86 "$ip" &
}

box6 192.168.4.45 0    a  # dgx2
box6 192.168.4.58 520  b  # asus1
box6 192.168.4.32 1040 c  # asus3
box6 192.168.4.39 1560 d  # asus2
box6 192.168.4.56 2080 e  # asus4

# dgx3: 579 items as 12 streams (3x49 + 9x48).
IP3=192.168.4.49
run_shard f01 2600 49 "$IP3" &
run_shard f02 2649 49 "$IP3" &
run_shard f03 2698 49 "$IP3" &
run_shard f04 2747 48 "$IP3" &
run_shard f05 2795 48 "$IP3" &
run_shard f06 2843 48 "$IP3" &
run_shard f07 2891 48 "$IP3" &
run_shard f08 2939 48 "$IP3" &
run_shard f09 2987 48 "$IP3" &
run_shard f10 3035 48 "$IP3" &
run_shard f11 3083 48 "$IP3" &
run_shard f12 3131 48 "$IP3" &
wait

BUNDLES=()
for d in a1 a2 a3 a4 a5 a6 b1 b2 b3 b4 b5 b6 c1 c2 c3 c4 c5 c6 \
         d1 d2 d3 d4 d5 d6 e1 e2 e3 e4 e5 e6 \
         f01 f02 f03 f04 f05 f06 f07 f08 f09 f10 f11 f12; do
  [ -f "$SR/$d/manifest.json" ] || { echo "missing bundle $d" >&2; exit 1; }
  BUNDLES+=(--local-bundle "$SR/$d")
done

uv run --python 3.11 python "$ROOT/scripts/build_datasheet_frontier_candidates.py" \
  --work-queue "$Q" \
  "${BUNDLES[@]}" \
  --require-complete \
  --output-directory "$SR/frontier-candidates" \
  >"$SR/logs/frontier.log" 2>&1 || exit 1

uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" prepare \
  --candidates "$SR/frontier-candidates/candidates.jsonl" \
  --allowed-root "$SR" \
  --model claude-sonnet-5 \
  --input-price-per-million 3.0 --output-price-per-million 15.0 \
  --batch-discount 0.5 --spend-cap-usd "$SPEND_CAP" \
  --output "$SR/frontier-prepared" \
  >>"$SR/logs/frontier.log" 2>&1 || exit 1

uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" submit \
  --bundle "$SR/frontier-prepared" \
  --state-directory "$SR/frontier-submission" \
  --approved-spend-cap-usd "$SPEND_CAP" --resume \
  >>"$SR/logs/frontier.log" 2>&1 || exit 1

echo ti-power-summary-teacher-batch-submitted
