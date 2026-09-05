#!/usr/bin/env bash
# Poll all outstanding Anthropic teacher batches; as each submission's
# batches all end, run retrieve -> reconcile -> verify -> finalize so
# training pairs are sealed the moment results exist. Exits when every
# listed run is harvested. Safe to re-run (skips harvested runs).
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GT=/Volumes/M5_4TB/DigiKey_Reference_Designs/claude_ground_truth

# name|state_dir|prepared|queue|page_evidence|pillar_evidence|registry
RUNS=(
"v2pin|results/datasheet-six-node-local-pin-volume-v2-locator-20260902/frontier-submission|results/datasheet-six-node-local-pin-volume-v2-locator-20260902/frontier-prepared|results/datasheet-structural-pin-volume-v2-locator-20260902.json|results/datasheet-page-evidence-20260901|results/datasheet-six-node-local-pin-volume-v2-locator-20260902/frontier-candidates/pillar-evidence.jsonl|results/datasheet-corpus-registry-20260901.json"
"v3pin|results/datasheet-six-node-local-pin-volume-v3-overflow-20260902/frontier-submission|results/datasheet-six-node-local-pin-volume-v3-overflow-20260902/frontier-prepared|results/datasheet-structural-pin-volume-v3-overflow-20260902.json|results/datasheet-page-evidence-20260901|results/datasheet-six-node-local-pin-volume-v3-overflow-20260902/frontier-candidates/pillar-evidence.jsonl|results/datasheet-corpus-registry-20260901.json"
"mcu|results/datasheet-six-node-mcu-v1-20260903/frontier-submission|results/datasheet-six-node-mcu-v1-20260903/frontier-prepared|results/datasheet-mcu-factory-v1b-20260903/structural-queue.json|results/datasheet-mcu-factory-v1b-20260903/page-evidence|results/datasheet-six-node-mcu-v1-20260903/frontier-candidates/pillar-evidence.jsonl|results/datasheet-mcu-factory-v1b-20260903/corpus-registry.json"
"tipar|results/datasheet-ti-power-factory-v1-20260902/frontier-submission-parametric|results/datasheet-ti-power-factory-v1-20260902/frontier-prepared-parametric|results/datasheet-ti-power-factory-v1-20260902/structural-parametric-vision-v2.json|results/datasheet-ti-power-factory-v1-20260902/page-evidence|results/datasheet-ti-power-factory-v1-20260902/frontier-candidates-parametric/pillar-evidence.jsonl|results/datasheet-ti-power-factory-v1-20260902/corpus-registry.json"
"tisum|results/datasheet-ti-power-summary-local-20260903/frontier-submission|results/datasheet-ti-power-summary-local-20260903/frontier-prepared|results/datasheet-ti-power-factory-v1-20260902/structural-summary-opn-v1.json|results/datasheet-ti-power-factory-v1-20260902/page-evidence|results/datasheet-ti-power-summary-local-20260903/frontier-candidates/pillar-evidence.jsonl|results/datasheet-ti-power-factory-v1-20260902/corpus-registry.json"
"mcupin3|results/datasheet-mcu-pin-v3-local-20260903/frontier-submission|results/datasheet-mcu-pin-v3-local-20260903/frontier-prepared|results/datasheet-mcu-structural-pin-v3-receipts-20260903.json|results/datasheet-mcu-factory-v1b-20260903/page-evidence|results/datasheet-mcu-pin-v3-local-20260903/frontier-candidates/pillar-evidence.jsonl|results/datasheet-mcu-factory-v1b-20260903/corpus-registry.json"
"tipin|results/datasheet-ti-power-pin-v1-local-20260905/frontier-submission|results/datasheet-ti-power-pin-v1-local-20260905/frontier-prepared|results/datasheet-ti-power-structural-pin-v1-20260905.json|results/datasheet-ti-power-page-evidence-pin-v3-20260905|results/datasheet-ti-power-pin-v1-local-20260905/frontier-candidates/pillar-evidence.jsonl|results/datasheet-ti-power-factory-v1-20260902/corpus-registry.json"
"mcupin4|results/datasheet-mcu-pin-v4-local-20260905/frontier-submission|results/datasheet-mcu-pin-v4-local-20260905/frontier-prepared|results/datasheet-mcu-structural-pin-v4-borderless-20260905.json|results/datasheet-mcu-page-evidence-v4-borderless-20260905|results/datasheet-mcu-pin-v4-local-20260905/frontier-candidates/pillar-evidence.jsonl|results/datasheet-mcu-factory-v1b-20260903/corpus-registry.json"
"psv|results/datasheet-passives-factory-v1-20260905/frontier-submission|results/datasheet-passives-factory-v1-20260905/frontier-prepared|results/datasheet-passives-factory-v1-20260905/structural-queue.json|results/datasheet-passives-factory-v1-20260905/page-evidence|results/datasheet-passives-factory-v1-20260905/frontier-candidates/pillar-evidence.jsonl|results/datasheet-passives-factory-v1-20260905/corpus-registry.json"
)

all_ended() {
  local sd="$1" tmp
  # The batch CLI refuses to overwrite existing files, and mktemp creates
  # the file it names; ask for an unused name instead.
  tmp="$(mktemp -u)"
  uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" \
    status --state-directory "$ROOT/$sd" --output "$tmp" >/dev/null 2>&1 || {
    rm -f "$tmp"
    return 1
  }
  python3 - "$tmp" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
batches = d.get("batches") or []
ok = bool(batches) and all(
    b.get("processing_status") == "ended" for b in batches
)
raise SystemExit(0 if ok else 1)
EOF
  local rc=$?
  rm -f "$tmp"
  return $rc
}

harvest() {
  local name="$1" sd="$2" prep="$3" queue="$4" ev="$5" pillar="$6" reg="$7"
  local run_dir marker
  run_dir="$ROOT/$(dirname "$sd")"
  marker="$run_dir/harvest-$name.done"
  [ -f "$marker" ] && return 0
  all_ended "$sd" || return 1
  echo "harvesting $name"
  uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" retrieve \
    --state-directory "$ROOT/$sd" \
    --output-directory "$run_dir/frontier-results-$name" \
    >"$run_dir/harvest-$name.log" 2>&1 || return 1
  uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" reconcile \
    --bundle "$ROOT/$prep" \
    --results-directory "$run_dir/frontier-results-$name" \
    --input-price-per-million 3.0 --output-price-per-million 15.0 \
    --batch-discount 0.5 \
    --output "$run_dir/frontier-reconciled-$name.json" \
    >>"$run_dir/harvest-$name.log" 2>&1 || return 1
  uv run --python 3.11 python "$ROOT/scripts/verify_datasheet_frontier_teachers.py" \
    --bundle "$ROOT/$prep" \
    --reconciliation "$run_dir/frontier-reconciled-$name.json" \
    --work-queue "$ROOT/$queue" \
    --page-evidence "$ROOT/$ev" \
    --pillar-evidence "$ROOT/$pillar" \
    --corpus-registry "$ROOT/$reg" \
    --ground-truth-root "$GT" \
    --verifications-output "$run_dir/teacher-verifications-$name.jsonl" \
    --claims-output "$run_dir/teacher-claims-$name.jsonl" \
    >>"$run_dir/harvest-$name.log" 2>&1 || return 1
  uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" finalize \
    --bundle "$ROOT/$prep" \
    --reconciliation "$run_dir/frontier-reconciled-$name.json" \
    --verifications "$run_dir/teacher-verifications-$name.jsonl" \
    --local-results "$(dirname "$ROOT/$pillar")/local-results.jsonl" \
    --output-directory "$run_dir/training-pairs-$name" \
    >>"$run_dir/harvest-$name.log" 2>&1 || return 1
  date > "$marker"
  echo "HARVESTED_$name pairs=$(wc -l < "$run_dir/training-pairs-$name/training-pairs.jsonl" 2>/dev/null || echo 0)"
}

while :; do
  remaining=0
  for spec in "${RUNS[@]}"; do
    IFS='|' read -r name sd prep queue ev pillar reg <<<"$spec"
    harvest "$name" "$sd" "$prep" "$queue" "$ev" "$pillar" "$reg" || remaining=$((remaining + 1))
  done
  [ "$remaining" = 0 ] && break
  sleep 900
done
echo all-teacher-batches-harvested
