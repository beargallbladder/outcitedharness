#!/usr/bin/env bash
# Re-verify already-paid teacher batches under the per-fact summary salvage
# rule (2026-09-06). Zero frontier spend: consumes existing reconciliations.
# Writes v2 verifications/claims/training-pairs alongside the originals; the
# round-3 dataset builder must use v2 bundles INSTEAD of v1 for these runs.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GT=/Volumes/M5_4TB/DigiKey_Reference_Designs/claude_ground_truth

# name|run_dir|prepared|queue|page_evidence|pillar_evidence|registry
RUNS=(
"mcu|results/datasheet-six-node-mcu-v1-20260903|results/datasheet-six-node-mcu-v1-20260903/frontier-prepared|results/datasheet-mcu-factory-v1b-20260903/structural-queue.json|results/datasheet-mcu-factory-v1b-20260903/page-evidence|results/datasheet-six-node-mcu-v1-20260903/frontier-candidates/pillar-evidence.jsonl|results/datasheet-mcu-factory-v1b-20260903/corpus-registry.json"
"tisum|results/datasheet-ti-power-summary-local-20260903|results/datasheet-ti-power-summary-local-20260903/frontier-prepared|results/datasheet-ti-power-factory-v1-20260902/structural-summary-opn-v1.json|results/datasheet-ti-power-factory-v1-20260902/page-evidence|results/datasheet-ti-power-summary-local-20260903/frontier-candidates/pillar-evidence.jsonl|results/datasheet-ti-power-factory-v1-20260902/corpus-registry.json"
"apps|results/datasheet-apps-summary-v1-local-20260905|results/datasheet-apps-summary-v1-local-20260905/frontier-prepared|results/datasheet-apps-structural-summary-v1-20260905.json|results/datasheet-apps-page-evidence-v1-20260905|results/datasheet-apps-summary-v1-local-20260905/frontier-candidates/pillar-evidence.jsonl|results/datasheet-apps-corpus-registry-v1-20260905.json"
"psv|results/datasheet-passives-factory-v1-20260905|results/datasheet-passives-factory-v1-20260905/frontier-prepared|results/datasheet-passives-factory-v1-20260905/structural-queue.json|results/datasheet-passives-factory-v1-20260905/page-evidence|results/datasheet-passives-factory-v1-20260905/frontier-candidates/pillar-evidence.jsonl|results/datasheet-passives-factory-v1-20260905/corpus-registry.json"
)

for spec in "${RUNS[@]}"; do
  IFS='|' read -r name rd prep queue ev pillar reg <<<"$spec"
  out_pairs="$ROOT/$rd/training-pairs-v2-$name"
  [ -d "$out_pairs" ] && { echo "SKIP_$name (v2 exists)"; continue; }
  uv run --python 3.11 python "$ROOT/scripts/verify_datasheet_frontier_teachers.py" \
    --bundle "$ROOT/$prep" \
    --reconciliation "$ROOT/$rd/frontier-reconciled-$name.json" \
    --work-queue "$ROOT/$queue" \
    --page-evidence "$ROOT/$ev" \
    --pillar-evidence "$ROOT/$pillar" \
    --corpus-registry "$ROOT/$reg" \
    --ground-truth-root "$GT" \
    --verifications-output "$ROOT/$rd/teacher-verifications-v2-$name.jsonl" \
    --claims-output "$ROOT/$rd/teacher-claims-v2-$name.jsonl" \
    >"$ROOT/$rd/reverify-v2-$name.log" 2>&1 || { echo "VERIFY_FAILED_$name"; tail -3 "$ROOT/$rd/reverify-v2-$name.log"; continue; }
  uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" finalize \
    --bundle "$ROOT/$prep" \
    --reconciliation "$ROOT/$rd/frontier-reconciled-$name.json" \
    --verifications "$ROOT/$rd/teacher-verifications-v2-$name.jsonl" \
    --local-results "$(dirname "$ROOT/$pillar")/local-results.jsonl" \
    --output-directory "$out_pairs" \
    >>"$ROOT/$rd/reverify-v2-$name.log" 2>&1 || { echo "FINALIZE_FAILED_$name"; tail -3 "$ROOT/$rd/reverify-v2-$name.log"; continue; }
  v1=$(wc -l < "$ROOT/$rd/training-pairs-$name/training-pairs.jsonl" 2>/dev/null || echo 0)
  v2=$(wc -l < "$out_pairs/training-pairs.jsonl" 2>/dev/null || echo 0)
  echo "REVERIFIED_$name v1_pairs=$v1 v2_pairs=$v2"
done
echo reverify-v2-complete
