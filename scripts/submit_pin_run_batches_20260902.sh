#!/usr/bin/env bash
# One-shot watcher: submit sealed pin teacher candidates to Anthropic Batch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="claude-sonnet-5"
INPUT_PRICE="3.0"
OUTPUT_PRICE="15.0"
DISCOUNT="0.5"

submit_run() {
  local run_dir="$1"
  local bundle_root="$2"
  local spend_cap="$3"
  local candidates="$run_dir/frontier-candidates"
  local deadline=$((SECONDS + 21600))
  while [[ ! -f "$candidates/candidates.jsonl" ]]; do
    ((SECONDS < deadline)) || {
      echo "timed out waiting for $candidates" >&2
      return 1
    }
    sleep 30
  done
  local prepared="$run_dir/frontier-prepared"
  if [[ ! -f "$prepared/manifest.json" ]]; then
    uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" \
      prepare \
      --candidates "$candidates/candidates.jsonl" \
      --allowed-root "$bundle_root" \
      --model "$MODEL" \
      --input-price-per-million "$INPUT_PRICE" \
      --output-price-per-million "$OUTPUT_PRICE" \
      --batch-discount "$DISCOUNT" \
      --spend-cap-usd "$spend_cap" \
      --output "$prepared"
  fi
  uv run --python 3.11 python "$ROOT/scripts/datasheet_frontier_batch.py" \
    submit \
    --bundle "$prepared" \
    --state-directory "$run_dir/frontier-submission" \
    --approved-spend-cap-usd "$spend_cap" \
    --resume
  echo "submitted $run_dir"
}

submit_run \
  "$ROOT/results/datasheet-six-node-local-cr586-pin-v2-20260902" \
  "$ROOT/results/datasheet-six-node-local-cr586-pin-v2-20260902" \
  40
submit_run \
  "$ROOT/results/datasheet-six-node-local-pin-volume-v1-20260902" \
  "$ROOT/results/datasheet-six-node-local-pin-volume-v1-20260902" \
  30
