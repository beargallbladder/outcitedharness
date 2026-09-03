#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS="${RESULTS:-$ROOT/results}"
TAG="${TAG:-bf16-candidate-v1-20260901}"
RUN_DIR="${RUN_DIR:-$RESULTS/datasheet-frozen-qwen3vl30b-$TAG}"
PAGE_EVIDENCE="${PAGE_EVIDENCE:-$RESULTS/datasheet-page-evidence-20260901}"
PIN_COHORT="${PIN_COHORT:-$RESULTS/datasheet-extraction-frozen-cohort-v6-20260901}"
PARAMETRIC_COHORT="${PARAMETRIC_COHORT:-$RESULTS/datasheet-parametric-frozen-cohort-v2-20260901}"
BASE_MODEL="${BASE_MODEL:-qwen3-vl-30b-bf16-base}"
CANDIDATE_MODEL="${CANDIDATE_MODEL:-qwen3-vl-30b-bf16-candidate-v1}"

BASE_DGX3="${BASE_DGX3:-http://192.168.4.49:8912/v1}"
BASE_ASUS1="${BASE_ASUS1:-http://192.168.4.58:8912/v1}"
BASE_ASUS3="${BASE_ASUS3:-http://192.168.4.32:8912/v1}"
CANDIDATE_DGX2="${CANDIDATE_DGX2:-http://192.168.4.45:8912/v1}"
CANDIDATE_ASUS2="${CANDIDATE_ASUS2:-http://192.168.4.39:8912/v1}"
CANDIDATE_ASUS4="${CANDIDATE_ASUS4:-http://192.168.4.56:8912/v1}"

if [[ -e "$RUN_DIR" || -L "$RUN_DIR" ]]; then
  echo "Immutable evaluation run already exists: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pin" "$RUN_DIR/parametric"

wait_for_endpoint() {
  local base_url="$1"
  local expected_model="$2"
  local deadline=$((SECONDS + 1200))
  local payload
  while ((SECONDS < deadline)); do
    if payload="$(curl -fsS --max-time 5 "$base_url/models" 2>/dev/null)" \
      && [[ "$payload" == *"$expected_model"* ]]; then
      printf 'ready %s %s\n' "$base_url" "$expected_model"
      return 0
    fi
    sleep 10
  done
  printf 'timed out waiting for %s (%s)\n' "$base_url" "$expected_model" >&2
  return 1
}

wait_for_endpoint "$BASE_DGX3" "$BASE_MODEL" &
readiness_pids=("$!")
wait_for_endpoint "$BASE_ASUS1" "$BASE_MODEL" &
readiness_pids+=("$!")
wait_for_endpoint "$BASE_ASUS3" "$BASE_MODEL" &
readiness_pids+=("$!")
wait_for_endpoint "$CANDIDATE_DGX2" "$CANDIDATE_MODEL" &
readiness_pids+=("$!")
wait_for_endpoint "$CANDIDATE_ASUS2" "$CANDIDATE_MODEL" &
readiness_pids+=("$!")
wait_for_endpoint "$CANDIDATE_ASUS4" "$CANDIDATE_MODEL" &
readiness_pids+=("$!")
for pid in "${readiness_pids[@]}"; do
  wait "$pid"
done

run_shard() {
  local lane="$1"
  local variant="$2"
  local node="$3"
  local base_url="$4"
  local model="$5"
  local offset="$6"
  local limit="$7"
  local cohort="$8"
  local output="$RUN_DIR/$lane/$variant-$node"
  uv run --python 3.11 --extra vision python \
    "$ROOT/scripts/run_datasheet_structural_extraction.py" \
    --structural-queue "$cohort/work-queue.json" \
    --page-evidence "$PAGE_EVIDENCE" \
    --base-url "$base_url" \
    --model "$model" \
    --offset "$offset" \
    --limit "$limit" \
    --render-dpi 220 \
    --timeout-seconds 900 \
    --vision-policy always \
    --output-directory "$output" \
    >"$RUN_DIR/logs/$lane-$variant-$node.log" 2>&1
}

wait_for_shards() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if ((failed)); then
    echo "One or more extraction shards failed; inspect $RUN_DIR/logs" >&2
    return 1
  fi
}

run_lane() {
  local lane="$1"
  local cohort="$2"
  local first="$3"
  local second="$4"
  local third="$5"
  local expected="$6"

  run_shard "$lane" base dgx3 "$BASE_DGX3" "$BASE_MODEL" 0 "$first" "$cohort" &
  shard_pids=("$!")
  run_shard "$lane" base asus1 "$BASE_ASUS1" "$BASE_MODEL" "$first" "$second" "$cohort" &
  shard_pids+=("$!")
  run_shard "$lane" base asus3 "$BASE_ASUS3" "$BASE_MODEL" "$((first + second))" "$third" "$cohort" &
  shard_pids+=("$!")
  run_shard "$lane" candidate dgx2 "$CANDIDATE_DGX2" "$CANDIDATE_MODEL" 0 "$first" "$cohort" &
  shard_pids+=("$!")
  run_shard "$lane" candidate asus2 "$CANDIDATE_ASUS2" "$CANDIDATE_MODEL" "$first" "$second" "$cohort" &
  shard_pids+=("$!")
  run_shard "$lane" candidate asus4 "$CANDIDATE_ASUS4" "$CANDIDATE_MODEL" "$((first + second))" "$third" "$cohort" &
  shard_pids+=("$!")
  wait_for_shards "${shard_pids[@]}"

  local variant
  local nodes
  for variant in base candidate; do
    if [[ "$variant" == "base" ]]; then
      nodes=(dgx3 asus1 asus3)
    else
      nodes=(dgx2 asus2 asus4)
    fi
    uv run --python 3.11 python "$ROOT/scripts/merge_datasheet_local_bundles.py" \
      --input "$RUN_DIR/$lane/$variant-${nodes[0]}" \
      --input "$RUN_DIR/$lane/$variant-${nodes[1]}" \
      --input "$RUN_DIR/$lane/$variant-${nodes[2]}" \
      --expected-items "$expected" \
      --output "$RUN_DIR/$lane/$variant-merged" \
      >"$RUN_DIR/logs/$lane-$variant-merge.log"
    uv run --python 3.11 python "$ROOT/scripts/evaluate_datasheet_extraction.py" \
      --cohort "$cohort" \
      --local-bundle "$RUN_DIR/$lane/$variant-merged" \
      --output "$RUN_DIR/$lane-$variant-evaluation.json" \
      >"$RUN_DIR/logs/$lane-$variant-evaluation.log"
  done
}

cohort_size() {
  python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["counts"]["selected"])' \
    "$1/manifest.json"
}

run_balanced_lane() {
  local lane="$1"
  local cohort="$2"
  local expected
  local first
  local second
  local third
  expected="$(cohort_size "$cohort")"
  first="$(((expected + 2) / 3))"
  second="$(((expected + 1) / 3))"
  third="$((expected - first - second))"
  run_lane "$lane" "$cohort" "$first" "$second" "$third" "$expected"
}

run_balanced_lane pin "$PIN_COHORT"
run_balanced_lane parametric "$PARAMETRIC_COHORT"

decision_status=0
uv run --python 3.11 python \
  "$ROOT/scripts/compare_datasheet_extraction_evaluations.py" \
  --pin-base "$RUN_DIR/pin-base-evaluation.json" \
  --pin-candidate "$RUN_DIR/pin-candidate-evaluation.json" \
  --parametric-base "$RUN_DIR/parametric-base-evaluation.json" \
  --parametric-candidate "$RUN_DIR/parametric-candidate-evaluation.json" \
  --output "$RUN_DIR/candidate-decision.json" \
  >"$RUN_DIR/logs/candidate-decision.log" || decision_status=$?
if ((decision_status > 1)); then
  echo "Candidate decision could not be sealed" >&2
  exit "$decision_status"
fi

printf 'completed %s\n' "$RUN_DIR"
