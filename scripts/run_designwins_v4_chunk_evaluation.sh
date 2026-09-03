#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_designwins_v4_chunk_evaluation.sh --root ROOT --container NAME \
  --chunk-output-relative PATH --aggregate-output-relative PATH \
  --sealed-output-relative PATH [--evaluation-set full|canary] \
  [--adapter-relative PATH --adapter-manifest-relative PATH]
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

root=""
container=""
chunk_output_relative=""
aggregate_output_relative=""
sealed_output_relative=""
adapter_relative=""
adapter_manifest_relative=""
evaluation_set="full"
runtime_ref="nvcr.io/nvidia/vllm:26.05.post1-py3"

while (( $# )); do
  case "$1" in
    --root) root="${2:-}"; shift 2 ;;
    --container) container="${2:-}"; shift 2 ;;
    --chunk-output-relative) chunk_output_relative="${2:-}"; shift 2 ;;
    --aggregate-output-relative) aggregate_output_relative="${2:-}"; shift 2 ;;
    --sealed-output-relative) sealed_output_relative="${2:-}"; shift 2 ;;
    --adapter-relative) adapter_relative="${2:-}"; shift 2 ;;
    --adapter-manifest-relative) adapter_manifest_relative="${2:-}"; shift 2 ;;
    --evaluation-set) evaluation_set="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$root" =~ ^/[A-Za-z0-9._/-]+$ && "$root" != "/" ]] || die "invalid root"
[[ "$container" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] || die "invalid container"
for relative in \
  "$chunk_output_relative" \
  "$aggregate_output_relative" \
  "$sealed_output_relative"; do
  [[ "$relative" =~ ^evaluations/[A-Za-z0-9._/-]+\.json$ ]] ||
    die "invalid evaluation output path"
done
if [[ -n "$adapter_relative" || -n "$adapter_manifest_relative" ]]; then
  [[ "$adapter_relative" =~ ^checkpoints/[A-Za-z0-9._/-]+$ ]] ||
    die "invalid adapter path"
  [[ "$adapter_manifest_relative" =~ ^manifests/[A-Za-z0-9._/-]+\.json$ ]] ||
    die "invalid adapter manifest"
fi

root="$(realpath "$root")"
test -f "$root/.harness-training-owner-v1"
case "$evaluation_set" in
  full)
    chunk_root="$root/evaluations/frozen-designwins-v4-test-chunks"
    chunk_dataset="$chunk_root/llamafactory/designwins_text_test.json"
    canonical_chunks="$chunk_root/canonical/test.jsonl"
    parent_dataset="$root/datasets/designwins-v3-20260829/canonical/text/test.jsonl"
    chunk_manifest="$root/manifests/frozen-designwins-v4-test-chunks.sha256.json"
    chunk_samples=676
    parent_samples=141
    ;;
  canary)
    chunk_root="$root/evaluations/frozen-designwins-v4-canary-8"
    chunk_dataset="$chunk_root/llamafactory.json"
    canonical_chunks="$chunk_root/canonical.jsonl"
    parent_dataset="$chunk_root/parents.jsonl"
    chunk_manifest="$root/manifests/frozen-designwins-v4-canary-8.sha256.json"
    chunk_samples=19
    parent_samples=8
    ;;
  *) die "invalid evaluation set" ;;
esac
model_manifest="$root/manifests/qwen3-8b-model-20260830.sha256.json"
model="$root/models/Qwen3-8B"
for path in \
  "$chunk_dataset" \
  "$canonical_chunks" \
  "$parent_dataset" \
  "$chunk_manifest" \
  "$model_manifest" \
  "$root/scripts/evaluate_designwins_vllm.py" \
  "$root/scripts/evaluate_designwins_text.py" \
  "$root/scripts/aggregate_designwins_chunk_evaluation.py" \
  "$root/scripts/seal_designwins_evaluation.py" \
  "$root/scripts/training_manifest.py"; do
  test -f "$path" && test ! -L "$path"
done
python3 "$root/scripts/training_manifest.py" verify "$chunk_root" "$chunk_manifest"
python3 "$root/scripts/training_manifest.py" verify "$model" "$model_manifest"

chunk_output="$root/$chunk_output_relative"
aggregate_output="$root/$aggregate_output_relative"
sealed_output="$root/$sealed_output_relative"
test ! -e "$chunk_output"
test ! -e "$aggregate_output"
test ! -e "$sealed_output"
runtime_image_id="$(docker image inspect --format '{{.Id}}' "$runtime_ref")"
[[ "$runtime_image_id" =~ ^sha256:[0-9a-f]{64}$ ]] ||
  die "invalid vLLM runtime image"

docker_args=(
  run --rm
  --name "$container"
  --entrypoint python
  --gpus all
  --network none
  --ipc=host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn
  -v "$HOME/.cache/vllm-runtime:/tmp/.cache"
  -v "$model:/training/model:ro"
  -v "$chunk_dataset:/training/test.json:ro"
  -v "$root/scripts/evaluate_designwins_vllm.py:/training/evaluate_designwins_vllm.py:ro"
  -v "$root/scripts/evaluate_designwins_text.py:/training/evaluate_designwins_text.py:ro"
  -v "$root/evaluations:/output"
)
evaluate_args=(
  /training/evaluate_designwins_vllm.py
  --model /training/model
  --dataset /training/test.json
  --output "/output/${chunk_output_relative#evaluations/}"
  --max-samples "$chunk_samples"
  --cutoff-len 4096
  --max-new-tokens 2048
  --batch-size 16
  --generation-slack-tokens 256
  --gpu-memory-utilization 0.8
)
seal_args=()
if [[ -n "$adapter_relative" ]]; then
  adapter="$root/$adapter_relative"
  adapter_manifest="$root/$adapter_manifest_relative"
  test -d "$adapter" && test ! -L "$adapter"
  test -f "$adapter_manifest" && test ! -L "$adapter_manifest"
  python3 "$root/scripts/training_manifest.py" verify "$adapter" "$adapter_manifest"
  docker_args+=(-v "$adapter:/training/adapter:ro")
  evaluate_args+=(--adapter /training/adapter)
  seal_args+=(--adapter-manifest "$adapter_manifest")
fi

docker "${docker_args[@]}" "$runtime_image_id" "${evaluate_args[@]}"
python3 "$root/scripts/aggregate_designwins_chunk_evaluation.py" \
  --raw-evaluation "$chunk_output" \
  --chunks "$canonical_chunks" \
  --parents "$parent_dataset" \
  --generation-scorer "$root/scripts/evaluate_designwins_vllm.py" \
  --chunk-artifact-manifest "$chunk_manifest" \
  --output "$aggregate_output"
python3 "$root/scripts/seal_designwins_evaluation.py" \
  --source "$aggregate_output" \
  --output "$sealed_output" \
  --dataset "$canonical_chunks" \
  --model-manifest "$model_manifest" \
  --scorer "$root/scripts/aggregate_designwins_chunk_evaluation.py" \
  --runtime-image-id "$runtime_image_id" \
  "${seal_args[@]}" \
  --max-samples "$parent_samples" \
  --cutoff-len 4096 \
  --max-new-tokens 2048 \
  --batch-size 16 \
  --generation-slack-tokens 256
printf 'DESIGNWINS_V4_CHUNK_EVALUATION_COMPLETE output=%s\n' "$sealed_output"
