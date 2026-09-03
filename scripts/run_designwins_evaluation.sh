#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_designwins_evaluation.sh --root ROOT --output-relative PATH
    --container-name NAME [--adapter-relative PATH]
    [--image IMAGE] [--max-new-tokens N] [--batch-size N]
    [--generation-slack-tokens N]

Run the complete frozen DesignWins text holdout without network access.
The output path must be new and remain below ROOT/evaluations/.
EOF
}

root=""
output_relative=""
container_name=""
adapter_relative=""
image="sha256:728b352622d569b1343cb384dd8aa394917203d02631322b02d302d068e38139"
max_new_tokens=8192
batch_size=8
generation_slack_tokens=256

while (($#)); do
  case "$1" in
    --root) root="${2:-}"; shift 2 ;;
    --output-relative) output_relative="${2:-}"; shift 2 ;;
    --container-name) container_name="${2:-}"; shift 2 ;;
    --adapter-relative) adapter_relative="${2:-}"; shift 2 ;;
    --image) image="${2:-}"; shift 2 ;;
    --max-new-tokens) max_new_tokens="${2:-}"; shift 2 ;;
    --batch-size) batch_size="${2:-}"; shift 2 ;;
    --generation-slack-tokens) generation_slack_tokens="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$root" && -n "$output_relative" && -n "$container_name" ]] || {
  usage >&2
  exit 2
}
[[ "$output_relative" == evaluations/* ]] ||
  { echo "output must be below evaluations/" >&2; exit 2; }
[[ "$output_relative" != *".."* && "$adapter_relative" != *".."* ]] ||
  { echo "relative paths cannot contain '..'" >&2; exit 2; }
[[ "$max_new_tokens" =~ ^[0-9]+$ ]] && ((max_new_tokens >= 7398)) ||
  { echo "max-new-tokens must be at least 7398" >&2; exit 2; }
[[ "$batch_size" =~ ^[0-9]+$ ]] && ((batch_size >= 1 && batch_size <= 16)) ||
  { echo "batch-size must be between 1 and 16" >&2; exit 2; }
[[ "$generation_slack_tokens" =~ ^[0-9]+$ ]] &&
  ((generation_slack_tokens >= 32 && generation_slack_tokens <= 1024)) ||
  { echo "generation-slack-tokens must be between 32 and 1024" >&2; exit 2; }
[[ "$container_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]+$ ]] ||
  { echo "invalid container name" >&2; exit 2; }

root="$(cd "$root" && pwd -P)"
test -f "$root/.harness-training-owner-v1"
test -f "$root/models/Qwen3-8B/config.json"
test -f "$root/datasets/designwins-v3-20260829/llamafactory/designwins_text_test.json"
test -f "$root/scripts/evaluate_designwins_text.py"
output="$root/$output_relative"
test ! -e "$output"
output_dir="$(dirname "$output")"
output_name="$(basename "$output")"
mkdir -p "$output_dir"
docker image inspect "$image" >/dev/null

adapter_mount=()
adapter_args=()
if [[ -n "$adapter_relative" ]]; then
  [[ "$adapter_relative" == checkpoints/* ]] ||
    { echo "adapter must be below checkpoints/" >&2; exit 2; }
  test -s "$root/$adapter_relative/adapter_model.safetensors"
  adapter_mount=(-v "$root/$adapter_relative:/training/adapter:ro")
  adapter_args=(--adapter /training/adapter)
fi

log="${output%.json}.log"
docker run --rm \
  --name "$container_name" \
  --entrypoint /bin/bash \
  --gpus all \
  --network none \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TOKENIZERS_PARALLELISM=false \
  -v "$root/models/Qwen3-8B:/training/model:ro" \
  -v "$root/datasets/designwins-v3-20260829/llamafactory/designwins_text_test.json:/training/test.json:ro" \
  -v "$root/scripts/evaluate_designwins_text.py:/training/evaluate.py:ro" \
  -v "$output_dir:/output" \
  "${adapter_mount[@]}" \
  "$image" \
  -lc 'python /training/evaluate.py \
    --model /training/model \
    --dataset /training/test.json \
    "$@"' \
  evaluate-designwins \
  "${adapter_args[@]}" \
  --max-samples 141 \
  --max-new-tokens "$max_new_tokens" \
  --batch-size "$batch_size" \
  --generation-slack-tokens "$generation_slack_tokens" \
  --output "/output/$output_name" \
  2>&1 | tee "$log"

test -s "$output"
