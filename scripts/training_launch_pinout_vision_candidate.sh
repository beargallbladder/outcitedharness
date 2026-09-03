#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: training_launch_pinout_vision_candidate.sh --root DGX2_ROOT
  --config FILE --receipt-output FILE [--image IMAGE] [--launch]

Runs exactly one 1,101-example Qwen3-VL pinout candidate on DGX2. The container
is offline and the resulting adapter is not promoted.
EOF
}

die() {
  printf 'pinout vision candidate: %s\n' "$*" >&2
  exit 2
}

root=""
config=""
receipt=""
image="sha256:728b352622d569b1343cb384dd8aa394917203d02631322b02d302d068e38139"
launch=false

while (( $# )); do
  case "$1" in
    --root) [[ $# -ge 2 ]] || die "--root needs a value"; root="$2"; shift 2 ;;
    --config) [[ $# -ge 2 ]] || die "--config needs a value"; config="$2"; shift 2 ;;
    --receipt-output) [[ $# -ge 2 ]] || die "--receipt-output needs a value"; receipt="$2"; shift 2 ;;
    --image) [[ $# -ge 2 ]] || die "--image needs a value"; image="$2"; shift 2 ;;
    --launch) launch=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for value in "$root" "$config" "$receipt"; do
  [[ "$value" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "invalid absolute path"
done
[[ "$root" != "/" ]] || die "training root cannot be /"
[[ "$image" =~ ^sha256:[a-f0-9]{64}$ ]] || die "image must be a SHA-256 image ID"

if [[ "$launch" != true ]]; then
  printf 'PLAN only: offline 1,101-example Qwen3-VL pinout candidate on DGX2\n'
  printf 'config=%s image=%s receipt=%s\n' "$config" "$image" "$receipt"
  exit 0
fi

[[ "$(hostname -s | tr '[:upper:]' '[:lower:]')" == "spark-49af" ]] ||
  die "launch is restricted to DGX2 (spark-49af)"
[[ -f "$root/.harness-training-owner-v1" && ! -L "$root/.harness-training-owner-v1" ]] ||
  die "training root is not ownership-marked"
grep -qx 'role=dgx2' "$root/.harness-training-owner-v1" ||
  die "training root marker is not DGX2"
[[ -f "$config" && ! -L "$config" ]] || die "config is missing or unsafe"
[[ ! -e "$receipt" && ! -L "$receipt" ]] ||
  die "immutable preflight receipt already exists"
root_real="$(realpath "$root")"
config_real="$(realpath "$config")"
[[ "$config_real" == "$root_real/configs/"* ]] ||
  die "config must be below DGX2 configs"
docker image inspect "$image" >/dev/null 2>&1 ||
  die "qualified training image is unavailable"
preflight="$root_real/scripts/verify_pinout_vision_training_preflight_v2.py"
[[ -f "$preflight" && ! -L "$preflight" ]] ||
  die "candidate preflight v2 is not installed"
python3 "$preflight" \
  --root "$root_real" \
  --config "$config_real" \
  --mode candidate \
  --image "$image" \
  --receipt-output "$receipt" ||
  die "candidate preflight failed"

container_config="/training/configs/${config_real#"$root_real/configs/"}"
run_log="$root_real/runs/pinout-vision-qwen3-vl-8b-candidate-v1.log"
[[ ! -e "$run_log" && ! -L "$run_log" ]] ||
  die "immutable candidate log already exists"

set +e
docker run --rm \
  --name pinout-vision-qwen3-vl-8b-candidate-v1 \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env TOKENIZERS_PARALLELISM=false \
  --network none \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount "type=bind,src=$root_real,dst=/training" \
  --workdir /training \
  "$image" train "$container_config" 2>&1 | tee "$run_log"
status="${PIPESTATUS[0]}"
set -e
chmod 0444 "$run_log"
exit "$status"
