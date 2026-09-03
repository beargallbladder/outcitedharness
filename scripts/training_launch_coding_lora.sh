#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: training_launch_coding_lora.sh --root DGX2_ROOT --config FILE
  --sequence-audit FILE --database FILE --dataset-version-id ID
  [--image IMAGE] [--launch]

Runs a network-denied Qwen3-8B LoRA pipeline qualification on DGX2. This does
not promote or modify the production Qwen3-Coder-Next endpoint.
EOF
}

die() {
  printf 'coding LoRA: %s\n' "$*" >&2
  exit 2
}

root=""
config=""
sequence_audit=""
database=""
dataset_version_id=""
image="harness/llamafactory-gb10:20260829"
launch=false

while (( $# )); do
  case "$1" in
    --root) root="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --sequence-audit) sequence_audit="$2"; shift 2 ;;
    --database) database="$2"; shift 2 ;;
    --dataset-version-id) dataset_version_id="$2"; shift 2 ;;
    --image) image="$2"; shift 2 ;;
    --launch) launch=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$root" =~ ^/[A-Za-z0-9._/-]+$ && "$root" != "/" ]] ||
  die "invalid --root"
for value in "$config" "$sequence_audit" "$database"; do
  [[ "$value" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "invalid input path"
done
[[ "$dataset_version_id" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "invalid dataset version"
[[ "$image" =~ ^[A-Za-z0-9._/@:-]+$ ]] || die "invalid image"

if [[ "$launch" != true ]]; then
  printf 'PLAN only: offline coding LoRA on DGX2 config=%s\n' "$config"
  exit 0
fi

[[ "$(hostname -s | tr '[:upper:]' '[:lower:]')" == "spark-49af" ]] ||
  die "launch is restricted to DGX2"
[[ -f "$root/.harness-training-owner-v1" && ! -L "$root/.harness-training-owner-v1" ]] ||
  die "training root is not ownership-marked"
grep -qx 'role=dgx2' "$root/.harness-training-owner-v1" ||
  die "training root marker is not DGX2"
for value in "$config" "$sequence_audit" "$database"; do
  [[ -f "$value" && ! -L "$value" ]] || die "required input is missing or unsafe"
done
root_real="$(realpath "$root")"
config_real="$(realpath "$config")"
[[ "$config_real" == "$root_real/configs/"* ]] ||
  die "config must be below DGX2 configs"
[[ -s "$root_real/models/Qwen3-8B/config.json" ]] ||
  die "Qwen3-8B checkpoint is incomplete"
docker image inspect "$image" >/dev/null 2>&1 ||
  die "qualified training image is unavailable"
preflight="$root_real/scripts/verify_coding_training_preflight.py"
[[ -f "$preflight" && ! -L "$preflight" ]] ||
  die "coding training preflight is not installed"
python3 "$preflight" \
  --root "$root_real" \
  --config "$config_real" \
  --sequence-audit "$sequence_audit" \
  --database "$database" \
  --dataset-version-id "$dataset_version_id" ||
  die "coding training preflight failed"

container_config="/training/configs/${config_real#"$root_real/configs/"}"
exec docker run --rm \
  --name cursor-shadow-code-qwen3-8b-smoke-v1 \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --network none \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount "type=bind,src=$root_real,dst=/training" \
  --workdir /training \
  "$image" train "$container_config"
