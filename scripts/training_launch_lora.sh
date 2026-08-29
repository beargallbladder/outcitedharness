#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: training_launch_lora.sh --root DGX2_ROOT --config FILE [options]

Options:
  --image IMAGE    Qualified local LLaMA Factory image.
                   Default: harness/llamafactory-gb10:20260829
  --launch         Start training (default is validation/plan only).

The container has no network, receives no credentials, and mounts only the
ownership-marked DGX2 training root at /training.
EOF
}

die() {
  printf 'LoRA training: %s\n' "$*" >&2
  exit 2
}

root=""
config=""
image="harness/llamafactory-gb10:20260829"
launch=false

while (( $# )); do
  case "$1" in
    --root) [[ $# -ge 2 ]] || die "--root needs a value"; root="$2"; shift 2 ;;
    --config) [[ $# -ge 2 ]] || die "--config needs a value"; config="$2"; shift 2 ;;
    --image) [[ $# -ge 2 ]] || die "--image needs a value"; image="$2"; shift 2 ;;
    --launch) launch=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$root" =~ ^/[A-Za-z0-9._/-]+$ && "$root" != "/" ]] || die "invalid --root"
[[ "$config" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "invalid --config"
[[ "$image" =~ ^[A-Za-z0-9._/@:-]+$ ]] || die "invalid --image"

if [[ "$launch" != true ]]; then
  printf 'PLAN only: offline LoRA on DGX2 image=%s config=%s\n' "$image" "$config"
  printf '%s\n' 'Re-run with --launch after image, model, dataset, and config validation.'
  exit 0
fi

[[ "$(hostname -s | tr '[:upper:]' '[:lower:]')" == "spark-49af" ]] ||
  die "launch is restricted to DGX2 (spark-49af)"
[[ -f "$root/.harness-training-owner-v1" ]] || die "training root is not marked"
grep -qx 'role=dgx2' "$root/.harness-training-owner-v1" ||
  die "training root marker is not DGX2"
[[ -f "$config" && ! -L "$config" ]] || die "config must be a regular file"
root_real="$(realpath "$root")"
config_real="$(realpath "$config")"
[[ "$config_real" == "$root_real/configs/"* ]] ||
  die "config must be below DGX2_ROOT/configs"
docker image inspect "$image" >/dev/null 2>&1 || die "training image is unavailable"
[[ -s "$root/models/Qwen3-8B/config.json" ]] || die "Qwen3-8B is incomplete"
[[ -s "$root/datasets/designwins-v2-20260829/artifact.sha256.json" ]] ||
  die "DesignWins dataset is incomplete"

container_config="/training/configs/${config_real#"$root_real/configs/"}"
exec docker run --rm \
  --name harness-lora-designwins-pilot \
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
