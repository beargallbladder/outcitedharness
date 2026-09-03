#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: training_launch_lora.sh --root DGX2_ROOT --config FILE [options]

Options:
  --image IMAGE    Qualified local LLaMA Factory image.
                   Default: harness/llamafactory-gb10:20260829
  --sequence-audit FILE
                   Hash-bound sequence audit covering the configured cutoff.
  --database FILE  Durable training registry containing the dataset version.
  --dataset-version-id ID
                   Immutable eligible dataset version bound to the audit.
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
sequence_audit=""
database=""
dataset_version_id=""
launch=false

while (( $# )); do
  case "$1" in
    --root) [[ $# -ge 2 ]] || die "--root needs a value"; root="$2"; shift 2 ;;
    --config) [[ $# -ge 2 ]] || die "--config needs a value"; config="$2"; shift 2 ;;
    --image) [[ $# -ge 2 ]] || die "--image needs a value"; image="$2"; shift 2 ;;
    --sequence-audit) [[ $# -ge 2 ]] || die "--sequence-audit needs a value"; sequence_audit="$2"; shift 2 ;;
    --database) [[ $# -ge 2 ]] || die "--database needs a value"; database="$2"; shift 2 ;;
    --dataset-version-id) [[ $# -ge 2 ]] || die "--dataset-version-id needs a value"; dataset_version_id="$2"; shift 2 ;;
    --launch) launch=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$root" =~ ^/[A-Za-z0-9._/-]+$ && "$root" != "/" ]] || die "invalid --root"
[[ "$config" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "invalid --config"
[[ "$image" =~ ^[A-Za-z0-9._/@:-]+$ ]] || die "invalid --image"
[[ -z "$sequence_audit" || "$sequence_audit" =~ ^/[A-Za-z0-9._/-]+$ ]] ||
  die "invalid --sequence-audit"
[[ -z "$database" || "$database" =~ ^/[A-Za-z0-9._/-]+$ ]] ||
  die "invalid --database"
[[ -z "$dataset_version_id" || "$dataset_version_id" =~ ^[A-Za-z0-9._-]+$ ]] ||
  die "invalid --dataset-version-id"

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
[[ -n "$sequence_audit" ]] ||
  die "--sequence-audit is required for launch"
[[ -n "$database" ]] || die "--database is required for launch"
[[ -n "$dataset_version_id" ]] ||
  die "--dataset-version-id is required for launch"
[[ -f "$sequence_audit" && ! -L "$sequence_audit" ]] ||
  die "sequence audit must be a regular file"
[[ -f "$database" && ! -L "$database" ]] ||
  die "database must be a regular file"
root_real="$(realpath "$root")"
config_real="$(realpath "$config")"
audit_real="$(realpath "$sequence_audit")"
[[ "$config_real" == "$root_real/configs/"* ]] ||
  die "config must be below DGX2_ROOT/configs"
[[ "$audit_real" == "$root_real/"* ]] ||
  die "sequence audit must be below DGX2_ROOT"
docker image inspect "$image" >/dev/null 2>&1 || die "training image is unavailable"
[[ -s "$root/models/Qwen3-8B/config.json" ]] || die "Qwen3-8B is incomplete"
preflight="$root_real/scripts/verify_designwins_training_preflight.py"
[[ -f "$preflight" && ! -L "$preflight" ]] ||
  die "DesignWins sequence preflight is not installed"
python3 "$preflight" \
  --root "$root_real" \
  --config "$config_real" \
  --sequence-audit "$audit_real" \
  --database "$database" \
  --dataset-version-id "$dataset_version_id" ||
  die "DesignWins sequence preflight failed"

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
