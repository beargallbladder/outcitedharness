#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: training_launch_single.sh --root DGX2_ROOT --config FILE [options] [-- SPECFORGE_ARGS...]

Options:
  --specforge PATH       SpecForge CLI (default: ROOT/specforge-venv/bin/specforge).
  --launch               Start the job (default is validation/plan only).

Run this template on DGX2. It never contacts another host or changes services.
Credentials must be supplied through the process environment; they are not
printed or read from config templates by this script.
EOF
}

die() {
  printf 'single-node training: %s\n' "$*" >&2
  exit 2
}

root=""
config=""
specforge=""
launch=false
declare -a job_args=()

while (( $# )); do
  case "$1" in
    --root) [[ $# -ge 2 ]] || die "--root needs a value"; root="$2"; shift 2 ;;
    --config) [[ $# -ge 2 ]] || die "--config needs a value"; config="$2"; shift 2 ;;
    --specforge) [[ $# -ge 2 ]] || die "--specforge needs a value"; specforge="$2"; shift 2 ;;
    --launch) launch=true; shift ;;
    --) shift; job_args=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument before --: $1" ;;
  esac
done

[[ "$root" =~ ^/[A-Za-z0-9._/-]+$ && "$root" != "/" ]] || die "invalid --root"
[[ "$config" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "--config must be a safe absolute path"
specforge="${specforge:-$root/specforge-venv/bin/specforge}"
[[ "$specforge" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "--specforge must be a safe absolute path"

if [[ "$launch" != true ]]; then
  printf 'PLAN only: single-node SpecForge on DGX2, cli=%s\n' "$specforge"
  printf '%s\n' 'Re-run with --launch after validating storage and configuration.'
  exit 0
fi

hostname_short="$(hostname -s | tr '[:upper:]' '[:lower:]')"
[[ "$hostname_short" == "spark-49af" ]] ||
  die "launch is restricted to DGX2 (spark-49af)"
[[ -f "$root/.harness-training-owner-v1" ]] || die "DGX2 root is not ownership-marked"
marker_role=""
marker_root=""
while IFS='=' read -r key value; do
  case "$key" in
    role) marker_role="$value" ;;
    root) marker_root="$value" ;;
  esac
done < "$root/.harness-training-owner-v1"
[[ "$marker_role" == "dgx2" && "$marker_root" == "$root" ]] ||
  die "root marker does not match dgx2 at $root"
[[ -f "$config" && ! -L "$config" ]] || die "config must be a regular non-symlink file"
config_real="$(realpath "$config")"
root_real="$(realpath "$root")"
[[ "$config_real" == "$root_real/configs/"* ]] ||
  die "config must be stored below DGX2_ROOT/configs"
[[ -x "$specforge" ]] || die "SpecForge CLI is not installed at $specforge"

export HF_HOME="$root/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$root/cache/huggingface/hub"
export TRANSFORMERS_CACHE="$root/cache/huggingface/transformers"
export HF_DATASETS_CACHE="$root/cache/huggingface/datasets"
export TORCH_HOME="$root/cache/torch"
export TORCH_EXTENSIONS_DIR="$root/cache/torch/extensions"
export XDG_CACHE_HOME="$root/cache/xdg"
export PYTHONUNBUFFERED=1

printf '%s\n' 'Launching single-node SpecForge on DGX2.'
exec "$specforge" train --config "$config_real" "${job_args[@]}"
