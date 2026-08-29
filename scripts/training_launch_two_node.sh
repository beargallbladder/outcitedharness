#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  training_launch_two_node.sh --node-rank 0|1 --store-root PATH
    --scratch-root PATH --config FILE --master-addr ADDRESS --interface IFACE
    [--master-port PORT] [--specforge PATH] [--launch]
    [-- SPECFORGE_ARGS...]

Run rank 0 on DGX2 and rank 1 on ASUS1. DGX2_ROOT must be visible at the same
path on both nodes; it remains authoritative. ASUS1 uses only its marked
scratch root for rank-local caches and temporary work. The default is a plan.
EOF
}

die() {
  printf 'two-node training: %s\n' "$*" >&2
  exit 2
}

read_marker() {
  local marker="$1"
  local expected_role="$2"
  local expected_root="$3"
  local key value marker_role="" marker_root=""
  [[ -f "$marker" && ! -L "$marker" ]] || die "missing regular marker: $marker"
  while IFS='=' read -r key value; do
    case "$key" in
      role) marker_role="$value" ;;
      root) marker_root="$value" ;;
    esac
  done < "$marker"
  [[ "$marker_role" == "$expected_role" ]] ||
    die "marker $marker does not match role $expected_role"
  [[ "$expected_root" == "-" || "$marker_root" == "$expected_root" ]] ||
    die "marker $marker does not match $expected_role at $expected_root"
}

rank=""
store_root=""
scratch_root=""
config=""
master_addr=""
master_port=29500
interface=""
specforge=""
launch=false
declare -a job_args=()

while (( $# )); do
  case "$1" in
    --node-rank) [[ $# -ge 2 ]] || die "--node-rank needs a value"; rank="$2"; shift 2 ;;
    --store-root) [[ $# -ge 2 ]] || die "--store-root needs a value"; store_root="$2"; shift 2 ;;
    --scratch-root) [[ $# -ge 2 ]] || die "--scratch-root needs a value"; scratch_root="$2"; shift 2 ;;
    --config) [[ $# -ge 2 ]] || die "--config needs a value"; config="$2"; shift 2 ;;
    --master-addr) [[ $# -ge 2 ]] || die "--master-addr needs a value"; master_addr="$2"; shift 2 ;;
    --master-port) [[ $# -ge 2 ]] || die "--master-port needs a value"; master_port="$2"; shift 2 ;;
    --interface) [[ $# -ge 2 ]] || die "--interface needs a value"; interface="$2"; shift 2 ;;
    --specforge) [[ $# -ge 2 ]] || die "--specforge needs a value"; specforge="$2"; shift 2 ;;
    --launch) launch=true; shift ;;
    --) shift; job_args=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument before --: $1" ;;
  esac
done

[[ "$rank" == "0" || "$rank" == "1" ]] || die "--node-rank must be 0 or 1"
[[ "$store_root" =~ ^/[A-Za-z0-9._/-]+$ && "$store_root" != "/" ]] ||
  die "invalid --store-root"
[[ "$scratch_root" =~ ^/[A-Za-z0-9._/-]+$ && "$scratch_root" != "/" ]] ||
  die "invalid --scratch-root"
[[ "$config" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "--config must be a safe absolute path"
[[ "$master_addr" =~ ^[A-Fa-f0-9:.]+$ ]] || die "--master-addr must be an IP address"
[[ "$master_port" =~ ^[0-9]+$ && "$master_port" -ge 1024 && "$master_port" -le 65535 ]] ||
  die "invalid --master-port"
[[ "$interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || die "invalid --interface"
specforge="${specforge:-$store_root/specforge-venv/bin/specforge}"
[[ "$specforge" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "--specforge must be a safe absolute path"

expected_host="$([[ "$rank" == "0" ]] && printf dgx2 || printf asus1)"
if [[ "$launch" != true ]]; then
  printf 'PLAN only: two-node SpecForge consumer rank=%s host=%s cli=%s\n' \
    "$rank" "$expected_host" "$specforge"
  printf '%s\n' 'Re-run with --launch on each node after both doctor reports pass.'
  exit 0
fi

hostname_short="$(hostname -s | tr '[:upper:]' '[:lower:]')"
expected_hostname="$([[ "$rank" == "0" ]] && printf spark-49af || printf gx10-fc2e)"
[[ "$hostname_short" == "$expected_hostname" ]] ||
  die "rank $rank must launch on hostname $expected_host"
read_marker "$store_root/.harness-training-owner-v1" "dgx2" "-"
[[ -f "$config" && ! -L "$config" ]] || die "config must be a regular non-symlink file"
config_real="$(realpath "$config")"
store_real="$(realpath "$store_root")"
[[ "$config_real" == "$store_real/configs/"* ]] ||
  die "config must be stored below DGX2_ROOT/configs"
[[ -x "$specforge" ]] || die "SpecForge CLI is not installed at $specforge"

"$SCRIPT_DIR/training_link_doctor.sh" \
  --interface "$interface" \
  --peer "$master_addr" \
  --require-ready

if [[ "$rank" == "0" ]]; then
  cache_root="$store_root/cache"
else
  read_marker "$scratch_root/.harness-training-owner-v1" "asus1" "$scratch_root"
  python3 "$SCRIPT_DIR/training_check_scratch.py" "$scratch_root"
  cache_root="$scratch_root/cache"
fi

export HF_HOME="$cache_root/huggingface"
export HUGGINGFACE_HUB_CACHE="$cache_root/huggingface/hub"
export TRANSFORMERS_CACHE="$cache_root/huggingface/transformers"
export HF_DATASETS_CACHE="$cache_root/huggingface/datasets"
export TORCH_HOME="$cache_root/torch"
export TORCH_EXTENSIONS_DIR="$cache_root/torch/extensions"
export XDG_CACHE_HOME="$cache_root/xdg"
export NCCL_SOCKET_IFNAME="$interface"
export GLOO_SOCKET_IFNAME="$interface"
export NCCL_IB_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

printf 'Launching two-node SpecForge consumer rank %s on %s.\n' \
  "$rank" "$expected_host"
exec "$specforge" train \
  --config "$config_real" \
  --role consumer \
  --node-rank "$rank" \
  "deployment.trainer.master_addr=$master_addr" \
  "deployment.trainer.master_port=$master_port" \
  "${job_args[@]}"
