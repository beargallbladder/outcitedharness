#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_DEFAULT="nvcr.io/nvidia/pytorch@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1"

usage() {
  cat <<'EOF'
Usage:
  training_launch_nccl_smoke.sh --node-rank 0|1 --root PATH
    --master-addr ADDRESS --interface IFACE --hcas HCA[,HCA]
    [--master-port PORT] [--image IMAGE] [--launch]

Run rank 0 on DGX2 and rank 1 on ASUS1 after both direct-link interfaces have
been configured. The default is a no-op plan. Rank 0 writes the qualification
JSON below DGX2_ROOT/runs; rank 1 writes no durable result.
EOF
}

die() {
  printf 'NCCL smoke: %s\n' "$*" >&2
  exit 2
}

rank=""
root=""
master_addr=""
master_port=29501
interface=""
hcas=""
image="$IMAGE_DEFAULT"
launch=false

while (( $# )); do
  case "$1" in
    --node-rank) rank="${2:?missing --node-rank value}"; shift 2 ;;
    --root) root="${2:?missing --root value}"; shift 2 ;;
    --master-addr) master_addr="${2:?missing --master-addr value}"; shift 2 ;;
    --master-port) master_port="${2:?missing --master-port value}"; shift 2 ;;
    --interface) interface="${2:?missing --interface value}"; shift 2 ;;
    --hcas) hcas="${2:?missing --hcas value}"; shift 2 ;;
    --image) image="${2:?missing --image value}"; shift 2 ;;
    --launch) launch=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$rank" == "0" || "$rank" == "1" ]] ||
  die "--node-rank must be 0 or 1"
[[ "$root" =~ ^/[A-Za-z0-9._/-]+$ && "$root" != "/" ]] ||
  die "--root must be a safe absolute path"
[[ "$master_addr" =~ ^[A-Fa-f0-9:.]+$ ]] ||
  die "--master-addr must be an IP address"
[[ "$master_port" =~ ^[0-9]+$ && "$master_port" -ge 1024 && "$master_port" -le 65535 ]] ||
  die "invalid --master-port"
[[ "$interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || die "invalid --interface"
[[ "$hcas" =~ ^[A-Za-z0-9_.:-]+(,[A-Za-z0-9_.:-]+)*$ ]] ||
  die "--hcas must be a comma-separated HCA allowlist"
[[ "$image" == "$IMAGE_DEFAULT" ]] ||
  die "qualification image must remain pinned by digest"

expected_hostname="$([[ "$rank" == "0" ]] && printf spark-49af || printf gx10-fc2e)"
role="$([[ "$rank" == "0" ]] && printf dgx2 || printf asus1)"
output="/training/runs/cable-qualification-nccl.json"

if [[ "$launch" != true ]]; then
  printf 'PLAN only: NCCL rank=%s host=%s interface=%s hcas=%s image=%s\n' \
    "$rank" "$role" "$interface" "$hcas" "$image"
  printf '%s\n' 'Start rank 1 first, then rank 0; add --launch only after both link doctors pass.'
  exit 0
fi

[[ "$(hostname -s | tr '[:upper:]' '[:lower:]')" == "$expected_hostname" ]] ||
  die "rank $rank must run on $expected_hostname"
[[ -f "$root/.harness-training-owner-v1" && ! -L "$root/.harness-training-owner-v1" ]] ||
  die "training root lacks a regular owner marker"
marker_role="$(awk -F= '$1 == "role" {print $2}' "$root/.harness-training-owner-v1")"
[[ "$marker_role" == "$role" ]] || die "training root is not owned by $role"
[[ -d "/sys/class/net/$interface" ]] || die "interface $interface is absent"
[[ "$(<"/sys/class/net/$interface/operstate")" == "up" ]] ||
  die "interface $interface is not up"
for hca in ${hcas//,/ }; do
  [[ -d "/sys/class/infiniband/$hca" ]] || die "HCA $hca is absent"
done
docker image inspect "$image" >/dev/null 2>&1 ||
  die "pinned PyTorch image is not staged"

mkdir -p "$root/runs"
docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --device /dev/infiniband \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --env "NCCL_SOCKET_IFNAME=$interface" \
  --env "GLOO_SOCKET_IFNAME=$interface" \
  --env "NCCL_IB_HCA=$hcas" \
  --env NCCL_IB_GID_INDEX=3 \
  --env NCCL_IB_ROCE_VERSION_NUM=2 \
  --env NCCL_IB_DISABLE=0 \
  --env NCCL_ASYNC_ERROR_HANDLING=1 \
  --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  --env NCCL_DEBUG=INFO \
  --mount "type=bind,src=$SCRIPT_DIR/training_nccl_smoke.py,dst=/opt/harness/training_nccl_smoke.py,readonly" \
  --mount "type=bind,src=$root,dst=/training" \
  --entrypoint torchrun \
  "$image" \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank="$rank" \
  --master-addr="$master_addr" \
  --master-port="$master_port" \
  /opt/harness/training_nccl_smoke.py \
  --output "$output"
