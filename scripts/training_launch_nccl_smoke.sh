#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_ID="sha256:728b352622d569b1343cb384dd8aa394917203d02631322b02d302d068e38139"
IMAGE_DEFAULT="nvcr.io/nvidia/pytorch@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1"

usage() {
  cat <<'EOF'
Usage:
  training_launch_nccl_smoke.sh --node-rank RANK --root PATH
    --master-addr ADDRESS --interface IFACE --hcas HCA[,HCA]
    [--world-size 2|4|6] [--gid-index INDEX] [--master-port PORT]
    [--image IMAGE] [--launch]

Run rank 0 on DGX2 and rank 1 on ASUS1 for the legacy pair qualification.
With --world-size 4, ranks 2 and 3 run on DGX3 and ASUS3. With world size 6,
ranks 4 and 5 run on ASUS2 and ASUS4. The default is a no-op plan. Rank 0
writes the qualification JSON below DGX2_ROOT/runs; other ranks write no
durable result. The known-good RoCE-v2 GID index defaults to 5 for ranks
0/1/4/5 and 3 for ranks 2/3; --gid-index overrides it after inspection.
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
world_size=2
gid_index=""
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
    --world-size) world_size="${2:?missing --world-size value}"; shift 2 ;;
    --gid-index) gid_index="${2:?missing --gid-index value}"; shift 2 ;;
    --interface) interface="${2:?missing --interface value}"; shift 2 ;;
    --hcas) hcas="${2:?missing --hcas value}"; shift 2 ;;
    --image) image="${2:?missing --image value}"; shift 2 ;;
    --launch) launch=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$world_size" == "2" || "$world_size" == "4" || "$world_size" == "6" ]] ||
  die "--world-size must be 2, 4, or 6"
[[ "$rank" =~ ^[0-5]$ && "$rank" -lt "$world_size" ]] ||
  die "--node-rank must be between 0 and world-size minus one"
if [[ -z "$gid_index" ]]; then
  gid_indices=(5 5 3 3 5 5)
  gid_index="${gid_indices[$rank]}"
fi
[[ "$root" =~ ^/[A-Za-z0-9._/-]+$ && "$root" != "/" ]] ||
  die "--root must be a safe absolute path"
[[ "$master_addr" =~ ^[A-Fa-f0-9:.]+$ ]] ||
  die "--master-addr must be an IP address"
[[ "$master_port" =~ ^[0-9]+$ && "$master_port" -ge 1024 && "$master_port" -le 65535 ]] ||
  die "invalid --master-port"
[[ "$gid_index" =~ ^[0-9]+$ && "$gid_index" -le 31 ]] ||
  die "invalid --gid-index"
[[ "$interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || die "invalid --interface"
[[ "$hcas" =~ ^[A-Za-z0-9_.:-]+(,[A-Za-z0-9_.:-]+)*$ ]] ||
  die "--hcas must be a comma-separated HCA allowlist"
[[ "$image" == "$IMAGE_DEFAULT" || "$image" == "$IMAGE_ID" ]] ||
  die "qualification image must remain pinned by registry digest or image ID"

expected_hostnames=(
  spark-49af gx10-fc2e spark-69c8 gx10-0309 gx10-26b6 gx10-33af
)
roles=(dgx2 asus1 dgx3 asus3 asus2 asus4)
expected_hostname="${expected_hostnames[$rank]}"
role="${roles[$rank]}"
output="/training/runs/${world_size}-rank-switch-nccl.json"

if [[ "$launch" != true ]]; then
  printf 'PLAN only: NCCL rank=%s/%s host=%s interface=%s hcas=%s gid=%s image=%s\n' \
    "$rank" "$world_size" "$role" "$interface" "$hcas" "$gid_index" "$image"
  printf '%s\n' 'Start nonzero ranks first, then rank 0; add --launch only after all link doctors pass.'
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
actual_image_id="$(docker image inspect "$image" --format '{{.Id}}' 2>/dev/null)" ||
  die "pinned PyTorch image is not staged"
[[ "$actual_image_id" == "$IMAGE_ID" ]] ||
  die "staged PyTorch image ID does not match the qualified image"

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
  --env "NCCL_IB_GID_INDEX=$gid_index" \
  --env NCCL_IB_ROCE_VERSION_NUM=2 \
  --env NCCL_IB_DISABLE=0 \
  --env NCCL_ASYNC_ERROR_HANDLING=1 \
  --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  --env NCCL_DEBUG=INFO \
  --mount "type=bind,src=$SCRIPT_DIR/training_nccl_smoke.py,dst=/opt/harness/training_nccl_smoke.py,readonly" \
  --mount "type=bind,src=$root,dst=/training" \
  --entrypoint torchrun \
  "$image" \
  --nnodes="$world_size" \
  --nproc-per-node=1 \
  --node-rank="$rank" \
  --master-addr="$master_addr" \
  --master-port="$master_port" \
  /opt/harness/training_nccl_smoke.py \
  --expected-world-size "$world_size" \
  --output "$output"
