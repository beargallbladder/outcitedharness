#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="harness/qwen38-native-tp:tf5.16.1-fla0.5.2"
IMAGE_ID="sha256:2456858d68a54153e1fc4c04da9f18c7405787f6a7cab768a3bf5c9956cce74e"
CONFIG_SHA256="889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
MODEL_RELATIVE="models/Qwen3.8-Flash-Next-BF16"

usage() {
  cat <<'EOF'
Usage:
  training_launch_qwen38_tp_load_smoke.sh --node-rank 0|1|2|3
    --root PATH --master-addr ADDRESS --interface IFACE --hcas HCA[,HCA]
    [--master-port PORT] [--image IMAGE] [--launch]

Run a four-rank native tensor-parallel load and 32-token forward only. This
launcher cannot train. Rank mapping is DGX2, ASUS1, DGX3, ASUS3. The default
is a no-op plan.
EOF
}

die() {
  printf 'Qwen3.8 TP load smoke: %s\n' "$*" >&2
  exit 2
}

rank=""
root=""
master_addr=""
master_port=29546
interface=""
hcas=""
image="$IMAGE_TAG"
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

[[ "$rank" =~ ^[0-3]$ ]] || die "--node-rank must be 0, 1, 2, or 3"
[[ "$root" =~ ^/[A-Za-z0-9._/-]+$ && "$root" != "/" ]] ||
  die "--root must be a safe absolute path"
[[ "$master_addr" =~ ^[A-Fa-f0-9:.]+$ ]] ||
  die "--master-addr must be an IP address"
[[ "$master_port" =~ ^[0-9]+$ && "$master_port" -ge 1024 && "$master_port" -le 65535 ]] ||
  die "invalid --master-port"
[[ "$interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || die "invalid --interface"
[[ "$hcas" =~ ^[A-Za-z0-9_.:-]+(,[A-Za-z0-9_.:-]+)*$ ]] ||
  die "--hcas must be a comma-separated HCA allowlist"
[[ "$image" == "$IMAGE_TAG" || "$image" == "$IMAGE_ID" ]] ||
  die "probe image must remain pinned"

expected_hostnames=(spark-49af gx10-fc2e spark-69c8 gx10-0309)
roles=(dgx2 asus1 dgx3 asus3)
gid_indices=(5 5 3 3)
expected_hostname="${expected_hostnames[$rank]}"
role="${roles[$rank]}"
gid_index="${gid_indices[$rank]}"
model="$root/$MODEL_RELATIVE"
output="$root/runs/qwen38-native-tp-load-smoke-v1.json"

if [[ "$launch" != true ]]; then
  printf 'PLAN only: Qwen3.8 TP rank=%s/4 host=%s gid=%s model=%s image=%s\n' \
    "$rank" "$role" "$gid_index" "$model" "$image"
  exit 0
fi

[[ "$(hostname -s | tr '[:upper:]' '[:lower:]')" == "$expected_hostname" ]] ||
  die "rank $rank must run on $expected_hostname"
marker="$root/.harness-training-owner-v1"
[[ -f "$marker" && ! -L "$marker" ]] || die "training root owner marker is absent"
marker_role="$(awk -F= '$1 == "role" {print $2}' "$marker")"
marker_root="$(awk -F= '$1 == "root" {print $2}' "$marker")"
[[ "$marker_role" == "$role" && "$marker_root" == "$root" ]] ||
  die "training root owner marker does not match this rank"
[[ -d "/sys/class/net/$interface" && "$(<"/sys/class/net/$interface/operstate")" == "up" ]] ||
  die "fabric interface is absent or down"
for hca in ${hcas//,/ }; do
  [[ -d "/sys/class/infiniband/$hca" ]] || die "HCA $hca is absent"
done
[[ -f "$model/config.json" && ! -L "$model/config.json" ]] ||
  die "model config is absent or unsafe"
[[ "$(sha256sum "$model/config.json" | awk '{print $1}')" == "$CONFIG_SHA256" ]] ||
  die "model config digest does not match the qualified revision"
[[ -f "$model/model.safetensors.index.json" ]] ||
  die "model download is incomplete: weight index is absent"
shopt -s nullglob
shards=("$model"/model-*-of-00131.safetensors)
(( ${#shards[@]} == 131 )) ||
  die "model download is incomplete: expected 131 weight files"
for shard in "${shards[@]}"; do
  [[ -f "$shard" && ! -L "$shard" ]] || die "model weight is absent or unsafe"
done
[[ -f "$SCRIPT_DIR/qwen38_tp_load_smoke.py" ]] ||
  die "load probe is absent"
[[ -f "$SCRIPT_DIR/verify_qwen38_training_preflight.py" ]] ||
  die "training preflight is absent"
actual_image_id="$(docker image inspect "$image" --format '{{.Id}}' 2>/dev/null)" ||
  die "probe image is not staged"
[[ "$actual_image_id" == "$IMAGE_ID" ]] ||
  die "probe image ID does not match the qualified runtime"
[[ "$rank" != "0" || ! -e "$output" ]] ||
  die "rank-zero output already exists"

python3 "$SCRIPT_DIR/verify_qwen38_training_preflight.py" \
  --config "$model/config.json" \
  --strategy native-tp-load \
  --world-size 4 >/dev/null

docker run --rm \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 16g \
  --read-only \
  --tmpfs /tmp:rw,exec,nosuid,size=8589934592 \
  --device /dev/infiniband \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --env HOME=/tmp \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env "NCCL_SOCKET_IFNAME=$interface" \
  --env "GLOO_SOCKET_IFNAME=$interface" \
  --env "NCCL_IB_HCA=$hcas" \
  --env "NCCL_IB_GID_INDEX=$gid_index" \
  --env NCCL_IB_ROCE_VERSION_NUM=2 \
  --env NCCL_IB_DISABLE=0 \
  --env NCCL_CUMEM_ENABLE=0 \
  --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  --mount "type=bind,src=$model,dst=/model,readonly" \
  --mount "type=bind,src=$SCRIPT_DIR/qwen38_tp_load_smoke.py,dst=/opt/harness/qwen38_tp_load_smoke.py,readonly" \
  --mount "type=bind,src=$root/runs,dst=/results" \
  --entrypoint torchrun \
  "$image" \
  --nnodes=4 \
  --nproc-per-node=1 \
  --node-rank="$rank" \
  --master-addr="$master_addr" \
  --master-port="$master_port" \
  /opt/harness/qwen38_tp_load_smoke.py \
  --model /model \
  --output /results/qwen38-native-tp-load-smoke-v1.json \
  --sequence-length 32
