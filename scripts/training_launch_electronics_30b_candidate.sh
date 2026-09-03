#!/usr/bin/env bash
set -euo pipefail

IMAGE_ID="sha256:728b352622d569b1343cb384dd8aa394917203d02631322b02d302d068e38139"

usage() {
  cat <<'EOF'
Usage:
  training_launch_electronics_30b_candidate.sh --node-rank 0..5
    --root PATH --config FILE --model-manifest FILE
    --master-addr ADDRESS --interface IFACE --hcas HCA[,HCA]
    [--master-port PORT] [--gid-index INDEX] [--image IMAGE_ID] [--launch]

Launch one rank of the sealed six-node BF16 Qwen3-VL-30B candidate run.
The default is a no-op plan. Start ranks 1-5 first, then rank 0. Every node
hash-verifies its model, dataset, validation split, recipe, and runtime before
joining the job.
EOF
}

die() {
  printf 'electronics 30B candidate training: %s\n' "$*" >&2
  exit 2
}

rank=""
root=""
config=""
model_manifest=""
master_addr=""
master_port=29563
interface=""
hcas=""
gid_index=""
image="$IMAGE_ID"
launch=false

while (( $# )); do
  case "$1" in
    --node-rank) rank="${2:?missing --node-rank value}"; shift 2 ;;
    --root) root="${2:?missing --root value}"; shift 2 ;;
    --config) config="${2:?missing --config value}"; shift 2 ;;
    --model-manifest) model_manifest="${2:?missing --model-manifest value}"; shift 2 ;;
    --master-addr) master_addr="${2:?missing --master-addr value}"; shift 2 ;;
    --master-port) master_port="${2:?missing --master-port value}"; shift 2 ;;
    --interface) interface="${2:?missing --interface value}"; shift 2 ;;
    --hcas) hcas="${2:?missing --hcas value}"; shift 2 ;;
    --gid-index) gid_index="${2:?missing --gid-index value}"; shift 2 ;;
    --image) image="${2:?missing --image value}"; shift 2 ;;
    --launch) launch=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$rank" =~ ^[0-5]$ ]] || die "--node-rank must be between 0 and 5"
for value in "$root" "$config" "$model_manifest"; do
  [[ "$value" =~ ^/[A-Za-z0-9._/-]+$ && "$value" != "/" ]] ||
    die "root, config, and model manifest must be safe absolute paths"
done
[[ "$master_addr" =~ ^[A-Fa-f0-9:.]+$ ]] ||
  die "--master-addr must be an IP address"
[[ "$master_port" =~ ^[0-9]+$ && "$master_port" -ge 1024 && "$master_port" -le 65535 ]] ||
  die "invalid --master-port"
[[ "$interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || die "invalid --interface"
[[ "$hcas" =~ ^[A-Za-z0-9_.:-]+(,[A-Za-z0-9_.:-]+)*$ ]] ||
  die "--hcas must be a comma-separated HCA allowlist"
[[ "$image" == "$IMAGE_ID" ]] || die "training image must remain pinned"

expected_hostnames=(
  spark-49af gx10-fc2e spark-69c8 gx10-0309 gx10-26b6 gx10-33af
)
roles=(dgx2 asus1 dgx3 asus3 asus2 asus4)
gid_indices=(5 5 3 3 5 5)
expected_hostname="${expected_hostnames[$rank]}"
role="${roles[$rank]}"
gid_index="${gid_index:-${gid_indices[$rank]}}"
[[ "$gid_index" =~ ^[0-9]+$ && "$gid_index" -le 31 ]] ||
  die "invalid --gid-index"

if [[ "$launch" != true ]]; then
  printf 'PLAN only: BF16 Qwen3-VL-30B candidate rank=%s/6 host=%s gid=%s\n' \
    "$rank" "$role" "$gid_index"
  printf 'model=%s config=%s image=%s\n' "$model_manifest" "$config" "$image"
  exit 0
fi

[[ "$(hostname -s | tr '[:upper:]' '[:lower:]')" == "$expected_hostname" ]] ||
  die "rank $rank must run on $expected_hostname"
marker="$root/.harness-training-owner-v1"
[[ -f "$marker" && ! -L "$marker" ]] || die "training owner marker is absent"
marker_role="$(awk -F= '$1 == "role" {print $2}' "$marker")"
marker_root="$(awk -F= '$1 == "root" {print $2}' "$marker")"
[[ "$marker_role" == "$role" ]] || die "training root role does not match rank"
[[ "$marker_root" == "$root" ]] || die "training owner marker root does not match"
[[ -f "$config" && ! -L "$config" ]] || die "training config is absent or unsafe"
[[ -f "$model_manifest" && ! -L "$model_manifest" ]] ||
  die "model manifest is absent or unsafe"
[[ -d "/sys/class/net/$interface" && "$(<"/sys/class/net/$interface/operstate")" == "up" ]] ||
  die "fabric interface is absent or down"
for hca in ${hcas//,/ }; do
  [[ -d "/sys/class/infiniband/$hca" ]] || die "HCA $hca is absent"
done

preflight="$root/scripts/verify_electronics_30b_candidate_preflight.py"
[[ -f "$preflight" && ! -L "$preflight" ]] || die "candidate preflight is absent"
actual_image="$(docker image inspect "$image" --format '{{.Id}}' 2>/dev/null)" ||
  die "pinned LLaMA Factory image is not staged"
[[ "$actual_image" == "$IMAGE_ID" ]] || die "training image ID changed"

receipt="$root/runs/electronics-30b-bf16-candidate-rank-${rank}-preflight-v1.json"
run_log="$root/runs/electronics-30b-bf16-candidate-rank-${rank}-train-v1.log"
container_name="electronics-30b-bf16-candidate-rank-${rank}-v1"
[[ ! -e "$receipt" && ! -L "$receipt" ]] ||
  die "immutable candidate preflight receipt already exists"
[[ ! -e "$run_log" && ! -L "$run_log" ]] ||
  die "immutable candidate rank log already exists"
[[ -z "$(docker ps -aq -f "name=^/${container_name}$")" ]] ||
  die "candidate training container name already exists"

python3 "$preflight" \
  --root "$root" \
  --config "$config" \
  --model-manifest "$model_manifest" \
  --image "$image" \
  --node-rank "$rank" \
  --receipt-output "$receipt" >/dev/null ||
  die "rank $rank candidate preflight failed"

root_real="$(realpath "$root")"
config_real="$(realpath "$config")"
[[ "$config_real" == "$root_real/configs/"* ]] ||
  die "config must be below the training root configs directory"
container_config="/training/configs/${config_real#"$root_real/configs/"}"

set +e
docker run --rm \
  --name "$container_name" \
  --gpus all \
  --user "$(id -u):$(id -g)" \
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
  --env TOKENIZERS_PARALLELISM=false \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env FORCE_TORCHRUN=1 \
  --env NNODES=6 \
  --env "NODE_RANK=$rank" \
  --env NPROC_PER_NODE=1 \
  --env "MASTER_ADDR=$master_addr" \
  --env "MASTER_PORT=$master_port" \
  --env "NCCL_SOCKET_IFNAME=$interface" \
  --env "GLOO_SOCKET_IFNAME=$interface" \
  --env "NCCL_IB_HCA=$hcas" \
  --env "NCCL_IB_GID_INDEX=$gid_index" \
  --env NCCL_IB_ROCE_VERSION_NUM=2 \
  --env NCCL_IB_DISABLE=0 \
  --env NCCL_CROSS_NIC=0 \
  --env NCCL_CUMEM_ENABLE=0 \
  --env NCCL_ASYNC_ERROR_HANDLING=1 \
  --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  --env NCCL_DEBUG=INFO \
  --mount "type=bind,src=$root_real,dst=/training" \
  "$image" train "$container_config" 2>&1 | tee "$run_log"
status="${PIPESTATUS[0]}"
set -e
chmod 0444 "$run_log"
exit "$status"
