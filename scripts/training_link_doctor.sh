#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: training_link_doctor.sh --interface IFACE [options]

Options:
  --host ROLE         Inspect one explicit six-node training host over SSH.
  --peer ADDRESS      Also test route and non-fatal reachability to the peer.
  --mtu BYTES         Expected MTU (default: 9000).
  --container-image   Pinned PyTorch/NCCL image already staged on both hosts.
  --require-ready     Return nonzero when any readiness warning is found.

The doctor is read-only. A missing cable, interface, RDMA utility, or NCCL
runtime is reported as a warning by default, so this can run before cabling.
EOF
}

die() {
  printf 'training link doctor: %s\n' "$*" >&2
  exit 2
}

warning_count=0
warn() {
  warning_count=$((warning_count + 1))
  printf 'WARN: %s\n' "$*"
}

doctor_local() {
  local interface="$1"
  local expected_mtu="$2"
  local peer="$3"
  local require_ready="$4"
  local container_image="$5"
  local actual_mtu state
  [[ "$peer" == "-" ]] && peer=""

  printf 'Direct-link readiness report for %s on %s\n' "$interface" "$(hostname -s)"

  if [[ ! -d "/sys/class/net/$interface" ]]; then
    warn "interface $interface is absent (acceptable before hardware installation)"
  else
    actual_mtu="$(<"/sys/class/net/$interface/mtu")"
    state="$(<"/sys/class/net/$interface/operstate")"
    printf 'interface=%s state=%s mtu=%s expected_mtu=%s\n' \
      "$interface" "$state" "$actual_mtu" "$expected_mtu"
    [[ "$actual_mtu" == "$expected_mtu" ]] ||
      warn "MTU mismatch on $interface"
    [[ "$state" == "up" ]] ||
      warn "$interface is not up; the direct cable may not be present"
    if command -v ip >/dev/null 2>&1; then
      ip -brief address show dev "$interface" || warn "could not inspect interface addresses"
      ip route show dev "$interface" || warn "could not inspect interface routes"
    else
      warn "iproute2 is unavailable"
    fi
  fi

  if command -v rdma >/dev/null 2>&1; then
    printf '%s\n' '--- RDMA links ---'
    rdma link show || warn "rdma link query failed"
  elif command -v ibv_devinfo >/dev/null 2>&1; then
    printf '%s\n' '--- RDMA devices ---'
    ibv_devinfo || warn "ibv_devinfo failed"
  else
    warn "neither rdma nor ibv_devinfo is installed"
  fi

  if command -v ibdev2netdev >/dev/null 2>&1; then
    printf '%s\n' '--- RDMA/netdev mapping ---'
    ibdev2netdev || warn "ibdev2netdev failed"
  else
    warn "ibdev2netdev is unavailable; interface-to-HCA mapping is unknown"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' '--- GPU topology ---'
    nvidia-smi topo -m || warn "nvidia-smi topology query failed"
  else
    warn "nvidia-smi is unavailable"
  fi

  if command -v docker >/dev/null 2>&1 &&
    docker image inspect "$container_image" >/dev/null 2>&1; then
    if ! docker run --rm --gpus all --network none "$container_image" python - <<'PY'
try:
    import torch
except ImportError:
    print("PyTorch: not installed")
    raise SystemExit(1)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"torch.distributed NCCL available: {torch.distributed.is_nccl_available()}")
if not torch.cuda.is_available() or not torch.distributed.is_nccl_available():
    raise SystemExit(1)
PY
    then
      warn "containerized PyTorch NCCL probe failed"
    fi
  else
    warn "pinned PyTorch/NCCL qualification image is not staged"
  fi

  printf 'Recommended launch setting: NCCL_SOCKET_IFNAME=%s\n' "$interface"
  printf '%s\n' 'Review NCCL_DEBUG=INFO output during the first controlled launch.'

  if [[ -n "$peer" ]]; then
    if command -v ip >/dev/null 2>&1; then
      ip route get "$peer" || warn "no route to peer $peer"
    fi
    if command -v ping >/dev/null 2>&1; then
      ping -c 1 -W 1 "$peer" >/dev/null 2>&1 ||
        warn "peer $peer did not answer; cable or peer configuration may be absent"
    else
      warn "ping is unavailable"
    fi
  fi

  if (( warning_count == 0 )); then
    printf '%s\n' 'READY: all inspected direct-link prerequisites passed.'
  else
    printf 'NOT READY: %d warning(s); no changes were made.\n' "$warning_count"
  fi
  [[ "$require_ready" != true || "$warning_count" == 0 ]]
}

if [[ "${1:-}" == "__doctor" ]]; then
  shift
  [[ "$#" == 5 ]] || die "invalid internal invocation"
  doctor_local "$@"
  exit $?
fi

host=""
interface=""
peer=""
expected_mtu=9000
container_image="nvcr.io/nvidia/pytorch@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1"
require_ready=false

while (( $# )); do
  case "$1" in
    --host) [[ $# -ge 2 ]] || die "--host needs a value"; host="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
    --interface) [[ $# -ge 2 ]] || die "--interface needs a value"; interface="$2"; shift 2 ;;
    --peer) [[ $# -ge 2 ]] || die "--peer needs a value"; peer="$2"; shift 2 ;;
    --mtu) [[ $# -ge 2 ]] || die "--mtu needs a value"; expected_mtu="$2"; shift 2 ;;
    --container-image) [[ $# -ge 2 ]] || die "--container-image needs a value"; container_image="$2"; shift 2 ;;
    --require-ready) require_ready=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || die "a safe --interface is required"
[[ "$expected_mtu" =~ ^[0-9]+$ && "$expected_mtu" -ge 1500 ]] ||
  die "--mtu must be an integer of at least 1500"
[[ -z "$peer" || "$peer" =~ ^[A-Fa-f0-9:.]+$ ]] || die "invalid peer address"
case "$host" in
  ""|dgx2|asus1|dgx3|asus3|asus2|asus4) ;;
  *) die "--host must name one of the six training nodes" ;;
esac
[[ "$container_image" =~ ^nvcr\.io/nvidia/pytorch@sha256:[a-f0-9]{64}$ ]] ||
  die "--container-image must be a pinned NVIDIA PyTorch digest"

if [[ -n "$host" ]]; then
  peer_arg="${peer:--}"
  ssh -o BatchMode=yes -o ConnectTimeout=10 -- "$host" \
    bash -s -- __doctor "$interface" "$expected_mtu" "$peer_arg" "$require_ready" "$container_image" < "$0"
else
  doctor_local "$interface" "$expected_mtu" "$peer" "$require_ready" "$container_image"
fi
