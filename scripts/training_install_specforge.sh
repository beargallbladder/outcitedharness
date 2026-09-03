#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_DEFAULT="harness/specforge-gb10:c4395469"

usage() {
  cat <<'EOF'
Usage:
  training_install_specforge.sh --role dgx2|asus1 --root PATH [--image IMAGE]
    [--build] [--install-wrapper] [--apply]

Build the digest-pinned ARM64 SpecForge runtime on DGX2 or ASUS1. On DGX2,
--install-wrapper creates ROOT/specforge-venv/bin/specforge, which runs the
container with the authoritative training root mounted at the same path.
The default is a no-op plan.
EOF
}

die() {
  printf 'SpecForge installation: %s\n' "$*" >&2
  exit 2
}

role=""
root=""
image="$IMAGE_DEFAULT"
build=false
install_wrapper=false
apply=false

while (( $# )); do
  case "$1" in
    --role) role="${2:?missing --role value}"; shift 2 ;;
    --root) root="${2:?missing --root value}"; shift 2 ;;
    --image) image="${2:?missing --image value}"; shift 2 ;;
    --build) build=true; shift ;;
    --install-wrapper) install_wrapper=true; shift ;;
    --apply) apply=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$role" == "dgx2" || "$role" == "asus1" ]] ||
  die "--role must be dgx2 or asus1"
[[ "$root" =~ ^/[A-Za-z0-9._/-]+$ && "$root" != "/" ]] ||
  die "--root must be a safe absolute path"
[[ "$image" == "$IMAGE_DEFAULT" ]] ||
  die "image must remain pinned to $IMAGE_DEFAULT"
[[ "$build" == true || "$install_wrapper" == true ]] ||
  die "select --build, --install-wrapper, or both"
if [[ "$install_wrapper" == true && "$role" != "dgx2" ]]; then
  die "only DGX2 may install the authoritative wrapper"
fi

if [[ "$apply" != true ]]; then
  printf 'PLAN only: host=%s root=%s image=%s build=%s wrapper=%s\n' \
    "$role" "$root" "$image" "$build" "$install_wrapper"
  exit 0
fi

hostname_short="$(hostname -s | tr '[:upper:]' '[:lower:]')"
expected_hostname="$([[ "$role" == "dgx2" ]] && printf spark-49af || printf gx10-fc2e)"
[[ "$hostname_short" == "$expected_hostname" ]] ||
  die "role $role must run on $expected_hostname"

marker="$root/.harness-training-owner-v1"
[[ -f "$marker" && ! -L "$marker" ]] ||
  die "training root lacks a regular owner marker"
marker_role="$(awk -F= '$1 == "role" {print $2}' "$marker")"
[[ "$marker_role" == "$role" ]] ||
  die "training root is not owned by $role"

if [[ "$build" == true ]]; then
  docker build \
    --file "$REPO_ROOT/deploy/training/SpecForge.GB10.Dockerfile" \
    --tag "$image" \
    "$REPO_ROOT"
  docker run --rm --gpus all --entrypoint python "$image" -c \
    'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
fi

if [[ "$install_wrapper" == true ]]; then
  docker image inspect "$image" >/dev/null 2>&1 ||
    die "build or load $image before installing the wrapper"
  wrapper_dir="$root/specforge-venv/bin"
  mkdir -p "$wrapper_dir"
  wrapper="$wrapper_dir/specforge"
  temporary="$wrapper.tmp.$$"
  cat >"$temporary" <<EOF
#!/usr/bin/env bash
set -euo pipefail
root="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")/../.." && pwd)"
image="$image"
docker_args=(
  run --rm --gpus all --network host --ipc host
  --ulimit memlock=-1 --ulimit stack=67108864
  --user "\$(id -u):\$(id -g)"
  --env HOME=/tmp
  --mount "type=bind,src=\$root,dst=\$root"
  --workdir /opt/specforge
)
[[ -d /dev/infiniband ]] && docker_args+=(--device /dev/infiniband)
for name in \
  HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE \
  TORCH_HOME TORCH_EXTENSIONS_DIR XDG_CACHE_HOME \
  NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME NCCL_IB_DISABLE NCCL_IB_HCA \
  NCCL_IB_GID_INDEX NCCL_IB_ROCE_VERSION_NUM NCCL_ASYNC_ERROR_HANDLING \
  TORCH_NCCL_ASYNC_ERROR_HANDLING NCCL_DEBUG PYTHONUNBUFFERED \
  PYTORCH_CUDA_ALLOC_CONF; do
  [[ -v "\$name" ]] && docker_args+=(--env "\$name")
done
exec docker "\${docker_args[@]}" "\$image" "\$@"
EOF
  chmod 0755 "$temporary"
  mv "$temporary" "$wrapper"
  "$wrapper" --help >/dev/null
  printf 'Installed %s backed by %s\n' "$wrapper" "$image"
fi
