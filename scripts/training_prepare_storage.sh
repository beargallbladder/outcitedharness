#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  training_prepare_storage.sh --role dgx2|asus1 --root ABS_PATH [options]

Options:
  --apply             Perform changes (default is a no-op plan).
  --host HOST         Run on the matching remote host through SSH.
  --owner USER        Owner for newly created directories (default: current user).
  --group GROUP       Group for newly created directories (default: current group).
  --sudo              Use sudo on the explicitly named remote host.
  --min-bytes BYTES   Override the role's capacity/free-space threshold.

Remote mutation requires both --host and --apply. HOST must resolve to the same
hostname as --role. This script never removes files or changes services.
EOF
}

die() {
  printf 'training storage: %s\n' "$*" >&2
  exit 2
}

valid_word() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]]
}

valid_path() {
  [[ "$1" =~ ^/[A-Za-z0-9._/-]+$ && "$1" != "/" ]]
}

assert_host_role() {
  local expected="$1"
  local actual expected_hostname
  actual="$(hostname -s | tr '[:upper:]' '[:lower:]')"
  case "$expected" in
    dgx2) expected_hostname="spark-49af" ;;
    asus1) expected_hostname="gx10-fc2e" ;;
    *) die "unknown host role '$expected'" ;;
  esac
  [[ "$actual" == "$expected_hostname" ]] ||
    die "hostname '$actual' does not match $expected ($expected_hostname)"
}

create_dir_if_missing() {
  local path="$1"
  local owner="$2"
  local group="$3"
  if [[ ! -d "$path" ]]; then
    install -d -m 0750 -o "$owner" -g "$group" "$path"
  fi
}

claim_root() {
  local role="$1"
  local root="$2"
  local owner="$3"
  local group="$4"
  local marker="$root/.harness-training-owner-v1"
  local entry key value marker_role="" marker_root=""

  if [[ ! -e "$root" ]]; then
    create_dir_if_missing "$root" "$owner" "$group"
  elif [[ ! -d "$root" ]]; then
    die "$root exists and is not a directory"
  fi

  if [[ -f "$marker" ]]; then
    while IFS='=' read -r key value; do
      case "$key" in
        role) marker_role="$value" ;;
        root) marker_root="$value" ;;
      esac
    done < "$marker"
    [[ "$marker_role" == "$role" && "$marker_root" == "$root" ]] ||
      die "ownership marker does not match requested role/root"
    return
  fi

  shopt -s nullglob dotglob
  local entries=("$root"/*)
  shopt -u nullglob dotglob
  if (( ${#entries[@]} != 0 )); then
    die "refusing to claim non-empty unowned directory $root"
  fi

  printf 'schema=1\nrole=%s\nroot=%s\n' "$role" "$root" > "$marker"
  chmod 0640 "$marker"
  if [[ "$(id -u)" == "0" ]]; then
    chown "$owner:$group" "$marker"
  fi
}

filesystem_bytes() {
  local root="$1"
  df -PB1 "$root" | awk 'NR == 2 {print $2, $4}'
}

apply_local() {
  local role="$1"
  local root="$2"
  local owner="$3"
  local group="$4"
  local minimum="$5"
  local total available probe
  local -a directories

  assert_host_role "$role"
  probe="$root"
  while [[ ! -e "$probe" ]]; do
    probe="$(dirname "$probe")"
  done
  [[ -d "$probe" ]] || die "nearest existing parent is not a directory: $probe"
  read -r total available < <(filesystem_bytes "$probe")
  [[ "$total" =~ ^[0-9]+$ && "$available" =~ ^[0-9]+$ ]] ||
    die "could not determine filesystem capacity for $probe"

  if [[ "$role" == "dgx2" ]]; then
    (( total >= minimum )) ||
      die "DGX2 filesystem has $total bytes; at least $minimum are required"
    directories=(
      "$root/datasets"
      "$root/models"
      "$root/cache/huggingface/hub"
      "$root/cache/huggingface/transformers"
      "$root/cache/huggingface/datasets"
      "$root/cache/torch/extensions"
      "$root/cache/xdg"
      "$root/artifacts"
      "$root/checkpoints"
      "$root/manifests"
      "$root/runs"
      "$root/configs"
    )
  else
    (( available >= minimum )) ||
      die "ASUS1 scratch has $available free bytes; floor is $minimum"
    directories=(
      "$root/staging"
      "$root/cache/huggingface/hub"
      "$root/cache/huggingface/transformers"
      "$root/cache/huggingface/datasets"
      "$root/cache/torch/extensions"
      "$root/cache/xdg"
      "$root/work"
      "$root/checkpoints"
    )
  fi

  claim_root "$role" "$root" "$owner" "$group"
  if [[ "$role" == "asus1" ]]; then
    printf '%s\n' "$minimum" > "$root/.minimum-free-bytes"
    chmod 0640 "$root/.minimum-free-bytes"
    if [[ "$(id -u)" == "0" ]]; then
      chown "$owner:$group" "$root/.minimum-free-bytes"
    fi
  fi

  for entry in "${directories[@]}"; do
    create_dir_if_missing "$entry" "$owner" "$group"
  done
  printf 'Prepared %s training storage at %s (no files removed).\n' "$role" "$root"
}

if [[ "${1:-}" == "__local_apply" ]]; then
  shift
  [[ "$#" == 5 ]] || die "invalid internal invocation"
  apply_local "$@"
  exit 0
fi

role=""
root=""
host=""
owner="$(id -un)"
group="$(id -gn)"
minimum=""
apply=false
use_sudo=false

while (( $# )); do
  case "$1" in
    --role) [[ $# -ge 2 ]] || die "--role needs a value"; role="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
    --root) [[ $# -ge 2 ]] || die "--root needs a value"; root="$2"; shift 2 ;;
    --host) [[ $# -ge 2 ]] || die "--host needs a value"; host="$2"; shift 2 ;;
    --owner) [[ $# -ge 2 ]] || die "--owner needs a value"; owner="$2"; shift 2 ;;
    --group) [[ $# -ge 2 ]] || die "--group needs a value"; group="$2"; shift 2 ;;
    --min-bytes) [[ $# -ge 2 ]] || die "--min-bytes needs a value"; minimum="$2"; shift 2 ;;
    --apply) apply=true; shift ;;
    --sudo) use_sudo=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$role" == "dgx2" || "$role" == "asus1" ]] || die "--role must be dgx2 or asus1"
valid_path "$root" || die "--root must be a safe absolute path other than /"
valid_word "$owner" || die "invalid owner"
valid_word "$group" || die "invalid group"
minimum="${minimum:-$([[ "$role" == "dgx2" ]] && printf 4000000000000 || printf 268435456000)}"
[[ "$minimum" =~ ^[0-9]+$ && "$minimum" -gt 0 ]] || die "--min-bytes must be a positive integer"

if [[ -n "$host" ]]; then
  valid_word "$host" || die "invalid host"
  host_short="${host%%.*}"
  host_short="$(printf '%s' "$host_short" | tr '[:upper:]' '[:lower:]')"
  [[ "$host_short" == "$role" ]] ||
    die "--host must name the selected role exactly (dgx2 or asus1)"
fi

if [[ "$apply" != true ]]; then
  printf 'PLAN only: prepare role=%s root=%s threshold=%s target=%s\n' \
    "$role" "$root" "$minimum" "${host:-local}"
  printf 'Re-run with --apply to make additive, ownership-marked changes.\n'
  exit 0
fi

if [[ -z "$host" ]]; then
  [[ "$use_sudo" == false ]] || die "--sudo is only valid with --host"
  apply_local "$role" "$root" "$owner" "$group" "$minimum"
  exit 0
fi

remote=(bash -s -- __local_apply "$role" "$root" "$owner" "$group" "$minimum")
if [[ "$use_sudo" == true ]]; then
  remote=(sudo "${remote[@]}")
fi
ssh -- "$host" "${remote[@]}" < "$0"
