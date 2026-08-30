#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  training_configure_link.sh --role dgx2|asus1
    --primary-interface IFACE --primary-cidr ADDRESS/PREFIX
    --secondary-interface IFACE --secondary-cidr ADDRESS/PREFIX
    [--mtu BYTES] [--apply]

Configure only the two ConnectX direct-link interfaces. The default is a
no-op plan. Run with sudo and --apply after the interconnect is installed.
This does not alter the LAN, Wi-Fi, Tailscale, routes, or system services.
EOF
}

die() {
  printf 'training link configuration: %s\n' "$*" >&2
  exit 2
}

role=""
primary_interface=""
primary_cidr=""
secondary_interface=""
secondary_cidr=""
mtu=9000
apply=false

while (( $# )); do
  case "$1" in
    --role) role="${2:?missing --role value}"; shift 2 ;;
    --primary-interface) primary_interface="${2:?missing --primary-interface value}"; shift 2 ;;
    --primary-cidr) primary_cidr="${2:?missing --primary-cidr value}"; shift 2 ;;
    --secondary-interface) secondary_interface="${2:?missing --secondary-interface value}"; shift 2 ;;
    --secondary-cidr) secondary_cidr="${2:?missing --secondary-cidr value}"; shift 2 ;;
    --mtu) mtu="${2:?missing --mtu value}"; shift 2 ;;
    --apply) apply=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$role" == "dgx2" || "$role" == "asus1" ]] ||
  die "--role must be dgx2 or asus1"
for interface in "$primary_interface" "$secondary_interface"; do
  [[ "$interface" =~ ^[A-Za-z0-9_.:-]+$ ]] || die "invalid interface"
done
[[ "$primary_interface" != "$secondary_interface" ]] ||
  die "primary and secondary interfaces must differ"
for cidr in "$primary_cidr" "$secondary_cidr"; do
  [[ "$cidr" =~ ^10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}/(24|30)$ ]] ||
    die "direct-link addresses must be private 10/8 IPv4 /24 or /30 CIDRs"
done
[[ "$mtu" =~ ^[0-9]+$ && "$mtu" -ge 1500 && "$mtu" -le 9216 ]] ||
  die "MTU must be between 1500 and 9216"

expected_hostname="$([[ "$role" == "dgx2" ]] && printf spark-49af || printf gx10-fc2e)"
if [[ "$apply" != true ]]; then
  printf 'PLAN only: host=%s primary=%s:%s secondary=%s:%s mtu=%s\n' \
    "$role" "$primary_interface" "$primary_cidr" \
    "$secondary_interface" "$secondary_cidr" "$mtu"
  printf '%s\n' 'Re-run as root with --apply after the cable is installed.'
  exit 0
fi

[[ "$(id -u)" == "0" ]] || die "--apply must run as root"
[[ "$(hostname -s | tr '[:upper:]' '[:lower:]')" == "$expected_hostname" ]] ||
  die "role $role must run on $expected_hostname"
for interface in "$primary_interface" "$secondary_interface"; do
  [[ -d "/sys/class/net/$interface" ]] ||
    die "ConnectX interface $interface is absent; verify the cable and hotplug"
  if ip route show default dev "$interface" | awk 'NF {found=1} END {exit !found}'; then
    die "refusing to modify default-route interface $interface"
  fi
done

ip link set dev "$primary_interface" mtu "$mtu" up
ip address replace "$primary_cidr" dev "$primary_interface"
ip link set dev "$secondary_interface" mtu "$mtu" up
ip address replace "$secondary_cidr" dev "$secondary_interface"

printf 'Configured %s direct link:\n' "$role"
ip -brief address show dev "$primary_interface"
ip -brief address show dev "$secondary_interface"
