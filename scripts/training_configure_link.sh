#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  training_configure_link.sh --role dgx2|asus1|dgx3|asus3|asus2|asus4
    --primary-interface IFACE --primary-cidr ADDRESS/PREFIX
    --secondary-interface IFACE --secondary-cidr ADDRESS/PREFIX
    [--mtu BYTES] [--apply]

Configure only the two ConnectX direct-link interfaces. The default is a
no-op plan. Run with sudo and --apply after the interconnect is installed.
On NetworkManager hosts, the existing device profiles are updated so the
addresses and MTU survive profile reactivation and reboot. This does not alter
the LAN, Wi-Fi, Tailscale, default route, or system services.
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

case "$role" in
  dgx2|asus1|dgx3|asus3|asus2|asus4) ;;
  *) die "--role must name one of the six training nodes" ;;
esac
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

case "$role" in
  dgx2) expected_hostname="spark-49af" ;;
  asus1) expected_hostname="gx10-fc2e" ;;
  dgx3) expected_hostname="spark-69c8" ;;
  asus3) expected_hostname="gx10-0309" ;;
  asus2) expected_hostname="gx10-26b6" ;;
  asus4) expected_hostname="gx10-33af" ;;
esac
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

configure_interface() {
  local interface="$1"
  local cidr="$2"
  local connection=""

  if command -v nmcli >/dev/null 2>&1 &&
    systemctl is-active --quiet NetworkManager 2>/dev/null; then
    connection="$(nmcli -g GENERAL.CONNECTION device show "$interface")"
    [[ -n "$connection" && "$connection" != "--" ]] ||
      die "NetworkManager has no active profile for $interface"
    nmcli connection modify "$connection" \
      ipv4.method manual \
      ipv4.addresses "$cidr" \
      ipv4.gateway "" \
      ipv4.dns "" \
      ipv4.never-default yes \
      ipv6.method link-local \
      802-3-ethernet.mtu "$mtu" \
      connection.autoconnect yes
    nmcli connection up "$connection" >/dev/null
  else
    ip link set dev "$interface" mtu "$mtu" up
    ip address replace "$cidr" dev "$interface"
  fi
}

configure_interface "$primary_interface" "$primary_cidr"
configure_interface "$secondary_interface" "$secondary_cidr"

printf 'Configured %s direct link:\n' "$role"
ip -brief address show dev "$primary_interface"
ip -brief address show dev "$secondary_interface"
