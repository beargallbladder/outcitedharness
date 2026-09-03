#!/usr/bin/env bash
set -euo pipefail

RECIPE="${QWEN38_RECIPE:-$HOME/qwen38-sglang-recipe}"
STATE_DIR="${QWEN38_WATCHDOG_STATE:-$HOME/.local/state/qwen38-watchdog}"
READY_URL="${QWEN38_READY_URL:-http://127.0.0.1:8888/v1/models}"
FAILURE_LIMIT="${QWEN38_FAILURE_LIMIT:-3}"
RESTART_COOLDOWN="${QWEN38_RESTART_COOLDOWN:-900}"
START_TIMEOUT="${QWEN38_START_TIMEOUT:-1200}"

API_KEY="$(
  python3 - "$RECIPE/.env" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
if path.is_file():
    for line in path.read_text().splitlines():
        if line.startswith("API_KEY="):
            print(line.split("=", 1)[1])
            break
PY
)"
auth_curl=()
[[ -n "${API_KEY:-}" ]] &&
  auth_curl=(-H "Authorization: Bearer $API_KEY")

mkdir -p "$STATE_DIR/incidents"
chmod 700 "$STATE_DIR" "$STATE_DIR/incidents"
exec 9>"$STATE_DIR/lock"
flock -n 9 || exit 0

failures_file="$STATE_DIR/consecutive-failures"
last_restart_file="$STATE_DIR/last-restart-epoch"
disabled_file="$STATE_DIR/disabled"
log_file="$STATE_DIR/watchdog.log"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$log_file"
}

healthy() {
  curl -fsS --max-time 8 "${auth_curl[@]}" "$READY_URL" >/dev/null 2>&1
}

read_integer() {
  local path="$1" fallback="$2" value
  value="$(<"$path")" 2>/dev/null || value="$fallback"
  [[ "$value" =~ ^[0-9]+$ ]] || value="$fallback"
  printf '%s' "$value"
}

redact_log() {
  local source="$1" destination="$2"
  python3 - "$source" "$destination" "$RECIPE/.env" <<'PY'
from pathlib import Path
import sys

source, destination, environment = sys.argv[1:]
secret = ""
for line in Path(environment).read_text().splitlines():
    if line.startswith("API_KEY="):
        secret = line.split("=", 1)[1]
        break
text = Path(source).read_text(errors="replace") if Path(source).exists() else ""
if secret:
    text = text.replace(secret, "[REDACTED_API_KEY]")
Path(destination).write_text(text)
PY
}

capture_incident() {
  local incident="$STATE_DIR/incidents/$(date -u +%Y%m%dT%H%M%SZ)"
  local head_raw worker_raw
  mkdir -p "$incident"
  head_raw="$(mktemp)"
  worker_raw="$(mktemp)"
  {
    printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'failures=%s\n' "$failures"
    docker inspect \
      --format 'head_status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} error={{.State.Error}}' \
      qwen38-flash-next-head 2>&1 || true
    ssh -o BatchMode=yes -o ConnectTimeout=8 samkimasus4@10.10.10.2 \
      "docker inspect --format 'worker_status={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} error={{.State.Error}}' qwen38-flash-next-worker" \
      2>&1 || true
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader 2>&1 || true
  } >"$incident/metadata.txt"
  docker logs --tail 200 qwen38-flash-next-head >"$head_raw" 2>&1 || true
  ssh -o BatchMode=yes -o ConnectTimeout=8 samkimasus4@10.10.10.2 \
    "docker logs --tail 200 qwen38-flash-next-worker" \
    >"$worker_raw" 2>&1 || true
  redact_log "$head_raw" "$incident/head.log"
  redact_log "$worker_raw" "$incident/worker.log"
  rm -f "$head_raw" "$worker_raw"
  chmod -R go-rwx "$incident"
  log "captured incident $incident"
}

case "${1:-check}" in
  disable)
    : >"$disabled_file"
    log "watchdog disabled"
    exit 0
    ;;
  enable)
    rm -f "$disabled_file"
    printf '0\n' >"$failures_file"
    log "watchdog enabled"
    exit 0
    ;;
  status)
    if [[ -e "$disabled_file" ]]; then
      printf 'disabled\n'
    elif healthy; then
      printf 'healthy\n'
    else
      printf 'unhealthy failures=%s\n' "$(read_integer "$failures_file" 0)"
    fi
    exit 0
    ;;
  check) ;;
  *) printf 'usage: %s [check|status|enable|disable]\n' "$0" >&2; exit 2 ;;
esac

[[ ! -e "$disabled_file" ]] || exit 0

if healthy; then
  printf '0\n' >"$failures_file"
  exit 0
fi

failures="$(( $(read_integer "$failures_file" 0) + 1 ))"
printf '%s\n' "$failures" >"$failures_file"
log "health failure $failures/$FAILURE_LIMIT"
(( failures >= FAILURE_LIMIT )) || exit 0

now="$(date +%s)"
last_restart="$(read_integer "$last_restart_file" 0)"
if (( now - last_restart < RESTART_COOLDOWN )); then
  log "restart suppressed by ${RESTART_COOLDOWN}s cooldown"
  exit 0
fi

capture_incident
printf '%s\n' "$now" >"$last_restart_file"
log "starting coordinated two-rank restart"
if timeout "$START_TIMEOUT" "$RECIPE/start.sh" stop >>"$log_file" 2>&1 &&
   timeout "$START_TIMEOUT" "$RECIPE/start.sh" serve >>"$log_file" 2>&1 &&
   healthy; then
  printf '0\n' >"$failures_file"
  log "coordinated restart succeeded"
  exit 0
fi

log "coordinated restart failed"
exit 1
