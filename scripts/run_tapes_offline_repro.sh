#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "__container" ]]; then
  server_pid=""
  cleanup() {
    if [[ -n "$server_pid" ]]; then
      kill "$server_pid" >/dev/null 2>&1 || true
      wait "$server_pid" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT INT TERM

  python /training/scripts/serve_bge_checkpoint.py \
    --model /training/models/bge-m3-cr-tapes-v1/checkpoints_20260427T183216Z \
    --host 127.0.0.1 \
    --port 18881 \
    --device cuda \
    --max-length 128 \
    --max-batch 256 \
    >/training/logs/tapes-offline-v1-server.log 2>&1 &
  server_pid=$!

  python - <<'PY'
import json
import time
import urllib.request

deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:18881/health", timeout=2) as response:
            payload = json.loads(response.read())
        if payload.get("status") == "ok":
            break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("offline BGE server did not become healthy")
PY

  python /training/scripts/evaluate_tapes_open_set.py \
    --endpoint http://127.0.0.1:18881/embed \
    --mask /training/evaluations/tapes-open-set-v1-repro/input/tapes_l3_keyword_relevance_mask_2026-W34.json \
    --kim-split /training/evaluations/tapes-open-set-v1-repro/input/kim_tag_agreement_eval_split_v1.json \
    --kim-baseline /training/evaluations/tapes-open-set-v1-repro/input/kim_tag_agreement_eval_v1_bge-m3-cr-tapes-v1.json \
    --retrieval-split /training/evaluations/tapes-open-set-v1-repro/input/bge-m3-eval-split-v1.1.jsonl \
    --retrieval-baseline /training/evaluations/tapes-open-set-v1-repro/input/tapes_bge_m3_retrieval_eval_2026-W34-v1-recheck.json \
    --output /training/evaluations/tapes-open-set-v1-repro/offline-v1-reproduction.json \
    --batch-size 128 \
    --timeout 120
  exit $?
fi

root="${DGX2_TRAINING_ROOT:-$HOME/harness-training}"
image="${LLAMAFACTORY_IMAGE:-harness/llamafactory-gb10:20260829}"
output="$root/evaluations/tapes-open-set-v1-repro/offline-v1-reproduction.json"

[[ "$(hostname -s)" == "spark-49af" ]] ||
  { printf '%s\n' "offline reproduction must run on DGX2" >&2; exit 2; }
[[ -s "$root/models/bge-m3-cr-tapes-v1/checkpoints_20260427T183216Z/model.safetensors" ]] ||
  { printf '%s\n' "immutable v1 checkpoint is incomplete" >&2; exit 2; }
[[ -s "$root/evaluations/tapes-open-set-v1-repro/input/bge-m3-eval-split-v1.1.jsonl" ]] ||
  { printf '%s\n' "pinned evaluation inputs are incomplete" >&2; exit 2; }
docker image inspect "$image" >/dev/null 2>&1 ||
  { printf '%s\n' "offline evaluation image is unavailable" >&2; exit 2; }

set +e
docker run --rm \
  --gpus all \
  --network none \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --mount "type=bind,src=$root,dst=/training" \
  --entrypoint bash \
  "$image" \
  /training/scripts/run_tapes_offline_repro.sh __container
result=$?
set -e

if (( result == 0 )); then
  printf '%s\n' "TAPES_OFFLINE_V1_REPRO_DONE"
elif [[ -s "$output" ]]; then
  printf '%s\n' "TAPES_OFFLINE_V1_REPRO_MISMATCH"
  exit "$result"
else
  printf '%s\n' "TAPES_OFFLINE_V1_REPRO_FAILED"
  exit "$result"
fi
