#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/model}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen38-flash-next-nvfp4}"
MASTER_ADDR="${MASTER_ADDR:-10.10.10.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
NODE_RANK="${NODE_RANK:-0}"
SPEC_TOKENS="${SPEC_TOKENS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
PORT="${PORT:-8888}"

args=(
  "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --quantization modelopt_fp4
  --tensor-parallel-size 2
  --pipeline-parallel-size 1
  --nnodes 2
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
  --node-rank "$NODE_RANK"
  --distributed-executor-backend mp
  --enforce-eager
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens 8192
  --enable-chunked-prefill
  --gpu-memory-utilization 0.85
  --tool-call-parser qwen3_coder
  --enable-auto-tool-choice
  --reasoning-parser qwen3
  --host 0.0.0.0
  --port "$PORT"
)

if (( SPEC_TOKENS > 0 )); then
  args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}")
fi

# Prefix caching remains disabled during qualification. On GB10, vLLM #54173
# requires two additional Mamba cache fixes before it is safe to enable.
if [[ "${ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
  args+=(--enable-prefix-caching)
fi
if [[ "${HEADLESS:-0}" == "1" ]]; then
  args+=(--headless)
fi

exec vllm serve "${args[@]}" "$@"
