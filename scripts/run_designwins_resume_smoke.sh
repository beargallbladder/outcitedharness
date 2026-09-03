#!/usr/bin/env bash
set -euo pipefail

root="${TRAINING_ROOT:-$HOME/harness-training}"
image="${TRAINING_IMAGE:-sha256:728b352622d569b1343cb384dd8aa394917203d02631322b02d302d068e38139}"
container_name="${CONTAINER_NAME:-designwins-resume-smoke-20260830}"
checkpoint_relative="checkpoints/designwins-text-qwen3-8b-full-20260830"
source_config_relative="scripts/llamafactory_designwins_text_full.yaml"
output_relative="runs/designwins-resume-smoke-20260830"
config_relative="runs/designwins-resume-smoke-20260830.yaml"
summary_relative="runs/designwins-resume-smoke-20260830.json"
log_relative="runs/designwins-resume-smoke-20260830.log"

test -f "$root/.harness-training-owner-v1"
test -f "$root/$source_config_relative"
test -f "$root/scripts/designwins_resume_smoke.py"
test -d "$root/$checkpoint_relative"
test ! -e "$root/$output_relative"
test ! -e "$root/$config_relative"
test ! -e "$root/$summary_relative"
docker image inspect "$image" >/dev/null

common=(
  --rm
  --entrypoint /bin/bash
  --network none
  --user "$(id -u):$(id -g)"
  -e HOME=/tmp
  -v "$root:/training"
  "$image"
)

docker run "${common[@]}" -lc \
  'python /training/scripts/designwins_resume_smoke.py prepare \
    --source-config /training/scripts/llamafactory_designwins_text_full.yaml \
    --checkpoint-root /training/checkpoints/designwins-text-qwen3-8b-full-20260830 \
    --output-dir /training/runs/designwins-resume-smoke-20260830 \
    --destination-config /training/runs/designwins-resume-smoke-20260830.yaml'

docker run \
  --name "$container_name" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TOKENIZERS_PARALLELISM=false \
  "${common[@]}" \
  -lc 'llamafactory-cli train \
    /training/runs/designwins-resume-smoke-20260830.yaml' \
  2>&1 | tee "$root/$log_relative"

docker run "${common[@]}" -lc \
  'python /training/scripts/designwins_resume_smoke.py verify \
    --source-config /training/scripts/llamafactory_designwins_text_full.yaml \
    --checkpoint-root /training/checkpoints/designwins-text-qwen3-8b-full-20260830 \
    --output-dir /training/runs/designwins-resume-smoke-20260830 \
    --destination-config /training/runs/designwins-resume-smoke-20260830.yaml \
    --summary /training/runs/designwins-resume-smoke-20260830.json'

test -s "$root/$summary_relative"
