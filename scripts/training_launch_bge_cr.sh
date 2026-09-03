#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' \
  "CategoryRank/Tapes processing is suspended pending new owner guidance." >&2
exit 2

root="${DGX2_TRAINING_ROOT:-$HOME/harness-training}"
image="${BGE_REPRO_IMAGE:-harness/bge-repro-gb10:prod-20260829}"
output=""
epochs=1
launch=false
corpus_sha256="d822f07c7a0458424daa3cc18b88bb6b936f091acb6bc16cfa9c13c8ab66e61d"

usage() {
  cat <<'EOF'
Usage: training_launch_bge_cr.sh [options]
  --root PATH       DGX2-owned training root
  --image IMAGE     pinned FlagEmbedding training image
  --output PATH     empty output directory below ROOT/checkpoints
  --epochs N        training epochs (default: 1)
  --launch          execute; otherwise print the validated plan
EOF
}

while (($#)); do
  case "$1" in
    --root) root="${2:?missing --root value}"; shift 2 ;;
    --image) image="${2:?missing --image value}"; shift 2 ;;
    --output) output="${2:?missing --output value}"; shift 2 ;;
    --epochs) epochs="${2:?missing --epochs value}"; shift 2 ;;
    --launch) launch=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$(hostname -s)" == "spark-49af" ]] ||
  { printf '%s\n' "CR training must run on DGX2 (spark-49af)" >&2; exit 2; }
[[ -f "$root/.harness-training-owner-v1" ]] ||
  { printf '%s\n' "training root is not Harness-owned" >&2; exit 2; }
[[ "$epochs" =~ ^[1-9][0-9]*$ ]] && ((epochs <= 10)) ||
  { printf '%s\n' "epochs must be between 1 and 10" >&2; exit 2; }

corpus="$root/datasets/cr-local-train-v0/cr_bge_m3_joint_corpus_v2_20260818T171746Z.flagembedding.jsonl"
model="$root/models/bge-m3"
output="${output:-$root/checkpoints/cr-bge-m3-language-geometry-pilot}"
case "$output" in
  "$root"/checkpoints/*) ;;
  *) printf '%s\n' "output must be below $root/checkpoints" >&2; exit 2 ;;
esac
[[ -f "$corpus" ]] ||
  { printf '%s\n' "owner-approved CR corpus is missing" >&2; exit 2; }
printf '%s  %s\n' "$corpus_sha256" "$corpus" | sha256sum --check --strict
[[ -f "$model/config.json" && -f "$model/pytorch_model.bin" ]] ||
  { printf '%s\n' "stock BGE-M3 checkpoint is incomplete" >&2; exit 2; }
if [[ -e "$output" && -n "$(ls -A "$output" 2>/dev/null)" ]]; then
  printf '%s\n' "output directory is not empty: $output" >&2
  exit 2
fi

command=(
  docker run --rm
  --gpus all
  --network none
  --ipc host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --user "$(id -u):$(id -g)"
  --env HOME=/tmp
  --mount "type=bind,src=$root,dst=/training"
  --entrypoint python
  "$image"
  -m torch.distributed.run
  --nproc_per_node 1
  -m FlagEmbedding.finetune.embedder.encoder_only.m3
  --output_dir "/training/${output#"$root"/}"
  --model_name_or_path "/training/${model#"$root"/}"
  --train_data "/training/${corpus#"$root"/}"
  --learning_rate 1e-5
  --fp16
  --num_train_epochs "$epochs"
  --per_device_train_batch_size 4
  --gradient_accumulation_steps 8
  --dataloader_drop_last True
  --normalize_embeddings True
  --temperature 0.02
  --query_max_len 128
  --passage_max_len 128
  --train_group_size 8
  --negatives_cross_device
  --logging_steps 25
  --save_steps 1000
  --save_total_limit 2
  --query_instruction_for_retrieval ""
  --query_instruction_format '{}{}'
  --knowledge_distillation False
  --unified_finetuning True
  --use_self_distill True
  --self_distill_start_step 0
  --warmup_ratio 0.1
  --weight_decay 0.01
)

if [[ "$launch" != true ]]; then
  printf 'Validated CR owner-contract pilot plan:\n'
  printf '  corpus_sha256=%s\n' "$corpus_sha256"
  printf '  output=%s\n' "$output"
  printf '  '
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$output"
exec "${command[@]}"
