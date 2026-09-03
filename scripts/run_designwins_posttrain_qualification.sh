#!/usr/bin/env bash
set -euo pipefail

root="${TRAINING_ROOT:-$HOME/harness-training}"
peer="${PEER_SSH:-samkimasus1@10.77.0.2}"
peer_root="${PEER_ROOT:-/home/samkimasus1/harness-training}"
peer_key="${PEER_KEY:-$HOME/.ssh/harness_training_ed25519}"
deadline_epoch="${DEADLINE_EPOCH:-1788159600}" # 2026-08-31T07:00:00Z
runtime_image="sha256:728b352622d569b1343cb384dd8aa394917203d02631322b02d302d068e38139"
job_id="${JOB_ID:-designwins-text-qwen3-8b-full-20260830}"
adapter_relative="checkpoints/designwins-text-qwen3-8b-full-20260830"
adapter="$root/$adapter_relative"
baseline_relative="evaluations/baseline-full-141.json"
baseline_raw_relative="evaluations/baseline-full-141.raw.json"
candidate_relative="evaluations/candidate-full-141.json"
candidate_raw_relative="evaluations/candidate-full-141.raw.json"
repeat_relative="evaluations/candidate-repeat-full-141.json"
repeat_raw_relative="evaluations/candidate-repeat-full-141.raw.json"
qualification_relative="evaluations/designwins-full-qualification.json"
checkpoint_manifest_relative="manifests/$job_id.checkpoint.sha256.json"
model_manifest_relative="manifests/qwen3-8b-model-20260830.sha256.json"
training_container="designwins-full-train-20260830"
baseline_container="designwins-baseline-full-141-20260830"
candidate_container="designwins-candidate-full-141-20260830"
repeat_container="designwins-candidate-repeat-full-141-20260830"
resume_container="designwins-resume-smoke-20260830"
lease_file="$root/ledger/$job_id.lease.json"
lease_container="/training/ledger/$job_id.lease.json"

ssh_peer=(ssh -i "$peer_key" -o BatchMode=yes -o ConnectTimeout=10 "$peer")
rsync_peer=(rsync -a -e "ssh -i $peer_key -o BatchMode=yes -o ConnectTimeout=10")

test -f "$root/.harness-training-owner-v1"
test -f "$peer_key"
test -f "$lease_file"
test -x "$root/scripts/run_designwins_evaluation.sh"
test -x "$root/scripts/run_designwins_resume_smoke.sh"
test -f "$root/scripts/designwins_resume_smoke.py"
test -f "$root/scripts/compare_designwins_qualification.py"
test -f "$root/scripts/seal_designwins_evaluation.py"
test -f "$root/scripts/training_manifest.py"
test -f "$root/scripts/llamafactory_designwins_text_full.yaml"
test -f "$root/control/learning-factory-20260830/training_queue_admin.py"
test -f "$root/control/learning-factory-20260830/record_designwins_evaluation.py"

remaining() {
  local value=$((deadline_epoch - $(date -u +%s)))
  ((value > 0)) || return 1
  printf '%s\n' "$value"
}

control_python() {
  docker run --rm \
    --entrypoint python \
    --network none \
    --user "$(id -u):$(id -g)" \
    -e PYTHONPATH=/workspace/Harnessv1 \
    -v "$root/control/learning-factory-20260830:/workspace/Harnessv1:ro" \
    -v "$root/ledger:/training/ledger" \
    -v "$root/evaluations:/training/evaluations:ro" \
    -v "$root/runs:/training/runs:ro" \
    -v "$root/checkpoints:/training/checkpoints:ro" \
    "$runtime_image" "$@"
}

queue_admin() {
  control_python \
    /workspace/Harnessv1/training_queue_admin.py \
    --database /training/ledger/learning.db "$@"
}

reject_job() {
  local expected="$1"
  local reason="$2"
  if [[ "$expected" == "assigned" ]]; then
    queue_admin fail \
      --job-id "$job_id" \
      --lease-file "$lease_container" \
      --error "$reason" \
      --terminal || true
    rm -f "$lease_file"
  else
    queue_admin transition \
      --job-id "$job_id" --expected "$expected" --target rejected || true
  fi
}

cleanup() {
  docker rm -f "$resume_container" >/dev/null 2>&1 || true
  docker rm -f "$candidate_container" >/dev/null 2>&1 || true
  "${ssh_peer[@]}" docker rm -f "$repeat_container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

while docker inspect "$training_container" >/dev/null 2>&1; do
  remaining >/dev/null || {
    docker stop -t 30 "$training_container" >/dev/null 2>&1 || true
    reject_job assigned "training missed the immutable midnight deadline"
    echo "Training missed the immutable midnight deadline" >&2
    exit 1
  }
  sleep 30
done
test -s "$adapter/adapter_model.safetensors" || {
  reject_job assigned "training exited without an adapter checkpoint"
  echo "Training exited without an adapter checkpoint" >&2
  exit 1
}
sudo chown -R "$(id -u):$(id -g)" "$adapter"
adapter_sha="$(sha256sum "$adapter/adapter_model.safetensors" | awk '{print $1}')"

seconds="$(remaining)" || {
  reject_job assigned "deadline expired before resume verification"
  exit 1
}
if ! timeout --foreground --kill-after=60 "$seconds" \
  env CONTAINER_NAME="$resume_container" \
    bash "$root/scripts/run_designwins_resume_smoke.sh"; then
  reject_job assigned "final adapter resume reproduction failed"
  echo "Final adapter resume reproduction failed or exceeded the deadline" >&2
  exit 1
fi
resume_adapter_sha="$(
  python3 - "$root/runs/designwins-resume-smoke-20260830.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["sha256"]["candidate_adapter"])
PY
)"
test "$resume_adapter_sha" = "$adapter_sha" || {
  reject_job assigned "resume proof does not bind the final adapter"
  exit 1
}

checkpoint_manifest="$root/$checkpoint_manifest_relative"
model_manifest="$root/$model_manifest_relative"
test ! -e "$checkpoint_manifest"
test ! -e "$model_manifest"
python3 "$root/scripts/training_manifest.py" \
  create "$adapter" "$checkpoint_manifest"
python3 "$root/scripts/training_manifest.py" \
  create "$root/models/Qwen3-8B" "$model_manifest"

queue_admin complete \
  --job-id "$job_id" \
  --lease-file "$lease_container" \
  --checkpoint-uri "file:///training/$adapter_relative/adapter_model.safetensors" \
  --checkpoint-sha256 "$adapter_sha"
rm -f "$lease_file"

while ! "${ssh_peer[@]}" test -s "$peer_root/$baseline_relative"; do
  remaining >/dev/null || {
    reject_job trained "baseline missed the immutable midnight deadline"
    echo "Baseline missed the immutable midnight deadline" >&2
    exit 1
  }
  "${ssh_peer[@]}" docker inspect "$baseline_container" >/dev/null 2>&1 || {
    reject_job trained "baseline exited without a complete evaluation"
    echo "Baseline exited without a complete evaluation" >&2
    exit 1
  }
  sleep 30
done

"${ssh_peer[@]}" mkdir -p \
  "$peer_root/$adapter_relative" \
  "$peer_root/manifests"
"${rsync_peer[@]}" "$adapter/" "$peer:$peer_root/$adapter_relative/"
"${rsync_peer[@]}" \
  "$checkpoint_manifest" "$model_manifest" \
  "$peer:$peer_root/manifests/"
"${ssh_peer[@]}" \
  python3 "$peer_root/scripts/training_manifest.py" verify \
    "$peer_root/$adapter_relative" \
    "$peer_root/$checkpoint_manifest_relative"
"${ssh_peer[@]}" \
  python3 "$peer_root/scripts/training_manifest.py" verify \
    "$peer_root/models/Qwen3-8B" \
    "$peer_root/$model_manifest_relative"

test ! -e "$root/$candidate_raw_relative"
test ! -e "$root/$candidate_relative"
test ! -e "$root/$repeat_raw_relative"
test ! -e "$root/$repeat_relative"
"${ssh_peer[@]}" test ! -e "$peer_root/$baseline_raw_relative"
"${ssh_peer[@]}" test ! -e "$peer_root/$repeat_raw_relative"
"${ssh_peer[@]}" test ! -e "$peer_root/$repeat_relative"
"${ssh_peer[@]}" \
  "sudo chown \"\$(id -u):\$(id -g)\" '$peer_root/$baseline_relative'"
"${ssh_peer[@]}" mv \
  "$peer_root/$baseline_relative" "$peer_root/$baseline_raw_relative"
"${ssh_peer[@]}" \
  python3 "$peer_root/scripts/seal_designwins_evaluation.py" \
    --source "$peer_root/$baseline_raw_relative" \
    --output "$peer_root/$baseline_relative" \
    --dataset "$peer_root/datasets/designwins-v3-20260829/llamafactory/designwins_text_test.json" \
    --model-manifest "$peer_root/$model_manifest_relative" \
    --scorer "$peer_root/scripts/evaluate_designwins_text.py" \
    --runtime-image-id "$runtime_image" \
    --max-samples 141 \
    --cutoff-len 4096 \
    --max-new-tokens 8192 \
    --batch-size 8 \
    --generation-slack-tokens 256

seconds="$(remaining)" || {
  reject_job trained "deadline expired before candidate evaluation"
  exit 1
}
timeout --foreground --kill-after=60 "$seconds" \
  bash "$root/scripts/run_designwins_evaluation.sh" \
    --root "$root" \
    --output-relative "$candidate_raw_relative" \
    --adapter-relative "$adapter_relative" \
    --container-name "$candidate_container" &
candidate_pid=$!
timeout --foreground --kill-after=60 "$seconds" \
  "${ssh_peer[@]}" \
    bash "$peer_root/scripts/run_designwins_evaluation.sh" \
      --root "$peer_root" \
      --output-relative "$repeat_raw_relative" \
      --adapter-relative "$adapter_relative" \
      --container-name "$repeat_container" &
repeat_pid=$!

set +e
wait "$candidate_pid"
candidate_status=$?
wait "$repeat_pid"
repeat_status=$?
set -e
if ((candidate_status != 0 || repeat_status != 0)); then
  reject_job trained "candidate evaluation failed or exceeded the deadline"
  echo "Candidate evaluation failed or exceeded the deadline" >&2
  exit 1
fi

python3 "$root/scripts/seal_designwins_evaluation.py" \
  --source "$root/$candidate_raw_relative" \
  --output "$root/$candidate_relative" \
  --dataset "$root/datasets/designwins-v3-20260829/llamafactory/designwins_text_test.json" \
  --model-manifest "$model_manifest" \
  --scorer "$root/scripts/evaluate_designwins_text.py" \
  --runtime-image-id "$runtime_image" \
  --adapter-manifest "$checkpoint_manifest" \
  --max-samples 141 \
  --cutoff-len 4096 \
  --max-new-tokens 8192 \
  --batch-size 8 \
  --generation-slack-tokens 256
"${ssh_peer[@]}" \
  python3 "$peer_root/scripts/seal_designwins_evaluation.py" \
    --source "$peer_root/$repeat_raw_relative" \
    --output "$peer_root/$repeat_relative" \
    --dataset "$peer_root/datasets/designwins-v3-20260829/llamafactory/designwins_text_test.json" \
    --model-manifest "$peer_root/$model_manifest_relative" \
    --scorer "$peer_root/scripts/evaluate_designwins_text.py" \
    --runtime-image-id "$runtime_image" \
    --adapter-manifest "$peer_root/$checkpoint_manifest_relative" \
    --max-samples 141 \
    --cutoff-len 4096 \
    --max-new-tokens 8192 \
    --batch-size 8 \
    --generation-slack-tokens 256

test ! -e "$root/$baseline_relative"
"${rsync_peer[@]}" \
  "$peer:$peer_root/$baseline_relative" \
  "$root/$baseline_relative"
"${rsync_peer[@]}" \
  "$peer:$peer_root/$baseline_raw_relative" \
  "$root/$baseline_raw_relative"
"${rsync_peer[@]}" \
  "$peer:$peer_root/$repeat_relative" \
  "$root/$repeat_relative"
"${rsync_peer[@]}" \
  "$peer:$peer_root/$repeat_raw_relative" \
  "$root/$repeat_raw_relative"
"${rsync_peer[@]}" \
  "$peer:$peer_root/${repeat_raw_relative%.json}.log" \
  "$root/${repeat_raw_relative%.json}.log"

set +e
python3 "$root/scripts/compare_designwins_qualification.py" \
  --baseline "$root/$baseline_relative" \
  --candidate "$root/$candidate_relative" \
  --candidate-repeat "$root/$repeat_relative" \
  --output "$root/$qualification_relative"
qualification_status=$?
set -e

run_root="$root/runs/$job_id-qualification"
run_manifest="$root/manifests/$job_id.qualification.sha256.json"
promotion_decision="$root/evaluations/designwins-offline-promotion.json"
test ! -e "$run_root"
test ! -e "$run_manifest"
test ! -e "$promotion_decision"

set +e
control_python \
  /workspace/Harnessv1/record_designwins_evaluation.py \
  --database /training/ledger/learning.db \
  --qualification "/training/$qualification_relative" \
  --resume-summary /training/runs/designwins-resume-smoke-20260830.json \
  --job-id "$job_id" \
  --dataset-version-id designwins-text-v3-20260829 \
  --candidate-sha256 "$adapter_sha" \
  --evaluation-id designwins-full-offline-20260830 \
  >"$promotion_decision"
promotion_status=$?
set -e
test -s "$promotion_decision" || {
  reject_job trained "offline promotion evidence could not be recorded"
  exit 1
}
queue_admin transition \
  --job-id "$job_id" --expected trained --target evaluated

mkdir "$run_root"
rsync -a \
  "$root/scripts/llamafactory_designwins_text_full.yaml" \
  "$root/runs/designwins-full-train-20260830.log" \
  "$root/runs/designwins-resume-smoke-20260830.yaml" \
  "$root/runs/designwins-resume-smoke-20260830.json" \
  "$root/runs/designwins-resume-smoke-20260830.log" \
  "$root/$baseline_relative" \
  "$root/$baseline_raw_relative" \
  "$root/$candidate_relative" \
  "$root/$candidate_raw_relative" \
  "$root/${candidate_raw_relative%.json}.log" \
  "$root/$repeat_relative" \
  "$root/$repeat_raw_relative" \
  "$root/${repeat_raw_relative%.json}.log" \
  "$root/$qualification_relative" \
  "$promotion_decision" \
  "$checkpoint_manifest" \
  "$model_manifest" \
  "$run_root/"
python3 "$root/scripts/training_manifest.py" create "$run_root" "$run_manifest"
python3 "$root/scripts/training_manifest.py" verify "$run_root" "$run_manifest"

remaining >/dev/null || {
  queue_admin transition \
    --job-id "$job_id" --expected evaluated --target rejected
  echo "Evidence completed after the immutable midnight deadline" >&2
  exit 1
}
if ((qualification_status == 0 && promotion_status == 0)); then
  queue_admin transition \
    --job-id "$job_id" --expected evaluated --target shadow
  echo "DESIGNWINS_QUALIFICATION_COMPLETE passed=true state=shadow"
  exit 0
fi
queue_admin transition \
  --job-id "$job_id" --expected evaluated --target rejected
echo "DESIGNWINS_QUALIFICATION_COMPLETE passed=false state=rejected"
exit 1
