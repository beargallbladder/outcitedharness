# Training deployment assets

These assets prepare only DGX2 and ASUS1. DGX2 is the authoritative durable
store; ASUS1 is disposable scratch with a hard 250 GiB free-space floor.
Nothing here starts, stops, or edits services on ASUS2, ASUS4, or DGX3.

## Safety model

- Storage preparation is a no-op unless `--apply` is present.
- Remote preparation additionally requires an explicit `--host dgx2` or
  `--host asus1`; the remote hostname must match the selected role.
- A non-empty directory without `.harness-training-owner-v1` is never claimed.
- Existing directories are not recursively re-owned or re-permissioned.
- No script performs cleanup. Below-floor ASUS1 scratch blocks launch and
  requires an operator to decide what owned data can be archived or removed.
- Tokens come only from the launch process environment. Templates contain no
  credential fields, and launch scripts do not print argument values.
- Launches are plans unless `--launch` is present.

## 1. Prepare storage

Copy `storage.env.example` to a private operator file and adjust paths. The
integrated Spark NVMe filesystems are valid: the capacity and free-space gates
matter, not whether Linux mounted the device at `/`. Do not add credentials.
The DGX2 filesystem must report at least 4,000,000,000,000 bytes total. ASUS1
must report at least 268,435,456,000 bytes (250 GiB) free.

Review plans locally:

```shell
bash scripts/training_prepare_storage.sh \
  --role dgx2 --root /home/samkim2/harness-training
bash scripts/training_prepare_storage.sh \
  --role asus1 --root /home/samkimasus1/harness-training
```

Apply remotely only after reviewing the paths and ownership:

```shell
bash scripts/training_prepare_storage.sh \
  --role dgx2 --host dgx2 --root /home/samkim2/harness-training \
  --owner samkim2 --group samkim2 --apply
bash scripts/training_prepare_storage.sh \
  --role asus1 --host asus1 --root /home/samkimasus1/harness-training \
  --owner samkimasus1 --group samkimasus1 --apply
```

DGX2 receives durable `datasets`, `models`, `artifacts`, `checkpoints`,
`manifests`, `runs`, and `configs` directories. Its explicit cache roots are:

```text
HF_HOME=<DGX2_ROOT>/cache/huggingface
HUGGINGFACE_HUB_CACHE=<DGX2_ROOT>/cache/huggingface/hub
TRANSFORMERS_CACHE=<DGX2_ROOT>/cache/huggingface/transformers
HF_DATASETS_CACHE=<DGX2_ROOT>/cache/huggingface/datasets
TORCH_HOME=<DGX2_ROOT>/cache/torch
TORCH_EXTENSIONS_DIR=<DGX2_ROOT>/cache/torch/extensions
XDG_CACHE_HOME=<DGX2_ROOT>/cache/xdg
```

The launch scripts set these directly instead of relying on user defaults.
ASUS1 uses the equivalent hierarchy below its scratch root only for its
rank-local cache.

## 2. Record and verify artifacts

Manifests are deterministic, contain SHA-256 plus byte size for every regular
file, reject symlinks, and refuse to overwrite a changed manifest:

```shell
python3 scripts/training_manifest.py create \
  /mnt/dgx2/training/artifacts/run-001 \
  /mnt/dgx2/training/manifests/run-001.sha256.json
python3 scripts/training_manifest.py verify \
  /mnt/dgx2/training/artifacts/run-001 \
  /mnt/dgx2/training/manifests/run-001.sha256.json
```

Run verification before and after any intentional artifact transfer. Use a new
manifest path when artifact content changes.

## 3. Check a future direct link

The doctor is read-only and defaults to warnings, so it is useful before a
cable exists. Run it on both training nodes:

```shell
bash scripts/training_link_doctor.sh \
  --host dgx2 --interface enp65s0f0 --mtu 9000 --peer 10.77.0.2
bash scripts/training_link_doctor.sh \
  --host asus1 --interface enp65s0f0 --mtu 9000 --peer 10.77.0.1
```

It inspects interface state/address/routes, MTU, RDMA tools and mapping, GPU
topology, the NCCL library, and PyTorch CUDA/NCCL support. Add
`--require-ready` only once absence should fail automation.

## 4. Launch SpecForge

Place the real SpecForge configuration below
`<DGX2_ROOT>/configs/`. Config syntax belongs to the installed SpecForge
version, so this repository does not provide a guessed schema. The canonical
entry point is `specforge train --config`; the CLI itself creates the audited
process topology from `deployment.trainer`.

Single-node DGX2 plan, then launch:

```shell
bash scripts/training_launch_single.sh \
  --root /home/samkim2/harness-training \
  --config /home/samkim2/harness-training/configs/job.yaml
bash scripts/training_launch_single.sh \
  --root /home/samkim2/harness-training \
  --config /home/samkim2/harness-training/configs/job.yaml \
  --launch
```

For a two-node consumer, the DGX2 store and SpecForge installation must be
visible at the same paths on ASUS1 (the direct link can carry an NFS mount).
The YAML must declare `deployment.trainer.nnodes: 2` and
`nproc_per_node: 1`, matching the one GB10 GPU in each Spark. Launch any
required producer separately, then start consumer rank 0 on DGX2 and rank 1 on
ASUS1 with identical config, master, interface, and overrides:

```shell
# DGX2
bash scripts/training_launch_two_node.sh \
  --node-rank 0 --store-root /mnt/dgx2-training \
  --scratch-root /home/samkimasus1/harness-training \
  --config /mnt/dgx2-training/configs/job.yaml \
  --master-addr 10.77.0.1 --master-port 29500 \
  --interface enp65s0f0 --launch

# ASUS1
bash scripts/training_launch_two_node.sh \
  --node-rank 1 --store-root /mnt/dgx2-training \
  --scratch-root /home/samkimasus1/harness-training \
  --config /mnt/dgx2-training/configs/job.yaml \
  --master-addr 10.77.0.1 --master-port 29500 \
  --interface enp65s0f0 --launch
```

Extra SpecForge arguments go after `--`. Keep tokens out of command-line
arguments because process listings may expose them.

## 5. Reproduce the owner-signed Tapes v1 gate

Never evaluate v1 through live `:8800` or the `latest` symlink. The immutable
checkpoint is
`models/bge-m3-cr-tapes-v1/checkpoints_20260427T183216Z`; its
`model.safetensors` SHA-256 is
`f1e18565298c5f528b9c4aabb04b5460b9b5ec1cb5dd6f653ad0ea754e977ed8`.

Build the isolated runtime and execute the no-network gate on DGX2:

```shell
docker build \
  --file configs/BGERepro.GB10.Dockerfile \
  --tag harness/bge-repro-gb10:prod-20260829 \
  configs
BGE_REPRO_IMAGE=harness/bge-repro-gb10:prod-20260829 \
  bash scripts/run_tapes_offline_repro.sh
```

The owner contract pins FlagEmbedding fp16, `text[:512]`, Kim-tag batch 64,
and retrieval batch 256. Acceptance permits at most one changed Kim-tag
decision and one changed retrieval query; every category-alignment recall
metric must match exactly. The script exits successfully only when that
contract passes. It binds the checkpoint server to container loopback and
runs the entire evaluation with `--network none`.

## 6. Run the CR-owned language-geometry pilot

The only approved CR training input is
`datasets/cr-local-train-v0/cr_bge_m3_joint_corpus_v2_20260818T171746Z.flagembedding.jsonl`
(11,990 rows, SHA-256
`d822f07c7a0458424daa3cc18b88bb6b936f091acb6bc16cfa9c13c8ab66e61d`).
It contains contrastive language geometry, not raw brand/category memory
facts. Raw `category_mentions_v2`, FOUNDATION snapshots, successor/acquisition
identity pairs, and the packaged eval files remain forbidden as training
inputs.

Review the validated plan and then launch on DGX2:

```shell
bash scripts/training_launch_bge_cr.sh
bash scripts/training_launch_bge_cr.sh --launch
```

The launcher verifies the owner marker and corpus checksum, requires the
sealed local stock BGE-M3 checkpoint, writes only below the DGX2 checkpoint
root, and runs without network access. It never changes live `:8800`; any
trained checkpoint remains non-production until the owner-defined frozen
evaluations pass.

Post-training qualification uses the checksum-pinned contrastive holdout and
then the decision-level open-set gate:

```shell
docker run --rm --gpus all --network none --ipc host \
  --user "$(id -u):$(id -g)" --env HOME=/tmp \
  --mount "type=bind,src=$HOME/harness-training,dst=/training" \
  --entrypoint python harness/bge-repro-gb10:prod-20260829 \
  /training/scripts/evaluate_bge_holdout.py \
  --holdout /training/datasets/tapes-holdout-v1/joint_corpus_v2_tapes_holdout_eval.jsonl \
  --holdout-sha256 27d986bd1a9a6c8713c2c58400017cab1fd561ab191a2316d143fd2ed5adb7d0 \
  --baseline /training/models/bge-m3 \
  --candidate /training/checkpoints/cr-bge-m3-language-geometry-pilot \
  --output /training/evaluations/cr-bge-m3-language-geometry-pilot-holdout.json

BGE_MODEL_RELATIVE=checkpoints/cr-bge-m3-language-geometry-pilot \
TAPES_REPRO_OUTPUT_RELATIVE=evaluations/cr-bge-m3-language-geometry-pilot-open-set.json \
  bash scripts/run_tapes_offline_repro.sh
```

The first one-epoch pilot improved broad holdout pos@1 from 73.33% to 75.87%
but regressed two task types and failed the open-set gate (Kim top1 76.50% to
70.07%; retrieval R@1 31.03% to 26.60%). It is sealed as a no-promote
experiment. Do not resume or alter its curriculum without a new CR owner
ruling.
