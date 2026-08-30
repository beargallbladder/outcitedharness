# Harness v1 Qualification

Evidence date: 2026-08-29

## Release decision

The M5 control plane, local model fleet, LiteLLM boundary, optional Cline
frontend, and ephemeral sandbox lifecycle are qualified. Training and vision
outputs remain evidence-gated: neither the MCU LoRA nor the CR BGE pilot is
approved for production. The only unfinished hardware requirement is the
ASUS1/DGX2 direct-link cable qualification, which cannot run before the cable
is installed.

## Qualified boundaries

- LiteLLM and Harness launch services are running on loopback-only ports 7410
  and 8787.
- `scripts/litellm_qualification.py` passed authentication, model discovery,
  basic completion, usage, JSON, tool calls, follow-up, streaming,
  orchestration, direct/proxy latency, no-automatic-fallback, and three-way
  concurrency checks.
- Direct/proxy overhead was 5.1 ms for Qwen Coder, 24.9 ms for Qwen3.8, and
  16.5 ms for Nemotron. Three concurrent coder requests produced 768 tokens at
  an aggregate 78.799 tokens/s.
- Live fleet validation passed for DGX3 Qwen Coder, M5 Qwen foreman,
  dual-Spark Qwen3.8 peer foreman/primary critic, ASUS3 Nemotron fallback
  critic, protected Spark embedder, and configured hosted fallbacks.
- Cline is installed and configured as an optional OpenAI-compatible client of
  LiteLLM. It has no direct physical-worker route.
- The repository test suite passed: 328 tests, one third-party deprecation
  warning, zero failures.

## Sandbox qualification

The M5 sandbox CLI has durable SQLite lifecycle evidence, bounded resources,
TTL cleanup, Caddy ingress, Tailscale preview integration, and greenfield
promotion gates. Qualification canaries were created, tested, stopped, and
removed; the registry retains all records in `removed` state and has no active
canary containers.

## Data and training controls

- DGX2 is the authoritative training host. Its owner marker is present and
  3.52 TB was free at qualification.
- ASUS1 has its owner-marked scratch root and 789 GB free, but is not an
  authoritative checkpoint host.
- CategoryRank and Tapes data are usable only under explicit owner contracts.
  Raw `category_mentions_v2` is not a training set.
- The only approved CR pilot input was the pinned 11,990-row language-geometry
  pack with SHA-256
  `d822f07c7a0458424daa3cc18b88bb6b936f091acb6bc16cfa9c13c8ab66e61d`.
- Tapes v1 reproduction uses the immutable v1 checkpoint, FlagEmbedding 1.4.0,
  fp16, and `text[:512]` in a no-network container. It never evaluates v1
  through live port 8800.
- Owners confirmed that live Spark port 8800 has loaded FAE v4 weights behind
  a legacy `bge-m3-cr-tapes-v1` request label since 2026-08-22. Configuration
  preserves the wire label but documents the real identity. Do not restart it,
  flip its symlink, retag CR foundation data from it, or use it as a v1
  baseline.
- DesignWins vision workers remain disabled until the 200-page bakeoff and
  explicit cutover.

## Pilot decisions

### MCU LoRA

The schema-correct evaluation found no improvement over the base model. The
adapter is sealed offline and is not promoted.

### CR BGE language geometry

One epoch completed on DGX2: 11,990 rows, 375 steps, 19m25s runtime, final
training loss 0.8386434.

The pilot improved owner-pinned broad holdout positive-at-1 from 73.33% to
75.87%, but regressed two task types. It then failed the exact open-set gate:

- Kim top1: 76.50% to 70.07%; 1,176 changed decisions.
- Kim top3: 97.26% to 92.28%.
- Retrieval R@1: 31.03% to 26.60%; 45 changed queries.
- Category alignment R@1: 79.31% to 51.72%; 12 changed queries.

Decision: **no promote**. The sealed evidence is under
`$HOME/harness-training/evaluations/cr-bge-m3-language-geometry-pilot` on
DGX2, with manifest
`$HOME/harness-training/manifests/cr-bge-m3-language-geometry-pilot-evidence.sha256.json`.
A next experiment requires a CR owner ruling on taxonomy replay/regularization
or narrower per-family adapters.

## Rollback

- LiteLLM can be stopped with `scripts/install_litellm.sh uninstall`; Harness
  remains independently managed by `scripts/install_harness_orch.sh`.
- Cline can be disabled without affecting Cursor, LiteLLM, or Harness.
- Sandbox rollback is `harness sandbox stop ID` followed by
  `harness sandbox remove ID`; durable event records remain.
- No trained checkpoint is referenced by a live serving configuration.
  Rollback for either failed pilot is therefore non-action: leave it sealed
  offline.
- Physical SGLang services remain independently owned and can be restored from
  their pinned launch scripts. Protected ports 8800 and 8810 are excluded from
  automated restart or repurpose paths.

## Remaining hardware gate

After the direct-link cable is installed between ASUS1 and DGX2:

1. Confirm the hot-plugged ConnectX-7 functions expose the proven
   `enp1s0f1np1`/`enP2p1s0f1np1` rails and
   `rocep1s0f1`/`roceP2p1s0f1` HCAs.
2. Apply the isolated 10.77.0.0/24 and 10.77.1.0/24 addresses with
   `training_configure_link.sh`.
3. Run link, MTU, route, RDMA, NCCL correctness, bandwidth, and
   failure-recovery qualification. The digest-pinned two-rank smoke is staged
   and checksum-verified on both hosts.
4. Confirm DGX2 remains the only authoritative dataset/checkpoint writer and
   ASUS1 remains ephemeral scratch.
5. Seal the qualification logs and manifest on DGX2.

Until those steps pass, do not advertise two-node training as production-ready.
