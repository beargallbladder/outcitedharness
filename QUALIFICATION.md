# Harness v1 Qualification

Evidence date: 2026-08-30

## Release decision

The M5 control plane, local model fleet, LiteLLM boundary, dual-node Qwen
OpenAI-compatible contract, and ephemeral sandbox lifecycle are qualified.
Cline is configured to use that Qwen endpoint directly; the API contract is
qualified, while the VS Code GUI remains a manual UX canary. Training and
vision outputs remain evidence-gated: neither the MCU LoRA nor the CR BGE
pilot is approved for production. The ASUS1/DGX2 direct-link transport is now
qualified for two-node experiments; production promotion still requires a
successful workload-specific training run.

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
- The ASUS1/DGX2 training link exposes two active 200 Gb/s ConnectX-7/RoCE
  rails. Both use MTU 9000, passed bidirectional jumbo packets, survived
  NetworkManager profile reactivation, and passed the pinned two-rank NCCL
  correctness test.
- Dual-rail NCCL measured 23.057 Gb/s at 256 MiB. Primary-only and
  secondary-only runs both remained correct at 10.8146 and 10.6932 Gb/s,
  respectively, proving that either rail can carry the two-rank workload.
- VS Code and Cursor have Cline 4.1.16 pinned and configured directly to
  `http://100.68.133.1:8888/v1`, model
  `qwen38-flash-next-nvfp4-sglang`. The direct contract passed model discovery,
  exact usage fields, chat, JSON, native tool calls, tool-result follow-up,
  determinism, streaming, and concurrency. Cline—not Harness—owns this
  interactive tool loop.
- The repository test suite passed: 337 tests, one third-party deprecation
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
- All CategoryRank and Tapes export, evaluation, backfill, and training is
  suspended pending new owner guidance on cleanliness, lineage, and approved
  use. Historical contracts and checksums are provenance only; they do not
  authorize a smoke test or reproduction.
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

## DGX2-ASUS1 direct-link qualification

The hardware gate completed on 2026-08-30:

1. `enp1s0f1np1` and `enP2p1s0f1np1` negotiated 200 Gb/s full duplex on both
   systems and mapped to active `rocep1s0f1` and `roceP2p1s0f1` HCAs.
2. DGX2 uses `10.77.0.1`/`10.77.1.1`; ASUS1 uses
   `10.77.0.2`/`10.77.1.2`. Both isolated profiles use MTU 9000 and are
   prohibited from becoming default routes.
3. Link doctors, normal and 8972-byte ICMP payloads, two-rank all-reduce, and
   both single-rail fallback runs passed.
4. DGX2 remains the authoritative dataset/checkpoint writer; ASUS1 remains
   ephemeral scratch.
5. Results and their SHA-256 manifest are sealed below
   `/home/samkim2/harness-training/{runs,manifests}` on DGX2. The manifest
   digest is
   `1f7be26b29b5c8e7d0bebf2fdc9f1db738b0628c58420c0134db82b2dd95218f`.

This qualifies the transport, not an arbitrary training recipe. A model- or
framework-specific distributed run must still pass before that workload is
called production-ready.
