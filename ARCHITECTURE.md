# Model Harness v0.1

Measurement harness plus a Cline gateway. Not a Cursor clone. Hardware today is one M5 control host and one DGX Spark. Extra workers are config only until the boxes exist.

## Live boxes

| Role | Where | Model | Cline id | Worker id |
|---|---|---|---|---|
| Worker | ASUS gx10-33af (asus4) `:8900` | `qwen3-coder-next` | `harness-local` | `primary_coder` |
| Worker | Spark 49af `:8900` | `qwen3-coder-next` | `harness-dgx2` | `dgx2_coder` |
| Worker | ASUS gx10-fc2e `:8900` | `qwen3-coder-next` | `harness-asus` | `asus_coder` |
| Worker | Spark 69c8 `:8900` | `qwen3-coder-next` | `harness-dgx3` | `dgx3_coder` |
| Orchestrator | M5 `:8082` | Qwen3.8-27B-8bit + MTP | `harness-m5` | `fallback_reasoner` |
| Senior | Anthropic | Claude Sonnet 4.6 | `harness-frontier` | `frontier_senior` |
| Critic | ASUS gx10-0309 `:8900` | Nemotron 3.5 Lightning 30B-A3B NVFP4 | internal | `researcher` |
| Peer foreman | ASUS gx10-26b6 `:8900` | Qwen3-Coder-Next NVFP4 | — (auto, chain only) | `asus2_foreman` |
| Embedder | Spark e10b `:8800` | `bge-m3-cr-tapes-v1` (1024-dim) | — (not a chat model) | `spark_embedder` |

The Spark `:8800` embedder is the CR team's weekly embedding service (`/v1/embeddings`, `/semantic-search`). spark-e10b (.38) is dedicated to embedding/search/training — its coder was retired 2026-08-27. **Never restart or repurpose the embedder.** asus4 is the interim primary coder until the DeepSeek two-Spark pair forms.

Gateway: `http://127.0.0.1:8787/v1` (`com.samkim.harness-cline`). Cline must use this URL, not `:8900` or `:8082`.

`secondary` / `fast` / `monster` are in `config/workers.yaml` with `enabled: false`. Arrival day is flip `enabled: true`, not a rewrite.

## Two ladders (do not mix)

**Cline `harness-auto`** (dead box only): asus4 → M5 → Claude. Next hop only on connect / 5xx / empty. HTTP 200 with a bad answer stays on asus4. `harness-local` does not fail over.

**Cline `harness-orch`**: you type; the harness decides. If the repo is needed, the foreman returns Cline tool calls (Cline is the hands). After those results are in the thread, the foreman slices packets, idle GB10s write answers, and local Nemotron grades factual grounding. Failed packets receive one bounded repair on a different local worker. Only after local QA is exhausted does the harness construct a bounded evidence packet and make at most one frontier rescue call. The frontier answer must pass the local critic before it is returned. Accepted concrete patches can be converted into one real Cline edit call; Cline tool results return for verification.

**Foreman chain**: M5 (`m5_qwen`) orchestrates when reachable; if the M5 is off-network or its LLM is down, asus2 (`asus2_qwen`, gx10-26b6, Qwen3-Coder-Next) takes over as a peer — same prompts, same authority. Health-checked per turn (20s cache), automatic, no user action. If both are down, orch fails closed. Nemotron (asus3) is deliberately NOT in the chain: the grader must stay independent of the planner. Nemotron is the synchronous first critic after a 2026-08-27 retest found and removed an obsolete thinking retry: it scored 7/7 in 3.78s alone and 7/7 in 5.47s or less across four concurrent grades. GLM 5.2 remains the hosted fallback.

MiniMax-M2.5 REAP 139B is **parked** on gx10-26b6 (weights on disk, container stopped, restart=no): this vLLM build has no native FP4 on GB10, and the Marlin repack needs ~2× the 75GB weights — OOM crash-loop on one 121GB box. Revisit with a newer NGC image or a two-Spark pair.

**Dispatch CLI** is the same graph without Cline: `harness dispatch "intent"`.

**Learning loop**: dispatch stages, local attempts, critic grades, frontier cost, and rescue verification are linked to a task in `results/harness.db`. `harness promote TASK_ID --pack cases/learned` turns a locally verified frontier rescue into a human-reviewed regression case. Frontier output is never treated as training truth merely because it came from a frontier model.

**Model rotation**: models and physical workers remain explicit in `config/models.yaml` and `config/workers.yaml`. Role order comes from each worker's `priority`; no foreman or critic chain is hardcoded. Change the YAML, run `harness fleet validate`, then manually restart the gateway. There is no automatic network discovery or hot reload.

**Client boundary**: remote callers see only `harness-orch`, generic readiness, and harness-owned model identifiers. Detailed workers, endpoints, upstream model names, and critic attribution remain in loopback health output, dispatch artifacts, and the internal database.

**Tournament** (`config/models.yaml` tiers) is a separate measurement list (DeepSeek, MiniMax, etc.). It is not what Cline `harness-auto` calls.

Quality escalate is `harness rescue PACKET.md` or `dispatch --senior`. It is not `harness-auto`.

**Fleet optimize** is `harness optimize`: same packet to every enabled coder (A/B the boxes). `dispatch` splits work so the pool stays busy.

`supportsPromptCache: false` because these local endpoints do not speak Cline's cache protocol. Cline already resends the full thread. The flag stays off so Cline does not report fake `cacheReads`.

## Task / evidence

- `log_turn` writes `cline_turns` then an `Attempt` via `TaskService.session_task()`.
- Session reuse: latest **open** task with intent `cline session` and a turn in the last 30 minutes. `harness task start "fix geocode"` is a different intent and is not reused.
- HTTP 200 does not close a Cline session.
- Alias map: `harness-local` / `harness-auto` → `primary_coder`; `harness-dgx2` → `dgx2_coder`; `harness-asus` → `asus_coder`; `harness-dgx3` → `dgx3_coder`; `harness-m5` → `fallback_reasoner`; `harness-frontier` → `frontier_senior`.
- CLI: `harness dispatch|optimize|task start|list|show|current|packet|record|decide`

## Failover

`should_failover(status, error, has_next)` is true only when there is a next worker and (`status >= 500` or `error`).

## Measured (2026-08-24)

Spark vs M5 3.8: pong 70ms vs 369ms; ~66 vs 41 tok/s. Chair 15: both 2/15, no overlap. Spark median 5.6s, M5 7.2s. Prior Spark 27B on that pack was 4/15 at 29.7s.

## Storage

`results/harness.db`: `runs`, `case_runs`, `model_results`, `cline_turns`, `tasks`, `attempts`, `evidence`, `decisions`.

## Do not

Dashboard, Kubernetes, swarm, a second Cline, serve 122B as Cline, restart `:8800`–`:8803` or `:8902` from this Mac.
