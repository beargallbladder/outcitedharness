# Model Harness v0.1

Measurement harness plus a Cline gateway. Not a Cursor clone. Hardware today is one M5 control host and one DGX Spark. Extra workers are config only until the boxes exist.

## Live boxes

| Role | Where | Model | Cline id | Worker id |
|---|---|---|---|---|
| Worker | Spark `:8900` | `qwen3-coder-next` (80B MoE, ~3B active) | `harness-local` | `primary_coder` |
| Orchestrator | M5 `:8082` | Qwen3.8-27B-8bit + MTP | `harness-m5` | `fallback_reasoner` |
| Senior | Anthropic | Claude Sonnet 4.6 | `harness-frontier` | `frontier_senior` |

Gateway: `http://127.0.0.1:8787/v1` (`com.samkim.harness-cline`). Cline must use this URL, not `:8900` or `:8082`.

`secondary` / `fast` / `monster` are in `config/workers.yaml` with `enabled: false`. Arrival day is flip `enabled: true`, not a rewrite.

## Two ladders (do not mix)

**Cline `harness-auto`** (dead box only): Spark → M5 → Claude. Next hop only on connect / 5xx / empty. HTTP 200 with a bad answer stays on Spark. `harness-local` does not fail over.

**Tournament** (`config/models.yaml` tiers) is a separate measurement list (DeepSeek, MiniMax, etc.). It is not what Cline `harness-auto` calls.

Quality escalate is `harness rescue PACKET.md` (structured packet to Claude, reject >20k / missing sections). It is not `harness-auto`.

`supportsPromptCache: false` because these local endpoints do not speak Cline's cache protocol. Cline already resends the full thread. The flag stays off so Cline does not report fake `cacheReads`.

## Task / evidence

- `log_turn` writes `cline_turns` then an `Attempt` via `TaskService.session_task()`.
- Session reuse: latest **open** task with intent `cline session` and a turn in the last 30 minutes. `harness task start "fix geocode"` is a different intent and is not reused.
- HTTP 200 does not close a Cline session.
- Alias map: `harness-local` / `harness-auto` → `primary_coder`; `harness-m5` → `fallback_reasoner`; `harness-frontier` → `frontier_senior`.
- CLI: `harness task start|list|show|current|packet|record|decide`

## Failover

`should_failover(status, error, has_next)` is true only when there is a next worker and (`status >= 500` or `error`).

## Measured (2026-08-24)

Spark vs M5 3.8: pong 70ms vs 369ms; ~66 vs 41 tok/s. Chair 15: both 2/15, no overlap. Spark median 5.6s, M5 7.2s. Prior Spark 27B on that pack was 4/15 at 29.7s.

## Storage

`results/harness.db`: `runs`, `case_runs`, `model_results`, `cline_turns`, `tasks`, `attempts`, `evidence`, `decisions`.

## Do not

Dashboard, Kubernetes, swarm, a second Cline, serve 122B as Cline, restart `:8800`–`:8803` or `:8902` from this Mac.
