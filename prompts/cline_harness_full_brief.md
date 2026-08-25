Workspace is /Users/samkim/Harnessv1. Do not find(1) $HOME. Do not restart Spark, M5, embeddings, or OCR.

You are Cline talking to the harness gateway at http://127.0.0.1:8787/v1 (Model ID harness-auto or harness-local). You are NOT talking to a raw :8900 or :8082 port.

JOB after you read this packet:
1. Answer the five questions at the bottom in 10 lines or less.
2. Write /Users/samkim/Harnessv1/ARCHITECTURE.md (max 80 lines) from this packet only.
3. Run: cd /Users/samkim/Harnessv1 && .venv/bin/python -m pytest tests/test_workers.py tests/test_failover.py tests/test_task.py -q
4. Stop. Do not enable secondary/fast/monster. Do not edit config/workers.yaml failover order.

# WHAT THIS IS

A measurement harness + Cline gateway. Not a Cursor clone. Not Kubernetes. Not a five-node cluster yet.

Hardware that exists TODAY:
- Control host: Apple M5 Max 128GB. Serves Qwen3.8-27B-8bit + MTP on :8082 (mlx-vlm). Orchestrator / vision. Alias harness-m5 → fallback_reasoner.
- Worker: NVIDIA DGX Spark at 192.168.4.38. vLLM qwen3-coder-next on :8900 (80B MoE, ~3B active, 131k ctx). Alias harness-local → primary_coder.
- Senior: Claude Sonnet 4.6 via ANTHROPIC_API_KEY. Alias harness-frontier → frontier_senior. Quality escalate is `harness rescue PACKET.md`, not auto_ladder.
- Future boxes (secondary, fast, monster) are in workers.yaml with enabled: false. Scheduler must say "unavailable", not crash.

Cline in VS Code:
- Provider: OpenAI Compatible
- Base URL: http://127.0.0.1:8787/v1
- Key: sk-harness-local
- harness-auto = dead-box failover only: Spark → M5 → Claude on connect/5xx/empty. Bad answers do NOT move.
- harness-local = Spark only (no failover). 180s timeout if Spark is hung on auto.

Git: local main, tag v0.1-single-worker. First commit 6488cb8.

Do NOT: start/stop/reconfigure :8800–:8803 (embeddings) or :8902 (OCR). Do not serve 122B. Do not download 35B-A3B. Do not build a dashboard or a second Cline.

# MEASURED

Spark coder vs M5 3.8 (live bang): pong 70ms vs 369ms; medium 66 tok/s vs 41 tok/s; both emit native OpenAI tool_calls.
Chair 15 cases (20260824_060426): both 2/15, 0 overlap, 11 neither. Spark median 5.6s, M5 7.2s.
Older Spark 27B on same pack was 4/15 at 29.7s. Coder-next is faster, not smarter on chair.
Cline resends the full thread every turn (supportsPromptCache: false on purpose). 80k-token tasks make 27B look dead. New Cline task per job.

# CONFIG (live)

## config/workers.yaml
```yaml
workers:
  primary_coder:
    enabled: true
    model_key: dgx_qwen
    endpoint: http://192.168.4.38:8900/v1
    capabilities: [coding, tool_calling, long_context]
    failover_order: 1
  fallback_reasoner:
    enabled: true
    model_key: m5_qwen
    endpoint: http://127.0.0.1:8082/v1
    capabilities: [reasoning, vision, tool_calling]
    failover_order: 2
  frontier_senior:
    enabled: true
    model_key: frontier
    endpoint: https://api.anthropic.com/v1
    capabilities: [review]
    failover_order: 3
    notes: dead-box last hop only; quality escalate is harness rescue
  secondary:
    enabled: false
    endpoint: http://future-dgx-pair-b:8000/v1
    failover_order: 4
  fast:
    enabled: false
    endpoint: http://future-dgx-5:8000/v1
  monster:
    enabled: false
    endpoint: http://future-monster:8000/v1
```

## config/cline.yaml
aliases: harness-auto→auto, harness-local→dgx_qwen, harness-m5→m5_qwen, harness-frontier→frontier
auto_ladder: [dgx_qwen, m5_qwen, frontier]  # MUST match enabled failover_order (tested)
listen: 127.0.0.1:8787
context_window: 131072
max_output_tokens: 8192

## config/models.yaml (live locals only)
dgx_qwen: openai_compatible http://192.168.4.38:8900/v1 model=qwen3-coder-next timeout 180
m5_qwen: openai_compatible http://127.0.0.1:8082/v1 model=/Volumes/M5_4TB/models/Qwen3.8-27B-8bit extra_body.enable_thinking=false vision=true
frontier: anthropic claude-sonnet-4-6 max_tokens 4096

# CODE (core — files are on disk, this is the live text)

## harness/workers/router.py
should_failover(status, error, has_next): True only if has_next and (status>=500 or bool(error)).
route_models → ladder_for.

## harness/gateway/spec.py ladder_for
harness-auto: registry.failover_keys() or spec.auto_ladder
harness-local: [dgx_qwen] only — does NOT fail over
If load_registry(explicit_root) has no workers.yaml, do not walk up into the real repo (tests).

## harness/gateway/server.py
_complete_with_fallback loops models_to_try; should_failover; log_turn on the chosen result.
LaunchAgent: com.samkim.harness-cline → .venv/bin/harness serve

## harness/gateway/proxy.py log_turn
insert_cline_turn then TaskService.session_task() + record_turn.
_worker_for_alias: harness-local|harness-auto→primary_coder, harness-m5→fallback_reasoner, harness-frontier→frontier_senior.
Session: only reuse latest OPEN task with intent "cline session" if last attempt within 30 minutes. Do not attach to "fix geocode". HTTP 200 does not close the session.

## harness/task/models.py
Task, WorkPacket (to_markdown rescue-shaped sections: TASK, RELEVANT ARCHITECTURE, FILES, OBSERVED FAILURE, ATTEMPTS, TEST EVIDENCE, FOREMAN HYPOTHESIS, QUESTION), AttemptRecord.to_evidence_json() shape:
{task_id, worker, attempt, files_changed, commands, tests:{passed,failed}, result, ttft_ms, tokens_per_sec, tool_calls}
Evidence, Decision.

## harness/task/service.py
CLINE_INTENT="cline session" SESSION_IDLE=30min
start/get/list/latest_session/session_task/record(close=True|False)/record_turn/packet/add_evidence/add_decision
record_turn keeps status=open.

## harness/task/search.py
search_code(query, repo, mode=auto|grep|ast|semantic|hybrid)
auto: identifier→grep, class/def→ast, else hybrid.
semantic: empty hits, detail "unavailable (BGE-M3 worker not wired)".
ast: ast-grep if installed else unavailable.
grep/hybrid: ripgrep or python scan.

## harness/rescue.py
harness rescue PACKET.md → frontier only. Rejects missing sections or >20k chars (no Cline dump).

## CLI
harness serve | health | workers | task start|list|show|packet|record|decide | search | rescue --template | tournament

# PRD ORDER (do not jump)

M1 done: WorkerRegistry wraps DGX→M5→frontier. Tests lock harness-local no-failover.
M2 done: Task/WorkPacket/Attempt/Evidence/Decision + sqlite.
M3 partial: ContextManager builds packets from structured fields only.
M4 partial: ripgrep live; ast/semantic unavailable not missing.
M5 partial: future workers declared disabled.
M6 not done: M4 Mac Cline → this gateway over LAN (still listen 127.0.0.1).
Not building: dashboard, k8s, swarm, own Cline, 122B as Cline.

When more DGXs arrive: enabled:true on secondary/fast/monster. Software change should be boring.

# FIVE QUESTIONS (answer these first)

1. If Spark :8900 returns 200 with a wrong answer, does harness-auto call M5?
2. If I `harness task start "fix geocode"` then a Cline turn logs, which task_id gets the Attempt?
3. What do I change when DGX pair-b arrives?
4. Why is supportsPromptCache false?
5. What command sends a fail packet to Claude without the chat thread?

Then write ARCHITECTURE.md and run the pytest line. Stop.
