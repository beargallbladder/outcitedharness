# Model Harness v0.1

Experimental instrument for one question:

> For real work, what is the cheapest model tier that reliably solves the task — and does the ladder save frontier spend, or only add latency?

This is a client of existing local/cloud model endpoints. Cursor remains the
agent and owns IDE tools. Loopback LiteLLM (`http://127.0.0.1:7410/v1`)
exposes the local models and sends `harness-orch` to the durable Harness
orchestrator on `:8787`.

## What it measures

- Independent tournament: same task packet to every enabled model
- Escalation ladder: stop at the first objective `PASS`
- Direct-frontier baseline vs the ladder
- `minimum_model_that_solved`
- Incremental value per tier
- `local_waste_ms`: time spent in unsuccessful cheaper tiers

## Safety

The Harness runtime never starts, stops, restarts, or reconfigures inference services.
If a model is down, it reports that and continues. GCI may call only the
existing `:8800/v1/embeddings` route through its bounded encoder client; all
other live embedding/search and OCR routes remain out of scope.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # only if you enable cloud providers
```

Edit:

- `config/models.yaml` — endpoints, model IDs, enable flags
- `config/pricing.yaml` — USD per million tokens (`null` = unknown)
- `config/settings.yaml` — timeouts, results path

Cloud providers stay disabled until you fill `base_url`, `model`, and the matching API key env var.

## Live stack

The approved allocation is one dedicated coder plus independent planning and
criticism:

- `local-coder`: DGX3 Qwen3-Coder-Next SGLang
- `local-qwen38`: ASUS2 + ASUS4 Qwen3.8 Flash Next SGLang TP2
- `local-critic`: ASUS3 Nemotron 3.5 Lightning SGLang
- `harness-orch`: decomposition, parallel dispatch, grading, repair, and verification
- `frontier-claude`: manual paid route only; never an automatic fallback

Install or repair the loopback LiteLLM service, then qualify the stack:

```shell
scripts/install_litellm.sh install
uv run python scripts/litellm_qualification.py
```

No Cline extension is required. Use Cursor's native agent and tools; invoke
local orchestration through `harness dispatch`, the `harness-orch` API route,
or project automation.

For service ownership, ports, SGLang rollback rules, billing controls, and the
MCP allowlist, see `ARCHITECTURE.md`.

## Commands

```bash
harness serve
harness health
harness tournament cases/example_001
harness escalate cases/example_001
harness benchmark cases/
harness baseline cases/
harness results
harness inspect RUN_ID
harness compare CASE_ID
harness economics
harness judge RUN_ID CASE_ID MODEL_KEY PASS|PARTIAL|FAIL
```

`harness health` probes configured chat endpoints only. It does not repair anything.

## Repository verification contract

For coding tasks, `harness-orch` first uses an explicitly named allowlisted
verification command. Otherwise it deterministically compiles a repository
contract from `pyproject.toml`, `pytest.ini`, `package.json`, workspace/lock
files, TypeScript/Ruff configuration, and single-line verifier commands in CI
workflows. If no safe command can be selected, the task is blocked before any
edit.

Repositories can override inference with a checked-in `.harness.toml`:

```toml
[verification]
required = ["unit", "lint"]

[verification.commands.unit]
argv = [".venv/bin/pytest", "-q"]
timeout = 60

[verification.commands.lint]
argv = [".venv/bin/ruff", "check", "."]
timeout = 60
```

Only verifier executables and named package scripts on the harness allowlist
are accepted. Every required command must exit zero against the same current
file-state hash; any mutation clears earlier verification results.

When verification fails, the loop expands its existing working set from
concrete diagnostics before repairing. It reads unseen workspace files named
by tracebacks, pytest, TypeScript, and linter output; searches exact unresolved
symbols; and uses the workspace-scoped code index only after exact evidence is
insufficient. Expansion never restarts broad gather and does not consume a
coding iteration.

Before each coding or repair lease, a deterministic Context Compiler turns the
persistent working set into a model-specific bounded packet. It preserves the
current objective, verifier contract, latest machine failure, changed source,
diff, directly relevant tests, and causal expansion evidence in that order.
Peripheral evidence is ranked and sliced only after current state; prior
failures are retained as compact structured attempt summaries rather than
replayed transcripts. Configure the conservative fleet default with
`coder_context_tokens` and override it per model when needed. Captured packets
use provenance tags such as `<CURRENT_FILE path="…" hash="…">`,
`<CURRENT_DIFF hash="…">`, and `<EXPANDED_EVIDENCE path="…">`, and report
their configured budget and compiled size.

Before each returned edit, the orchestrator captures a content-addressed task
baseline for every attributable path. Logical checkpoints are finalized from
the exact post-edit filesystem state used for verification, including created
and deleted untracked files. Exhausted changes are preserved. To explicitly
restore only that task's files, run:

```shell
harness rollback-task TASK_ID
```

Rollback requires confirmation (or `--yes`), refuses the entire operation if
any task path changed after the latest checkpoint, and preserves unrelated
dirty and untracked work. It never uses `git reset`, `checkout`, or `clean`.

## Global Code Intelligence

GCI is an independent authenticated service on Spark port `8810`. Its SQLite
database lives under `/data/harness-gci`; it never opens CategoryRank index
paths. The only shared dependency is a bounded request to
`http://127.0.0.1:8800/v1/embeddings`, with one request in flight and index
batches capped at 16.

```shell
harness gci status
harness gci scan
harness gci refresh
harness gci auto-run
harness gci search "repository verification contract"
harness gci search --mode symbol RepoContract
harness gci pause
harness gci resume
```

Approved roots come from `code_index_repos`. Scans include dirty and untracked
source files but exclude ignored/vendor files and reject symlink escapes.
Global results are discovery evidence only. A hit becomes a workspace read
path only when its source host and canonical repository root exactly match the
active workspace.

### Refresh automation

Refresh automation runs on the machine that owns the configured working
copies. It never asks the Spark to clone or pull repositories. Active
repositories are checked every five minutes; after 30 days without observed
Git or working-tree activity, they remain searchable but are checked only
daily. Any detected activity returns a repository to the active cadence.

The cheap activity probe uses Git HEAD, status, and dirty/untracked path
metadata. If its fingerprint is unchanged, the pass does not contact GCI and
performs no embedding work. Changed repositories reuse the manifest protocol
and send only changed source files and deletions. Scheduling and failure state
live in the separate local SQLite database configured by
`gci_refresh.state_path`.

Add an owned local repository by adding its absolute path to
`code_index_repos`, then run `harness gci auto-run --force`. Paths are never
discovered automatically.

```shell
scripts/install_gci_refresh.sh install
scripts/install_gci_refresh.sh status
scripts/install_gci_refresh.sh uninstall
```

The launchd job runs one short-lived, locked pass every five minutes. One
repository failure does not block the others; bounded retry state is shown by
`harness gci status`, and logs are written under `~/.harness/logs`.

Deployment uses `scripts/deploy_gci.sh spark`. It installs only
`harness-gci.service`, with a separate virtual environment, bearer-token
environment file, resource controls, and restart lifecycle. It does not change
or restart `bge-m3-embed.service`.

## Autonomous greenfield builds

Greenfield planning runs bounded GCI queries for architecture, domain, testing,
and tooling analogues. The resulting excerpts retain `gci://` provenance and
are advisory only: they cannot authorize source-repository reads, edits,
copying, or runtime coupling.

```shell
harness build new \
  --name electronics-family-api \
  --stack python \
  --dest ~/Projects/electronics-family-api \
  --intent "Build a FastAPI service for managing electronics-family comparisons"

harness build approve GREENFIELD_RUN_ID
harness build status GREENFIELD_RUN_ID
harness build resume GREENFIELD_RUN_ID
```

`build new` creates only durable planning state. It runs no package command and
creates neither the destination nor an execution workspace. The single
approval freezes the discovery, product specification, dependency list, and
milestone plan; material drift blocks execution.

After approval, M0 is bootstrapped and mechanically verified in
`~/.harness/runs/<run-id>/repo`. Open that isolated folder in Cursor and send
`Continue greenfield <run-id>`. The Cursor agent executes later
milestones, while the controller verifies and commits each milestone, survives
restarts, performs a final same-state repository gate, and publishes only if
the reserved destination fingerprint is unchanged.

Verified applications that provide a Dockerfile can be deployed to a hardened,
tailnet-only M5 preview before publication:

```shell
harness build preview GREENFIELD_RUN_ID --container-port 8000
harness sandbox list
harness sandbox down greenfield-GREENFIELD_RUN_ID
```

The preview gate rechecks the final state hash, builds offline for ARM64, runs
without application egress or host secrets, and records a TTL-bound lifecycle.
See `deploy/sandbox/README.md`.

## Case format

```text
cases/example_001/
  case.yaml
  prompt.md
  inputs/
  expected/
  notes.md
```

Evaluators: `exact_text`, `exact_json`, `numeric_fields`, `required_fields`, `regex`, `command`, `human`.

There is no LLM-as-judge in v0.1. Prefer an objective check. Use `human` when you must review later.

## Ladder

| Key | Default | Role |
|---|---|---|
| `dgx_qwen` | on | L0 DGX Qwen3-Coder-Next |
| `m5_qwen` | on | L1 M5 Qwen3.8-27B 8-bit |
| `deepseek` | off | L2 cloud proxy |
| `minimax` | off | L3 cloud proxy |
| `frontier` | off | L4 OpenAI- or Anthropic-compatible |

Tournament mode never shows one model another model's answer.

## Results

SQLite: `results/harness.db`

Raw answers: `results/runs/<run_id>/<case_id>/<model_key>/`
