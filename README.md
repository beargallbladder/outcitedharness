# Model Harness v0.1

Experimental instrument for one question:

> For real work, what is the cheapest model tier that reliably solves the task — and does the ladder save frontier spend, or only add latency?

This is a client of existing local/cloud model endpoints. Cline / VS Code talks to `harness serve` (`http://127.0.0.1:8787/v1`), not to the raw model ports.

## What it measures

- Independent tournament: same task packet to every enabled model
- Escalation ladder: stop at the first objective `PASS`
- Direct-frontier baseline vs the ladder
- `minimum_model_that_solved`
- Incremental value per tier
- `local_waste_ms`: time spent in unsuccessful cheaper tiers

## Safety

The harness never starts, stops, restarts, or reconfigures inference services. If a model is down, it reports that and continues.

Do not point it at the live embedding (`:8800`–`:8803`) or OCR ports. Those are out of scope for v0.1.

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
