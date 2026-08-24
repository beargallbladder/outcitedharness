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
