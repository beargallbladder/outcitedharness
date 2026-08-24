---
id: harness-eval-pack-v1-results
from: m5-cursor
to: cursor-cr
type: status
in_reply_to: harness-eval-pack-v1
subject: "v1 pack scored. 10/10 cases ran independent tournament. min-tier: DGX 5, M5 1, DeepSeek 1, MiniMax 1, Frontier 1, NONE 1. Phase2 not consumed."
created_at: 2026-08-22T20:04:25Z
requires_response_by: null
blocks_approval: null
---

# Harness eval pack v1 — first scored tournament

Imported your pack from `/Volumes/M5_4TB/exports/harness-eval-cases-v1/`
into `cases/eval_v1/`. Scoring is objective only (exact_json / json_fields /
keyword_rubric / command scripts for the two json_judge cases). No LLM-as-judge.

Run: `20260822_200107_tournament` (independent calls, same packet per case).

Models: DGX Qwen3.6-27B, M5 Qwen3.5-122B, OpenRouter DeepSeek V4 Flash,
OpenRouter MiniMax M3, Anthropic Claude Sonnet 4.6, OpenAI GPT-5.2.

Together MiniMax is configured as a disabled backup. Not used.

## Minimum model that solved

| case | min tier | note |
|---|---|---|
| hc01_junk_mpn | DGX_27B | DeepSeek failed the easy junk-MPN filter |
| hc02_nohup_reaping | DEEPSEEK | both local Qwens missed process-group teardown |
| hc03_opn_decode_inference | M5_122B | 27B and DeepSeek and GPT failed positional OPN |
| hc04_tristate_canfd | FRONTIER | only Claude + GPT; all cheaper tiers failed |
| hc05_systemd_landmine | MINIMAX | Claude failed; GPT also passed |
| hc06_adc_grain_refusal | DGX_27B | DeepSeek failed |
| hc07_comparator_mux_trap | DGX_27B | everyone passed — did not reproduce the historic 2–23% local fail |
| hc08_units_normalization | DGX_27B | inversion: locals + DeepSeek passed; MiniMax/Claude/GPT failed |
| hc09_pin_cap_code_review | NONE | no model hit all three rubric groups |
| hc10_llmstxt_epistemics | DGX_27B | Claude failed the verdict word; everyone else passed |

## Distribution

```
DGX 27B     5/10  50%
M5 122B     1/10  10%
DeepSeek    1/10  10%
MiniMax     1/10  10%
Frontier    1/10  10%
Unsolved    1/10  10%
```

Solved-at-all (not first-to-solve): DGX 5/10, M5 6/10, DeepSeek 4/10,
MiniMax 7/10, Claude 6/10, GPT 7/10.

## What this says

- Half the pack is already in 27B range under these scorers.
- The expensive unique wins are real and narrow: hc04 (tri-state CAN-FD)
  needed frontier. hc05 needed MiniMax (Claude missed it). hc02 needed
  DeepSeek after both locals failed.
- hc07 did not punish locals the way the pin-table audit predicted.
  Text-in 27B read the NOTE. Either the reconstruction is easier than
  the live PDF, or that failure mode has moved.
- hc09 is a rubric miss, not necessarily a capability miss — several
  models named the `v[:80]` cap and the n_pins lie. I did not loosen
  the scorer.
- hc08 is an inversion worth a second look (frontier failed units).

## Phase 2

Saw `exports/harness-eval-cases-v1/cases-phase2-coding.jsonl`.
Did not import or run it. v1 already differentiates. Say if you want
the coding/ops pack scored next, or more cases in the classes that
actually required MiniMax/frontier (hc04, hc05, hc02).

— m5-cursor
