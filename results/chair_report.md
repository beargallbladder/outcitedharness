# THREE-SPARK CHAIR REPORT

27B is CONTROL and is not on trial. **PASS only. PARTIAL / ERROR / VOID = miss.**

Run: `20260823_000733_tournament` (15 one-shot 27B-miss cases) plus prior 27B / M5 / serial rows. Flash hc13 = **VOID** (hung, killed). hc14 OpenRouter quartet = hard timeout (Claude PARTIAL only). hc20 Flash = hard timeout.

VISION pack: **empty — no image cases. Do not invent a vision %.**

## Headline

| model | Total PASS | 27B-fails solved | residual | unique on 27B-fails |
|---|---|---|---|---|
| dgx_qwen (control) | 6/23 | 0/17 | 0% | — |
| m5_qwen | 6/23 | 2/17 | 12% | hc03 shared w/ GLM; hc12 shared |
| deepseek_flash 0731 | 2/23 | 2/17 | 12% | none (hc12 + hc18 both shared) |
| hy3 | 4/23 | 2/17 | 12% | none (hc13 shared w/ Claude) |
| minimax | 7/23 | 3/17 | 18% | **hc11** |
| glm52 | 5/23 | 3/17 | 18% | **hc17** |
| frontier (Claude, not a Spark chair) | 6/23 | 4/17 | 24% | **hc04, hc15** |

Highest cloud residual is MiniMax / GLM at **3 of 17** (18%). Claude is 4/17 and is not deployable on Sparks.

## By bucket (PASS / n)

| model | CODE (9) | PDF (5) | VISION | REASONING (9) |
|---|---|---|---|---|
| dgx_qwen | 0/9 | 3/5 | n/a | 3/9 |
| deepseek_flash | 1/9 | 0/5 | n/a | 1/9 |
| hy3 | 1/9 | 0/5 | n/a | 3/9 |
| minimax | 2/9 | 2/5 | n/a | 3/9 |
| glm52 | 1/9 | 1/5 | n/a | 3/9 |
| frontier | 1/9 | 2/5 | n/a | 3/9 |

CODE is where a 3-boxer would have to win. Nobody did. Serial tickets **sr01 wallet** and **sr02 ingest** are still FAIL for every model that ran them. 0731 / MiniMax / Claude have **no serial rows**.

## 27B misses: who PASSed

| case | 27B | who PASSed |
|---|---|---|
| hc11 dict-sort | PARTIAL | MiniMax |
| hc12 pipe-exit | PARTIAL | Flash, MiniMax, M5 |
| hc13 symlink | FAIL | Hy3, Claude (Flash VOID) |
| hc14 base64 | PARTIAL | nobody (4 timeouts) |
| hc16 gz-twin | FAIL | nobody |
| hc17 allowlist | PARTIAL | GLM |
| hc20 dual-write | PARTIAL | nobody (Flash timeout) |
| sr01 wallet | FAIL | nobody (0731/M3/Claude = GAP) |
| sr02 ingest sha | FAIL | nobody (0731/M3/Claude = GAP) |
| hc03 OPN | FAIL | GLM, M5 |
| hc09 pin-cap | FAIL | nobody |
| hc04 tri-state | FAIL | Claude |
| hc05 systemd | PARTIAL | nobody |
| hc15 sha-vs-presence | PARTIAL | Claude |
| hc18 env-race | PARTIAL | Flash, Hy3, MiniMax, GLM, Claude |
| hc24 authz review | PARTIAL | nobody |
| hc25 serverless 503 | PARTIAL | nobody |

Hard unsolved even by Claude: hc05, hc09, hc14, hc16, hc20, hc24, hc25, both serial tickets.

## Instrument notes

- 27B 6/23 is **by construction** — this pack is weighted to 27B misses. Do not read it as “27B is weak.”
- hc18 is a pile-on, not a monster advantage.
- hc24/hc25 PARTIAL is the known phrase-rubric pattern from phase 3; not evidence a 3-boxer would close them.
- 27B+critic was **not** run. Even if it recovered some misses, that would shrink residual further.

## THREE-SPARK CHAIR: NO MODEL EARNS THREE SPARKS

Residual tops out at 18% for a deployable monster, with one unique quiz each (MiniMax hc11, GLM hc17). That is not a dedicated 3-Spark inference pool. Flash-class (2-box) is not earned either — 0731 residual is 12% and it hung.

**Cancel-down is the honest call** unless you want to spend more money proving serial + critic cannot close the same 17. The leftover use of extra boxes is still training + spare, not a monster chair.
