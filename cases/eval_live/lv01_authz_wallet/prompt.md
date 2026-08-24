These are LIVE files from our sandbox. The 27B already sits on one Spark
and stays there. You are auditioning for the 3-Spark monster slot.

Review lib/api/v1/auth.ts slugInWallet and its callers.
Q1: api_tier='custom', wallet_slugs=['microcontrollers'] — which aisles can it read?
Q2: why is that dangerous in THIS paid-API business (ops sets wallets, unpublished aisles exist)?
Q3: write the minimal patch that keeps a genuine all-access path for internal keys.

Score is conceptual. Name the hole, the ops/money trap, and an explicit
all-access mechanism (internal flag/tier/wildcard) — not the exact memo words.
