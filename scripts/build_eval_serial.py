#!/usr/bin/env python3
"""Snapshot three production tickets into isolated serial checkouts."""

from __future__ import annotations

import shutil
from pathlib import Path

SANDBOX = Path("/Volumes/M5_4TB/repos/outcited-ai-sandbox-20260822")
DEST = Path("cases/eval_serial")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n")


def copy_into(ticket: str, rel: str) -> None:
    src = SANDBOX / rel
    dest = DEST / ticket / "repo" / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


WALLET_ORACLE = r'''#!/usr/bin/env python3
import os, re, sys
from pathlib import Path
root = Path(os.environ["HARNESS_WORKSPACE"])
auth = (root / "lib/api/v1/auth.ts").read_text()
fails = []
if re.search(r"api_tier === ['\"]custom['\"]\)\s*return true", auth):
    fails.append("custom still short-circuits to true")
if "wallet_slugs" not in auth:
    fails.append("wallet_slugs check is gone")
if "free_slug" not in auth:
    fails.append("free tier check is gone")
if not re.search(r"is_internal|['\"]internal['\"]|all_access|wallet_slugs\.includes\(['\"]\*", auth):
    fails.append("no explicit all-access path for internal keys")
for rel in ("lib/api/v1/with-auth.ts", "lib/designwins/facade/auth.ts"):
    text = (root / rel).read_text()
    if "slugInWallet" not in text:
        fails.append(f"{rel} no longer calls slugInWallet")
if fails:
    print("; ".join(fails))
    raise SystemExit(1)
print("ok")
'''

INGEST_ORACLE = r'''#!/usr/bin/env python3
import os, sys
from pathlib import Path
root = Path(os.environ["HARNESS_WORKSPACE"])
text = (root / "scripts/audit/ingest-core-ai-index-and-below-floor-artifacts.ts").read_text()
low = text.lower()
fails = []
if "not re-ingesting" in text:
    fails.append("presence skip remains — still skipping on kind+week existence")
if fails:
    print("; ".join(fails))
    raise SystemExit(1)
print("ok")
'''

WEEK_ORACLE = r'''#!/usr/bin/env python3
import os, re, sys
from pathlib import Path
root = Path(os.environ["HARNESS_WORKSPACE"])
text = (root / "scripts/ops/week-status.ts").read_text()
fails = []
if re.search(r"missingVsReference\s*=\s*\[\s*\.\.\.\s*currentLaneNames", text):
    fails.append("set difference is inverted — new lanes counted as missing")
if not re.search(r"missingVsReference\s*=\s*\[\s*\.\.\.\s*referenceLaneNames", text):
    fails.append("missing lanes must be reference minus current")
if fails:
    print("; ".join(fails))
    raise SystemExit(1)
print("ok")
'''


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)

    copy_into("sr01_wallet", "lib/api/v1/auth.ts")
    copy_into("sr01_wallet", "lib/api/v1/with-auth.ts")
    copy_into("sr01_wallet", "lib/designwins/facade/auth.ts")
    write(DEST / "sr01_wallet" / "ticket.yaml", "id: sr01_wallet\ntitle: Fix slugInWallet custom short-circuit\nmax_turns: 16\n")
    write(
        DEST / "sr01_wallet" / "prompt.md",
        """Production checkout (isolated). Task: fix entitlement.

`lib/api/v1/auth.ts` `slugInWallet` returns true for every `api_tier==='custom'`
key, so `wallet_slugs` is decorative. Callers in `with-auth.ts` and
`lib/designwins/facade/auth.ts` trust that function.

Requirements:
1. A custom key with wallet_slugs=['microcontrollers'] must NOT read other aisles.
2. Free tier still uses free_slug.
3. Keep a genuine all-access path for internal keys (is_internal flag or
   api_tier==='internal' or equivalent). Do not make custom mean unlimited.
4. Do not delete the callers. Do not mock the function.

Use `run` until it prints PASS, then finish.
""",
    )
    write(DEST / "sr01_wallet" / "oracle" / "check.py", WALLET_ORACLE)

    copy_into("sr02_ingest_sha", "scripts/audit/ingest-core-ai-index-and-below-floor-artifacts.ts")
    write(DEST / "sr02_ingest_sha" / "ticket.yaml", "id: sr02_ingest_sha\ntitle: Presence-check ingest must become sha-check\nmax_turns: 16\n")
    write(
        DEST / "sr02_ingest_sha" / "prompt.md",
        """Production checkout (isolated). House rule 10: sha-check, never presence-check.

`scripts/audit/ingest-core-ai-index-and-below-floor-artifacts.ts` skips the
Full Substrate upload when kind+week is already in cr_artifacts
("not re-ingesting"). That keeps a stale blob after an upstream re-ship.

Fix it: compare on-disk sha256 to the stored row; re-upload when they differ.
Do not leave a presence-only skip. `run` until PASS, then finish.
""",
    )
    write(DEST / "sr02_ingest_sha" / "oracle" / "check.py", INGEST_ORACLE)

    copy_into("sr03_week_status", "scripts/ops/week-status.ts")
    week = DEST / "sr03_week_status" / "repo" / "scripts/ops/week-status.ts"
    text = week.read_text()
    broken = text.replace(
        "const missingVsReference = [...referenceLaneNames].filter(m => !currentLaneNames.has(m))",
        "const missingVsReference = [...currentLaneNames].filter(m => !referenceLaneNames.has(m))",
    )
    if broken == text:
        raise SystemExit("failed to inject week-status inversion")
    week.write_text(broken)
    write(DEST / "sr03_week_status" / "ticket.yaml", "id: sr03_week_status\ntitle: Restore missing-lanes set difference\nmax_turns: 16\n")
    write(
        DEST / "sr03_week_status" / "prompt.md",
        """Production checkout (isolated). `scripts/ops/week-status.ts` is the weekly
truth-teller. A regression inverted MISSING VS reference: it now lists lanes
that are NEW in the current week (gemini-flash trap) instead of lanes that
were in the reference week and vanished.

Restore the correct set difference: missing = in reference, absent in current.
Do not invert it the other way. `run` until PASS, then finish.
""",
    )
    write(DEST / "sr03_week_status" / "oracle" / "check.py", WEEK_ORACLE)
    print(f"wrote 3 tickets under {DEST}")


if __name__ == "__main__":
    main()
