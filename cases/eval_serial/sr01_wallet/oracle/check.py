#!/usr/bin/env python3
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
