#!/usr/bin/env python3
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
