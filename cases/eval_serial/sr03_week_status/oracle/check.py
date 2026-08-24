#!/usr/bin/env python3
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
