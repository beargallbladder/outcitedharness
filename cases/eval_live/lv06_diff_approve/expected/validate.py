#!/usr/bin/env python3
import os, re, sys
from pathlib import Path
text = Path(os.environ["HARNESS_ANSWER_PATH"]).read_text()
verdicts = re.findall(r"\b(APPROVE|BLOCK)\b", text.upper())
if not verdicts:
    print("missing APPROVE or BLOCK")
    raise SystemExit(1)
if verdicts[-1] != "APPROVE":
    print(f"final verdict {verdicts[-1]}, must APPROVE")
    raise SystemExit(1)
print("ok")
