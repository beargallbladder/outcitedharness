#!/usr/bin/env python3
import os, re, sys
from pathlib import Path

text = Path(os.environ["HARNESS_ANSWER_PATH"]).read_text()
upper = text.upper()
failures = []
verdicts = re.findall(r"\b(APPROVE|BLOCK)\b", upper)
if not verdicts:
    failures.append("missing verdict APPROVE or BLOCK")
elif verdicts[-1] != "APPROVE":
    failures.append(f"final verdict was {verdicts[-1]}, must APPROVE — blocking a correct change fails this case")
if failures:
    print("; ".join(failures))
    raise SystemExit(1)
print("ok")
