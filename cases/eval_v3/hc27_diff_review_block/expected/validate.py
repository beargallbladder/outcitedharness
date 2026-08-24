#!/usr/bin/env python3
import os, re, sys
from pathlib import Path

text = Path(os.environ["HARNESS_ANSWER_PATH"]).read_text()
upper = text.upper()
low = text.lower()
failures = []
verdicts = re.findall(r"\b(APPROVE|BLOCK)\b", upper)
if not verdicts:
    failures.append("missing verdict APPROVE or BLOCK")
elif verdicts[-1] != "BLOCK":
    failures.append(f"final verdict was {verdicts[-1]}, must BLOCK")
needles = [
    ("silence to false", ["silence", "omission", "omits", "absent", "absence", "does not prove", "never proves absence"]),
    ("false denial", ["false", "explicit denial", "explicit no", "has_can_fd=false", "has_can_fd = false", "writes false"]),
]
for label, keys in needles:
    if not any(k in low for k in keys):
        failures.append(f"did not name {label}")
if failures:
    print("; ".join(failures))
    raise SystemExit(1)
print("ok")
