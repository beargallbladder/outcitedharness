#!/usr/bin/env python3
import os, re, sys
from pathlib import Path
text = Path(os.environ["HARNESS_ANSWER_PATH"]).read_text()
upper = text.upper()
low = text.lower()
failures = []
verdicts = re.findall(r"\b(APPROVE|BLOCK)\b", upper)
if not verdicts:
    failures.append("missing APPROVE or BLOCK")
elif verdicts[-1] != "BLOCK":
    failures.append(f"final verdict {verdicts[-1]}, must BLOCK")
if not any(k in low for k in ["silence", "omission", "omits", "absent", "absence", "never proves", "does not prove", "unknown"]):
    failures.append("did not treat summary silence as non-proof")
if not any(k in low for k in ["false", "explicit denial", "has_can_fd=false", "has_can_fd = false", "writes false"]):
    failures.append("did not name the False-from-silence write")
if failures:
    print("; ".join(failures))
    raise SystemExit(1)
print("ok")
