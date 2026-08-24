#!/usr/bin/env python3
import json, os, re, sys
from pathlib import Path

FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)

def extract_json(text: str):
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for match in FENCE_RE.finditer(stripped):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = stripped.find(opener), stripped.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no json")

text = Path(os.environ["HARNESS_ANSWER_PATH"]).read_text()
low = text.lower()
failures = []
want = {"kimi-thinking", "qwen3.7-plus"}
got = set()
try:
    parsed = extract_json(text)
    if isinstance(parsed, dict):
        models = parsed.get("missing_models") or parsed.get("models") or parsed.get("result")
        if isinstance(models, list):
            got = {str(x) for x in models}
    elif isinstance(parsed, list):
        got = {str(x) for x in parsed}
except Exception as exc:
    failures.append(f"parse fail: {exc}")

if "gemini-flash" in {x.lower() for x in got}:
    failures.append("gemini-flash is new in W34, not missing")
if got != want:
    failures.append(f"missing_models must be {sorted(want)}, got {sorted(got)}")
if not re.search(r"\bexcept\b|not\s+in|left\s+join|is\s+null|anti-?join", low):
    failures.append("query must be set difference (EXCEPT / NOT IN / LEFT JOIN IS NULL)")
if failures:
    print("; ".join(failures))
    raise SystemExit(1)
print("ok")
