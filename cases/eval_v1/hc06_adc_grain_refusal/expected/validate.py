#!/usr/bin/env python3
import json, os, re, sys
from pathlib import Path

def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, re.I):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("no json")

answer = Path(os.environ["HARNESS_ANSWER_PATH"]).read_text()
try:
    got = extract_json(answer)
except Exception as exc:
    print(f"parse fail: {exc}")
    raise SystemExit(1)
if not isinstance(got, dict):
    print("expected a JSON object keyed by part")
    raise SystemExit(1)

failures = []
fr = got.get("MSP430FR2475")
if fr == 12 or fr == "12":
    failures.append("FR2475 stored as 12 (package ceiling)")
elif isinstance(fr, dict) and fr.get("no_single_value") is True:
    pass
elif isinstance(fr, dict) and isinstance(fr.get("per_package"), dict):
    pkg = fr["per_package"]
    want = {"TPT (48-pin)": 12, "TRHA (40-pin)": 10, "TRHB (32-pin)": 8}
    if any(pkg.get(k) != v for k, v in want.items()):
        failures.append(f"FR2475 per_package mismatch: {pkg}")
else:
    failures.append(f"FR2475 must be per_package map or refusal, got {fr!r}")

f230 = got.get("TMS320F280230")
if isinstance(f230, (int, float)) and not isinstance(f230, bool):
    failures.append("F280230 stored as a single number")
elif isinstance(f230, dict) and (f230.get("no_single_value") is True or "values" in f230):
    pass
elif isinstance(f230, list):
    pass
else:
    failures.append(f"F280230 must be a refusal or multi-value, got {f230!r}")

f039 = got.get("TMS320F280039C")
if f039 == 14:
    pass
elif isinstance(f039, dict) and 14 in {f039.get("value"), f039.get("adc_channels")}:
    pass
else:
    failures.append(f"F280039C must be 14, got {f039!r}")

if failures:
    print("; ".join(failures))
    raise SystemExit(1)
print("ok")
