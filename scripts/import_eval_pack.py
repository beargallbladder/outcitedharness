#!/usr/bin/env python3
"""Import the mailbox eval pack into harness case directories."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

PACK = Path("/Volumes/M5_4TB/exports/harness-eval-cases-v1/cases.jsonl")
DEST = Path("cases/eval_v1")


def dump_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def write_case(raw: dict) -> None:
    case_id = raw["case_id"]
    root = DEST / case_id
    (root / "inputs").mkdir(parents=True, exist_ok=True)
    (root / "expected").mkdir(parents=True, exist_ok=True)

    (root / "prompt.md").write_text(raw["prompt"].rstrip() + "\n")
    (root / "inputs" / "evidence.json").write_text(
        json.dumps(raw["evidence"], indent=2, ensure_ascii=False) + "\n"
    )
    (root / "notes.md").write_text((raw.get("history") or "").rstrip() + "\n")

    scoring = raw["scoring"]
    eval_spec: dict = {"type": scoring["type"]}
    if scoring["type"] == "exact_json":
        eval_spec["ignore_order"] = False
        if scoring.get("tolerance") is not None:
            eval_spec["tolerance"] = scoring["tolerance"]
        (root / "expected" / "answer.json").write_text(
            json.dumps(raw["expected"], indent=2, ensure_ascii=False) + "\n"
        )
        reference = {"file": "expected/answer.json"}
    elif scoring["type"] == "json_fields":
        eval_spec["fields"] = scoring["fields"]
        (root / "expected" / "answer.json").write_text(
            json.dumps(scoring["fields"], indent=2, ensure_ascii=False) + "\n"
        )
        reference = {"file": "expected/answer.json"}
    elif scoring["type"] == "keyword_rubric":
        eval_spec["groups"] = scoring["must_contain_any_of_each"]
        reference = None
    elif scoring["type"] == "json_judge":
        eval_spec["type"] = "command"
        eval_spec["command"] = ["python3", "expected/validate.py"]
        (root / "expected" / "rubric.md").write_text(scoring["rubric"].rstrip() + "\n")
        if raw.get("expected") is not None:
            (root / "expected" / "answer.json").write_text(
                json.dumps(raw["expected"], indent=2, ensure_ascii=False) + "\n"
            )
        reference = {"file": "expected/answer.json"} if raw.get("expected") else None
    else:
        raise ValueError(scoring["type"])

    spec = {
        "id": case_id,
        "title": case_id.replace("_", " "),
        "category": raw.get("category", "other"),
        "tags": [raw.get("difficulty", ""), raw.get("category", "")],
        "input_files": ["inputs/evidence.json"],
        "evaluation": eval_spec,
        "historical": {
            "notes": (raw.get("history") or "")[:400],
        },
    }
    if reference:
        spec["reference_answer"] = reference
    dump_yaml(root / "case.yaml", spec)


def write_hc06_validator() -> None:
    Path("cases/eval_v1/hc06_adc_grain_refusal/expected/validate.py").write_text(
        '''#!/usr/bin/env python3
import json, os, re, sys
from pathlib import Path

def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"```(?:json)?\\s*([\\s\\S]*?)```", text, re.I):
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
'''
    )


def write_hc10_validator() -> None:
    Path("cases/eval_v1/hc10_llmstxt_epistemics/expected/validate.py").write_text(
        '''#!/usr/bin/env python3
import os, re, sys
from pathlib import Path

text = Path(os.environ["HARNESS_ANSWER_PATH"]).read_text()
upper = text.upper()
failures = []
if "CONSISTENT_BUT_UNPROVEN" not in upper:
    failures.append("missing verdict CONSISTENT_BUT_UNPROVEN")
verdicts = re.findall(r"CONFIRMED|CONSISTENT_BUT_UNPROVEN|REFUTED", upper)
if verdicts and verdicts[-1] != "CONSISTENT_BUT_UNPROVEN":
    failures.append(f"final verdict was {verdicts[-1]}")
low = text.lower()
if not any(k in low for k in ["omission", "omit", "silence", "silent", "curated", "proves nothing", "does not prove", "fail to prove", "editorial"]):
    failures.append("did not treat curated-index silence as non-proof")
if not any(k in low for k in ["orderable", "parametric", "feed", "product-status", "product status", "known_fact", "known fact"]):
    failures.append("did not name settling evidence")
if failures:
    print("; ".join(failures))
    raise SystemExit(1)
print("ok")
'''
    )


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for line in PACK.read_text().splitlines():
        if line.strip():
            write_case(json.loads(line))
    write_hc06_validator()
    write_hc10_validator()
    print(f"imported {len(list(DEST.glob('*/case.yaml')))} cases into {DEST}")


if __name__ == "__main__":
    main()
