#!/usr/bin/env python3
"""Import CR phase-3 repo-grounded pack into cases/eval_v3."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

PACK = Path("/Volumes/M5_4TB/exports/harness-eval-cases-v1/cases-phase3-repo-grounded.jsonl")
DEST = Path("cases/eval_v3")


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
    reference = None
    if scoring["type"] == "exact_json":
        eval_spec["ignore_order"] = False
        if scoring.get("tolerance") is not None:
            eval_spec["tolerance"] = scoring["tolerance"]
        (root / "expected" / "answer.json").write_text(
            json.dumps(raw["expected"], indent=2, ensure_ascii=False) + "\n"
        )
        reference = {"file": "expected/answer.json"}
    elif scoring["type"] == "keyword_rubric":
        eval_spec["groups"] = scoring["must_contain_any_of_each"]
    elif scoring["type"] == "json_judge":
        eval_spec["type"] = "command"
        eval_spec["command"] = ["python3", "expected/validate.py"]
        (root / "expected" / "rubric.md").write_text(scoring["rubric"].rstrip() + "\n")
        if raw.get("expected") is not None:
            (root / "expected" / "answer.json").write_text(
                json.dumps(raw["expected"], indent=2, ensure_ascii=False) + "\n"
            )
            reference = {"file": "expected/answer.json"}
    else:
        raise ValueError(scoring["type"])

    spec = {
        "id": case_id,
        "title": case_id.replace("_", " "),
        "category": raw.get("category", "other"),
        "tags": [raw.get("difficulty", ""), raw.get("category", "")],
        "input_files": ["inputs/evidence.json"],
        "evaluation": eval_spec,
        "historical": {"notes": (raw.get("history") or "")[:400]},
    }
    if reference:
        spec["reference_answer"] = reference
    dump_yaml(root / "case.yaml", spec)


HC23_VALIDATOR = r'''#!/usr/bin/env python3
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
'''

HC27_VALIDATOR = r'''#!/usr/bin/env python3
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
'''

HC28_VALIDATOR = r'''#!/usr/bin/env python3
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
'''


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in PACK.read_text().splitlines():
        if line.strip():
            write_case(json.loads(line))
            n += 1
    (DEST / "hc23_sql_missing_lanes" / "expected" / "validate.py").write_text(HC23_VALIDATOR)
    (DEST / "hc27_diff_review_block" / "expected" / "validate.py").write_text(HC27_VALIDATOR)
    (DEST / "hc28_diff_review_approve" / "expected" / "validate.py").write_text(HC28_VALIDATOR)
    print(f"imported {n} cases into {DEST}")


if __name__ == "__main__":
    main()
