#!/usr/bin/env python3
"""Standing smoke: run CR eval cases hc01 + hc12 against harness-auto.

Runs from launchd (com.samkim.harness-smoke). Results append to
smoke-results.jsonl next to the eval pack; failures also post a mail to
the cursor-cr inbox so a regression is seen without anyone polling.

Scoring follows the pack README: hc01 is exact_json, hc12 is a
keyword_rubric (every OR-group needs one case-insensitive substring hit).
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

GATEWAY = "http://127.0.0.1:8787/v1/chat/completions"
PACK = Path("/Volumes/M5_4TB/configs/harness-evals")
RESULTS = PACK / "smoke-results.jsonl"
INBOX = Path("/Volumes/M5_4TB/agent-inbox/cursor-cr")
CASES = [
    ("cases.jsonl", "hc01_junk_mpn"),
    ("cases-phase2-coding.jsonl", "hc12_pipe_swallows_exit"),
]


def load_case(filename: str, case_id: str) -> dict:
    for line in (PACK / filename).read_text().splitlines():
        line = line.strip()
        if line:
            case = json.loads(line)
            if case["case_id"] == case_id:
                return case
    raise KeyError(f"{case_id} not in {filename}")


def ask(prompt: str, evidence: dict) -> tuple[str, str, float]:
    body = json.dumps(
        {
            "model": "harness-auto",
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nEvidence:\n{json.dumps(evidence, indent=1)}",
                }
            ],
            "max_tokens": 800,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(GATEWAY, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    latency = (time.perf_counter() - started) * 1000
    text = data["choices"][0]["message"].get("content") or ""
    served = str(data.get("model") or "?")
    return text, served, latency


def score_exact_json(text: str, expected) -> tuple[bool, str]:
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return False, "no JSON array in answer"
    try:
        got = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return False, f"unparseable JSON: {exc}"
    return (got == expected), f"got {got}"


def score_keyword_rubric(text: str, groups: list[list[str]]) -> tuple[bool, str]:
    # Strip markdown emphasis and collapse whitespace so "the *last* command"
    # still matches the rubric phrase "the last command".
    lowered = re.sub(r"\s+", " ", text.replace("*", "").replace("`", "").lower())
    misses = [group for group in groups if not any(k.lower() in lowered for k in group)]
    if misses:
        return False, f"missing any of: {misses}"
    return True, "all keyword groups hit"


def run_case(filename: str, case_id: str) -> dict:
    case = load_case(filename, case_id)
    try:
        text, served, latency = ask(case["prompt"], case.get("evidence") or {})
    except Exception as exc:
        return {
            "case_id": case_id,
            "pass": False,
            "why": f"request failed: {type(exc).__name__}: {exc}",
            "served_by": None,
            "latency_ms": None,
        }
    scoring = case.get("scoring") or {}
    if scoring.get("type") == "exact_json":
        ok, why = score_exact_json(text, case["expected"])
    elif scoring.get("type") == "keyword_rubric":
        ok, why = score_keyword_rubric(text, scoring["must_contain_any_of_each"])
    else:
        ok, why = False, f"unsupported scoring type {scoring.get('type')}"
    return {
        "case_id": case_id,
        "pass": ok,
        "why": why,
        "served_by": served,
        "latency_ms": round(latency),
        "answer_head": text[:200],
    }


def post_failure_mail(results: list[dict]) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    failed = [r["case_id"] for r in results if not r["pass"]]
    body = (
        "---\n"
        f"id: harness-smoke-fail-{stamp}\n"
        "from: m5-cursor\n"
        "to: cursor-cr\n"
        "type: status\n"
        f'subject: "Harness smoke FAIL: {", ".join(failed)}"\n'
        f"created_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        "---\n\n"
        "Standing smoke (hc01+hc12 vs harness-auto) failed:\n\n"
        f"```json\n{json.dumps(results, indent=1)}\n```\n"
    )
    name = f"{stamp}_m5-cursor_cursor-cr_status_harness-smoke-fail.md"
    (INBOX / name).write_text(body)


def main() -> int:
    results = [run_case(f, cid) for f, cid in CASES]
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "harness-auto",
        "results": results,
    }
    try:
        with RESULTS.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        print(f"could not append results: {exc}", file=sys.stderr)
    if not all(r["pass"] for r in results):
        try:
            post_failure_mail(results)
        except OSError as exc:
            print(f"could not post failure mail: {exc}", file=sys.stderr)
        print(json.dumps(record, indent=1))
        return 1
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
