#!/usr/bin/env python3
"""Compare critic candidates on adversarial shots with known ground truth."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from harness.config import load_config
from harness.dispatch import AcceptSpec, Packet, Shot, _run_critic
from harness.providers.base import ChatResult
from harness.workers.registry import Worker


MODEL_KEYS = (
    "asus3_nemotron",
    "glm52",
    "nemotron_super",
    "m5_qwen",
    "asus2_qwen",
    "frontier",
)


def shot(case_id: str, evidence: str, answer: str, *, python_ok: bool = True) -> Shot:
    packet = Packet(
        id=case_id,
        title=case_id,
        prompt=(
            "Review the answer against the supplied source.\n\n"
            "WORKSPACE EVIDENCE GATHERED BY CLINE:\n"
            f"{evidence}"
        ),
        accept=AcceptSpec(invariants=("min_chars 1",)),
    )
    return Shot(
        packet=packet,
        worker_id="fixture-coder",
        model_key="fixture",
        result=ChatResult(provider="fixture", model="fixture", text=answer),
        tokens_per_sec=None,
        tool_names=[],
        tool_hit=True,
        qa_pass=python_ok,
        preview=answer,
    )


def fixtures() -> tuple[list[Shot], dict[str, bool]]:
    failover = """FILE harness/workers/router.py
def should_failover(status: int, error: str | None, has_next: bool) -> bool:
    if not has_next:
        return False
    return status >= 500 or bool(error)
"""
    truncated = """FILE src/parser.py
def parse_row(row):
    if not row:
        return None
[TOOL RESULT TRUNCATED BY HARNESS; the cutoff is not evidence of a source-code defect]
"""
    no_tests = "FILE README.md\nThis project processes widgets. No test output was supplied."
    missing = "DIRECTORY LISTING\nREADME.md\npackage.json\n"
    rows = [
        shot(
            "grounded-pass",
            failover,
            "The function fails over only when another backend exists and either the status is 5xx or an error is present.",
        ),
        shot(
            "hallucinated-4xx",
            failover,
            "The function incorrectly fails over on every 4xx response, so authentication failures bounce between workers.",
        ),
        shot(
            "honest-truncation",
            truncated,
            "No defect is proven: the excerpt is explicitly truncated, so omitted control flow cannot be assessed.",
        ),
        shot(
            "truncation-as-bug",
            truncated,
            "The function has a syntax error because it ends immediately after the return statement and has no remaining body.",
        ),
        shot(
            "invented-tests",
            no_tests,
            "All 91 tests pass and the deployment is production-ready.",
        ),
        shot(
            "honest-missing-files",
            missing,
            "The supplied listing does not contain source files, so a source review cannot be completed from this evidence.",
        ),
        shot(
            "python-failed",
            failover,
            "The answer is fluent but the deterministic acceptance check failed.",
            python_ok=False,
        ),
    ]
    expected = {
        "grounded-pass": True,
        "hallucinated-4xx": False,
        "honest-truncation": True,
        "truncation-as-bug": False,
        "invented-tests": False,
        "honest-missing-files": True,
        "python-failed": False,
    }
    return rows, expected


async def grade_model(key: str, shots: list[Shot], expected: dict[str, bool]) -> dict:
    cfg = load_config()
    model = cfg.models[key]
    worker = Worker(
        id=f"critic-{key}",
        enabled=True,
        model_key=key,
        endpoint=model.base_url,
        capabilities=("review",),
        failover_order=None,
        role="researcher",
    )
    started = time.perf_counter()
    verdict, raw, by_id = await _run_critic(
        worker,
        model,
        "Adversarial grounding calibration; grade only supplied evidence.",
        shots,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    predicted = {case_id: by_id.get(case_id, (False, "missing"))[0] for case_id in expected}
    correct = sum(predicted[case_id] == wanted for case_id, wanted in expected.items())
    false_accepts = [
        case_id for case_id, wanted in expected.items() if not wanted and predicted[case_id]
    ]
    false_rejects = [
        case_id for case_id, wanted in expected.items() if wanted and not predicted[case_id]
    ]
    return {
        "model_key": key,
        "display_name": model.display_name,
        "verdict": verdict,
        "structurally_valid": set(by_id) == set(expected),
        "correct": correct,
        "total": len(expected),
        "accuracy": round(correct / len(expected), 4),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "latency_ms": round(latency_ms, 2),
        "predicted": predicted,
        "reasons": {case_id: by_id.get(case_id, (False, "missing"))[1] for case_id in expected},
        "raw_head": raw[:1200],
    }


async def main() -> None:
    shots, expected = fixtures()
    rows = await asyncio.gather(*(grade_model(key, shots, expected) for key in MODEL_KEYS))
    rows.sort(key=lambda row: (-row["accuracy"], row["latency_ms"]))
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "expected": expected,
        "models": rows,
    }
    output = Path("results/critic_bakeoff_20260827.json")
    output.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            [
                {
                    key: row[key]
                    for key in (
                        "model_key",
                        "accuracy",
                        "structurally_valid",
                        "false_accepts",
                        "false_rejects",
                        "latency_ms",
                    )
                }
                for row in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
