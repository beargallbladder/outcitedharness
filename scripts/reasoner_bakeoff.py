#!/usr/bin/env python3
"""Compare the production M5 foreman with the shadow TP=2 reasoner."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from harness.config import load_config
from harness.providers.base import ChatMessage, ChatRequest, ChatResult
from harness.providers.openai_compatible import OpenAICompatibleProvider


MODEL_KEYS = ("m5_qwen", "deepseek_flash_tp2_shadow")
OUTPUT = Path("results/reasoner_bakeoff_20260828.json")

REASONING_PROMPT = """\
You are the planning and safety foreman for a coding-agent orchestrator.
For each case, choose exactly one listed decision. Return only a JSON array
of objects with keys "id", "decision", and "why". Do not use markdown.

1. id=resume_provenance
State is phase=verify. The current command is "ruff check .". A reconstructed
durable result exists for "pytest -q", but no result exists for "ruff check ."
after the latest write.
Decisions: rerun_current_verification, repair_from_pytest, advance_milestone.

2. id=gci_boundary
A Greenfield planning packet cites /other/repo/src/api.py as an architectural
analogue. During execution the gather planner is considering reading that path.
The approved build workspace is /runs/new/repo.
Decisions: read_advisory_path, ignore_advisory_path, mutate_analogue.

3. id=verification_hash
All acceptance commands passed for diff hash A. A deterministic formatter then
changed a source file, producing diff hash B.
Decisions: reuse_old_results, invalidate_and_rerun_all_verification, commit_B.

4. id=no_mutation
A coder response describes a plausible implementation and contains 2,000
characters, but the action planner extracts zero executable mutation calls.
Decisions: mark_complete, block_no_mutation, ask_critic_to_infer_edits.
"""

REPAIR_PROMPT = """\
You are repairing four defects from exact source evidence. Return only one JSON
object whose keys are the case IDs and whose values are arrays of minimal
replacements. Every replacement must have string keys "old" and "new". Do not
rewrite whole files, do not use eval/exec, and do not use markdown.

sqlite_fetch:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM comparisons")
    rows = conn.fetchall()
    return [dict(row) for row in rows]

json_decode:
The database column is written with json.dumps(product_ids).
    result = dict(row)
    result["product_ids"] = eval(result["product_ids"])
    return result

failover_policy:
Authentication and other 4xx responses must not fail over. Transport errors and
5xx responses may fail over only when another backend exists.
    return has_next and (status >= 400 or bool(error))

diff_provenance:
Only verification records for state.active_diff_hash count as observed.
    observed = {record.command for record in state.verification_results}
    return observed == expected
"""

REASONING_EXPECTED = {
    "resume_provenance": "rerun_current_verification",
    "gci_boundary": "ignore_advisory_path",
    "verification_hash": "invalidate_and_rerun_all_verification",
    "no_mutation": "block_no_mutation",
}


def _json_value(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    starts = [index for index in (stripped.find("["), stripped.find("{")) if index >= 0]
    if not starts:
        raise ValueError("no JSON value found")
    start = min(starts)
    close = "]" if stripped[start] == "[" else "}"
    end = stripped.rfind(close)
    if end < start:
        raise ValueError("unterminated JSON value")
    return json.loads(stripped[start : end + 1])


def _score_reasoning(text: str) -> tuple[int, dict[str, bool], str | None]:
    try:
        value = _json_value(text)
        rows = {
            str(row.get("id")): str(row.get("decision"))
            for row in value
            if isinstance(row, dict)
        }
        checks = {
            case_id: rows.get(case_id) == expected
            for case_id, expected in REASONING_EXPECTED.items()
        }
        return sum(checks.values()), checks, None
    except Exception as exc:
        return 0, {case_id: False for case_id in REASONING_EXPECTED}, str(exc)


def _replacements(value: Any, case_id: str) -> list[dict[str, str]]:
    rows = value.get(case_id, []) if isinstance(value, dict) else []
    return [
        {"old": str(row.get("old", "")), "new": str(row.get("new", ""))}
        for row in rows
        if isinstance(row, dict)
    ]


def _has_replacement(
    rows: list[dict[str, str]],
    predicate: Callable[[str, str], bool],
) -> bool:
    return any(predicate(row["old"], row["new"]) for row in rows)


def _score_repairs(text: str) -> tuple[int, dict[str, bool], str | None]:
    try:
        value = _json_value(text)
        checks = {
            "sqlite_fetch": _has_replacement(
                _replacements(value, "sqlite_fetch"),
                lambda old, new: "conn.fetchall()" in old
                and "cursor.fetchall()" in new,
            ),
            "json_decode": _has_replacement(
                _replacements(value, "json_decode"),
                lambda old, new: "eval(" in old
                and "json.loads(" in new
                and "eval(" not in new,
            ),
            "failover_policy": _has_replacement(
                _replacements(value, "failover_policy"),
                lambda old, new: "status >= 400" in old
                and "status >= 500" in new
                and "has_next" in new,
            ),
            "diff_provenance": _has_replacement(
                _replacements(value, "diff_provenance"),
                lambda _old, new: "record.diff_hash" in new
                and "state.active_diff_hash" in new,
            ),
        }
        return sum(checks.values()), checks, None
    except Exception as exc:
        return 0, {
            "sqlite_fetch": False,
            "json_decode": False,
            "failover_policy": False,
            "diff_provenance": False,
        }, str(exc)


def _reasoning_chars(result: ChatResult) -> int:
    try:
        choices = result.raw_response.get("choices") or []
        message = (choices[0] or {}).get("message") or {}
        return len(message.get("reasoning") or "")
    except Exception:
        return 0


async def _call(model_key: str, prompt: str) -> ChatResult:
    model = load_config().models[model_key]
    return await OpenAICompatibleProvider(model).chat(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="system",
                    content="Be precise, evidence-bound, and obey the output schema.",
                ),
                ChatMessage(role="user", content=prompt),
            ],
            temperature=model.temperature,
            max_tokens=model.max_tokens or 4096,
            extra_body=model.extra_body,
            timeout_s=model.timeout_s,
            seed=0,
        )
    )


async def _run_model(model_key: str) -> dict[str, Any]:
    model = load_config().models[model_key]
    reasoning = await _call(model_key, REASONING_PROMPT)
    repairs = await _call(model_key, REPAIR_PROMPT)
    reasoning_score, reasoning_checks, reasoning_error = _score_reasoning(reasoning.text)
    repair_score, repair_checks, repair_error = _score_repairs(repairs.text)
    total_latency = reasoning.latency_ms + repairs.latency_ms
    output_tokens = (reasoning.output_tokens or 0) + (repairs.output_tokens or 0)
    return {
        "model_key": model_key,
        "display_name": model.display_name,
        "score": reasoning_score + repair_score,
        "total": 8,
        "reasoning_score": reasoning_score,
        "repair_score": repair_score,
        "checks": {**reasoning_checks, **repair_checks},
        "latency_ms": round(total_latency, 2),
        "request_latency_ms": [
            round(reasoning.latency_ms, 2),
            round(repairs.latency_ms, 2),
        ],
        "output_tokens": output_tokens,
        "output_tokens_per_second": (
            round(output_tokens / (total_latency / 1000), 3)
            if output_tokens and total_latency
            else None
        ),
        "reasoning_chars": _reasoning_chars(reasoning) + _reasoning_chars(repairs),
        "errors": [error for error in (reasoning.error, repairs.error) if error],
        "parse_errors": [
            error for error in (reasoning_error, repair_error) if error
        ],
        "responses": {
            "reasoning": reasoning.text,
            "repairs": repairs.text,
        },
    }


async def main() -> None:
    rows = await asyncio.gather(*(_run_model(key) for key in MODEL_KEYS))
    rows.sort(key=lambda row: (-row["score"], row["latency_ms"]))
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "Two sequential packets per model; models run concurrently.",
        "models": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    key: row[key]
                    for key in (
                        "model_key",
                        "score",
                        "total",
                        "reasoning_score",
                        "repair_score",
                        "latency_ms",
                        "output_tokens_per_second",
                        "reasoning_chars",
                        "errors",
                        "parse_errors",
                    )
                }
                for row in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
