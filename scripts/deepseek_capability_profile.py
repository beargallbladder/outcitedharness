#!/usr/bin/env python3
"""Find the best role for the two-node DeepSeek shadow reasoner."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from critic_bakeoff import fixtures
from reasoner_bakeoff import (
    REASONING_PROMPT,
    REPAIR_PROMPT,
    _json_value,
    _reasoning_chars,
    _score_reasoning,
    _score_repairs,
)

from harness.config import ModelConfig, load_config
from harness.dispatch import _run_critic
from harness.providers.base import ChatMessage, ChatRequest, ChatResult
from harness.providers.openai_compatible import OpenAICompatibleProvider
from harness.workers.registry import Worker


OUTPUT = Path("results/deepseek_capability_profile_20260828.json")

DIAGNOSIS_PROMPT = """\
Choose the root-cause action for each incident. Return only a JSON array with
objects containing "id", "decision", and "why". Do not use markdown.

1. id=worker_alignment
Code creates tasks from all workers, awaits results, then removes disabled
workers before zip(workers, results). Decisions: filter_before_dispatch,
sort_results, retry_disabled_workers.

2. id=sqlite_lock
A transaction starts, then awaits a remote embedding request for 30 seconds
before writing and committing. Other writers time out. Decisions:
increase_sqlite_timeout, move_network_outside_transaction, add_second_database.

3. id=durable_result
On resume, a verification result exists in durable command history but the
reconstructed message list contains no write exchange. A helper searches only
after the latest write and reports no result. Decisions: fabricate_success,
search_durable_command_history, reset_repository.

4. id=reboot_monitor
An SSH reboot monitor reconnects successfully, then invokes
"systemctl is-system-running --wait" and appears hung. Decisions:
increase_ssh_timeout, remove_inner_wait, reboot_again.
"""

DIAGNOSIS_EXPECTED = {
    "worker_alignment": "filter_before_dispatch",
    "sqlite_lock": "move_network_outside_transaction",
    "durable_result": "search_durable_command_history",
    "reboot_monitor": "remove_inner_wait",
}

ARCHITECTURE_PROMPT = """\
Choose the architecture that satisfies each constraint. Return only a JSON
array with objects containing "id", "decision", and "why". Do not use markdown.

1. id=isolated_search
A new global code index may reuse an existing embedding API but must not share
process lifecycle or persistence with the live ranking service. Decisions:
separate_service_shared_encoder, plugin_inside_ranker, shared_sqlite_tables.

2. id=two_node_model
One 155 GiB model cannot fit on either 121 GiB node. A 200 Gb/s fabric connects
them and one OpenAI-compatible endpoint is required. Decisions:
round_robin_independent_models, tensor_parallel_single_endpoint,
duplicate_full_model.

3. id=atomic_publish
A verified project must never appear partially copied at its destination.
Decisions: copy_files_in_place, sibling_staging_then_atomic_rename,
publish_before_verification.

4. id=role_routing
A candidate is excellent at grounded review but weak at mutation-boundary
decisions. Decisions: replace_foreman, route_only_review, route_every_request.
"""

ARCHITECTURE_EXPECTED = {
    "isolated_search": "separate_service_shared_encoder",
    "two_node_model": "tensor_parallel_single_endpoint",
    "atomic_publish": "sibling_staging_then_atomic_rename",
    "role_routing": "route_only_review",
}


def _long_context_prompt() -> str:
    rows = [
        f"component-{index:03d}: owner=team-{index % 11}; port={7000 + index}; "
        f"policy=ordinary-{index % 7}"
        for index in range(220)
    ]
    rows[23] = "FACT orchestrator_owner=M5"
    rows[81] = "FACT gci_port=8810"
    rows[147] = "FACT greenfield_boundary=advisory_paths_never_become_execution_reads"
    rows[209] = "FACT publication_method=sibling_staging_then_atomic_rename"
    evidence = "\n".join(rows)
    return f"""\
Read the inventory and return only one JSON object with exactly these keys:
orchestrator_owner, gci_port, greenfield_boundary, publication_method.
Use the exact values from FACT rows. Do not use markdown.

INVENTORY
{evidence}
"""


def _variant(model: ModelConfig, effort: str | None) -> ModelConfig:
    extra = dict(model.extra_body)
    if effort is not None:
        extra["reasoning_effort"] = effort
        extra.pop("enable_thinking", None)
    return model.model_copy(update={"extra_body": extra})


async def _chat(model: ModelConfig, prompt: str, max_tokens: int) -> ChatResult:
    return await OpenAICompatibleProvider(model).chat(
        ChatRequest(
            messages=[
                ChatMessage(
                    role="system",
                    content="Be concise, evidence-bound, and obey the output schema.",
                ),
                ChatMessage(role="user", content=prompt),
            ],
            temperature=0,
            max_tokens=max_tokens,
            extra_body=model.extra_body,
            timeout_s=model.timeout_s,
            seed=0,
        )
    )


def _score_decisions(
    text: str,
    expected: dict[str, str],
) -> tuple[int, dict[str, bool], str | None]:
    try:
        rows = _json_value(text)
        decisions = {
            str(row.get("id")): str(row.get("decision"))
            for row in rows
            if isinstance(row, dict)
        }
        checks = {
            case_id: decisions.get(case_id) == decision
            for case_id, decision in expected.items()
        }
        return sum(checks.values()), checks, None
    except Exception as exc:
        return 0, {case_id: False for case_id in expected}, str(exc)


def _score_long_context(
    text: str,
) -> tuple[int, dict[str, bool], str | None]:
    expected: dict[str, Any] = {
        "orchestrator_owner": "M5",
        "gci_port": 8810,
        "greenfield_boundary": "advisory_paths_never_become_execution_reads",
        "publication_method": "sibling_staging_then_atomic_rename",
    }
    try:
        value = _json_value(text)
        checks = {
            key: str(value.get(key)) == str(wanted)
            for key, wanted in expected.items()
        }
        return sum(checks.values()), checks, None
    except Exception as exc:
        return 0, {key: False for key in expected}, str(exc)


def _finish_reason(result: ChatResult) -> str | None:
    try:
        return str(result.raw_response["choices"][0]["finish_reason"])
    except Exception:
        return None


async def _calibrate(name: str, model: ModelConfig, effort: str | None) -> dict[str, Any]:
    max_tokens = 8192 if effort == "max" else 4096
    reasoning = await _chat(model, REASONING_PROMPT, max_tokens)
    repairs = await _chat(model, REPAIR_PROMPT, max_tokens)
    reasoning_score, reasoning_checks, reasoning_error = _score_reasoning(reasoning.text)
    repair_score, repair_checks, repair_error = _score_repairs(repairs.text)
    latency = reasoning.latency_ms + repairs.latency_ms
    tokens = (reasoning.output_tokens or 0) + (repairs.output_tokens or 0)
    return {
        "name": name,
        "effort": effort or "off",
        "score": reasoning_score + repair_score,
        "total": 8,
        "checks": {**reasoning_checks, **repair_checks},
        "latency_ms": round(latency, 2),
        "output_tokens": tokens,
        "tokens_per_second": round(tokens / (latency / 1000), 3) if latency else None,
        "reasoning_chars": _reasoning_chars(reasoning) + _reasoning_chars(repairs),
        "finish_reasons": [_finish_reason(reasoning), _finish_reason(repairs)],
        "errors": [item for item in (reasoning.error, repairs.error) if item],
        "parse_errors": [
            item for item in (reasoning_error, repair_error) if item
        ],
    }


async def _critic_profile(name: str, model: ModelConfig) -> dict[str, Any]:
    shots, expected = fixtures()
    worker = Worker(
        id=f"capability-{name}",
        enabled=True,
        model_key=name,
        endpoint=model.base_url,
        capabilities=("review",),
        failover_order=None,
        role="researcher",
    )
    verdict, raw, by_id = await _run_critic(
        worker,
        model,
        "Grounded review calibration. Grade only the supplied evidence.",
        shots,
    )
    if not by_id:
        return {
            "score": 0,
            "total": 0,
            "available": False,
            "checks": {},
            "verdict": verdict,
            "raw_head": raw[:800],
        }
    checks = {
        case_id: by_id.get(case_id, (False, "missing"))[0] == wanted
        for case_id, wanted in expected.items()
    }
    return {
        "score": sum(checks.values()),
        "total": len(checks),
        "available": True,
        "checks": checks,
        "verdict": verdict,
        "raw_head": raw[:800],
    }


async def _broad_profile(name: str, model: ModelConfig) -> dict[str, Any]:
    diagnosis, architecture, long_context = await asyncio.gather(
        _chat(model, DIAGNOSIS_PROMPT, 4096),
        _chat(model, ARCHITECTURE_PROMPT, 4096),
        _chat(model, _long_context_prompt(), 4096),
    )
    critic = await _critic_profile(name, model)
    diagnosis_score, diagnosis_checks, diagnosis_error = _score_decisions(
        diagnosis.text, DIAGNOSIS_EXPECTED
    )
    architecture_score, architecture_checks, architecture_error = _score_decisions(
        architecture.text, ARCHITECTURE_EXPECTED
    )
    context_score, context_checks, context_error = _score_long_context(
        long_context.text
    )
    categories = {
        "hard_diagnosis": {
            "score": diagnosis_score,
            "total": 4,
            "checks": diagnosis_checks,
        },
        "architecture": {
            "score": architecture_score,
            "total": 4,
            "checks": architecture_checks,
        },
        "long_context_retrieval": {
            "score": context_score,
            "total": 4,
            "checks": context_checks,
        },
        "grounded_review": critic,
    }
    results = (diagnosis, architecture, long_context)
    return {
        "name": name,
        "score": sum(row["score"] for row in categories.values()),
        "total": sum(row["total"] for row in categories.values()),
        "categories": categories,
        "latency_ms": round(sum(result.latency_ms for result in results), 2),
        "output_tokens": sum(result.output_tokens or 0 for result in results),
        "reasoning_chars": sum(_reasoning_chars(result) for result in results),
        "finish_reasons": [_finish_reason(result) for result in results],
        "errors": [result.error for result in results if result.error],
        "parse_errors": [
            item
            for item in (diagnosis_error, architecture_error, context_error)
            if item
        ],
        "responses": {
            "diagnosis": diagnosis.text,
            "architecture": architecture.text,
            "long_context": long_context.text,
        },
    }


async def main() -> None:
    cfg = load_config()
    qwen_tp2 = _variant(cfg.models["asus2_qwen"], None)
    deepseek_base = cfg.models["deepseek_flash_tp2_shadow"]
    calibration = await asyncio.gather(
        _calibrate("qwen_tp2", qwen_tp2, None),
        *(
            _calibrate(
                f"deepseek_{effort}",
                _variant(deepseek_base, effort),
                effort,
            )
            for effort in ("low", "high", "max")
        ),
    )
    deepseek_rows = [row for row in calibration if row["name"].startswith("deepseek_")]
    best = sorted(deepseek_rows, key=lambda row: (-row["score"], row["latency_ms"]))[0]
    best_effort = str(best["effort"])
    profiles = await asyncio.gather(
        _broad_profile("qwen_tp2", qwen_tp2),
        _broad_profile(
            f"deepseek_{best_effort}",
            _variant(deepseek_base, best_effort),
        ),
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_deepseek_effort": best_effort,
        "calibration": calibration,
        "profiles": profiles,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected_deepseek_effort": best_effort,
                "calibration": [
                    {
                        key: row[key]
                        for key in (
                            "name",
                            "score",
                            "total",
                            "latency_ms",
                            "output_tokens",
                            "reasoning_chars",
                            "finish_reasons",
                        )
                    }
                    for row in calibration
                ],
                "profiles": [
                    {
                        "name": row["name"],
                        "score": row["score"],
                        "total": row["total"],
                        "categories": {
                            key: f"{value['score']}/{value['total']}"
                            for key, value in row["categories"].items()
                        },
                        "latency_ms": row["latency_ms"],
                    }
                    for row in profiles
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
