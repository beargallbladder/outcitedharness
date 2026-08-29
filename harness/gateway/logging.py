from __future__ import annotations

import json
from typing import Any

from harness.cost import estimate_cost
from harness.storage.db import Store, utcnow


def _worker_for_alias(alias: str) -> str:
    if alias in ("harness-local", "harness-auto"):
        return "primary_coder"
    if alias == "harness-dgx2":
        return "dgx2_coder"
    if alias == "harness-asus":
        return "asus_coder"
    if alias == "harness-dgx3":
        return "dgx3_coder"
    if alias in ("harness-orch", "harness-m5"):
        return "fallback_reasoner"
    if alias == "harness-frontier":
        return "frontier_senior"
    if alias == "harness-researcher":
        return "researcher"
    return "primary_coder"


def _record_attempt(
    store: Store,
    alias: str,
    status: int,
    latency_ms: float,
    input_tokens: int | None,
    output_tokens: int | None,
) -> str:
    from harness.task.models import AttemptRecord
    from harness.task.service import TaskService

    svc = TaskService(store)
    task = svc.session_task()
    svc.record_turn(
        AttemptRecord(
            task_id=task.task_id,
            attempt=0,
            worker=_worker_for_alias(alias),
            result="success" if status < 400 else "failed",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=latency_ms,
        )
    )
    return task.task_id


def log_turn(
    store: Store,
    *,
    alias: str,
    model_key: str,
    upstream_model: str,
    stream: bool,
    status: int,
    latency_ms: float,
    input_tokens: int | None,
    output_tokens: int | None,
    cost: float | None,
    error: str | None,
    body: dict[str, Any],
) -> None:
    messages = body.get("messages") or []
    prompt_chars = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        prompt_chars += (
            len(content)
            if isinstance(content, str)
            else len(json.dumps(content or ""))
        )
    task_id = _record_attempt(
        store,
        alias,
        status,
        latency_ms,
        input_tokens,
        output_tokens,
    )
    store.insert_gateway_turn(
        {
            "task_id": task_id,
            "started_at": utcnow(),
            "alias": alias,
            "model_key": model_key,
            "upstream_model": upstream_model,
            "stream": int(stream),
            "status": status,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": cost,
            "error": error,
            "message_count": len(messages),
            "has_tools": int(bool(body.get("tools"))),
            "prompt_chars": prompt_chars,
        }
    )


def turn_cost(
    cfg: Any,
    model_key: str,
    inbound: int | None,
    outbound: int | None,
) -> float | None:
    return estimate_cost(cfg.pricing_for(model_key), inbound, outbound)
