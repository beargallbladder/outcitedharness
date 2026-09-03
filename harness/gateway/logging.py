from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING
from typing import Any

from harness.cost import estimate_cost
from harness.storage.db import Store, utcnow

if TYPE_CHECKING:
    from harness.training.ledger import LearningLedger


LOGGER = logging.getLogger(__name__)


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
    response: dict[str, Any] | str | None = None,
    learning_ledger: LearningLedger | None = None,
    source_revision: str | None = None,
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
    started_at = utcnow()
    turn_id = store.insert_gateway_turn(
        {
            "task_id": task_id,
            "started_at": started_at,
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
    if learning_ledger is not None and response is not None:
        _capture_learning_turn(
            learning_ledger,
            turn_id=turn_id,
            task_id=task_id,
            started_at=started_at,
            alias=alias,
            model_key=model_key,
            upstream_model=upstream_model,
            stream=stream,
            status=status,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            body=body,
            response=response,
            source_revision=source_revision,
        )


def _capture_learning_turn(
    ledger: LearningLedger,
    *,
    turn_id: int,
    task_id: str,
    started_at: str,
    alias: str,
    model_key: str,
    upstream_model: str,
    stream: bool,
    status: int,
    latency_ms: float,
    input_tokens: int | None,
    output_tokens: int | None,
    cost: float | None,
    body: dict[str, Any],
    response: dict[str, Any] | str,
    source_revision: str | None,
) -> None:
    from harness.training.ledger import ArtifactPayload, VerificationPayload
    from harness.training.models import (
        LearningEvent,
        SourceKind,
        is_excluded_learning_source,
    )

    capture_source = f"harness://gateway-turns/{turn_id}"
    if is_excluded_learning_source(
        SourceKind.HARNESS,
        capture_source,
        {"request": body, "response": response},
    ):
        LOGGER.warning(
            "learning capture skipped excluded source markers for gateway turn %s",
            turn_id,
        )
        return

    event = LearningEvent(
        event_id=f"gateway-turn-{turn_id}",
        event_type="gateway_turn",
        source_kind=SourceKind.HARNESS,
        source_uri=capture_source,
        source_revision=source_revision,
        task_id=task_id,
        lineage_id=task_id,
        authorization_scope="settings.learning_capture_enabled",
        created_at=datetime.fromisoformat(started_at),
        estimated_cost=cost,
        metadata={
            "alias": alias,
            "model_key": model_key,
            "upstream_model": upstream_model,
            "stream": stream,
            "status": status,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "data_use": "quarantine",
            "disposition": "quarantine",
        },
    )
    request_text = json.dumps(body, ensure_ascii=False, sort_keys=True)
    response_text = (
        response
        if isinstance(response, str)
        else json.dumps(response, ensure_ascii=False, sort_keys=True)
    )
    try:
        ledger.capture(
            event,
            [
                ArtifactPayload(
                    kind="gateway_request",
                    content=request_text,
                    media_type="application/json",
                ),
                ArtifactPayload(
                    kind="gateway_response",
                    content=response_text,
                    media_type=(
                        "text/plain"
                        if isinstance(response, str)
                        else "application/json"
                    ),
                ),
            ],
            [
                VerificationPayload(
                    kind="gateway_transport",
                    status="unknown",
                    verifier="harness.gateway",
                    output_kind="gateway_response",
                    metadata={
                        "http_status": status,
                        "transport_succeeded": status < 400,
                        "proof_scope": "transport_only",
                    },
                )
            ],
        )
    except Exception:
        LOGGER.exception("learning capture failed for gateway turn %s", turn_id)


def turn_cost(
    cfg: Any,
    model_key: str,
    inbound: int | None,
    outbound: int | None,
) -> float | None:
    return estimate_cost(cfg.pricing_for(model_key), inbound, outbound)
