from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from harness.config import ModelConfig
from harness.cost import estimate_cost
from harness.gateway.anthropic_compat import anthropic_json_to_openai, openai_to_anthropic_payload
from harness.gateway.qwen_tools import rewrite_openai_completion
from harness.storage.db import Store, utcnow


class ProxyResult:
    def __init__(self) -> None:
        self.status = 200
        self.headers: dict[str, str] = {"content-type": "application/json"}
        self.body: bytes = b""
        self.model_key = ""
        self.upstream_model = ""
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.latency_ms = 0.0
        self.error: str | None = None


def _openai_headers(model: ModelConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if model.api_key:
        headers["Authorization"] = f"Bearer {model.api_key}"
    headers.update(model.extra_headers)
    return headers


def _anthropic_headers(model: ModelConfig) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if model.api_key:
        headers["x-api-key"] = model.api_key
    return headers


def _rewrite_openai_body(body: dict[str, Any], model: ModelConfig) -> dict[str, Any]:
    payload = dict(body)
    payload["model"] = model.model
    if model.extra_body:
        payload.update(model.extra_body)
    if model.temperature is not None and "temperature" not in body:
        payload["temperature"] = model.temperature
    return payload


async def complete_openai(
    model: ModelConfig,
    body: dict[str, Any],
    timeout_s: float,
) -> ProxyResult:
    result = ProxyResult()
    result.model_key = model.key
    result.upstream_model = model.model
    started = time.perf_counter()
    payload = _rewrite_openai_body(body, model)
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                f"{model.base_url}/chat/completions",
                headers=_openai_headers(model),
                json=payload,
            )
        result.latency_ms = (time.perf_counter() - started) * 1000
        result.status = response.status_code
        result.body = response.content
        if response.status_code >= 400:
            result.error = response.text[:400]
            return result
        data = rewrite_openai_completion(response.json())
        result.body = json.dumps(data).encode()
        usage = data.get("usage") or {}
        result.input_tokens = usage.get("prompt_tokens")
        result.output_tokens = usage.get("completion_tokens")
        if not _has_content(data):
            result.error = "empty completion"
        return result
    except Exception as exc:
        result.latency_ms = (time.perf_counter() - started) * 1000
        result.status = 502
        result.error = f"{type(exc).__name__}: {exc}"
        result.body = json.dumps({"error": {"message": result.error, "type": "proxy_error"}}).encode()
        return result


async def stream_openai(
    model: ModelConfig,
    body: dict[str, Any],
    timeout_s: float,
) -> AsyncIterator[bytes]:
    payload = _rewrite_openai_body(body, model)
    payload["stream"] = True
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        async with client.stream(
            "POST",
            f"{model.base_url}/chat/completions",
            headers=_openai_headers(model),
            json=payload,
        ) as response:
            if response.status_code >= 400:
                text = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status_code}: {text[:300]}")
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk


async def complete_anthropic(
    model: ModelConfig,
    body: dict[str, Any],
    timeout_s: float,
    requested_model: str,
    max_tokens: int,
) -> ProxyResult:
    result = ProxyResult()
    result.model_key = model.key
    result.upstream_model = model.model
    started = time.perf_counter()
    payload = openai_to_anthropic_payload(body, model.model, max_tokens)
    payload["stream"] = False
    url = (
        f"{model.base_url}/messages"
        if model.base_url.endswith("/v1")
        else f"{model.base_url}/v1/messages"
    )
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(url, headers=_anthropic_headers(model), json=payload)
        result.latency_ms = (time.perf_counter() - started) * 1000
        result.status = response.status_code
        if response.status_code >= 400:
            result.error = response.text[:400]
            result.body = response.content
            return result
        converted = anthropic_json_to_openai(response.json(), requested_model)
        usage = converted.get("usage") or {}
        result.input_tokens = usage.get("prompt_tokens")
        result.output_tokens = usage.get("completion_tokens")
        result.body = json.dumps(converted).encode()
        return result
    except Exception as exc:
        result.latency_ms = (time.perf_counter() - started) * 1000
        result.status = 502
        result.error = f"{type(exc).__name__}: {exc}"
        result.body = json.dumps({"error": {"message": result.error, "type": "proxy_error"}}).encode()
        return result


async def stream_anthropic(
    model: ModelConfig,
    body: dict[str, Any],
    timeout_s: float,
    requested_model: str,
    max_tokens: int,
) -> AsyncIterator[bytes]:
    payload = openai_to_anthropic_payload(body, model.model, max_tokens)
    payload["stream"] = True
    url = (
        f"{model.base_url}/messages"
        if model.base_url.endswith("/v1")
        else f"{model.base_url}/v1/messages"
    )
    chunk_id = "harness-anthropic"
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        async with client.stream(
            "POST",
            url,
            headers=_anthropic_headers(model),
            json=payload,
        ) as response:
            if response.status_code >= 400:
                text = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status_code}: {text[:300]}")
            tool_index = -1
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield _sse_openai_delta(chunk_id, requested_model, {"content": delta["text"]})
                    elif delta.get("type") == "input_json_delta" and delta.get("partial_json") is not None:
                        yield _sse_openai_delta(
                            chunk_id,
                            requested_model,
                            {
                                "tool_calls": [
                                    {
                                        "index": max(tool_index, 0),
                                        "function": {"arguments": delta.get("partial_json") or ""},
                                    }
                                ]
                            },
                        )
                elif etype == "content_block_start":
                    block = event.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        tool_index += 1
                        yield _sse_openai_delta(
                            chunk_id,
                            requested_model,
                            {
                                "tool_calls": [
                                    {
                                        "index": tool_index,
                                        "id": block.get("id") or "",
                                        "type": "function",
                                        "function": {"name": block.get("name") or "", "arguments": ""},
                                    }
                                ]
                            },
                        )
                elif etype == "message_delta":
                    stop = (event.get("delta") or {}).get("stop_reason")
                    if stop:
                        reason = "tool_calls" if stop == "tool_use" else "stop"
                        yield _sse_openai_delta(chunk_id, requested_model, {}, finish=reason)
            yield b"data: [DONE]\n\n"


def _sse_openai_delta(
    chunk_id: str,
    model: str,
    delta: dict[str, Any],
    finish: str | None = None,
) -> bytes:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _has_content(data: dict[str, Any]) -> bool:
    choices = data.get("choices") or []
    if not choices:
        return False
    message = (choices[0] or {}).get("message") or {}
    if message.get("content"):
        return True
    if message.get("tool_calls"):
        return True
    return False


def _worker_for_alias(alias: str) -> str:
    if alias in ("harness-local", "harness-auto"):
        return "primary_coder"
    if alias == "harness-m5":
        return "fallback_reasoner"
    if alias == "harness-frontier":
        return "frontier_senior"
    return "primary_coder"


def _record_attempt(
    store: Store,
    alias: str,
    status: int,
    latency_ms: float,
    input_tokens: int | None,
    output_tokens: int | None,
) -> None:
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
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
            prompt_chars += len(content) if isinstance(content, str) else len(json.dumps(content or ""))
    store.insert_cline_turn(
        {
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
    _record_attempt(store, alias, status, latency_ms, input_tokens, output_tokens)


def turn_cost(cfg, model_key: str, inbound: int | None, outbound: int | None) -> float | None:
    return estimate_cost(cfg.pricing_for(model_key), inbound, outbound)
