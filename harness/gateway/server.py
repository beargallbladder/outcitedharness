from __future__ import annotations

import asyncio
import hmac
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from harness.config import AppConfig, load_config
from harness.gateway.logging import log_turn
from harness.gateway.spec import (
    GatewaySpec,
    is_orch_alias,
    listed_models,
    load_gateway_spec,
)
from harness.storage.db import Store
from harness.workers.registry import load_registry


def create_app(
    cfg: AppConfig | None = None,
    spec: GatewaySpec | None = None,
) -> Starlette:
    cfg = cfg or load_config()
    spec = spec or load_gateway_spec(cfg.root)
    registry = load_registry(cfg.root)
    store = Store(cfg.settings.db_path)
    learning_ledger = None
    if cfg.settings.learning_capture_enabled:
        from harness.training.ledger import LearningLedger

        learning_ledger = LearningLedger(
            store,
            cfg.settings.learning_artifact_root,
        )

    async def healthz(request: Request) -> JSONResponse:
        chain = registry.failover_keys() or spec.auto_ladder
        workers = registry.summary()
        async with httpx.AsyncClient(timeout=2.5) as client:
            await asyncio.gather(*(_probe_worker(client, row) for row in workers))
        if not _fleet_visible(request, spec):
            return JSONResponse(
                {
                    "ready": True,
                    "service": "harness",
                    "model": "harness-orch",
                }
            )
        return JSONResponse(
            {
                "ready": True,
                "service": "harness-orch-gateway",
                "listen": f"{spec.listen_host}:{spec.listen_port}",
                "aliases": spec.aliases,
                "auto_ladder": chain,
                "workers": workers,
            }
        )

    async def index(request: Request) -> JSONResponse:
        orchestration_models = sorted(
            alias for alias, target in spec.aliases.items() if target == "orch"
        )
        return JSONResponse(
            {
                "service": "harness",
                "auth": "send the provided API key as a Bearer token",
                "models": orchestration_models,
                "chat": "POST /v1/chat/completions (OpenAI-compatible)",
                "health": "GET /healthz",
            }
        )

    async def models(request: Request) -> JSONResponse:
        rows = [
            row for row in listed_models(spec) if is_orch_alias(spec, row["id"])
        ]
        return JSONResponse({"object": "list", "data": rows})

    async def chat(request: Request) -> Response:
        if not _authorized(request, spec):
            return JSONResponse({"error": {"message": "invalid api key", "type": "auth"}}, status_code=401)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": {"message": "invalid json", "type": "invalid_request"}}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": {"message": "body must be an object"}}, status_code=400)

        requested = str(body.get("model") or "harness-orch")
        if not is_orch_alias(spec, requested):
            return JSONResponse(
                {
                    "error": {
                        "message": "unknown harness model",
                        "type": "model_not_found",
                    }
                },
                status_code=404,
            )
        stream = bool(body.get("stream"))
        if stream:
            return StreamingResponse(
                _stream_orch(
                    cfg,
                    spec,
                    store,
                    requested,
                    body,
                    learning_ledger=learning_ledger,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return await _complete_orch(
            cfg,
            spec,
            store,
            requested,
            body,
            learning_ledger=learning_ledger,
        )

    return Starlette(
        routes=[
            Route("/", index, methods=["GET"]),
            Route("/v1", index, methods=["GET"]),
            Route("/healthz", healthz, methods=["GET"]),
            Route("/health", healthz, methods=["GET"]),
            Route("/v1/health", healthz, methods=["GET"]),
            Route("/v1/healthz", healthz, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/models", models, methods=["GET"]),
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/chat/completions", chat, methods=["POST"]),
        ]
    )


async def _probe_worker(client: httpx.AsyncClient, row: dict[str, Any]) -> None:
    """Live-probe one worker endpoint and annotate the summary row in place.

    Static registry status said "healthy" for any enabled worker, which hid
    dead backends (a down primary coder only surfaced as a 502 on the next
    request). Probe GET {endpoint}/models; a non-5xx answer means the socket
    is alive even if the service does not route /models (the embedder 404s).
    """
    row["live"] = "unknown"
    row["served_model"] = None
    row["probe_error"] = None
    endpoint = (row.get("endpoint") or "").rstrip("/")
    if not row.get("enabled"):
        row["live"] = "disabled"
        return
    if not endpoint.startswith("http://"):
        # Hosted APIs (frontier) need auth to probe; do not burn tokens here.
        row["live"] = "unprobed"
        return
    try:
        resp = await client.get(f"{endpoint}/models")
    except Exception as exc:
        row["live"] = "down"
        row["probe_error"] = f"{type(exc).__name__}: {exc}"
        return
    if resp.status_code >= 500:
        row["live"] = "down"
        row["probe_error"] = f"HTTP {resp.status_code}"
        return
    row["live"] = "up"
    if resp.status_code != 200:
        row["probe_error"] = f"HTTP {resp.status_code} on /models (socket alive)"
        return
    try:
        data = resp.json().get("data") or []
        row["served_model"] = str(data[0].get("id")) if data else None
    except Exception:
        pass


def _presented_keys(request: Request) -> list[str]:
    keys: list[str] = []
    header = (request.headers.get("authorization") or "").strip()
    if header:
        kind, _, rest = header.partition(" ")
        if rest and kind.lower() == "bearer":
            keys.append(rest.strip())
        else:
            keys.append(header)
    for name in ("x-api-key", "api-key"):
        value = (request.headers.get(name) or "").strip()
        if value:
            keys.append(value)
    return keys


def _loopback(request: Request, spec: GatewaySpec) -> bool:
    peer = request.client.host if request.client else ""
    if peer == "testclient":
        return spec.listen_host in {"127.0.0.1", "::1", "localhost"}
    return peer in {"127.0.0.1", "::1", "localhost"}


def _fleet_visible(request: Request, spec: GatewaySpec) -> bool:
    """Allow internal fleet detail only to loopback or authenticated callers."""
    return _loopback(request, spec) or _authorized(request, spec)


def _authorized(request: Request, spec: GatewaySpec) -> bool:
    if _loopback(request, spec):
        return True
    if not spec.api_key:
        return False
    allowed = (spec.api_key, f"sk-{spec.api_key}")
    return any(
        hmac.compare_digest(token, candidate)
        for token in _presented_keys(request)
        for candidate in allowed
    )


async def _complete_orch(
    cfg,
    spec,
    store,
    requested,
    body,
    *,
    learning_ledger=None,
):
    from harness.gateway.orch import (
        compact_thread,
        completion_body,
        completion_body_tools,
        last_user_text,
        run_orch,
    )

    started = time.perf_counter()
    messages = body.get("messages") or []
    intent = last_user_text(messages)
    thread = compact_thread(messages)
    if not intent:
        payload = {"error": {"message": "orch needs a user message", "type": "invalid_request"}}
        return JSONResponse(payload, status_code=400)
    try:
        result = await run_orch(
            cfg,
            intent,
            thread=thread,
            messages=messages,
            tools=body.get("tools") or body.get("functions"),
            extra=body if isinstance(body, dict) else None,
        )
        status = 200
        error = result.error
    except Exception as exc:
        from harness.gateway.orch import OrchResult

        result = OrchResult(text=f"Harness orch failed: {type(exc).__name__}: {exc}", error=str(exc))
        status = 502
        error = result.error
    latency_ms = (time.perf_counter() - started) * 1000
    if result.tool_calls:
        data = completion_body_tools(result.tool_calls, requested)
    else:
        data = completion_body(result.text, requested)
    log_turn(
        store,
        alias=requested,
        model_key="orch",
        upstream_model="gather" if result.tool_calls else "dispatch",
        stream=False,
        status=status,
        latency_ms=latency_ms,
        input_tokens=data["usage"]["prompt_tokens"],
        output_tokens=data["usage"]["completion_tokens"],
        cost=None,
        error=error,
        body=body,
        response=data,
        learning_ledger=learning_ledger,
    )
    return JSONResponse(data, status_code=status)


async def _stream_orch(
    cfg,
    spec,
    store,
    requested,
    body,
    *,
    learning_ledger=None,
):
    from harness.gateway.orch import compact_thread, last_user_text, run_orch, sse_chunk
    from harness.gateway.qwen_tools import synthesize_tool_call_sse

    started = time.perf_counter()
    messages = body.get("messages") or []
    intent = last_user_text(messages)
    thread = compact_thread(messages)
    error = None
    status = 200
    try:
        if not intent:
            text = "Harness orch needs a user message."
            status = 400
            error = text
            result_calls = []
        else:
            outcome = await run_orch(
                cfg,
                intent,
                thread=thread,
                messages=messages,
                tools=body.get("tools") or body.get("functions"),
                extra=body if isinstance(body, dict) else None,
            )
            text = outcome.text
            result_calls = outcome.tool_calls
            error = outcome.error
    except Exception as exc:
        text = f"Harness orch failed: {type(exc).__name__}: {exc}"
        status = 502
        error = text
        result_calls = []
    if result_calls:
        for chunk in synthesize_tool_call_sse(requested, result_calls):
            yield chunk
    else:
        step = 80
        yield sse_chunk(requested, {"role": "assistant", "content": ""})
        for i in range(0, len(text), step):
            yield sse_chunk(requested, {"content": text[i : i + step]})
        yield sse_chunk(requested, {}, finish="stop")
        yield b"data: [DONE]\n\n"
    log_turn(
        store,
        alias=requested,
        model_key="orch",
        upstream_model="gather" if result_calls else "dispatch",
        stream=True,
        status=status,
        latency_ms=(time.perf_counter() - started) * 1000,
        input_tokens=None,
        output_tokens=None,
        cost=None,
        error=error,
        body=body,
        response={"text": text, "tool_calls": result_calls},
        learning_ledger=learning_ledger,
    )


def serve(host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    cfg = load_config()
    spec = load_gateway_spec(cfg.root)
    app = create_app(cfg, spec)
    uvicorn.run(
        app,
        host=host or spec.listen_host,
        port=port or spec.listen_port,
        log_level="info",
    )
