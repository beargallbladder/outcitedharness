from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from harness.config import AppConfig, load_config
from harness.cost import estimate_cost
from harness.gateway.proxy import (
    complete_anthropic,
    complete_openai,
    log_turn,
    stream_anthropic,
    stream_openai,
)
from harness.gateway.qwen_tools import (
    collect_stream_text,
    parse_qwen_tool_text,
    stream_already_has_tool_calls,
    synthesize_tool_call_sse,
)
from harness.gateway.spec import ClineSpec, is_orch_alias, ladder_for, listed_models, load_cline_spec
from harness.storage.db import Store
from harness.workers.registry import load_registry
from harness.workers.router import should_failover


def create_app(cfg: AppConfig | None = None, spec: ClineSpec | None = None) -> Starlette:
    cfg = cfg or load_config()
    spec = spec or load_cline_spec(cfg.root)
    registry = load_registry(cfg.root)
    store = Store(cfg.settings.db_path)

    async def healthz(request: Request) -> JSONResponse:
        chain = registry.failover_keys() or spec.auto_ladder
        workers = registry.summary()
        async with httpx.AsyncClient(timeout=2.5) as client:
            await asyncio.gather(*(_probe_worker(client, row) for row in workers))
        if not _loopback(request, spec):
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
                "service": "harness-cline-gateway",
                "listen": f"{spec.listen_host}:{spec.listen_port}",
                "aliases": spec.aliases,
                "auto_ladder": chain,
                "workers": workers,
            }
        )

    async def index(request: Request) -> JSONResponse:
        public = not _loopback(request, spec)
        return JSONResponse(
            {
                "service": "harness",
                "auth": "send the provided API key as a Bearer token",
                "models": ["harness-orch"] if public else sorted(spec.aliases),
                "chat": "POST /v1/chat/completions (OpenAI-compatible)",
                "health": "GET /healthz",
            }
        )

    async def models(request: Request) -> JSONResponse:
        rows = listed_models(spec)
        if not _loopback(request, spec):
            rows = [row for row in rows if row["id"] == "harness-orch"]
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

        requested = str(body.get("model") or "harness-local")
        public_models = {
            alias for alias, target in spec.aliases.items() if target == "orch"
        }
        if not _loopback(request, spec) and requested not in public_models:
            return JSONResponse(
                {
                    "error": {
                        "message": "unknown harness model",
                        "type": "model_not_found",
                    }
                },
                status_code=404,
            )
        if is_orch_alias(spec, requested):
            stream = bool(body.get("stream"))
            if stream:
                return StreamingResponse(
                    _stream_orch(cfg, spec, store, requested, body),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            return await _complete_orch(cfg, spec, store, requested, body)
        try:
            models_to_try = ladder_for(spec, cfg, requested, registry)
        except KeyError as exc:
            return JSONResponse({"error": {"message": str(exc), "type": "model_not_found"}}, status_code=404)

        stream = bool(body.get("stream"))
        if stream:
            return StreamingResponse(
                _stream_with_fallback(cfg, spec, store, requested, models_to_try, body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return await _complete_with_fallback(cfg, spec, store, requested, models_to_try, body)

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


def _loopback(request: Request, spec: ClineSpec) -> bool:
    peer = request.client.host if request.client else ""
    return spec.listen_host in {"127.0.0.1", "localhost", "::1"} or peer in {"127.0.0.1", "::1", "localhost"}


def _authorized(request: Request, spec: ClineSpec) -> bool:
    # Dummy key is only so Cline's form is not empty. Cline 4 often sends
    # api-key, an sk- prefix, or a leftover secret on first verify.
    if not spec.api_key or _loopback(request, spec):
        return True
    allowed = {spec.api_key, f"sk-{spec.api_key}"}
    return any(token in allowed for token in _presented_keys(request))


async def _complete_with_fallback(
    cfg: AppConfig,
    spec: ClineSpec,
    store: Store,
    requested: str,
    models_to_try,
    body: dict[str, Any],
) -> Response:
    last = None
    errors: list[str] = []
    for model in models_to_try:
        started = time.perf_counter()
        if model.provider == "anthropic":
            result = await complete_anthropic(
                model, body, model.timeout_s or cfg.settings.default_timeout_s, requested, spec.max_output_tokens
            )
        else:
            result = await complete_openai(
                model,
                body,
                model.timeout_s or cfg.settings.default_timeout_s,
            )
            if result.status < 400:
                try:
                    data = json.loads(result.body)
                    data["model"] = requested
                    result.body = json.dumps(data).encode()
                except (TypeError, json.JSONDecodeError):
                    pass
        result.latency_ms = result.latency_ms or (time.perf_counter() - started) * 1000
        last = result
        retryable = should_failover(
            result.status, result.error, model is not models_to_try[-1]
        )
        if retryable and model is not models_to_try[-1]:
            errors.append(f"{model.key}:{result.error or result.status}")
            continue
        log_turn(
            store,
            alias=requested,
            model_key=result.model_key,
            upstream_model=result.upstream_model,
            stream=False,
            status=result.status,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost=estimate_cost(cfg.pricing_for(result.model_key), result.input_tokens, result.output_tokens),
            error=("; ".join(errors + ([result.error] if result.error else [])) or None),
            body=body,
        )
        content = result.body
        if result.status >= 400:
            content = json.dumps(
                {
                    "error": {
                        "message": "the harness could not complete this request",
                        "type": "harness_error",
                    }
                }
            ).encode()
        return Response(content=content, status_code=result.status, media_type="application/json")
    assert last is not None
    return Response(content=last.body, status_code=last.status, media_type="application/json")


async def _complete_orch(cfg, spec, store, requested, body):
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
    )
    return JSONResponse(data, status_code=status)


async def _stream_orch(cfg, spec, store, requested, body):
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
    )


async def _stream_with_fallback(
    cfg: AppConfig,
    spec: ClineSpec,
    store: Store,
    requested: str,
    models_to_try,
    body: dict[str, Any],
) -> AsyncIterator[bytes]:
    errors: list[str] = []
    for index, model in enumerate(models_to_try):
        started = time.perf_counter()
        try:
            if model.provider == "anthropic":
                agen = stream_anthropic(
                    model,
                    body,
                    model.timeout_s or cfg.settings.default_timeout_s,
                    requested,
                    spec.max_output_tokens,
                )
            else:
                agen = stream_openai(
                    model,
                    body,
                    model.timeout_s or cfg.settings.default_timeout_s,
                    requested,
                )
            if body.get("tools") and model.provider != "anthropic":
                buffered: list[bytes] = []
                async for chunk in agen:
                    buffered.append(chunk)
                if stream_already_has_tool_calls(buffered):
                    for chunk in buffered:
                        yield chunk
                else:
                    calls = parse_qwen_tool_text(collect_stream_text(buffered))
                    if calls:
                        for chunk in synthesize_tool_call_sse(requested, calls):
                            yield chunk
                    else:
                        for chunk in buffered:
                            yield chunk
            else:
                async for chunk in agen:
                    yield chunk
            log_turn(
                store,
                alias=requested,
                model_key=model.key,
                upstream_model=model.model,
                stream=True,
                status=200,
                latency_ms=(time.perf_counter() - started) * 1000,
                input_tokens=None,
                output_tokens=None,
                cost=None,
                error="; ".join(errors) or None,
                body=body,
            )
            return
        except Exception as exc:
            errors.append(f"{model.key}:{exc}")
            if index == len(models_to_try) - 1:
                log_turn(
                    store,
                    alias=requested,
                    model_key=model.key,
                    upstream_model=model.model,
                    stream=True,
                    status=502,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    input_tokens=None,
                    output_tokens=None,
                    cost=None,
                    error="; ".join(errors),
                    body=body,
                )
                payload = {
                    "error": {
                        "message": "the harness could not complete this request",
                        "type": "harness_error",
                    }
                }
                yield f"data: {json.dumps(payload)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return


def serve(host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    cfg = load_config()
    spec = load_cline_spec(cfg.root)
    app = create_app(cfg, spec)
    uvicorn.run(
        app,
        host=host or spec.listen_host,
        port=port or spec.listen_port,
        log_level="info",
    )
