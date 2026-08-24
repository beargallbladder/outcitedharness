from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

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
from harness.gateway.spec import ClineSpec, ladder_for, listed_models, load_cline_spec
from harness.storage.db import Store
from harness.workers.registry import load_registry
from harness.workers.router import should_failover


def create_app(cfg: AppConfig | None = None, spec: ClineSpec | None = None) -> Starlette:
    cfg = cfg or load_config()
    spec = spec or load_cline_spec(cfg.root)
    registry = load_registry(cfg.root)
    store = Store(cfg.settings.db_path)

    async def healthz(_request: Request) -> JSONResponse:
        chain = registry.failover_keys() or spec.auto_ladder
        return JSONResponse(
            {
                "ready": True,
                "service": "harness-cline-gateway",
                "aliases": spec.aliases,
                "auto_ladder": chain,
                "workers": registry.summary(),
            }
        )

    async def models(_request: Request) -> JSONResponse:
        return JSONResponse({"object": "list", "data": listed_models(spec)})

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
            Route("/healthz", healthz, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
            Route("/models", models, methods=["GET"]),
            Route("/v1/chat/completions", chat, methods=["POST"]),
            Route("/chat/completions", chat, methods=["POST"]),
        ]
    )


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
            result = await complete_openai(model, body, model.timeout_s or cfg.settings.default_timeout_s)
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
        return Response(content=result.body, status_code=result.status, media_type="application/json")
    assert last is not None
    return Response(content=last.body, status_code=last.status, media_type="application/json")


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
                agen = stream_openai(model, body, model.timeout_s or cfg.settings.default_timeout_s)
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
                        for chunk in synthesize_tool_call_sse(model.model, calls):
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
                payload = {"error": {"message": "; ".join(errors), "type": "proxy_error"}}
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
