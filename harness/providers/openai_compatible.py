from __future__ import annotations

import time
from typing import Any

import httpx

from harness.config import ModelConfig
from harness.providers.base import ChatRequest, ChatResult


def _usage_tokens(usage: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not usage:
        return None, None
    inbound = usage.get("prompt_tokens", usage.get("input_tokens"))
    outbound = usage.get("completion_tokens", usage.get("output_tokens"))
    return (
        int(inbound) if inbound is not None else None,
        int(outbound) if outbound is not None else None,
    )


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(item.get("text") or "")
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content or "")


class OpenAICompatibleProvider:
    def __init__(self, model: ModelConfig):
        self.model = model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.model.api_key:
            headers["Authorization"] = f"Bearer {self.model.api_key}"
        headers.update(self.model.extra_headers)
        return headers

    async def chat(self, request: ChatRequest) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.extra_body:
            payload.update(request.extra_body)

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=request.timeout_s) as client:
                response = await client.post(
                    f"{self.model.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
            latency_ms = (time.perf_counter() - started) * 1000
            body: Any
            try:
                body = response.json()
            except Exception:
                body = {"raw_text": response.text}

            if response.status_code >= 400:
                error = f"HTTP {response.status_code}: {_short_error(body, response.text)}"
                return ChatResult(
                    provider=self.model.provider,
                    model=self.model.model,
                    raw_response=body,
                    latency_ms=latency_ms,
                    error=error,
                )

            if not isinstance(body, dict):
                return ChatResult(
                    provider=self.model.provider,
                    model=self.model.model,
                    raw_response=body,
                    latency_ms=latency_ms,
                    error="Provider returned a non-object JSON body",
                )

            input_tokens, output_tokens = _usage_tokens(body.get("usage"))
            return ChatResult(
                provider=self.model.provider,
                model=self.model.model,
                text=_extract_text(body),
                raw_response=body,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                tool_calls=_extract_tool_calls(body),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            return ChatResult(
                provider=self.model.provider,
                model=self.model.model,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def health(self, timeout_s: float) -> tuple[bool, str]:
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.get(
                    f"{self.model.base_url}/models",
                    headers=self._headers(),
                )
                if response.status_code >= 400:
                    # Non-chat services (e.g. the BGE-M3 embedder) have no /v1/models
                    # but expose /healthz at the server root.
                    root = self.model.base_url.rsplit("/v1", 1)[0]
                    probe = await client.get(f"{root}/healthz", headers=self._headers())
                    if probe.status_code < 400:
                        return True, "healthz ok"
                    return False, f"HTTP {response.status_code}"
            try:
                body = response.json()
            except Exception:
                return True, "reachable (non-JSON /models)"
            names = _model_names(body)
            if names and self.model.model not in names:
                return True, f"reachable ({len(names)} models listed)"
            if self.model.model in names:
                return True, "model listed"
            return True, "reachable"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"


def _extract_tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    choices = payload.get("choices") or []
    if not choices:
        return []
    message = (choices[0] or {}).get("message") or {}
    calls = message.get("tool_calls") or []
    return [c for c in calls if isinstance(c, dict)]


def _model_names(body: Any) -> list[str]:
    if isinstance(body, dict):
        data = body.get("data") or body.get("models") or []
        names = []
        for item in data:
            if isinstance(item, dict):
                name = item.get("id") or item.get("name")
                if name:
                    names.append(str(name))
            elif isinstance(item, str):
                names.append(item)
        return names
    return []


def _short_error(body: Any, fallback: str) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
        if body.get("message"):
            return str(body["message"])
    text = fallback.strip().replace("\n", " ")
    return text[:240] if text else "unknown error"
