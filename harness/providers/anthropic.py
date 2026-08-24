from __future__ import annotations

import time
from typing import Any

import httpx

from harness.config import ModelConfig
from harness.providers.base import ChatRequest, ChatResult


class AnthropicProvider:
    def __init__(self, model: ModelConfig):
        self.model = model

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.model.api_key:
            headers["x-api-key"] = self.model.api_key
        return headers

    def _messages_url(self) -> str:
        base = self.model.base_url
        if base.endswith("/v1"):
            return f"{base}/messages"
        return f"{base}/v1/messages"

    async def chat(self, request: ChatRequest) -> ChatResult:
        system_parts = [m.content for m in request.messages if m.role == "system"]
        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
        payload: dict[str, Any] = {
            "model": self.model.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or 4096,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.extra_body:
            payload.update(request.extra_body)

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=request.timeout_s) as client:
                response = await client.post(
                    self._messages_url(),
                    headers=self._headers(),
                    json=payload,
                )
            latency_ms = (time.perf_counter() - started) * 1000
            try:
                body = response.json()
            except Exception:
                body = {"raw_text": response.text}

            if response.status_code >= 400:
                return ChatResult(
                    provider=self.model.provider,
                    model=self.model.model,
                    raw_response=body,
                    latency_ms=latency_ms,
                    error=f"HTTP {response.status_code}: {_short_error(body, response.text)}",
                )

            text = _extract_text(body if isinstance(body, dict) else {})
            usage = body.get("usage") if isinstance(body, dict) else {}
            inbound = (usage or {}).get("input_tokens")
            outbound = (usage or {}).get("output_tokens")
            return ChatResult(
                provider=self.model.provider,
                model=self.model.model,
                text=text,
                raw_response=body,
                input_tokens=int(inbound) if inbound is not None else None,
                output_tokens=int(outbound) if outbound is not None else None,
                latency_ms=latency_ms,
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
        if self.model.missing_key:
            return False, f"missing ${self.model.api_key_env}"
        if self.model.placeholder_url:
            return False, "placeholder base_url/model"
        url = (
            f"{self.model.base_url}/models"
            if self.model.base_url.endswith("/v1")
            else f"{self.model.base_url}/v1/models"
        )
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.get(url, headers=self._headers())
            if response.status_code >= 400:
                return False, f"HTTP {response.status_code}"
            body = response.json()
            names = [
                item.get("id")
                for item in (body.get("data") or [])
                if isinstance(item, dict) and item.get("id")
            ]
            if self.model.model in names:
                return True, "model listed"
            if names:
                return True, f"reachable ({len(names)} models listed)"
            return True, "reachable"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"


def _extract_text(body: dict[str, Any]) -> str:
    blocks = body.get("content") or []
    parts = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


def _short_error(body: Any, fallback: str) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
    text = (fallback or "").strip().replace("\n", " ")
    return text[:240] if text else "unknown error"
