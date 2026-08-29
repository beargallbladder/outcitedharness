#!/usr/bin/env python3
"""Qualify loopback LiteLLM without invoking paid routes."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:7410/v1").rstrip(
    "/"
)
KEY = os.environ.get("LITELLM_MASTER_KEY", "")
OUTPUT = Path(
    os.environ.get(
        "LITELLM_QUAL_OUTPUT", "results/litellm_qualification_20260829.json"
    )
)
TIMEOUT = float(os.environ.get("LITELLM_TIMEOUT", "1800"))


def _headers(key: str | None = None) -> dict[str, str]:
    token = KEY if key is None else key
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "lookup_ticket",
            "description": "Look up an incident ticket.",
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
                "additionalProperties": False,
            },
        },
    }


def _post(
    client: httpx.Client, model: str, messages: list[dict[str, Any]], **extra: Any
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 64,
    }
    payload.update(extra)
    response = client.post(
        f"{BASE_URL}/chat/completions", headers=_headers(), json=payload
    )
    if response.status_code >= 400:
        raise RuntimeError(f"{model}: HTTP {response.status_code}: {response.text[:800]}")
    return response.json()


def _sync_checks() -> dict[str, Any]:
    expected_models = {
        "local-coder",
        "local-qwen38",
        "local-critic",
        "harness-orch",
        "frontier-claude",
    }
    with httpx.Client(timeout=TIMEOUT) as client:
        unauthorized = client.get(
            f"{BASE_URL}/models", headers=_headers("definitely-invalid")
        )
        models_response = client.get(f"{BASE_URL}/models", headers=_headers())
        models_response.raise_for_status()
        listed = {
            str(item.get("id"))
            for item in models_response.json().get("data", [])
            if isinstance(item, dict)
        }

        basic: dict[str, str] = {}
        usage_ok = True
        for model in ("local-coder", "local-qwen38", "local-critic"):
            body = _post(
                client,
                model,
                [{"role": "user", "content": "Reply with exactly: LOCAL_READY"}],
                max_tokens=32,
            )
            basic[model] = str(
                body["choices"][0]["message"].get("content") or ""
            )
            usage = body.get("usage") or {}
            usage_ok = usage_ok and all(
                isinstance(usage.get(name), int)
                for name in ("prompt_tokens", "completion_tokens", "total_tokens")
            )

        json_body = _post(
            client,
            "local-critic",
            [{"role": "user", "content": 'Return JSON with status set to "ready".'}],
            response_format={"type": "json_object"},
            max_tokens=32,
        )
        json_text = str(
            json_body["choices"][0]["message"].get("content") or ""
        )
        try:
            json_ok = json.loads(json_text).get("status") == "ready"
        except (json.JSONDecodeError, AttributeError):
            json_ok = False

        tool_body = _post(
            client,
            "local-coder",
            [
                {
                    "role": "user",
                    "content": "Call lookup_ticket for INC-731. Do not answer directly.",
                }
            ],
            tools=[_tool()],
            tool_choice="auto",
            max_tokens=128,
        )
        calls = tool_body["choices"][0]["message"].get("tool_calls") or []
        tool_ok = False
        followup_ok = False
        if calls:
            function = calls[0].get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_ok = (
                function.get("name") == "lookup_ticket"
                and arguments.get("ticket_id") == "INC-731"
            )
            followup = _post(
                client,
                "local-coder",
                [
                    {
                        "role": "user",
                        "content": "Call lookup_ticket for INC-731. Do not answer directly.",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": calls,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": calls[0].get("id") or "call_0",
                        "content": '{"status":"resolved"}',
                    },
                ],
                tools=[_tool()],
            )
            followup_text = str(
                followup["choices"][0]["message"].get("content") or ""
            )
            followup_ok = "resolved" in followup_text.lower()

        stream_text = ""
        stream_finish = ""
        saw_done = False
        with client.stream(
            "POST",
            f"{BASE_URL}/chat/completions",
            headers=_headers(),
            json={
                "model": "local-coder",
                "messages": [{"role": "user", "content": "Reply exactly STREAM_OK"}],
                "temperature": 0,
                "max_tokens": 32,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    saw_done = True
                    continue
                event = json.loads(raw)
                choices = event.get("choices") or []
                if choices:
                    choice = choices[0]
                    stream_text += str((choice.get("delta") or {}).get("content") or "")
                    stream_finish = str(choice.get("finish_reason") or stream_finish)

        orch = _post(
            client,
            "harness-orch",
            [{"role": "user", "content": "Reply briefly: orchestration route ready."}],
            max_tokens=128,
        )
        orch_text = str(orch["choices"][0]["message"].get("content") or "")

        latency: dict[str, dict[str, float | bool]] = {}
        direct_routes = {
            "local-coder": (
                "http://100.73.119.63:8900/v1/chat/completions",
                "qwen3-coder-next",
            ),
            "local-qwen38": (
                "http://100.68.133.1:8888/v1/chat/completions",
                "qwen38-flash-next-nvfp4-sglang",
            ),
            "local-critic": (
                "http://100.89.118.36:8900/v1/chat/completions",
                "nemotron-lightning",
            ),
        }
        for route, (direct_url, served_model) in direct_routes.items():
            direct_payload = {
                "model": served_model,
                "messages": [{"role": "user", "content": "Reply exactly LATENCY_OK"}],
                "temperature": 0,
                "max_tokens": 32,
            }
            started = time.perf_counter()
            direct_response = client.post(direct_url, json=direct_payload)
            direct_response.raise_for_status()
            direct_seconds = time.perf_counter() - started
            started = time.perf_counter()
            _post(
                client,
                route,
                [{"role": "user", "content": "Reply exactly LATENCY_OK"}],
                max_tokens=32,
            )
            proxy_seconds = time.perf_counter() - started
            overhead = proxy_seconds - direct_seconds
            latency[route] = {
                "direct_seconds": round(direct_seconds, 4),
                "proxy_seconds": round(proxy_seconds, 4),
                "overhead_seconds": round(overhead, 4),
                "within_limit": overhead <= max(1.5, direct_seconds * 0.5),
            }

    config = yaml.safe_load((ROOT / "config" / "litellm.yaml").read_text()) or {}
    settings = config.get("litellm_settings") or {}
    no_fallbacks = not any(
        name in settings
        for name in ("fallbacks", "default_fallbacks", "context_window_fallbacks")
    )
    return {
        "unauthorized_status": unauthorized.status_code,
        "auth_ok": unauthorized.status_code in {400, 401, 403},
        "models": sorted(listed),
        "models_ok": expected_models.issubset(listed),
        "basic": basic,
        "basic_ok": all("LOCAL_READY" in text for text in basic.values()),
        "usage_ok": usage_ok,
        "json_text": json_text,
        "json_ok": json_ok,
        "tool_ok": tool_ok,
        "followup_ok": followup_ok,
        "stream_text": stream_text,
        "stream_finish": stream_finish,
        "stream_ok": "STREAM_OK" in stream_text and saw_done and bool(stream_finish),
        "orch_text": orch_text,
        "orch_ok": bool(orch_text.strip()),
        "latency": latency,
        "latency_ok": all(bool(row["within_limit"]) for row in latency.values()),
        "no_automatic_fallbacks": no_fallbacks,
    }


async def _one(client: httpx.AsyncClient, index: int) -> tuple[float, int]:
    started = time.perf_counter()
    response = await client.post(
        f"{BASE_URL}/chat/completions",
        headers=_headers(),
        json={
            "model": "local-coder",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write numbered deterministic Python glossary entries "
                        f"until the token limit. Nonce {index}."
                    ),
                }
            ],
            "temperature": 0,
            "max_tokens": 256,
            "ignore_eos": True,
        },
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    tokens = int((response.json().get("usage") or {}).get("completion_tokens") or 0)
    return elapsed, tokens


async def _concurrency() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        started = time.perf_counter()
        rows = await asyncio.gather(*(_one(client, index) for index in range(3)))
    wall = time.perf_counter() - started
    tokens = sum(row[1] for row in rows)
    return {
        "requests": 3,
        "wall_seconds": round(wall, 3),
        "completion_tokens": tokens,
        "aggregate_tokens_per_second": round(tokens / wall, 3),
        "all_nonzero": all(row[1] > 0 for row in rows),
    }


def main() -> None:
    if not KEY:
        raise SystemExit("LITELLM_MASTER_KEY is required")
    checks = _sync_checks()
    concurrency = asyncio.run(_concurrency())
    passed = all(
        bool(checks[name])
        for name in (
            "auth_ok",
            "models_ok",
            "basic_ok",
            "usage_ok",
            "json_ok",
            "tool_ok",
            "followup_ok",
            "stream_ok",
            "orch_ok",
            "latency_ok",
            "no_automatic_fallbacks",
        )
    ) and bool(concurrency["all_nonzero"])
    result = {"passed": passed, "checks": checks, "concurrency": concurrency}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
