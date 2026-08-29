#!/usr/bin/env python3
"""Qualify an SGLang endpoint for Harness chat and tool traffic."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://100.68.133.1:8888/v1").rstrip(
    "/"
)
MODEL = os.environ.get("SGLANG_MODEL", "qwen38-flash-next-nvfp4-sglang")
OUTPUT = Path(
    os.environ.get("SGLANG_QUAL_OUTPUT", "results/sglang_qualification.json")
)
TIMEOUT = float(os.environ.get("SGLANG_TIMEOUT", "900"))


def _payload(messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    payload.update(extra)
    return payload


def _post(client: httpx.Client, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(f"{BASE_URL}/chat/completions", json=payload)
    if response.status_code >= 400:
        raise RuntimeError(
            f"HTTP {response.status_code}: {response.text[:1000]}"
        )
    return response.json()


def _tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "lookup_ticket",
            "description": "Look up an incident ticket by identifier.",
            "parameters": {
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
                "additionalProperties": False,
            },
        },
    }


def _sync_checks() -> dict[str, Any]:
    with httpx.Client(timeout=TIMEOUT) as client:
        models_response = client.get(f"{BASE_URL}/models")
        models_response.raise_for_status()
        listed = [
            str(item.get("id"))
            for item in (models_response.json().get("data") or [])
            if isinstance(item, dict)
        ]

        chat = _post(
            client,
            _payload(
                [{"role": "user", "content": "Reply with exactly: SGLANG_READY"}],
                max_tokens=32,
            ),
        )
        chat_text = str(chat["choices"][0]["message"].get("content") or "")

        json_response = _post(
            client,
            _payload(
                [
                    {
                        "role": "user",
                        "content": 'Return a JSON object with status set to "ready".',
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=32,
            ),
        )
        json_text = str(
            json_response["choices"][0]["message"].get("content") or ""
        )
        try:
            json_object = json.loads(json_text)
        except json.JSONDecodeError:
            json_object = {}

        tool_response = _post(
            client,
            _payload(
                [
                    {
                        "role": "user",
                        "content": "Use lookup_ticket for ticket INC-731. Do not answer directly.",
                    }
                ],
                tools=[_tool()],
                tool_choice="auto",
            ),
        )
        tool_calls = tool_response["choices"][0]["message"].get("tool_calls") or []
        tool_ok = False
        tool_call: dict[str, Any] = {}
        if tool_calls:
            tool_call = tool_calls[0]
            function = tool_call.get("function") or {}
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
            _payload(
                [
                    {
                        "role": "user",
                        "content": "Use lookup_ticket for ticket INC-731. Do not answer directly.",
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id") or "call_0",
                        "content": '{"status":"resolved","owner":"ops"}',
                    },
                ],
                tools=[_tool()],
                max_tokens=64,
            ),
        )
        followup_text = str(
            followup["choices"][0]["message"].get("content") or ""
        )

        deterministic_answers = [
            str(
                _post(
                    client,
                    _payload(
                        [
                            {
                                "role": "user",
                                "content": "Return only the integer 731 multiplied by 17.",
                            }
                        ],
                        seed=42,
                        max_tokens=16,
                    ),
                )["choices"][0]["message"].get("content")
                or ""
            )
            for _ in range(3)
        ]

        stream_text = ""
        saw_done = False
        with client.stream(
            "POST",
            f"{BASE_URL}/chat/completions",
            json=_payload(
                [{"role": "user", "content": "Reply with exactly: STREAM_OK"}],
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=32,
            ),
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
                    stream_text += str(
                        ((choices[0] or {}).get("delta") or {}).get("content") or ""
                    )

    return {
        "models": listed,
        "model_listed": MODEL in listed,
        "chat_text": chat_text,
        "chat_ok": "SGLANG_READY" in chat_text,
        "json_text": json_text,
        "json_object_ok": json_object.get("status") == "ready",
        "tool_call": tool_call,
        "tool_ok": tool_ok,
        "followup_text": followup_text,
        "followup_ok": "resolved" in followup_text.lower(),
        "deterministic_answers": deterministic_answers,
        "deterministic": len(set(deterministic_answers)) == 1,
        "stream_text": stream_text,
        "stream_ok": "STREAM_OK" in stream_text and saw_done,
    }


async def _one_speed(client: httpx.AsyncClient, index: int) -> dict[str, Any]:
    started = time.perf_counter()
    response = await client.post(
        f"{BASE_URL}/chat/completions",
        json=_payload(
            [
                {
                    "role": "user",
                    "content": (
                        "Write a deterministic numbered glossary of Python terms. "
                        f"Continue until the output limit. Nonce {index}."
                    ),
                }
            ],
            max_tokens=384,
            ignore_eos=True,
        ),
    )
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    body = response.json()
    usage = body.get("usage") or {}
    return {
        "seconds": elapsed,
        "tokens": int(usage.get("completion_tokens") or 0),
    }


async def _speed() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for concurrency in (1, 2, 4):
            started = time.perf_counter()
            requests = await asyncio.gather(
                *(_one_speed(client, index) for index in range(concurrency))
            )
            wall = time.perf_counter() - started
            tokens = sum(row["tokens"] for row in requests)
            rows.append(
                {
                    "concurrency": concurrency,
                    "wall_seconds": round(wall, 3),
                    "completion_tokens": tokens,
                    "aggregate_tokens_per_second": round(tokens / wall, 3),
                    "median_request_seconds": round(
                        statistics.median(row["seconds"] for row in requests), 3
                    ),
                }
            )
    return rows


def main() -> None:
    checks = _sync_checks()
    speed = asyncio.run(_speed())
    required = (
        "model_listed",
        "chat_ok",
        "json_object_ok",
        "tool_ok",
        "followup_ok",
        "deterministic",
        "stream_ok",
    )
    passed = all(bool(checks[name]) for name in required)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": BASE_URL,
        "model": MODEL,
        "passed": passed,
        "checks": checks,
        "speed": speed,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
