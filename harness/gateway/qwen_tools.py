"""Turn Qwen XML tool text into OpenAI tool_calls Cline will execute."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

FUNCTION_RE = re.compile(
    r"<function=(?P<name>[^>\s]+)>(?P<body>.*?)</function>",
    re.DOTALL,
)
PARAM_RE = re.compile(
    r"<parameter=(?P<name>[^>\s]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def looks_like_qwen_tools(text: str) -> bool:
    if not text:
        return False
    return "<tool_call>" in text or "<function=" in text


def parse_qwen_tool_text(text: str) -> list[dict[str, Any]]:
    if not looks_like_qwen_tools(text):
        return []
    calls: list[dict[str, Any]] = []
    blocks = TOOL_CALL_RE.findall(text) or [text]
    for block in blocks:
        stripped = block.strip()
        if stripped.startswith("{"):
            parsed = _from_json_block(stripped)
            if parsed:
                calls.append(parsed)
                continue
        for match in FUNCTION_RE.finditer(block):
            params = {
                item.group("name"): item.group("value") for item in PARAM_RE.finditer(match.group("body"))
            }
            calls.append(_call(match.group("name").strip(), params))
    return calls


def rewrite_openai_completion(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices") or []
    if not choices:
        return data
    message = (choices[0] or {}).get("message") or {}
    if message.get("tool_calls"):
        return data
    calls = parse_qwen_tool_text(message.get("content") or "")
    if not calls:
        return data
    message["content"] = ""
    message["tool_calls"] = calls
    choices[0]["message"] = message
    choices[0]["finish_reason"] = "tool_calls"
    return data


def collect_stream_text(chunks: list[bytes]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        for line in chunk.split(b"\n"):
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if not raw or raw == b"[DONE]":
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            delta = ((obj.get("choices") or [{}])[0].get("delta") or {})
            content = delta.get("content")
            if content:
                parts.append(content)
    return "".join(parts)


def stream_already_has_tool_calls(chunks: list[bytes]) -> bool:
    for chunk in chunks:
        if b'"tool_calls"' in chunk:
            return True
    return False


def synthesize_tool_call_sse(model: str, calls: list[dict[str, Any]]) -> list[bytes]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    out = [_sse(chunk_id, model, {"role": "assistant", "content": ""})]
    for index, call in enumerate(calls):
        out.append(
            _sse(
                chunk_id,
                model,
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": call["function"]["arguments"],
                            },
                        }
                    ]
                },
            )
        )
    out.append(_sse(chunk_id, model, {}, finish="tool_calls"))
    out.append(b"data: [DONE]\n\n")
    return out


def _from_json_block(raw: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    name = obj.get("name") or obj.get("function")
    if not name:
        return None
    args = obj.get("arguments") or obj.get("parameters") or {}
    return _call(str(name), args)


def _call(name: str, args: Any) -> dict[str, Any]:
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _sse(chunk_id: str, model: str, delta: dict[str, Any], finish: str | None = None) -> bytes:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()
