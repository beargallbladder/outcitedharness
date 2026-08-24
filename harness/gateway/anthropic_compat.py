"""Translate Cline's OpenAI chat+tools payload to Anthropic Messages."""

from __future__ import annotations

import json
from typing import Any


def openai_to_anthropic_payload(body: dict[str, Any], model_id: str, max_tokens: int) -> dict[str, Any]:
    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for raw in body.get("messages") or []:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        if role == "system":
            text = _content_text(raw.get("content"))
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": raw.get("tool_call_id") or raw.get("id") or "",
                            "content": _content_text(raw.get("content")),
                        }
                    ],
                }
            )
            continue
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = _content_text(raw.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                args = fn.get("arguments") or "{}"
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except json.JSONDecodeError:
                        parsed = {"raw": args}
                else:
                    parsed = args
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id") or "",
                        "name": fn.get("name") or "",
                        "input": parsed,
                    }
                )
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        messages.append({"role": "user", "content": _user_content(raw.get("content"))})

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": _merge_adjacent(messages),
        "max_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or max_tokens),
        "stream": bool(body.get("stream")),
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if body.get("temperature") is not None:
        payload["temperature"] = body["temperature"]
    tools = _anthropic_tools(body.get("tools") or [])
    if tools:
        payload["tools"] = tools
    return payload


def anthropic_json_to_openai(body: dict[str, Any], requested_model: str) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in body.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(block.get("text") or "")
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = body.get("usage") or {}
    finish = "tool_calls" if tool_calls else "stop"
    return {
        "id": body.get("id") or "harness-anthropic",
        "object": "chat.completion",
        "model": requested_model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens") or 0,
            "completion_tokens": usage.get("output_tokens") or 0,
            "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
        },
    }


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
        return "".join(parts)
    return str(content)


def _user_content(content: Any) -> Any:
    if isinstance(content, str) or content is None:
        return content or ""
    if isinstance(content, list):
        blocks = []
        for item in content:
            if isinstance(item, str):
                blocks.append({"type": "text", "text": item})
            elif isinstance(item, dict) and item.get("type") == "image_url":
                url = ((item.get("image_url") or {}) if isinstance(item.get("image_url"), dict) else {})
                href = url.get("url") or ""
                if href.startswith("data:"):
                    header, _, b64 = href.partition(",")
                    media = header.split(";")[0].removeprefix("data:") or "image/png"
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media, "data": b64},
                        }
                    )
                else:
                    blocks.append({"type": "text", "text": href})
            elif isinstance(item, dict):
                blocks.append({"type": "text", "text": item.get("text") or ""})
        return blocks
    return str(content)


def _anthropic_tools(tools: list[Any]) -> list[dict[str, Any]]:
    out = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or tool
        name = fn.get("name")
        if not name:
            continue
        out.append(
            {
                "name": name,
                "description": fn.get("description") or "",
                "input_schema": fn.get("parameters") or fn.get("input_schema") or {"type": "object"},
            }
        )
    return out


def _merge_adjacent(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]["content"]
            cur = msg["content"]
            if isinstance(prev, str) and isinstance(cur, str):
                merged[-1]["content"] = prev + "\n" + cur
            else:
                merged[-1]["content"] = _as_blocks(prev) + _as_blocks(cur)
        else:
            merged.append(msg)
    return merged


def _as_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return list(content)
    return [{"type": "text", "text": str(content or "")}]
