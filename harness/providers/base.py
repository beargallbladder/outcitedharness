from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ImageAttachment:
    mime_type: str
    data_b64: str


@dataclass
class ChatMessage:
    role: str
    content: str
    images: list[ImageAttachment] = field(default_factory=list)


@dataclass
class ChatRequest:
    messages: list[ChatMessage]
    temperature: float = 0
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 180
    seed: int | None = None


@dataclass
class ChatResult:
    provider: str
    model: str
    text: str = ""
    raw_response: Any = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0
    error: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class ModelProvider(Protocol):
    async def chat(self, request: ChatRequest) -> ChatResult: ...

    async def health(self, timeout_s: float) -> tuple[bool, str]: ...
