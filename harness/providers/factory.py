from __future__ import annotations

from harness.config import ModelConfig
from harness.providers.anthropic import AnthropicProvider
from harness.providers.base import ModelProvider
from harness.providers.openai_compatible import OpenAICompatibleProvider


PROVIDERS = {
    "openai_compatible": OpenAICompatibleProvider,
    "openai": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
}


def build_provider(model: ModelConfig) -> ModelProvider:
    try:
        cls = PROVIDERS[model.provider]
    except KeyError as exc:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unknown provider '{model.provider}'. Known: {known}") from exc
    return cls(model)
