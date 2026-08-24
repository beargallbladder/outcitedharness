from __future__ import annotations

from harness.config import Pricing


def estimate_cost(
    pricing: Pricing,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if pricing.input_per_million is None or pricing.output_per_million is None:
        return None
    inbound = input_tokens or 0
    outbound = output_tokens or 0
    return (inbound / 1_000_000) * pricing.input_per_million + (
        outbound / 1_000_000
    ) * pricing.output_per_million
