#!/usr/bin/env python3
"""Measure raw decode speed for DeepSeek TP2 and equal hardware Qwen."""

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


OUTPUT = Path("results/raw_speed_bakeoff_20260828.json")
DEEPSEEK = ("http://100.68.133.1:9000/v1", "deepseek-v4-flash-0731")
QWEN_A = ("http://100.116.221.82:8900/v1", "qwen3-coder-next")
QWEN_B = ("http://100.124.181.13:8900/v1", "qwen3-coder-next")


async def _stream(
    endpoint: tuple[str, str],
    request_id: str,
    *,
    deepseek: bool,
) -> dict[str, Any]:
    base_url, model = endpoint
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Generate a long deterministic technical glossary. "
                    "Use short entries and continue until the output limit. "
                    f"Benchmark nonce: {request_id}"
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": 512,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if deepseek:
        payload["chat_template_kwargs"] = {"thinking": False}
    started = time.perf_counter()
    first_token_at: float | None = None
    completion_tokens = 0
    chunks = 0
    async with httpx.AsyncClient(timeout=180) as client:
        async with client.stream(
            "POST",
            f"{base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    continue
                body = json.loads(raw)
                usage = body.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
                choices = body.get("choices") or []
                if choices:
                    delta = (choices[0] or {}).get("delta") or {}
                    if delta.get("content") or delta.get("reasoning"):
                        chunks += 1
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
    finished = time.perf_counter()
    return {
        "request_id": request_id,
        "completion_tokens": completion_tokens,
        "chunks": chunks,
        "ttft_ms": (
            round((first_token_at - started) * 1000, 2)
            if first_token_at is not None
            else None
        ),
        "latency_ms": round((finished - started) * 1000, 2),
    }


async def _lane(
    name: str,
    endpoints: list[tuple[str, str]],
    concurrency: int,
    *,
    deepseek: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = await asyncio.gather(
        *(
            _stream(
                endpoints[index % len(endpoints)],
                f"{name}-{index}",
                deepseek=deepseek,
            )
            for index in range(concurrency)
        )
    )
    wall = time.perf_counter() - started
    tokens = sum(row["completion_tokens"] for row in rows)
    ttfts = [row["ttft_ms"] for row in rows if row["ttft_ms"] is not None]
    return {
        "name": name,
        "physical_nodes": 2 if deepseek or len(endpoints) == 2 else 1,
        "concurrency": concurrency,
        "completion_tokens": tokens,
        "wall_ms": round(wall * 1000, 2),
        "aggregate_tokens_per_second": round(tokens / wall, 3),
        "median_ttft_ms": round(statistics.median(ttfts), 2) if ttfts else None,
        "requests": rows,
    }


async def main() -> None:
    single_deepseek, single_qwen = await asyncio.gather(
        _lane("deepseek_tp2_c1", [DEEPSEEK], 1, deepseek=True),
        _lane("qwen_single_c1", [QWEN_A], 1, deepseek=False),
    )
    pair_deepseek, pair_qwen = await asyncio.gather(
        _lane("deepseek_tp2_c4", [DEEPSEEK], 4, deepseek=True),
        _lane("qwen_two_box_c4", [QWEN_A, QWEN_B], 4, deepseek=False),
    )
    rows = [single_deepseek, single_qwen, pair_deepseek, pair_qwen]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": os.environ.get(
            "BENCH_NOTE",
            "Measured while the Qwen runtime/model transfer was active on ASUS2.",
        ),
        "lanes": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    key: row[key]
                    for key in (
                        "name",
                        "physical_nodes",
                        "concurrency",
                        "completion_tokens",
                        "wall_ms",
                        "aggregate_tokens_per_second",
                        "median_ttft_ms",
                    )
                }
                for row in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
