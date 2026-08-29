#!/usr/bin/env python3
"""Test DeepSeek's likely specialties: long context and grounded review."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from critic_bakeoff import fixtures

from harness.config import load_config
from harness.dispatch import _run_critic
from harness.workers.registry import Worker


OUTPUT = Path("results/deepseek_specialty_bakeoff_20260828.json")


def _inventory(records: int, nonce: str) -> str:
    rows = [
        (
            f"module_{index:06d} owner=team_{index % 97:02d} "
            f"port={7000 + index % 2000} dependency=service_{index % 313:03d} "
            f"revision={index * 17 + 3}"
        )
        for index in range(records)
    ]
    rows[records // 2] = f"TARGET_{nonce}=verified_value_{records}"
    return (
        "Read this synthetic software inventory and return only the exact value "
        f"of TARGET_{nonce}.\n\n" + "\n".join(rows)
    )


async def _prefill(
    name: str,
    base_url: str,
    model: str,
    records: int,
    *,
    deepseek: bool = False,
) -> dict[str, Any]:
    nonce = f"{name}_{records}_{time.time_ns()}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": _inventory(records, nonce)}],
        "temperature": 0,
        "max_tokens": 32,
    }
    if deepseek:
        payload["chat_template_kwargs"] = {"thinking": False}
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
    elapsed = time.perf_counter() - started
    if response.status_code >= 400:
        return {
            "name": name,
            "records": records,
            "prompt_tokens": None,
            "latency_ms": round(elapsed * 1000, 2),
            "effective_prompt_tokens_per_second": None,
            "correct": False,
            "error": f"HTTP {response.status_code}: {response.text[:300]}",
        }
    body = response.json()
    usage = body.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    text = str(body["choices"][0]["message"].get("content") or "")
    expected = f"verified_value_{records}"
    return {
        "name": name,
        "records": records,
        "prompt_tokens": prompt_tokens,
        "latency_ms": round(elapsed * 1000, 2),
        "effective_prompt_tokens_per_second": (
            round(prompt_tokens / elapsed, 3) if prompt_tokens else None
        ),
        "correct": expected in text,
        "error": None,
        "response": text,
    }


async def _review(name: str, model_key: str) -> dict[str, Any]:
    cfg = load_config()
    model = cfg.models[model_key]
    worker = Worker(
        id=f"review-{name}",
        enabled=True,
        model_key=model_key,
        endpoint=model.base_url,
        capabilities=("review",),
        failover_order=None,
        role="critic",
    )
    shots, expected = fixtures()
    started = time.perf_counter()
    verdict, raw, by_id = await _run_critic(
        worker,
        model,
        "Grounded review speed test. Grade only supplied evidence.",
        shots,
    )
    elapsed = time.perf_counter() - started
    checks = {
        case_id: by_id.get(case_id, (False, "missing"))[0] == wanted
        for case_id, wanted in expected.items()
    }
    return {
        "name": name,
        "score": sum(checks.values()) if by_id else None,
        "total": len(checks),
        "available": bool(by_id),
        "latency_ms": round(elapsed * 1000, 2),
        "verdict": verdict,
        "checks": checks if by_id else {},
        "raw_head": raw[:800],
    }


async def main() -> None:
    cfg = load_config()
    deepseek = cfg.models["deepseek_flash_tp2_shadow"]
    qwen = cfg.models["dgx2_qwen"]
    m5 = cfg.models["m5_qwen"]
    include_m5 = os.environ.get("BENCH_INCLUDE_M5", "1") == "1"
    prefill_rows: list[dict[str, Any]] = []
    for records in (500, 2000):
        lanes = [
            _prefill(
                "deepseek_tp2",
                deepseek.base_url,
                deepseek.model,
                records,
                deepseek=True,
            ),
            _prefill("qwen_single", qwen.base_url, qwen.model, records),
        ]
        if include_m5:
            lanes.append(_prefill("m5", m5.base_url, m5.model, records))
        prefill_rows.extend(await asyncio.gather(*lanes))
    review_rows = await asyncio.gather(
        _review("deepseek_high", "deepseek_flash_tp2_shadow"),
        _review("nemotron", "asus3_nemotron"),
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": os.environ.get(
            "BENCH_NOTE",
            "Measured while the Qwen runtime/model transfer was active on ASUS2.",
        ),
        "prefill": prefill_rows,
        "grounded_review": review_rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
