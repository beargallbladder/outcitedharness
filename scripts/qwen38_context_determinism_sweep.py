#!/usr/bin/env python3
"""Locate the context-length threshold for Qwen TP=2 prefill instability."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


BASE_URL = os.environ.get("QWEN38_BASE_URL", "http://100.68.133.1:8888/v1")
URL = f"{BASE_URL.rstrip('/')}/chat/completions"
MODEL = os.environ.get("QWEN38_MODEL", "qwen38-flash-next-nvfp4")
API_KEY = os.environ.get("QWEN38_API_KEY", "")
OUTPUT = Path(
    os.environ.get(
        "QWEN38_SWEEP_OUTPUT",
        "results/qwen38_context_determinism_sweep_20260828.json",
    )
)
RECORD_COUNTS = (100, 300, 600, 1000)
RUNS = 5


def _prompt(record_count: int) -> str:
    target = record_count - 7
    records = " ".join(
        f"record_{index:05d}=value_{(index * 7919) % 104729:06d}"
        for index in range(record_count)
    )
    return (
        f"Read these records. Reply only with the value of record_{target:05d}.\n"
        + records
    )


def _request(client: httpx.Client, prompt: str) -> dict[str, Any]:
    response = client.post(
        URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 20,
            "chat_template_kwargs": {
                "enable_thinking": False,
                "preserve_thinking": False,
            },
        },
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    if not API_KEY:
        raise RuntimeError("QWEN38_API_KEY is required")
    rows: list[dict[str, Any]] = []
    with httpx.Client(
        timeout=300,
        headers={"Authorization": f"Bearer {API_KEY}"},
    ) as client:
        for record_count in RECORD_COUNTS:
            prompt = _prompt(record_count)
            started = time.perf_counter()
            responses = [_request(client, prompt) for _ in range(RUNS)]
            vectors = [
                response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                for response in responses
            ]
            fingerprints = [
                hashlib.sha256(
                    json.dumps(
                        vector, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest()
                for vector in vectors
            ]
            token_orders = [
                tuple(str(candidate["token"]) for candidate in vector)
                for vector in vectors
            ]
            first_tokens = [
                str(response["choices"][0]["logprobs"]["content"][0]["token"])
                for response in responses
            ]
            rows.append(
                {
                    "records": record_count,
                    "prompt_tokens": responses[0]["usage"]["prompt_tokens"],
                    "runs": RUNS,
                    "bit_exact_unique": len(set(fingerprints)),
                    "top20_order_unique": len(set(token_orders)),
                    "first_token_unique": len(set(first_tokens)),
                    "stable": len(set(fingerprints)) == 1,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
    payload = {
        "model": MODEL,
        "configuration": "TP2, exact QSA, MTP2, prefix cache off",
        "results": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
