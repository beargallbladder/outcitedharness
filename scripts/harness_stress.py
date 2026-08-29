#!/usr/bin/env python3
"""Sustained mixed-lane load test for the local harness gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


GATEWAY = "http://127.0.0.1:8787"
DIRECT_MODELS = (
    "harness-local",
    "harness-dgx2",
    "harness-dgx3",
    "harness-asus",
    "harness-m5",
    "harness-auto",
)


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil((percentile_value / 100) * len(ordered)) - 1)
    return ordered[index]


async def completion(
    client: httpx.AsyncClient,
    model: str,
    expected: str,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{GATEWAY}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Reply with exactly this token and nothing else: {expected}",
                    }
                ],
                "temperature": 0,
                "max_tokens": 32,
            },
            timeout=timeout_s,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        data = response.json()
        message = ((data.get("choices") or [{}])[0].get("message") or {})
        text = str(message.get("content") or "").strip()
        exact = text == expected if model != "harness-orch" else expected in text
        usage = data.get("usage") or {}
        return {
            "model": model,
            "status": response.status_code,
            "ok": response.status_code == 200 and bool(text),
            "exact": exact,
            "latency_ms": round(latency_ms, 2),
            "served_model": data.get("model"),
            "output_tokens": usage.get("completion_tokens"),
            "error": None,
            "answer_head": text[:160],
        }
    except Exception as exc:
        return {
            "model": model,
            "status": None,
            "ok": False,
            "exact": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "served_model": None,
            "output_tokens": None,
            "error": f"{type(exc).__name__}: {exc}",
            "answer_head": "",
        }


async def lane(
    client: httpx.AsyncClient,
    model: str,
    deadline: float,
    results: list[dict[str, Any]],
    *,
    timeout_s: float,
) -> None:
    sequence = 0
    while time.monotonic() < deadline:
        expected = f"STRESS_{model.replace('-', '_').upper()}_{sequence}"
        results.append(await completion(client, model, expected, timeout_s=timeout_s))
        sequence += 1
        await asyncio.sleep(0.1)


async def health_lane(
    client: httpx.AsyncClient,
    deadline: float,
    snapshots: list[dict[str, Any]],
) -> None:
    while time.monotonic() < deadline:
        started = time.perf_counter()
        try:
            response = await client.get(f"{GATEWAY}/healthz", timeout=15)
            data = response.json()
            workers = data.get("workers") or []
            snapshots.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "status": response.status_code,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "down": [
                        worker.get("id")
                        for worker in workers
                        if worker.get("enabled") and worker.get("live") == "down"
                    ],
                }
            )
        except Exception as exc:
            snapshots.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "status": None,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "down": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        await asyncio.sleep(5)


def summarize(results: list[dict[str, Any]], snapshots: list[dict[str, Any]], duration: float) -> dict:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_model[row["model"]].append(row)
    lanes: dict[str, Any] = {}
    for model, rows in sorted(by_model.items()):
        latencies = [float(row["latency_ms"]) for row in rows if row["ok"]]
        output_tokens = sum(int(row["output_tokens"] or 0) for row in rows)
        lanes[model] = {
            "requests": len(rows),
            "successes": sum(bool(row["ok"]) for row in rows),
            "exact": sum(bool(row["exact"]) for row in rows),
            "errors": sum(not bool(row["ok"]) for row in rows),
            "success_rate": round(sum(bool(row["ok"]) for row in rows) / len(rows), 4),
            "exact_rate": round(sum(bool(row["exact"]) for row in rows) / len(rows), 4),
            "p50_ms": round(statistics.median(latencies), 2) if latencies else None,
            "p95_ms": round(percentile(latencies, 95) or 0, 2) if latencies else None,
            "max_ms": round(max(latencies), 2) if latencies else None,
            "output_tokens": output_tokens,
        }
    health_latencies = [float(row["latency_ms"]) for row in snapshots if row.get("status") == 200]
    return {
        "duration_s": round(duration, 2),
        "total_requests": len(results),
        "total_successes": sum(bool(row["ok"]) for row in results),
        "total_exact": sum(bool(row["exact"]) for row in results),
        "throughput_requests_s": round(len(results) / duration, 3),
        "lanes": lanes,
        "health": {
            "snapshots": len(snapshots),
            "errors": sum(row.get("status") != 200 for row in snapshots),
            "snapshots_with_down_workers": sum(bool(row.get("down")) for row in snapshots),
            "p95_ms": round(percentile(health_latencies, 95) or 0, 2)
            if health_latencies
            else None,
        },
    }


async def run(duration_s: int, output: Path) -> None:
    started = time.monotonic()
    deadline = started + duration_s
    results: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=16)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            asyncio.create_task(
                lane(client, model, deadline, results, timeout_s=240 if model == "harness-orch" else 60)
            )
            for model in (*DIRECT_MODELS, "harness-orch")
        ]
        tasks.append(asyncio.create_task(health_lane(client, deadline, snapshots)))
        await asyncio.gather(*tasks)
    elapsed = time.monotonic() - started
    payload = {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summarize(results, snapshots, elapsed),
        "health_snapshots": snapshots,
        "requests": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.duration, args.output))


if __name__ == "__main__":
    main()
