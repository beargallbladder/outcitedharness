#!/usr/bin/env python3
"""Verify repeatability of the corrected Qwen3.8 QSA path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx


BASE_URL = os.environ.get("QWEN38_BASE_URL", "http://100.68.133.1:8888/v1")
URL = f"{BASE_URL.rstrip('/')}/chat/completions"
MODEL = os.environ.get("QWEN38_MODEL", "qwen38-flash-next-nvfp4")
REQUIRE_LOGIT_STABILITY = os.environ.get(
    "QWEN38_REQUIRE_LOGIT_STABILITY", "0"
).lower() in {"1", "true", "yes"}
OUTPUT = Path(
    os.environ.get(
        "QWEN38_PROBE_OUTPUT",
        "results/qwen38_determinism_probe_20260828.json",
    )
)


def _prompt() -> str:
    records = " ".join(
        f"record_{index:05d}=value_{(index * 7919) % 104729:06d}"
        for index in range(1000)
    )
    return (
        "Read these records. Reply only with the value of record_00731.\n" + records
    )


def _request(client: httpx.Client, max_tokens: int, logprobs: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": _prompt()}],
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "preserve_thinking": False,
        },
    }
    if logprobs:
        payload.update({"logprobs": True, "top_logprobs": 20})
    response = client.post(URL, json=payload)
    response.raise_for_status()
    return response.json()


def main() -> None:
    with httpx.Client(timeout=300) as client:
        token_runs = [_request(client, 1, True) for _ in range(10)]
        answer_runs = [_request(client, 64, False) for _ in range(3)]
    top_sets = [
        run["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        for run in token_runs
    ]
    top_fingerprints = [
        json.dumps(value, sort_keys=True, separators=(",", ":")) for value in top_sets
    ]
    token_orders = [
        tuple(str(candidate["token"]) for candidate in value) for value in top_sets
    ]
    baseline = {
        str(candidate["token"]): float(candidate["logprob"]) for candidate in top_sets[0]
    }
    max_logprob_drift = max(
        abs(float(candidate["logprob"]) - baseline[str(candidate["token"])])
        for value in top_sets[1:]
        for candidate in value
        if str(candidate["token"]) in baseline
    )
    answers = [
        str(run["choices"][0]["message"].get("content") or "")
        for run in answer_runs
    ]
    greedy_output_deterministic = len(set(answers)) == 1
    logit_rank_deterministic = len(set(token_orders)) == 1
    expected = f"value_{(731 * 7919) % 104729:06d}"
    answer_correct = bool(answers) and expected in answers[0]
    qualified = (
        greedy_output_deterministic
        and answer_correct
        and (logit_rank_deterministic or not REQUIRE_LOGIT_STABILITY)
    )
    payload = {
        "first_token_top20_unique": len(set(top_fingerprints)),
        "first_token_top20_order_unique": len(set(token_orders)),
        "logprob_values_exact": len(set(top_fingerprints)) == 1,
        "logit_rank_deterministic": logit_rank_deterministic,
        "max_logprob_drift": max_logprob_drift,
        "full_answer_unique": len(set(answers)),
        "greedy_output_deterministic": greedy_output_deterministic,
        "deterministic": greedy_output_deterministic,
        "strict_deterministic": (
            greedy_output_deterministic and logit_rank_deterministic
        ),
        "require_logit_stability": REQUIRE_LOGIT_STABILITY,
        "qualified": qualified,
        "answer": answers[0] if answers else "",
        "expected": expected,
        "answer_correct": answer_correct,
        "token_usage": [run.get("usage") for run in token_runs],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not qualified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
