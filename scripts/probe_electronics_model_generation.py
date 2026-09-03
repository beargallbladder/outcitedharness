#!/usr/bin/env python3
"""Run and seal one grounded multimodal generation sanity probe."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx


SCHEMA = "harness.electronics-model-generation-sanity.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("probe dataset must contain a non-empty JSON list")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError("probe dataset rows must be JSON objects")
    return value


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
            if stripped.lstrip().startswith("json"):
                stripped = stripped.lstrip()[4:].lstrip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model response contains no JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("model response JSON must be an object")
    return value


def _pin_identities(value: dict[str, Any]) -> set[tuple[str, str]]:
    pins = value.get("pins")
    if not isinstance(pins, list):
        raise ValueError("probe response must contain a pins list")
    identities: set[tuple[str, str]] = set()
    for pin in pins:
        if not isinstance(pin, dict):
            raise ValueError("every probe pin must be an object")
        number = str(pin.get("pin_no") or "").strip().casefold()
        name = str(pin.get("name") or "").strip().casefold()
        if not number or not name:
            raise ValueError("every probe pin requires pin_no and name")
        identities.add((number, name))
    return identities


def _score(
    predicted: set[tuple[str, str]],
    expected: set[tuple[str, str]],
) -> dict[str, float | int]:
    matched = len(predicted & expected)
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "predicted": len(predicted),
        "expected": len(expected),
        "matched": matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise ValueError(f"immutable probe output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode()
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def probe(
    *,
    base_url: str,
    model: str,
    dataset: Path,
    row_index: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    dataset = dataset.expanduser().resolve(strict=True)
    rows = _load_rows(dataset)
    if not 0 <= row_index < len(rows):
        raise ValueError("row index is outside the probe dataset")
    row = rows[row_index]
    messages = row.get("messages")
    images = row.get("images")
    if not isinstance(messages, list) or not isinstance(images, list):
        raise ValueError("probe row requires messages and images")
    if len(images) != 1:
        raise ValueError("generation sanity requires exactly one image")
    user_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    teacher_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if len(user_messages) != 1 or len(teacher_messages) != 1:
        raise ValueError("probe row requires one user and one teacher message")
    prompt = str(user_messages[0].get("content") or "")
    if prompt.count("<image>") != 1:
        raise ValueError("probe prompt must contain one image placeholder")
    prompt = prompt.replace("<image>", "", 1).lstrip()
    image = (dataset.parent / str(images[0])).resolve(strict=True)
    image.relative_to(dataset.parent.parent)
    image_bytes = image.read_bytes()
    teacher = _parse_json_object(str(teacher_messages[0].get("content") or ""))
    expected = _pin_identities(teacher)

    request = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract only source-grounded electronics facts. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                + base64.b64encode(image_bytes).decode("ascii")
                            )
                        },
                    },
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }
    started = time.perf_counter()
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            base_url.rstrip("/") + "/chat/completions",
            json=request,
        )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    body = response.json()
    try:
        content = str(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("model response has no assistant content") from error
    parsed = _parse_json_object(content)
    predicted = _pin_identities(parsed)
    metrics = _score(predicted, expected)
    passed = metrics["matched"] > 0
    core = {
        "schema": SCHEMA,
        "passed": passed,
        "model": model,
        "dataset": str(dataset),
        "row_index": row_index,
        "prompt_sha256": _sha256_bytes(prompt.encode()),
        "image_sha256": _sha256_bytes(image_bytes),
        "teacher_sha256": _sha256_bytes(_canonical(teacher)),
        "request_sha256": _sha256_bytes(_canonical(request)),
        "response_sha256": _sha256_bytes(content.encode()),
        "latency_ms": latency_ms,
        "metrics": metrics,
        "usage": body.get("usage"),
        "response": parsed,
    }
    core["evidence_sha256"] = _sha256_bytes(_canonical(core))
    if not passed:
        raise ValueError("generation sanity produced no grounded pin identity")
    return core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = probe(
        base_url=args.base_url,
        model=args.model,
        dataset=args.dataset,
        row_index=args.row_index,
        timeout_seconds=args.timeout_seconds,
    )
    _write_new(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
