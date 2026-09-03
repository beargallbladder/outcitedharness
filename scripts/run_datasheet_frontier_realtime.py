#!/usr/bin/env python3
"""Execute a sealed datasheet teacher bundle through Anthropic Messages now.

Successful per-request receipts are immutable and reusable, so interruption
never repeats completed paid calls. The sealed output matches the existing
batch-results contract and can enter the same reconciliation and verification
pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from harness.electronics.claims import canonical_json
from harness.electronics.frontier_batch import load_prepared_bundle


_THREAD_LOCAL = threading.local()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    parser.add_argument("--base-url", default="https://api.anthropic.com")
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input-price-per-million", type=float, required=True)
    parser.add_argument("--output-price-per-million", type=float, required=True)
    parser.add_argument("--spend-cap-usd", type=float, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, value: bytes, *, mode: int = 0o400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise ValueError(f"immutable output already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _load_candidates(bundle: Path) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    with (bundle / "candidates.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            candidates[str(value["candidate_id"])] = value
    return candidates


def _request_sha(request: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(request))).hexdigest()


def _receipt_path(state: Path, custom_id: str) -> Path:
    if not custom_id.startswith("teach-") or not custom_id[6:].isalnum():
        raise ValueError(f"unsafe custom request ID: {custom_id!r}")
    return state / "receipts" / f"{custom_id}.json"


def _load_receipt(
    state: Path,
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = _receipt_path(state, str(request["custom_id"]))
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("request_sha256") != _request_sha(request):
        raise ValueError(f"real-time receipt request mismatch: {path}")
    result = value.get("result")
    if not isinstance(result, Mapping) or result.get("type") != "succeeded":
        raise ValueError(f"real-time success receipt is malformed: {path}")
    return value


def _client(
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
):
    client = getattr(_THREAD_LOCAL, "anthropic_client", None)
    if client is None:
        from anthropic import Anthropic

        client = Anthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )
        _THREAD_LOCAL.anthropic_client = client
    return client


def _execute(
    request: Mapping[str, Any],
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
    max_attempts: int,
) -> dict[str, Any]:
    started = time.monotonic()
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            message = _client(
                api_key=api_key,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            ).messages.create(**dict(request["params"]))
            return {
                "schema": "harness.electronics-frontier-realtime-receipt.v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "custom_id": request["custom_id"],
                "request_sha256": _request_sha(request),
                "attempt": attempt,
                "latency_seconds": time.monotonic() - started,
                "result": {
                    "type": "succeeded",
                    "message": message.model_dump(mode="json"),
                },
            }
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            last_error = exc
            if attempt < max_attempts:
                delay = min(60.0, 2.0**attempt) + random.random()
                time.sleep(delay)
    assert last_error is not None
    return {
        "schema": "harness.electronics-frontier-realtime-error.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "custom_id": request["custom_id"],
        "request_sha256": _request_sha(request),
        "attempt": max_attempts,
        "latency_seconds": time.monotonic() - started,
        "error": {
            "type": type(last_error).__name__,
            "message": str(last_error)[:2000],
        },
    }


def _maximum_cost(
    requests: list[dict[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    input_price: float,
    output_price: float,
) -> float:
    input_tokens = 0
    output_tokens = 0
    for request in requests:
        candidate_id = str(request["_harness"]["candidate_id"])
        candidate = candidates[candidate_id]
        input_tokens += int(candidate["estimated_input_tokens"])
        output_tokens += int(request["params"]["max_tokens"])
    return (
        input_tokens * input_price / 1_000_000
        + output_tokens * output_price / 1_000_000
    )


def _seal_results(
    output: Path,
    receipts: list[dict[str, Any]],
    *,
    run_id: str,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        manifest = output / "manifest.json"
        if not manifest.is_file():
            raise ValueError(f"partial real-time result output exists: {output}")
        return json.loads(manifest.read_text(encoding="utf-8"))
    output.mkdir(parents=True, mode=0o700)
    try:
        rows = [
            {
                "custom_id": receipt["custom_id"],
                "result": receipt["result"],
            }
            for receipt in receipts
        ]
        payload = b"".join(canonical_json(row) + b"\n" for row in rows)
        result_name = f"{run_id}.jsonl"
        result_path = output / result_name
        _write_new(result_path, payload, mode=0o400)
        core = {
            "schema": "harness.electronics-frontier-results.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "local_training_pair_generation",
            "transport": "anthropic_messages_realtime",
            "batches": [
                {
                    "batch_id": run_id,
                    "result_file": result_name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "api_status": {
                        "processing_status": "ended",
                        "request_counts": {
                            "succeeded": len(rows),
                            "errored": 0,
                        },
                    },
                }
            ],
        }
        core["evidence_sha256"] = hashlib.sha256(
            canonical_json(core)
        ).hexdigest()
        _write_new(
            output / "manifest.json",
            json.dumps(
                core,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n",
            mode=0o400,
        )
        os.chmod(output, 0o500)
        return core
    except BaseException:
        print(f"partial real-time output retained: {output}", flush=True)
        raise


def main() -> int:
    args = _parser().parse_args()
    if not 1 <= args.concurrency <= 32:
        raise ValueError("--concurrency must be within 1..32")
    if not 1 <= args.max_attempts <= 10:
        raise ValueError("--max-attempts must be within 1..10")
    if args.offset < 0 or (args.limit is not None and args.limit < 1):
        raise ValueError("invalid request slice")
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise ValueError(f"missing ${args.api_key_env}")

    bundle = args.bundle.expanduser().resolve(strict=True)
    manifest, all_requests = load_prepared_bundle(bundle)
    stop = (
        len(all_requests)
        if args.limit is None
        else min(len(all_requests), args.offset + args.limit)
    )
    requests = all_requests[args.offset:stop]
    if not requests:
        raise ValueError("real-time request selection is empty")
    candidates = _load_candidates(bundle)
    maximum_cost = _maximum_cost(
        requests,
        candidates,
        input_price=args.input_price_per_million,
        output_price=args.output_price_per_million,
    )
    if maximum_cost > args.spend_cap_usd:
        raise ValueError(
            f"maximum real-time cost ${maximum_cost:.6f} exceeds "
            f"${args.spend_cap_usd:.6f} cap"
        )

    state = args.state_directory.expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    if state.is_symlink():
        raise ValueError("real-time state directory cannot be a symlink")
    os.chmod(state, 0o700)
    run_id = (
        "realtime-"
        + manifest["evidence_sha256"][:16]
        + f"-{args.offset:06d}-{len(requests):06d}"
    )
    intent_core = {
        "schema": "harness.electronics-frontier-realtime-intent.v1",
        "run_id": run_id,
        "prepared_evidence_sha256": manifest["evidence_sha256"],
        "offset": args.offset,
        "requests": len(requests),
        "concurrency": args.concurrency,
        "maximum_cost_usd": maximum_cost,
        "spend_cap_usd": args.spend_cap_usd,
    }
    intent_core["evidence_sha256"] = hashlib.sha256(
        canonical_json(intent_core)
    ).hexdigest()
    intent = state / "intent.json"
    if intent.is_file():
        existing = json.loads(intent.read_text(encoding="utf-8"))
        if existing != intent_core:
            raise ValueError("real-time intent conflicts with existing state")
    else:
        _write_new(
            intent,
            json.dumps(
                intent_core,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ).encode("utf-8")
            + b"\n",
            mode=0o400,
        )

    receipts: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for request in requests:
        receipt = _load_receipt(state, request)
        if receipt is None:
            pending.append(request)
        else:
            receipts[str(request["custom_id"])] = receipt
    print(
        json.dumps(
            {
                "run_id": run_id,
                "requests": len(requests),
                "resumed": len(receipts),
                "pending": len(pending),
                "maximum_cost_usd": maximum_cost,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    errors: list[dict[str, Any]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _execute,
                request,
                api_key=api_key,
                base_url=args.base_url,
                timeout_seconds=args.timeout_seconds,
                max_attempts=args.max_attempts,
            ): request
            for request in pending
        }
        for future in as_completed(futures):
            request = futures[future]
            value = future.result()
            if value.get("result", {}).get("type") == "succeeded":
                _write_new(
                    _receipt_path(state, str(request["custom_id"])),
                    canonical_json(value) + b"\n",
                    mode=0o400,
                )
                receipts[str(request["custom_id"])] = value
            else:
                errors.append(value)
                error_path = (
                    state
                    / "errors"
                    / f"{request['custom_id']}-{int(time.time())}.json"
                )
                _write_new(
                    error_path,
                    canonical_json(value) + b"\n",
                    mode=0o400,
                )
            completed = len(receipts)
            elapsed = max(0.001, time.monotonic() - started)
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(requests),
                        "errors_this_run": len(errors),
                        "requests_per_minute": 60.0
                        * (completed - (len(requests) - len(pending)))
                        / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if errors or len(receipts) != len(requests):
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "completed": len(receipts),
                    "errors": Counter(
                        value["error"]["type"] for value in errors
                    ),
                },
                default=dict,
                sort_keys=True,
            ),
            flush=True,
        )
        return 1

    ordered = [receipts[str(request["custom_id"])] for request in requests]
    output = args.output_directory.expanduser().resolve()
    result_manifest = _seal_results(output, ordered, run_id=run_id)
    usage = Counter()
    for receipt in ordered:
        message_usage = receipt["result"]["message"].get("usage") or {}
        usage["input_tokens"] += int(message_usage.get("input_tokens") or 0)
        usage["output_tokens"] += int(message_usage.get("output_tokens") or 0)
    actual_cost = (
        usage["input_tokens"] * args.input_price_per_million / 1_000_000
        + usage["output_tokens"] * args.output_price_per_million / 1_000_000
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "run_id": run_id,
                "requests": len(requests),
                "usage": dict(usage),
                "actual_cost_usd": actual_cost,
                "results_evidence_sha256": result_manifest["evidence_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
