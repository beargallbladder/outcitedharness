#!/usr/bin/env python3
"""Audit whether DesignWins prompt/response pairs fit the training context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from evaluate_designwins_text import load_records


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoff", type=int, action="append", default=[])
    parser.add_argument("--fail-on-truncation", action="store_true")
    args = parser.parse_args()
    cutoffs = sorted(set(args.cutoff or [4096, 8192, 12288]))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
    )
    rows = []
    for index, record in enumerate(load_records(args.dataset)):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(record["instruction"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = len(
            tokenizer(rendered, add_special_tokens=False)["input_ids"]
        )
        response_tokens = len(
            tokenizer(
                str(record["output"]), add_special_tokens=False
            )["input_ids"]
        ) + 1
        rows.append(
            {
                "index": index,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": prompt_tokens + response_tokens,
            }
        )
    totals = [row["total_tokens"] for row in rows]
    responses = [row["response_tokens"] for row in rows]
    result = {
        "schema": "harness.designwins-sequence-length-audit.v1",
        "dataset_sha256": _sha256(args.dataset),
        "records": len(rows),
        "total_tokens": {
            "p50": _percentile(totals, 0.5),
            "p90": _percentile(totals, 0.9),
            "p95": _percentile(totals, 0.95),
            "p99": _percentile(totals, 0.99),
            "max": max(totals),
        },
        "response_tokens": {
            "p50": _percentile(responses, 0.5),
            "p90": _percentile(responses, 0.9),
            "p95": _percentile(responses, 0.95),
            "p99": _percentile(responses, 0.99),
            "max": max(responses),
        },
        "cutoffs": {
            str(cutoff): {
                "truncated_records": sum(
                    row["total_tokens"] > cutoff for row in rows
                ),
                "truncated_rate": (
                    sum(row["total_tokens"] > cutoff for row in rows) / len(rows)
                ),
            }
            for cutoff in cutoffs
        },
        "details": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({key: value for key, value in result.items() if key != "details"}))
    if args.fail_on_truncation:
        required_cutoff = min(cutoffs)
        if result["cutoffs"][str(required_cutoff)]["truncated_records"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
