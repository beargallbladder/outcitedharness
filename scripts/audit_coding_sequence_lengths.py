#!/usr/bin/env python3
"""Prove that every coding SFT response fits the configured token cutoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "harness.coding-sequence-length-audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)]


def _load(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("coding dataset must be a regular file")
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not value:
        raise ValueError("coding dataset must be a non-empty JSON array")
    for row in value:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("messages"), list)
            or len(row["messages"]) != 2
        ):
            raise ValueError("coding dataset contains a malformed conversation")
    return value


def audit(
    *,
    model: Path,
    dataset: Path,
    cutoff: int,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if cutoff < 256:
        raise ValueError("coding cutoff must be at least 256")
    rows = _load(dataset)
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=True,
        trust_remote_code=True,
    )
    details = []
    for index, row in enumerate(rows):
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_tokens = len(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        total_tokens = len(
            tokenizer(full, add_special_tokens=False)["input_ids"]
        )
        details.append(
            {
                "index": index,
                "prompt_tokens": prompt_tokens,
                "response_tokens": total_tokens - prompt_tokens,
                "total_tokens": total_tokens,
                "fits": total_tokens <= cutoff,
            }
        )
    totals = [row["total_tokens"] for row in details]
    responses = [row["response_tokens"] for row in details]
    return {
        "schema": SCHEMA,
        "model_config_sha256": _sha256(model / "config.json"),
        "dataset_sha256": _sha256(dataset),
        "records": len(details),
        "cutoff_len": cutoff,
        "truncated_records": sum(not row["fits"] for row in details),
        "total_tokens": {
            "p50": _percentile(totals, 0.50),
            "p95": _percentile(totals, 0.95),
            "max": max(totals),
        },
        "response_tokens": {
            "p50": _percentile(responses, 0.50),
            "p95": _percentile(responses, 0.95),
            "max": max(responses),
        },
        "details": details,
    }


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("sequence audit output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.chmod(path, 0o444)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--cutoff", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = audit(
        model=arguments.model,
        dataset=arguments.dataset,
        cutoff=arguments.cutoff,
    )
    _write_once(arguments.output, result)
    summary = {key: value for key, value in result.items() if key != "details"}
    print(json.dumps(summary, sort_keys=True))
    return 1 if result["truncated_records"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
