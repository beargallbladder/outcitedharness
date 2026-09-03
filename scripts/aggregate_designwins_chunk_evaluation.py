#!/usr/bin/env python3
"""Aggregate bounded DesignWins chunk generations into parent-case scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from evaluate_designwins_text import extract_json, score_json, summarize_details


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pins(response: str) -> list[dict[str, Any]]:
    value = json.loads(response)
    pins = value.get("pins") if isinstance(value, dict) else None
    if not isinstance(pins, list) or any(not isinstance(pin, dict) for pin in pins):
        raise ValueError("expected response has no pin list")
    return pins


def _physical_identity(pin: dict[str, Any]) -> tuple[str, str]:
    return (
        str(pin.get("name") or ""),
        json.dumps(
            pin.get("pin_no"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def aggregate(
    raw_evaluation: dict[str, Any],
    chunks: list[dict[str, Any]],
    parents: list[dict[str, Any]],
) -> dict[str, Any]:
    details = raw_evaluation.get("details")
    if not isinstance(details, list) or len(details) != len(chunks):
        raise ValueError("raw evaluation and chunk dataset counts differ")
    if [row.get("index") for row in details] != list(range(len(chunks))):
        raise ValueError("raw chunk indices are not ordered and complete")
    parent_order = [str(row.get("pair_id") or "") for row in parents]
    if any(not pair_id for pair_id in parent_order):
        raise ValueError("parent dataset contains an empty pair identity")
    if len(parent_order) != len(set(parent_order)):
        raise ValueError("parent pair identities are not unique")
    parent_by_id = {str(row["pair_id"]): row for row in parents}
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for chunk, detail in zip(chunks, details, strict=True):
        metadata = chunk.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("chunk lacks metadata")
        parent_id = str(metadata.get("parent_pair_id") or "")
        if parent_id not in parent_by_id:
            raise ValueError(f"chunk references unknown parent {parent_id!r}")
        grouped.setdefault(parent_id, []).append((chunk, detail))

    parent_details: list[dict[str, Any]] = []
    original_pin_total = 0
    evaluated_pin_total = 0
    for index, parent_id in enumerate(parent_order):
        rows = grouped.get(parent_id)
        if not rows:
            raise ValueError(f"parent {parent_id!r} has no evaluation chunks")
        rows.sort(key=lambda row: int(row[0]["metadata"]["chunk_index"]))
        chunk_indices = [int(row[0]["metadata"]["chunk_index"]) for row in rows]
        declared_counts = {
            int(row[0]["metadata"]["chunk_count"]) for row in rows
        }
        if (
            chunk_indices != list(range(len(rows)))
            or declared_counts != {len(rows)}
        ):
            raise ValueError(f"parent {parent_id!r} has incomplete chunks")
        expected_pins = [
            pin
            for chunk, _detail in rows
            for pin in _pins(str(chunk["response"]))
        ]
        if len({_physical_identity(pin) for pin in expected_pins}) != len(
            expected_pins
        ):
            raise ValueError("expected chunks duplicate a physical pin")
        original_pins = _pins(str(parent_by_id[parent_id]["response"]))
        original_pin_total += len(original_pins)
        evaluated_pin_total += len(expected_pins)

        parsed = [extract_json(str(detail.get("response") or "")) for _, detail in rows]
        valid_chunks = all(
            isinstance(value, dict) and isinstance(value.get("pins"), list)
            for value in parsed
        )
        predicted = (
            {
                "pins": [
                    pin
                    for value in parsed
                    for pin in value["pins"]
                    if isinstance(pin, dict)
                ]
            }
            if valid_chunks
            else None
        )
        expected = {"pins": expected_pins}
        metadata = rows[0][0]["metadata"]
        part = str(metadata.get("part") or parent_id)
        parent_details.append(
            {
                "index": index,
                "part": part,
                "family": part.split("_", 1)[0].casefold(),
                "parent_pair_id": parent_id,
                "chunk_count": len(rows),
                "invalid_chunks": sum(
                    not (
                        isinstance(value, dict)
                        and isinstance(value.get("pins"), list)
                    )
                    for value in parsed
                ),
                "expected_tokens": sum(
                    int(detail.get("expected_tokens") or 0)
                    for _chunk, detail in rows
                ),
                "generated_tokens": sum(
                    int(detail.get("generated_tokens") or 0)
                    for _chunk, detail in rows
                ),
                "generation_budget": sum(
                    int(detail.get("generation_budget") or 0)
                    for _chunk, detail in rows
                ),
                "hit_generation_limit": any(
                    bool(detail.get("hit_generation_limit"))
                    for _chunk, detail in rows
                ),
                "score": score_json(expected, predicted),
                "response": (
                    json.dumps(predicted, ensure_ascii=False, sort_keys=True)
                    if predicted is not None
                    else ""
                ),
            }
        )
    return {
        "schema": "harness.designwins-chunk-aggregation.v1",
        "model": raw_evaluation.get("model"),
        "adapter": raw_evaluation.get("adapter"),
        "dataset": raw_evaluation.get("dataset"),
        "summary": summarize_details(parent_details),
        "coverage": {
            "parent_cases": len(parent_details),
            "original_physical_pins": original_pin_total,
            "evaluated_grounded_physical_pins": evaluated_pin_total,
            "excluded_physical_pins": original_pin_total - evaluated_pin_total,
            "evaluated_pin_rate": (
                evaluated_pin_total / original_pin_total
                if original_pin_total
                else 0
            ),
        },
        "details": parent_details,
    }


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-evaluation", required=True, type=Path)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--parents", required=True, type=Path)
    parser.add_argument("--generation-scorer", required=True, type=Path)
    parser.add_argument("--chunk-artifact-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = json.loads(args.raw_evaluation.read_text(encoding="utf-8"))
    result = aggregate(raw, _load_jsonl(args.chunks), _load_jsonl(args.parents))
    result["identity_inputs"] = {
        "raw_evaluation_sha256": _sha256(args.raw_evaluation),
        "chunks_sha256": _sha256(args.chunks),
        "parents_sha256": _sha256(args.parents),
        "aggregator_sha256": _sha256(Path(__file__)),
        "generation_scorer_sha256": _sha256(args.generation_scorer),
        "chunk_artifact_manifest_sha256": _sha256(
            args.chunk_artifact_manifest
        ),
    }
    _write_once(args.output, result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
